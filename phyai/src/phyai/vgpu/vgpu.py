"""vGPU class + ``create_vgpus`` syntax sugar.

A :class:`vGPU` bundles a long-lived green-ctx :class:`Shard` with an
optional per-vGPU :class:`torch.cuda.MemPool`, and exposes a single
``activate()`` context manager that puts both into effect for the calling
thread.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch

from phyai.utils.cuda import device_capability
from phyai.vgpu.backend import get_backend, resolve
from phyai.vgpu.partition import Shard, split_device_by_sm_count, to_shard
from phyai.vgpu.topology import round_up_sm_count, validate_total


class vGPU:
    """A long-lived compute slice of a CUDA device.

    A vGPU bundles:
      - one :class:`Shard` (a green-ctx with its bound non-blocking
        :class:`torch.cuda.Stream`), and
      - optionally a per-vGPU :class:`torch.cuda.MemPool` for fragmentation
        isolation between concurrently-running models.

    .. warning::

       vGPUs **must be long-lived**. flashinfer's
       ``split_device_green_ctx`` leaks driver memory on every call;
       ``cuStreamDestroy + cuGreenCtxDestroy`` plus ``empty_cache``
       recover none of it. The leak is upstream — likely in flashinfer or
       the CUDA driver layer — so the defensive posture in phyai is to
       keep vGPU objects long-lived. Creating and destroying vGPUs on a
       request hot path will exhaust device memory.
    """

    name: str
    shard: Shard
    stream: torch.cuda.Stream
    mem_pool: "torch.cuda.MemPool | None"

    def __init__(
        self,
        *,
        name: str,
        sm_count: int,
        own_mem_pool: bool = True,
        device: "str | torch.device" = "cuda:0",
        backend: str | None = None,
    ) -> None:
        dev = torch.device(device)
        cc = device_capability(dev)
        rounded = round_up_sm_count(sm_count, cc)
        total_sms = torch.cuda.get_device_properties(dev).multi_processor_count
        validate_total([rounded], total_sms)
        b = resolve(backend) if backend is not None else get_backend()
        spec = b.create_single(dev, rounded)
        self.shard = to_shard(
            spec,
            name=name,
            requested=sm_count,
            device=dev,
        )
        self.name = name
        self.stream = self.shard.stream
        self.mem_pool = torch.cuda.MemPool() if own_mem_pool else None
        self._closed = False
        from phyai.vgpu.state import register_vgpu

        register_vgpu(self)

    @classmethod
    def from_shard(
        cls,
        shard: Shard,
        *,
        name: str | None = None,
        own_mem_pool: bool = True,
    ) -> "vGPU":
        """Build a :class:`vGPU` around an existing :class:`Shard`.

        Useful when callers want to control the split themselves and then
        attach a MemPool to each shard.
        """
        obj = cls.__new__(cls)
        obj.shard = shard
        obj.name = name or shard.name
        obj.stream = shard.stream
        obj.mem_pool = torch.cuda.MemPool() if own_mem_pool else None
        obj._closed = False
        from phyai.vgpu.state import register_vgpu

        register_vgpu(obj)
        return obj

    @contextmanager
    def activate(self) -> Iterator["vGPU"]:
        """Enter the vGPU's stream + mem-pool scope.

        ``torch.cuda.use_mem_pool`` is documented as thread-local; child
        threads do not inherit and each thread that wants to dispatch into
        this vGPU must call ``activate()`` itself.
        """
        if self._closed:
            raise RuntimeError(f"vGPU {self.name!r} is closed")
        with torch.cuda.stream(self.stream):
            if self.mem_pool is not None:
                with torch.cuda.use_mem_pool(self.mem_pool):
                    yield self
            else:
                yield self

    def close(self) -> None:
        """Release backend resources. Idempotent.

        Order: synchronise stream -> backend ``destroy`` -> drop pool ref.
        Sync ensures no enqueued op is still in flight when the green ctx
        goes away; the MemPool reclaims its segments once the allocator
        observes its refcount drop to zero (live tensors keep it alive
        regardless).
        """
        if self._closed:
            return
        self._closed = True
        try:
            self.stream.synchronize()
        except Exception:
            pass
        try:
            get_backend().destroy(self.shard._spec)
        except Exception:
            pass
        self.mem_pool = None


def create_vgpus(
    *,
    device: "str | torch.device",
    sm_counts: list[int],
    names: list[str] | None = None,
    own_mem_pool: bool = True,
    include_remainder_vgpu: bool = False,
    backend: str | None = None,
) -> list[vGPU]:
    """Convenience wrapper: split + :meth:`vGPU.from_shard` for each shard.

    .. warning::

       Long-lived usage only. flashinfer's per-call driver leak means
       re-creating vGPUs in a request hot path will exhaust device
       memory; see :class:`vGPU` docstring.

    Args:
        device: CUDA device (string or :class:`torch.device`).
        sm_counts: SM count requested per non-remainder shard.
        names: Optional per-vGPU names; defaults to ``["vgpu_0", ...]``.
        own_mem_pool: If True (the default), allocate a dedicated
            :class:`torch.cuda.MemPool` per vGPU.
        include_remainder_vgpu: If True, return one extra vGPU representing
            the remainder shard (whatever SMs flashinfer didn't allocate).
            Default False — the remainder is destroyed.
        backend: Optional backend override; default is the process-level
            backend.

    Returns:
        ``len(sm_counts)`` vGPUs (or ``len(sm_counts) + 1`` when
        ``include_remainder_vgpu=True``).
    """
    shards = split_device_by_sm_count(
        device,
        sm_counts=sm_counts,
        backend=backend,
    )
    if not include_remainder_vgpu:
        rem = [s for s in shards if s.is_remainder]
        shards = [s for s in shards if not s.is_remainder]
        b = resolve(backend) if backend is not None else get_backend()
        for r in rem:
            try:
                r.stream.synchronize()
            except Exception:
                pass
            try:
                b.destroy(r._spec)
            except Exception:
                pass
    if names is None:
        names = [f"vgpu_{i}" for i in range(len(shards))]
    elif len(names) != len(shards):
        raise ValueError(
            f"create_vgpus: names length ({len(names)}) must match number "
            f"of resulting vGPUs ({len(shards)}); "
            f"got sm_counts={sm_counts}, "
            f"include_remainder_vgpu={include_remainder_vgpu}"
        )
    return [
        vGPU.from_shard(s, name=names[i], own_mem_pool=own_mem_pool)
        for i, s in enumerate(shards)
    ]
