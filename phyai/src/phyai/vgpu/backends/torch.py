"""Torch-native :class:`GreenCtxBackend` (single-shard fallback).

torch's ``GreenContext.create`` only models a single shard at a time; ATen
exposes no public split / remainder API. Two back-to-back
``GreenContext.create`` calls produce streams that nearly serialise on
the same SMs, so this backend deliberately refuses ``split_by_*`` rather
than handing out non-disjoint shards.

For the single-shard path:
  - torch 2.11+: use ``g.Stream()`` directly (returns an ExternalStream
    bound to the green ctx).
  - torch 2.10: the pybind binding for ``Stream()`` is missing, so fall
    back to ``set_context -> torch.cuda.Stream -> pop_context``.
"""

from __future__ import annotations

import torch
from torch.cuda.green_contexts import GreenContext

from phyai.vgpu._spec import ShardSpec
from phyai.vgpu.backend import register_vgpu_backend
from phyai.vgpu.exceptions import BackendCapabilityError


@register_vgpu_backend("torch")
class TorchBackend:
    """Single-shard backend backed by ``torch.cuda.green_contexts``."""

    name = "torch"

    def create_single(
        self,
        device: torch.device,
        num_sms: int,
    ) -> ShardSpec:
        idx = device.index if device.index is not None else torch.cuda.current_device()
        g = GreenContext.create(num_sms=num_sms, device_id=idx)
        try:
            stream = g.Stream()
        except AttributeError:
            # torch 2.10 binding gap — fall back to the stack workaround.
            g.set_context()
            try:
                stream = torch.cuda.Stream(device=device)
            finally:
                g.pop_context()
        return ShardSpec(
            stream=stream,
            sm_count=int(num_sms),
            is_remainder=False,
            _backend_handle=g,
            _backend_name="torch",
        )

    def split_by_count(
        self,
        device: torch.device,
        num_groups: int,
        min_count: int,
    ) -> list[ShardSpec]:
        raise BackendCapabilityError(
            "torch backend does not support split_by_count "
            "(multiple GreenContext.create calls may overlap on the same "
            "SMs and serialise rather than running disjoint), "
            "and there is no remainder to return. "
            "Use backend='flashinfer' for splits; "
            "use vGPU(sm_count=...) for a single shard."
        )

    def split_by_sm_counts(
        self,
        device: torch.device,
        sm_counts: list[int],
    ) -> list[ShardSpec]:
        raise BackendCapabilityError(
            "torch backend does not support split_by_sm_counts (same reason "
            "as split_by_count). Use backend='flashinfer' for splits; "
            "use vGPU(sm_count=...) for a single shard."
        )

    def destroy(self, spec: ShardSpec) -> None:
        """Best-effort: drop our reference to the GreenContext.

        ATen's ``GreenContext`` destructor handles native cleanup when its
        Python refcount drops; the stream attached via ``g.Stream()`` is an
        ExternalStream owned by the GreenContext.
        """
        # Nothing to do here — the spec going out of scope releases ``g``.
        return None
