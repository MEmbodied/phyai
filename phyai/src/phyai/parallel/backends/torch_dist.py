"""NcclBackend — ``torch.distributed.*`` for NCCL process groups.

Handles all 8 primitives in eager and graph-capture modes.

Scope:
  - Only registers ``can_handle`` = True when the process group's backend
    is NCCL (i.e., ``dist.get_backend(pg) == "nccl"``).
  - Uses NCCL-optimal "tensor" variants (``all_gather_into_tensor``,
    ``reduce_scatter_tensor``) — they avoid the per-rank Python-list
    overhead of the list-based equivalents.

CPU / gloo paths live in ``GlooBackend`` (sibling file).

Backend convention:
  - ``execute()`` writes results into a caller-allocated ``output``
    tensor (when applicable) and returns it.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from phyai.parallel.backend import Op, Topology
from phyai.parallel.state import Mode


def _backend_of(pg: dist.ProcessGroup) -> str:
    try:
        return dist.get_backend(pg)
    except Exception:
        return "?"


class NcclBackend:
    """Backend for NCCL process groups."""

    name = "nccl"

    _OPS = {
        Op.ALL_REDUCE,
        Op.ALL_GATHER,
        Op.REDUCE_SCATTER,
        Op.BROADCAST,
        Op.ALL_TO_ALL,
        Op.SEND,
        Op.RECV,
        Op.BARRIER,
    }

    def __init__(self) -> None:
        # Op -> bound method dispatch table. Built per-instance so the
        # bound methods capture self.
        self._handlers: dict[Op, callable] = {
            Op.ALL_REDUCE: self._all_reduce,
            Op.ALL_GATHER: self._all_gather,
            Op.REDUCE_SCATTER: self._reduce_scatter,
            Op.BROADCAST: self._broadcast,
            Op.ALL_TO_ALL: self._all_to_all,
            Op.SEND: self._send,
            Op.RECV: self._recv,
            Op.BARRIER: self._barrier,
        }

    def can_handle(
        self,
        *,
        op: Op,
        mode: Mode,
        nbytes: int,
        dtype: torch.dtype,
        world_size: int,
        topology: Topology,
        pg: dist.ProcessGroup | None = None,
        **extra: object,
    ) -> bool:
        if op not in self._OPS:
            return False
        # Without a PG handle we can't be 100% sure, but at probe time the
        # registry pass `pg=None`. Be permissive there; the per-call
        # execute path will still use NCCL APIs only on NCCL PGs.
        if pg is not None and _backend_of(pg) != dist.Backend.NCCL:
            return False
        return True

    def supports_capture(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # execute
    # ------------------------------------------------------------------

    def execute(
        self,
        *,
        op: Op,
        pg: dist.ProcessGroup,
        **kwargs,
    ) -> torch.Tensor | None:
        handler = self._handlers.get(op)
        if handler is None:
            raise NotImplementedError(f"NcclBackend.execute: op={op}")
        return handler(pg, **kwargs)

    # ------------------------------------------------------------------
    # primitives — NCCL-optimal tensor variants
    # ------------------------------------------------------------------

    def _all_reduce(self, pg, *, input, output, reduce_op, **_):
        output.copy_(input)
        dist.all_reduce(output, op=reduce_op, group=pg)
        return output

    def _all_gather(self, pg, *, input, output, dim, **_):
        x = input.contiguous()
        if dim == 0:
            dist.all_gather_into_tensor(output, x, group=pg)
            return output
        ws = dist.get_world_size(group=pg)
        stacked = torch.empty(
            (ws,) + tuple(x.shape),
            dtype=x.dtype,
            device=x.device,
        )
        dist.all_gather_into_tensor(stacked, x, group=pg)
        out = stacked.movedim(0, dim).contiguous().flatten(dim, dim + 1)
        output.copy_(out)
        return output

    def _reduce_scatter(self, pg, *, input, output, dim, reduce_op, **_):
        x = input.contiguous()
        if dim == 0:
            dist.reduce_scatter_tensor(output, x, op=reduce_op, group=pg)
            return output
        ws = dist.get_world_size(group=pg)
        chunks = [c.contiguous() for c in x.chunk(ws, dim=dim)]
        dist.reduce_scatter(output, chunks, op=reduce_op, group=pg)
        return output

    def _broadcast(self, pg, *, input, output, src, **_):
        output.copy_(input)
        dist.broadcast(output, src=src, group=pg)
        return output

    def _all_to_all(self, pg, *, input, output, in_splits, out_splits, **_):
        x = input.contiguous()
        dist.all_to_all_single(
            output,
            x,
            output_split_sizes=out_splits,
            input_split_sizes=in_splits,
            group=pg,
        )
        return output

    def _send(self, pg, *, input, dst, **_):
        dist.send(input.contiguous(), dst=dst, group=pg)
        return None

    def _recv(self, pg, *, output, src, **_):
        dist.recv(output, src=src, group=pg)
        return output

    def _barrier(self, pg, **_):
        dist.barrier(group=pg)
        return None


# Alias: ``TorchDistBackend`` is the original name for ``NcclBackend``.
TorchDistBackend = NcclBackend
