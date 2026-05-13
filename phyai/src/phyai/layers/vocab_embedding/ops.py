"""Custom-op wrapper for the masked vocab-parallel embedding lookup.

Two reasons this is wrapped as ``torch.library.custom_op`` rather than called
inline from the layer's ``forward``:

* **Graph capture stability.** Dynamo / ``torch.compile`` see an opaque op and
  do not re-trace the masking + gather path on every call. SGLang's reference
  implementation relies on ``@torch.compile(dynamic=True)`` to fuse the same
  pointwise pattern; the custom-op approach is structurally cleaner and avoids
  having to ``disable=`` it on backends without a working compiler.

* **Backend pluggability.** The default implementation forwards to the Triton
  kernel on CUDA and to a pure-PyTorch fallback elsewhere; a future CUDA
  graph or low-bit-weight kernel can replace either path without disturbing
  the layer code.

The shape contract is ``out.shape == input_ids.shape + (weight.shape[1],)``;
positions whose id falls outside ``[shard_start, shard_end)`` read as zero.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _masked_embedding_lookup_eager(
    input_ids: Tensor,
    weight: Tensor,
    shard_start: int,
    shard_end: int,
) -> Tensor:
    """Three-pass reference: mask, gather with safe index, zero-out misses.

    Used as the CPU / non-CUDA fallback for the custom op and as the parity
    oracle in tests.
    """
    mask = (input_ids >= shard_start) & (input_ids < shard_end)
    local_ids = torch.where(mask, input_ids - shard_start, torch.zeros_like(input_ids))
    out = F.embedding(local_ids, weight)
    return out.masked_fill(~mask.unsqueeze(-1), 0)


@torch.library.custom_op("phyai::masked_embedding_lookup", mutates_args=())
def _masked_embedding_lookup_op(
    input_ids: Tensor,
    weight: Tensor,
    shard_start: int,
    shard_end: int,
) -> Tensor:
    if input_ids.is_cuda and weight.is_cuda:
        # The Triton kernel lives in phyai-kernel and is bandwidth-bound; it
        # fuses mask + gather + zero-on-miss into a single pass.
        from phyai_kernel import masked_embedding_lookup as _triton_lookup

        return _triton_lookup(input_ids, weight, int(shard_start), int(shard_end))
    return _masked_embedding_lookup_eager(input_ids, weight, shard_start, shard_end)


@_masked_embedding_lookup_op.register_fake
def _(input_ids: Tensor, weight: Tensor, shard_start: int, shard_end: int) -> Tensor:
    out_shape = (*input_ids.shape, weight.shape[1])
    return torch.empty(out_shape, dtype=weight.dtype, device=weight.device)


def masked_embedding_lookup(
    input_ids: Tensor,
    weight: Tensor,
    *,
    shard_start: int,
    shard_end: int,
) -> Tensor:
    """Gather ``weight[input_ids - shard_start]`` for in-shard positions, else 0.

    Out-of-shard positions return zero rows so that an all-reduce across TP
    ranks recovers the global embedding without an explicit second pass.
    """
    return torch.ops.phyai.masked_embedding_lookup.default(
        input_ids, weight, int(shard_start), int(shard_end)
    )


__all__ = ["masked_embedding_lookup"]
