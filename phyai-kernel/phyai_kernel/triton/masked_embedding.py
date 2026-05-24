"""Triton fused mask + embedding-lookup + zero-on-miss for vocab-parallel TP.

Vocab-parallel embedding splits the table along V across TP ranks: rank ``r``
holds rows ``[r * V_per, (r+1) * V_per)`` of the global table. For an input
``input_ids`` of shape ``[..., ]``, each rank produces partial output rows that
are non-zero only when the id falls in its shard, and the final embedding is
the sum across ranks (an all-reduce).

The classical formulation does this in three passes:

1. ``mask = (ids >= start) & (ids < end)``
2. ``out = F.embedding(where(mask, ids - start, 0), W)``
3. ``out = out.masked_fill_(~mask, 0)``

Step 2 reads ``W`` for every output position (out-of-shard positions hit row 0
unnecessarily); step 3 reads and writes the entire output again. This kernel
fuses the three steps so out-of-shard positions write 0 directly without ever
touching ``W``, saving one full pass over the output tensor.

A 2-D launch:
* program_id(0) over rows of the flattened input (BLOCK_M tokens per program)
* program_id(1) over columns of the embedding (BLOCK_D dims per program)

The kernel is bandwidth-bound; correctness is straightforward (no reductions,
no fp32 promotion needed).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _masked_embedding_kernel(
    ids_ptr,
    weight_ptr,
    out_ptr,
    M,
    D,
    shard_start,
    shard_end,
    weight_stride_v,
    weight_stride_d,
    out_stride_m,
    out_stride_d,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    m_mask = offs_m < M
    d_mask = offs_d < D

    # Load token ids; out-of-bounds tokens go to 0 but we mask them later.
    ids = tl.load(ids_ptr + offs_m, mask=m_mask, other=0)
    in_range = (ids >= shard_start) & (ids < shard_end)
    local_ids = tl.where(in_range, ids - shard_start, 0)

    # Gather weight rows. ``other=0`` is what makes the zero-on-miss part work
    # without a separate masked_fill_: rows that fail the combined mask never
    # get loaded and the result tile is initialised to 0.
    w_ptrs = (
        weight_ptr
        + local_ids[:, None] * weight_stride_v
        + offs_d[None, :] * weight_stride_d
    )
    valid = in_range[:, None] & m_mask[:, None] & d_mask[None, :]
    w = tl.load(w_ptrs, mask=valid, other=0.0)

    out_ptrs = out_ptr + offs_m[:, None] * out_stride_m + offs_d[None, :] * out_stride_d
    store_mask = m_mask[:, None] & d_mask[None, :]
    tl.store(out_ptrs, w, mask=store_mask)


def _check_inputs(
    input_ids: torch.Tensor, weight: torch.Tensor, shard_start: int, shard_end: int
) -> None:
    if not input_ids.is_cuda or not weight.is_cuda:
        raise RuntimeError(
            "phyai_kernel.triton.masked_embedding_lookup: tensors must live on CUDA"
        )
    if weight.dim() != 2:
        raise RuntimeError(
            f"phyai_kernel.triton.masked_embedding_lookup: weight must be 2D, "
            f"got {weight.dim()}D"
        )
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise RuntimeError(
            f"phyai_kernel.triton.masked_embedding_lookup: input_ids dtype must be "
            f"int32/int64, got {input_ids.dtype}"
        )
    if shard_end < shard_start:
        raise RuntimeError(
            f"phyai_kernel.triton.masked_embedding_lookup: shard_end={shard_end} "
            f"< shard_start={shard_start}"
        )
    if shard_end - shard_start > weight.shape[0]:
        raise RuntimeError(
            f"phyai_kernel.triton.masked_embedding_lookup: shard width "
            f"{shard_end - shard_start} > weight rows {weight.shape[0]}"
        )


def masked_embedding_lookup(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
    shard_start: int,
    shard_end: int,
) -> torch.Tensor:
    """Fused masked vocab-parallel embedding lookup.

    Args:
        input_ids: integer tensor of shape ``[..., ]``. Values outside
            ``[shard_start, shard_end)`` produce zero output rows.
        weight: ``(V_per_rank, D)`` per-rank embedding table for this shard.
            Row 0 corresponds to global token id ``shard_start``.
        shard_start: inclusive lower bound of this rank's shard.
        shard_end: exclusive upper bound of this rank's shard.

    Returns:
        Tensor of shape ``input_ids.shape + (D,)``, dtype/device matching
        ``weight``. Out-of-shard positions read as 0; in-shard positions hold
        the corresponding embedding row.
    """
    _check_inputs(input_ids, weight, shard_start, shard_end)

    orig_shape = input_ids.shape
    ids_flat = input_ids.reshape(-1)
    if not ids_flat.is_contiguous():
        ids_flat = ids_flat.contiguous()
    M = ids_flat.shape[0]
    D = weight.shape[1]

    out_flat = torch.empty((M, D), dtype=weight.dtype, device=weight.device)

    if M == 0:
        return out_flat.reshape(*orig_shape, D)

    # Block sizes — small enough to keep occupancy high on common D=2048..8192
    # widths, large enough that BLOCK_M overhead amortises. Autotune later if
    # profile shows it matters.
    BLOCK_M = 64
    BLOCK_D = 128 if D >= 128 else triton.next_power_of_2(D)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(D, BLOCK_D))
    _masked_embedding_kernel[grid](
        ids_flat,
        weight,
        out_flat,
        M,
        D,
        shard_start,
        shard_end,
        weight.stride(0),
        weight.stride(1),
        out_flat.stride(0),
        out_flat.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )

    return out_flat.reshape(*orig_shape, D)


__all__ = ["masked_embedding_lookup"]
