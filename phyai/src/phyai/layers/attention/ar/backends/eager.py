"""Pure-PyTorch reference paged-KV attention for the AR (LM-side) stack.

Gathers each sample's KV by its ``paged_kv_indices`` — an arbitrary slot
layout, contiguous static slabs and non-contiguous radix / eviction
layouts alike — via :meth:`KVCachePool.gather_kv`, then runs the shared
:func:`eager_attn` reference. CPU/CI only; not graph-captureable.

The append-prefill causal alignment (``q`` is the tail of ``kv``) is the
radix prefix-reuse case and is handled by :func:`eager_attn` /
:func:`build_padded_mask` (``q_pos[i] = i + (S_kv - S_q)``).

**Sibling**:
:class:`phyai.layers.attention.diffusion.backends.eager.EagerDiffusionBackend`
still requires per-sample *contiguous* slabs. The two intentionally
diverge: the AR stack is the radix-cache foundation and must read
non-contiguous prefix-reuse layouts; the diffusion stack has no radix
consumer yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from phyai.layers.attention.ar.base import (
    ARAttentionBackend,
    ARAttentionLayerProto,
    ARAttnCtx,
    ARAttnMetadata,
    ARAttnPlanHandle,
)
from phyai.layers.attention.ar.registry import register_backend
from phyai.layers.attention.common import eager_attn, repeat_kv


if TYPE_CHECKING:
    from phyai.runtime.model_runner import ModelRunner


@dataclass(frozen=True)
class EagerARPlan(ARAttnPlanHandle):
    """Per-step ragged plumbing for :class:`EagerARBackend`.

    Carries the query cu-seqlens and the per-sample KV slot lists, read
    directly via :meth:`KVCachePool.gather_kv` (no contiguity assumption).
    """

    cu_seqlens_q: torch.Tensor  # (B+1,) int64
    paged_kv_indptr: torch.Tensor  # (B+1,) int64
    paged_kv_indices: torch.Tensor  # (sum_kv,) int64


@register_backend("eager")
class EagerARBackend(ARAttentionBackend):
    """Eager AR attention — gather KV by slot id + masked-softmax matmul."""

    def __init__(self, runner: "ModelRunner | None" = None) -> None:
        del runner

    def supports_capture(self) -> bool:
        return False

    def init_forward_metadata(self, meta: ARAttnMetadata) -> ARAttnPlanHandle:
        if (
            meta.cu_seqlens_q is None
            or meta.paged_kv_indptr is None
            or meta.paged_kv_indices is None
        ):
            raise ValueError(
                "EagerARBackend.init_forward_metadata requires cu_seqlens_q, "
                "paged_kv_indptr, and paged_kv_indices on ARAttnMetadata."
            )
        return EagerARPlan(
            cu_seqlens_q=meta.cu_seqlens_q.to(torch.int64),
            paged_kv_indptr=meta.paged_kv_indptr.to(torch.int64),
            paged_kv_indices=meta.paged_kv_indices.to(torch.int64),
        )

    def forward(
        self,
        layer: ARAttentionLayerProto,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: ARAttnCtx,
    ) -> torch.Tensor:
        if ctx.mode.is_idle():
            return q.new_zeros(q.shape)
        if not isinstance(ctx.plan, EagerARPlan):
            raise TypeError(
                f"EagerARBackend expected ctx.plan: EagerARPlan, got "
                f"{type(ctx.plan).__name__}."
            )
        ctx.kv_pool.write_kv(layer.layer_id, ctx.write_indices, k, v)
        plan = ctx.plan
        cu_q = plan.cu_seqlens_q.tolist()
        kv_indptr = plan.paged_kv_indptr.tolist()
        kv_indices = plan.paged_kv_indices
        out = torch.empty_like(q)
        for b in range(len(cu_q) - 1):
            q_start, q_end = cu_q[b], cu_q[b + 1]
            if q_end == q_start:
                continue
            kv_lo, kv_hi = kv_indptr[b], kv_indptr[b + 1]
            if kv_hi == kv_lo:
                out[q_start:q_end] = 0
                continue
            slots = kv_indices[kv_lo:kv_hi]
            k_seg, v_seg = ctx.kv_pool.gather_kv(layer.layer_id, slots)
            qi = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
            ki = repeat_kv(
                k_seg.transpose(0, 1).unsqueeze(0),
                layer.num_heads,
                layer.num_kv_heads,
            )
            vi = repeat_kv(
                v_seg.transpose(0, 1).unsqueeze(0),
                layer.num_heads,
                layer.num_kv_heads,
            )
            oi = eager_attn(
                qi,
                ki,
                vi,
                scale=layer.scale,
                causal=layer.causal,
                sliding_window=None,
                logits_soft_cap=None,
            )
            out[q_start:q_end] = oi.squeeze(0).transpose(0, 1)
        return out


__all__ = ["EagerARBackend", "EagerARPlan"]
