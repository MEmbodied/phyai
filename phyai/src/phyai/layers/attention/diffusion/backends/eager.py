"""Pure-PyTorch reference paged-KV attention for the diffusion / action-expert stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from phyai.layers.attention.common import eager_attn, repeat_kv
from phyai.layers.attention.diffusion.base import (
    DiffusionAttentionBackend,
    DiffusionAttentionLayerProto,
    DiffusionAttnCtx,
    DiffusionAttnMetadata,
    DiffusionAttnPlanHandle,
)
from phyai.layers.attention.diffusion.registry import register_backend


if TYPE_CHECKING:
    from phyai.runtime.model_runner import ModelRunner


@dataclass(frozen=True)
class EagerDiffusionPlan(DiffusionAttnPlanHandle):
    """Per-step ragged plumbing for :class:`EagerDiffusionBackend`."""

    cu_seqlens_q: torch.Tensor  # (B+1,) int64
    paged_kv_indptr: torch.Tensor  # (B+1,) int64
    paged_kv_indices: torch.Tensor  # (N,) int64


@register_backend("eager")
class EagerDiffusionBackend(DiffusionAttentionBackend):
    """Eager diffusion attention — contiguous-slab K/V slice + matmul."""

    def __init__(self, runner: "ModelRunner | None" = None) -> None:
        del runner

    def supports_capture(self) -> bool:
        return False

    def init_forward_metadata(
        self, meta: DiffusionAttnMetadata
    ) -> DiffusionAttnPlanHandle:
        if (
            meta.cu_seqlens_q is None
            or meta.paged_kv_indptr is None
            or meta.paged_kv_indices is None
        ):
            raise ValueError(
                "EagerDiffusionBackend.init_forward_metadata requires "
                "cu_seqlens_q, paged_kv_indptr, and paged_kv_indices on "
                "DiffusionAttnMetadata."
            )
        cu_q = meta.cu_seqlens_q.to(torch.int64)
        indptr = meta.paged_kv_indptr.to(torch.int64)
        indices = meta.paged_kv_indices.to(torch.int64)
        return EagerDiffusionPlan(
            cu_seqlens_q=cu_q,
            paged_kv_indptr=indptr,
            paged_kv_indices=indices,
        )

    def forward(
        self,
        layer: DiffusionAttentionLayerProto,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: DiffusionAttnCtx,
    ) -> torch.Tensor:
        if ctx.mode.is_idle():
            return q.new_zeros(q.shape)
        if not isinstance(ctx.plan, EagerDiffusionPlan):
            raise TypeError(
                f"EagerDiffusionBackend expected ctx.plan: EagerDiffusionPlan, "
                f"got {type(ctx.plan).__name__}."
            )
        ctx.kv_pool.write_kv(layer.layer_id, ctx.write_indices, k, v)
        plan = ctx.plan
        K_pool, V_pool = ctx.kv_pool.kv_buffer(layer.layer_id)
        cu_q = plan.cu_seqlens_q.tolist()
        kv_indptr = plan.paged_kv_indptr.tolist()
        out = torch.empty_like(q)
        for b in range(len(cu_q) - 1):
            q_start, q_end = cu_q[b], cu_q[b + 1]
            kv_start, kv_end = kv_indptr[b], kv_indptr[b + 1]
            if q_end == q_start:
                continue
            if kv_end == kv_start:
                out[q_start:q_end] = 0
                continue
            kv_indices = plan.paged_kv_indices[kv_start:kv_end]
            k_seg = K_pool.index_select(0, kv_indices).squeeze(1)
            v_seg = V_pool.index_select(0, kv_indices).squeeze(1)
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


__all__ = ["EagerDiffusionBackend", "EagerDiffusionPlan"]
