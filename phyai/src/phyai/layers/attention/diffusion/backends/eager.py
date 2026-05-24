"""Pure-PyTorch reference paged-KV attention for the diffusion / action-expert stack.

Structurally identical to
:class:`phyai.layers.attention.ar.backends.eager.EagerARBackend`. Bug
fixes here MUST be mirrored to that file.
"""

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
    kv_starts: torch.Tensor  # (B,) int64
    kv_lens: torch.Tensor  # (B,) int64


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
        indptr = meta.paged_kv_indptr.to(torch.int64).tolist()
        indices = meta.paged_kv_indices.to(torch.int64)
        B = len(indptr) - 1
        kv_starts = torch.zeros(B, dtype=torch.int64, device=indices.device)
        kv_lens = torch.zeros(B, dtype=torch.int64, device=indices.device)
        for b in range(B):
            s, e = indptr[b], indptr[b + 1]
            seg = indices[s:e]
            n = seg.numel()
            if n == 0:
                continue
            start = int(seg[0].item())
            expected = torch.arange(
                start, start + n, dtype=seg.dtype, device=seg.device
            )
            if not torch.equal(seg, expected):
                raise ValueError(
                    f"EagerDiffusionBackend requires contiguous KV slots per "
                    f"sample; sample {b} has non-contiguous "
                    f"paged_kv_indices={seg.tolist()}. Use 'flashinfer' "
                    f"for non-contiguous (radix / eviction) cache layouts."
                )
            kv_starts[b] = start
            kv_lens[b] = n
        return EagerDiffusionPlan(
            cu_seqlens_q=cu_q, kv_starts=kv_starts, kv_lens=kv_lens
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
        kv_starts = plan.kv_starts.tolist()
        kv_lens = plan.kv_lens.tolist()
        out = torch.empty_like(q)
        for b in range(len(cu_q) - 1):
            q_start, q_end = cu_q[b], cu_q[b + 1]
            kv_start, kv_len = kv_starts[b], kv_lens[b]
            if q_end == q_start:
                continue
            if kv_len == 0:
                out[q_start:q_end] = 0
                continue
            k_seg = K_pool[kv_start : kv_start + kv_len].squeeze(1)
            v_seg = V_pool[kv_start : kv_start + kv_len].squeeze(1)
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
