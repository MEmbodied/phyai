"""SDPA backend — :func:`F.scaled_dot_product_attention` with optional torch.compile.

:class:`SdpaAttentionBackend` drives :class:`Attention` (padded +
ragged prefill). SDPA cannot read paged KV layouts; the AR / Diffusion
paged stacks intentionally have no SDPA backend.

torch.compile policy
--------------------
Defaults to ``compile=True`` and wraps the inner SDPA op with
:func:`torch.compile` (``dynamic=True``) so the mask build / soft-cap
fallback / SDPA call collapse into one compiled region. The first
call is bracketed by an :func:`~phyai.utils.logging.all_ranks_log`
pair — Inductor compiles independently in every rank's process and
a stuck compile is otherwise invisible. Pass ``compile=False`` to
keep the eager Python path.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn.functional as F

from phyai.layers.attention.attention.base import (
    AttentionBackend,
    AttentionLayerProto,
    AttnCtx,
    AttnMetadata,
    AttnPlanHandle,
)
from phyai.layers.attention.attention.registry import register_backend
from phyai.layers.attention.common import (
    build_padded_mask,
    eager_attn,
    repeat_kv,
)
from phyai.utils.logging import all_ranks_log


if TYPE_CHECKING:
    from phyai.runtime.model_runner import ModelRunner


_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SdpaAttentionPlan(AttnPlanHandle):
    """Empty plan handle — sdpa no-cache has no per-step state."""


def _sdpa_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    causal: bool,
    sliding_window: int | None,
    logits_soft_cap: float | None,
) -> torch.Tensor:
    """Inner SDPA call with mask + soft-cap dispatch. Inputs are ``(*, H, S, D)``."""
    if logits_soft_cap is not None:
        return eager_attn(
            q,
            k,
            v,
            scale=scale,
            causal=causal,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
        )
    S_q = q.shape[-2]
    S_kv = k.shape[-2]
    if not causal and sliding_window is None:
        return F.scaled_dot_product_attention(q, k, v, scale=scale)
    if causal and sliding_window is None and S_q == S_kv:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
    mask = build_padded_mask(
        S_q, S_kv, q.device, causal=causal, sliding_window=sliding_window
    )
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)


def _wrap_with_compile_log(
    fn: Callable[..., torch.Tensor],
    *,
    label: str,
    log_kwargs: dict | None = None,
) -> Callable[..., torch.Tensor]:
    """Compile ``fn`` with ``dynamic=True`` and log the first call on every rank."""
    compiled = torch.compile(fn, dynamic=True)
    state = {"logged": False}
    log_kwargs = log_kwargs or {}

    def _logged(*args, **kwargs):
        if state["logged"]:
            return compiled(*args, **kwargs)
        state["logged"] = True
        all_ranks_log(
            _logger,
            logging.INFO,
            "%s: torch.compile tracing sdpa kernel (dynamic=True%s)",
            label,
            "".join(f", {k}={v!r}" for k, v in log_kwargs.items()),
        )
        t0 = time.perf_counter()
        out = compiled(*args, **kwargs)
        dt = time.perf_counter() - t0
        all_ranks_log(
            _logger,
            logging.INFO,
            "%s: sdpa torch.compile first-call took %.2fs",
            label,
            dt,
        )
        return out

    return _logged


@register_backend("sdpa")
class SdpaAttentionBackend(AttentionBackend):
    """SDPA no-cache backend — padded + ragged prefill."""

    def __init__(
        self, runner: "ModelRunner | None" = None, *, compile: bool = True
    ) -> None:
        del runner
        self.compile = bool(compile)
        if self.compile:
            self._sdpa_call = _wrap_with_compile_log(
                _sdpa_attn,
                label="Attention[sdpa]",
            )
        else:
            self._sdpa_call = _sdpa_attn

    def supports_capture(self) -> bool:
        return True

    def init_forward_metadata(self, meta: AttnMetadata) -> AttnPlanHandle:
        return SdpaAttentionPlan()

    def forward(
        self,
        layer: AttentionLayerProto,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: AttnCtx,
    ) -> torch.Tensor:
        if ctx.mode.is_idle():
            return q.new_zeros(q.shape)
        if ctx.layout.is_padded():
            return self._forward_padded(layer, q, k, v)
        return self._forward_ragged(layer, q, k, v, ctx)

    def _forward_padded(
        self,
        layer: AttentionLayerProto,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        q_h = q.transpose(1, 2)
        k_h = repeat_kv(k.transpose(1, 2), layer.num_heads, layer.num_kv_heads)
        v_h = repeat_kv(v.transpose(1, 2), layer.num_heads, layer.num_kv_heads)
        out = self._sdpa_call(
            q_h,
            k_h,
            v_h,
            scale=layer.scale,
            causal=layer.causal,
            sliding_window=layer.sliding_window,
            logits_soft_cap=layer.logits_soft_cap,
        )
        return out.transpose(1, 2).contiguous()

    def _forward_ragged(
        self,
        layer: AttentionLayerProto,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: AttnCtx,
    ) -> torch.Tensor:
        if ctx.cu_seqlens_q is None:
            raise ValueError(
                "SdpaAttentionBackend ragged forward requires ctx.cu_seqlens_q."
            )
        cu_q = ctx.cu_seqlens_q
        cu_kv = ctx.cu_seqlens_kv if ctx.cu_seqlens_kv is not None else cu_q
        cu_q_list = cu_q.tolist()
        cu_kv_list = cu_kv.tolist()
        outs: list[torch.Tensor] = []
        for b in range(len(cu_q_list) - 1):
            qs, qe = cu_q_list[b], cu_q_list[b + 1]
            ks, ke = cu_kv_list[b], cu_kv_list[b + 1]
            qi = q[qs:qe].unsqueeze(0).transpose(1, 2)
            ki = repeat_kv(
                k[ks:ke].unsqueeze(0).transpose(1, 2),
                layer.num_heads,
                layer.num_kv_heads,
            )
            vi = repeat_kv(
                v[ks:ke].unsqueeze(0).transpose(1, 2),
                layer.num_heads,
                layer.num_kv_heads,
            )
            oi = self._sdpa_call(
                qi,
                ki,
                vi,
                scale=layer.scale,
                causal=layer.causal,
                sliding_window=layer.sliding_window,
                logits_soft_cap=layer.logits_soft_cap,
            )
            outs.append(oi.transpose(1, 2).squeeze(0))
        return torch.cat(outs, dim=0)


__all__ = ["SdpaAttentionBackend", "SdpaAttentionPlan"]
