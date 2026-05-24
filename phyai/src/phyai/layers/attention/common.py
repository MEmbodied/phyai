"""Shared attention kernels used by every backend.

The eager mask construction, KV-head expansion, and reference
matmul-softmax kernel are pure torch primitives. Lifting them out of
any single backend lets sdpa fall back to ``eager_attn`` for
soft-cap, eager use them as the reference path, and any future
backend reuse them without crossing module privacy.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def repeat_kv(x: torch.Tensor, num_heads: int, num_kv_heads: int) -> torch.Tensor:
    """``(B, H_kv, S, D) -> (B, H_q, S, D)``; identity for MHA."""
    if num_heads == num_kv_heads:
        return x
    rep = num_heads // num_kv_heads
    return x.repeat_interleave(rep, dim=1)


def build_padded_mask(
    S_q: int,
    S_kv: int,
    device: torch.device,
    *,
    causal: bool,
    sliding_window: int | None,
) -> torch.Tensor | None:
    """Bool mask ``(S_q, S_kv)``: True = attend.

    Returns ``None`` when no masking is needed (full non-causal). The
    layer's ``__init__`` rejects ``sliding_window`` with ``causal=False``,
    so the only mask shapes here are causal / causal+SWA. Append-prefill
    alignment is ``q_pos[i] = i + (S_kv - S_q)``.
    """
    if not causal and sliding_window is None:
        return None
    i = torch.arange(S_q, device=device).unsqueeze(1)
    j = torch.arange(S_kv, device=device).unsqueeze(0)
    q_pos = i + (S_kv - S_q)
    mask = q_pos >= j
    if sliding_window is not None:
        mask = mask & (q_pos - j < sliding_window)
    return mask


def eager_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    causal: bool,
    sliding_window: int | None,
    logits_soft_cap: float | None,
) -> torch.Tensor:
    """Reference matmul + masked softmax. Inputs are ``(*, H, S, D)``."""
    S_q = q.shape[-2]
    S_kv = k.shape[-2]
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    if logits_soft_cap is not None:
        cap = logits_soft_cap
        attn = cap * torch.tanh(attn / cap)
    mask = build_padded_mask(
        S_q, S_kv, q.device, causal=causal, sliding_window=sliding_window
    )
    if mask is not None:
        attn = attn.masked_fill(~mask, float("-inf"))
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(attn, v)


__all__ = ["build_padded_mask", "eager_attn", "repeat_kv"]
