"""RoPE with a selectable kernel backend.

Most modern transformer decoders rotate Q and K with the rotate-half
(NeoX-style) variant of RoPE; :class:`NoStateAttention` is by design a
pure attention op and expects rotation to happen before it. This module
fills that gap.

A :class:`RotaryEmbedding` holds an fp32 ``cos_sin_cache`` of shape
``(max_position_embeddings, rotary_dim)`` — first half cos, second half
sin — which is exactly what flashinfer's
:func:`flashinfer.rope.apply_rope_with_cos_sin_cache` consumes. The eager
backend reads the same cache in PyTorch with a ``rotate_half`` (or, when
``interleave=True``, an even/odd-split) op.

Backends
--------
* ``"flashinfer"`` (default) — one fused kernel for Q and K.
* ``"eager"`` — pure PyTorch reference, also handy on CPU.

Supported rope_types
--------------------
``"default"`` (no scaling), ``"linear"`` (PI), ``"llama3"``. ``"yarn"``,
``"dynamic"``, ``"longrope"`` are reserved and raise ``NotImplementedError``
to keep the dispatch table closed.

Forward shape contract
----------------------
Both Q and K must have the same number of tokens (``S`` for padded,
``nnz`` for ragged) — append-prefill (Q shorter than K) is the caller's
responsibility, who can call this layer twice with different
``position_ids``.

* 4-D padded: ``q (B, S, H_q, D)``, ``k (B, S, H_k, D)``,
  ``position_ids (B, S)`` or ``(S,)``.
* 3-D ragged: ``q (nnz, H_q, D)``, ``k (nnz, H_k, D)``,
  ``position_ids (nnz,)``.

Limitations
-----------
* No fused FP8 RoPE (flashinfer's ``rope_quantize_fp8``); not needed for
  prefill-only VLA today.
* No in-place variant — output buffers are always freshly allocated.
"""

from __future__ import annotations

import math
from typing import Callable, Tuple

import torch
import torch.nn as nn

from phyai.engine_config import get_engine_config


_VALID_BACKENDS: tuple[str, ...] = ("flashinfer", "eager")


def _resolve_backend(name: str) -> str:
    canonical = name.replace("_", "-").lower()
    if canonical not in _VALID_BACKENDS:
        raise ValueError(
            f"Unknown RoPE backend {name!r}; expected one of {_VALID_BACKENDS!r}."
        )
    return canonical


# ---------------------------------------------------------------------------
# inv_freq computation per rope_type. Each helper returns
# ``(inv_freq, attention_scaling)`` so that the cos/sin cache is built
# uniformly: ``cache = cat([cos, sin], -1) * attention_scaling``.
# ---------------------------------------------------------------------------


def _default_inv_freq(
    rotary_dim: int,
    theta: float,
    *,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, float]:
    """Original RoPE: ``inv_freq[i] = theta ** (-2i / rotary_dim)``."""
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.int64).to(
                device=device, dtype=torch.float32
            )
            / rotary_dim
        )
    )
    return inv_freq, 1.0


def _linear_inv_freq(
    rotary_dim: int,
    theta: float,
    *,
    factor: float,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, float]:
    """PI / linear scaling: divide every inv_freq by ``factor``."""
    inv_freq, _ = _default_inv_freq(rotary_dim, theta, device=device)
    return inv_freq / factor, 1.0


def _llama3_inv_freq(
    rotary_dim: int,
    theta: float,
    *,
    factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_max_position_embeddings: int,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, float]:
    """Llama 3.1 piecewise-smooth scaling.

    Implements the piecewise-smooth long-context scaling described in the
    Llama 3 technical report.
    """
    inv_freq, _ = _default_inv_freq(rotary_dim, theta, device=device)
    low_freq_wavelen = original_max_position_embeddings / low_freq_factor
    high_freq_wavelen = original_max_position_embeddings / high_freq_factor
    wavelen = 2 * math.pi / inv_freq
    inv_freq_llama = torch.where(
        wavelen > low_freq_wavelen, inv_freq / factor, inv_freq
    )
    smooth = (original_max_position_embeddings / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed = (1 - smooth) * inv_freq_llama / factor + smooth * inv_freq_llama
    is_medium = ~(wavelen < high_freq_wavelen) & ~(wavelen > low_freq_wavelen)
    inv_freq_llama = torch.where(is_medium, smoothed, inv_freq_llama)
    return inv_freq_llama, 1.0


def _unsupported_rope_type(name: str) -> Callable[..., tuple[torch.Tensor, float]]:
    def _raise(*_args, **_kwargs):
        raise NotImplementedError(
            f"rope_type={name!r} is reserved but not implemented yet; "
            f"supported types: {tuple(ROPE_INV_FREQ_FNS)!r}."
        )

    return _raise


ROPE_INV_FREQ_FNS: dict[str, Callable[..., tuple[torch.Tensor, float]]] = {
    "default": _default_inv_freq,
    "linear": _linear_inv_freq,
    "llama3": _llama3_inv_freq,
    "yarn": _unsupported_rope_type("yarn"),
    "dynamic": _unsupported_rope_type("dynamic"),
    "longrope": _unsupported_rope_type("longrope"),
}


# ---------------------------------------------------------------------------
# Standalone ops (work on caller-supplied cos/sin).
# ---------------------------------------------------------------------------


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """``[a, b] -> [-b, a]`` along the last dim — the NeoX-style RoPE rotation."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotate-half RoPE with caller-supplied ``cos`` / ``sin``.

    NeoX-style rotate-half convention. ``cos`` and ``sin`` must already be
    sized to the full ``head_dim`` (i.e. each half duplicated).
    ``unsqueeze_dim`` controls which axis becomes the head axis after
    broadcasting:

    * ``unsqueeze_dim=1`` for ``q (B, H, S, D)`` (HF default),
    * ``unsqueeze_dim=2`` for ``q (B, S, H, D)``,
    * ``unsqueeze_dim=-2`` for ragged ``q (nnz, H, D)`` or any "head is
      second-to-last" layout.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out, k_out


def _rotate_interleave(x: torch.Tensor) -> torch.Tensor:
    """GPT-J / GPT-NeoX-old rotation: interleaved even/odd dims.

    ``[a0, a1, a2, a3, ...] -> [-a1, a0, -a3, a2, ...]``.
    """
    even = x[..., 0::2]
    odd = x[..., 1::2]
    rotated = torch.stack((-odd, even), dim=-1)
    return rotated.flatten(-2)


def _apply_interleaved(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = -2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Interleaved-style RoPE on caller-supplied cos/sin.

    ``cos`` / ``sin`` here have layout
    ``[c0, c0, c1, c1, ...]`` / ``[s0, s0, s1, s1, ...]`` so that the
    same index multiplies its even and odd partners.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_out = (q * cos) + (_rotate_interleave(q) * sin)
    k_out = (k * cos) + (_rotate_interleave(k) * sin)
    return q_out, k_out


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    """Rotary position embedding with a selectable kernel backend.

    Parameters
    ----------
    head_dim:
        Per-head channel width of Q / K. Cache rotates the leading
        ``rotary_dim = int(head_dim * partial_rotary_factor)`` channels
        and the trailing ``head_dim - rotary_dim`` are passed through
        unchanged.
    max_position_embeddings:
        Cache size along the sequence axis. ``position_ids`` must be
        ``< max_position_embeddings``.
    rope_theta:
        RoPE base wavelength. Common values: ``1e4`` for short-context
        configs, ``5e5`` (or larger) for long-context configs.
    rope_type:
        ``"default"``, ``"linear"``, or ``"llama3"``. ``"yarn"``,
        ``"dynamic"``, ``"longrope"`` raise NotImplementedError.
    rope_scaling:
        Extra params for the chosen ``rope_type``. ``"linear"`` requires
        ``{"factor": <float>}``; ``"llama3"`` requires
        ``{"factor", "low_freq_factor", "high_freq_factor",
        "original_max_position_embeddings"}``.
    partial_rotary_factor:
        Fraction of ``head_dim`` that gets rotated. Default ``1.0``.
    interleave:
        ``False`` (default): NeoX-style rotate-half geometry (the
        common case for modern transformer decoders). ``True``:
        GPT-J / NeoX-old even/odd geometry. flashinfer maps
        ``is_neox = not interleave``.
    backend:
        ``"flashinfer"`` (default) or ``"eager"``.
    device:
        Where to allocate the cache. Defaults to
        :attr:`phyai.engine_config.EngineConfig.device` (typically
        ``"cuda"``); pass an explicit ``"cpu"`` / ``"cuda:N"`` to
        override. ``.to(device)`` later still works.
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 8192,
        *,
        rope_theta: float = 10000.0,
        rope_type: str = "default",
        rope_scaling: dict | None = None,
        partial_rotary_factor: float = 1.0,
        interleave: bool = False,
        backend: str = "flashinfer",
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2:
            raise ValueError(f"head_dim must be a positive even int, got {head_dim}.")
        if not 0.0 < partial_rotary_factor <= 1.0:
            raise ValueError(
                f"partial_rotary_factor must be in (0, 1], got {partial_rotary_factor}."
            )
        rotary_dim = int(head_dim * partial_rotary_factor)
        if rotary_dim <= 0 or rotary_dim % 2:
            raise ValueError(
                f"rotary_dim={rotary_dim} (=int(head_dim={head_dim} * "
                f"partial_rotary_factor={partial_rotary_factor})) must be a "
                f"positive even int."
            )
        if max_position_embeddings <= 0:
            raise ValueError(
                f"max_position_embeddings must be positive, got "
                f"{max_position_embeddings}."
            )
        if rope_type not in ROPE_INV_FREQ_FNS:
            raise ValueError(
                f"Unknown rope_type {rope_type!r}; expected one of "
                f"{tuple(ROPE_INV_FREQ_FNS)!r}."
            )

        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_type = rope_type
        self.rope_scaling = dict(rope_scaling or {})
        self.partial_rotary_factor = partial_rotary_factor
        self.interleave = interleave
        self.backend = _resolve_backend(backend)
        if device is None:
            device = get_engine_config().device

        if self.backend == "flashinfer":
            # Fail fast at construction rather than at first forward.
            try:
                import flashinfer.rope  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "backend='flashinfer' but flashinfer is not installed; "
                    "either install flashinfer-python or pass backend='eager'."
                ) from e

        inv_freq, attention_scaling = ROPE_INV_FREQ_FNS[rope_type](
            rotary_dim, rope_theta, device=device, **self.rope_scaling
        )
        self.attention_scaling = float(attention_scaling)

        # Build the cos/sin cache. flashinfer expects fp32 with first half
        # cos and second half sin; the eager path reads the same buffer.
        t = torch.arange(
            max_position_embeddings, dtype=torch.float32, device=inv_freq.device
        )
        freqs = torch.outer(t, inv_freq.float())  # (max_pos, rotary_dim/2)
        cos = freqs.cos() * self.attention_scaling
        sin = freqs.sin() * self.attention_scaling
        cos_sin_cache = torch.cat([cos, sin], dim=-1).contiguous()
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_sin_cache", cos_sin_cache, persistent=False)

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._check_shapes(q, k, position_ids)
        if self.backend == "flashinfer":
            return self._forward_flashinfer(q, k, position_ids)
        return self._forward_eager(q, k, position_ids)

    def _check_shapes(
        self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor
    ) -> None:
        if q.dim() not in (3, 4):
            raise ValueError(
                f"q must be 3-D (ragged) or 4-D (padded), got shape "
                f"{tuple(q.shape)}."
            )
        if q.dim() != k.dim():
            raise ValueError(
                f"q and k must have the same number of dims, got "
                f"q={tuple(q.shape)}, k={tuple(k.shape)}."
            )
        if q.shape[-1] != self.head_dim or k.shape[-1] != self.head_dim:
            raise ValueError(
                f"q/k last dim must equal head_dim={self.head_dim}, got "
                f"q={tuple(q.shape)}, k={tuple(k.shape)}."
            )
        # Q and K must have the same number of tokens. For 4-D this is
        # the (B, S) pair; for 3-D this is the leading dim.
        if q.dim() == 4:
            if q.shape[0] != k.shape[0] or q.shape[1] != k.shape[1]:
                raise ValueError(
                    f"q/k must share leading (B, S); got q={tuple(q.shape)}, "
                    f"k={tuple(k.shape)}. Append-prefill (Q shorter than K) "
                    f"is not supported here — call RotaryEmbedding twice "
                    f"with different position_ids."
                )
        else:  # 3-D
            if q.shape[0] != k.shape[0]:
                raise ValueError(
                    f"q/k ragged token counts differ: q={tuple(q.shape)}, "
                    f"k={tuple(k.shape)}."
                )

    # ---------------------------- flashinfer ---------------------------- #

    def _forward_flashinfer(
        self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from flashinfer.rope import apply_rope_with_cos_sin_cache

        if self.cos_sin_cache.device != q.device:
            raise RuntimeError(
                f"cos_sin_cache is on {self.cos_sin_cache.device} but q is on "
                f"{q.device}; call ``rotary_emb.to(q.device)`` once before "
                f"forward."
            )

        orig_q_shape = q.shape
        orig_k_shape = k.shape
        if q.dim() == 4:
            B, S = q.shape[0], q.shape[1]
            H_q = q.shape[2]
            H_k = k.shape[2]
            nnz = B * S
            flat_q = q.reshape(nnz, H_q * self.head_dim)
            flat_k = k.reshape(nnz, H_k * self.head_dim)
            pos = position_ids
            if pos.dim() == 1:
                if pos.shape[0] != S:
                    raise ValueError(
                        f"1-D position_ids length {pos.shape[0]} does not "
                        f"match S={S} for 4-D q."
                    )
                pos = pos.unsqueeze(0).expand(B, S)
            elif pos.shape != (B, S):
                raise ValueError(
                    f"position_ids shape {tuple(pos.shape)} != (B, S)=({B}, {S})."
                )
            flat_pos = pos.reshape(nnz)
        else:  # 3-D
            nnz = q.shape[0]
            H_q = q.shape[1]
            H_k = k.shape[1]
            flat_q = q.reshape(nnz, H_q * self.head_dim)
            flat_k = k.reshape(nnz, H_k * self.head_dim)
            if position_ids.dim() != 1 or position_ids.shape[0] != nnz:
                raise ValueError(
                    f"position_ids shape {tuple(position_ids.shape)} does not "
                    f"match (nnz,)=({nnz},) for 3-D q."
                )
            flat_pos = position_ids
        flat_pos = flat_pos.contiguous().to(torch.int32)

        q_out, k_out = apply_rope_with_cos_sin_cache(
            positions=flat_pos,
            query=flat_q.contiguous(),
            key=flat_k.contiguous(),
            head_size=self.head_dim,
            cos_sin_cache=self.cos_sin_cache,
            is_neox=not self.interleave,
        )
        return q_out.view(orig_q_shape), k_out.view(orig_k_shape)

    # ------------------------------ eager ------------------------------ #

    def _forward_eager(
        self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Look up cos/sin per token. Cache stays fp32; cast back to q.dtype
        # when we multiply.
        pos = position_ids.to(self.cos_sin_cache.device).long()
        cs = self.cos_sin_cache[pos]  # (..., rotary_dim)
        cos_h, sin_h = cs.chunk(2, dim=-1)  # each (..., rotary_dim/2)

        if self.interleave:
            # cos / sin doubled by repeat-interleave: [c0, c0, c1, c1, ...].
            cos = cos_h.repeat_interleave(2, dim=-1)
            sin = sin_h.repeat_interleave(2, dim=-1)
            apply_fn = _apply_interleaved
        else:
            cos = torch.cat([cos_h, cos_h], dim=-1)
            sin = torch.cat([sin_h, sin_h], dim=-1)
            apply_fn = apply_rotary_pos_emb

        # Broadcast position-axis cos/sin to the q/k shape.
        # 4-D q: (B, S, H, D); cos here is (B, S, rd) when pos is (B, S),
        # or (S, rd) when pos is (S,) — the latter broadcasts along B.
        # 3-D q: (nnz, H, D); cos is (nnz, rd).
        # In both cases head axis is -2, so unsqueeze_dim=-2.
        cos = cos.to(dtype=q.dtype, device=q.device)
        sin = sin.to(dtype=q.dtype, device=q.device)

        if self.rotary_dim < self.head_dim:
            q_rot, q_pass = q[..., : self.rotary_dim], q[..., self.rotary_dim :]
            k_rot, k_pass = k[..., : self.rotary_dim], k[..., self.rotary_dim :]
            q_rot, k_rot = apply_fn(q_rot, k_rot, cos, sin, unsqueeze_dim=-2)
            return (
                torch.cat([q_rot, q_pass], dim=-1),
                torch.cat([k_rot, k_pass], dim=-1),
            )
        return apply_fn(q, k, cos, sin, unsqueeze_dim=-2)

    # ------------------------------------------------------------------ #

    def extra_repr(self) -> str:
        s = (
            f"head_dim={self.head_dim}, rotary_dim={self.rotary_dim}, "
            f"max_position_embeddings={self.max_position_embeddings}, "
            f"rope_theta={self.rope_theta}, rope_type={self.rope_type!r}, "
            f"backend={self.backend!r}"
        )
        if self.partial_rotary_factor != 1.0:
            s += f", partial_rotary_factor={self.partial_rotary_factor}"
        if self.interleave:
            s += ", interleave=True"
        return s


__all__ = [
    "RotaryEmbedding",
    "apply_rotary_pos_emb",
    "rotate_half",
    "ROPE_INV_FREQ_FNS",
]
