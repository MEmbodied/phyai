"""Stateless prefill attention — no KV cache, no radix.

The math is ``softmax(Q K^T / sqrt(D) + mask) V``; nothing is cached
between calls. Three kernel backends share one forward contract,
selectable at construction time via ``backend=``:

* ``"flashinfer"`` (default) — :mod:`flashinfer.prefill`. Uses
  ``single_prefill_with_kv_cache`` when the batch is one sequence and
  :class:`BatchPrefillWithRaggedKVCacheWrapper` otherwise.
* ``"sdpa"`` — :func:`torch.nn.functional.scaled_dot_product_attention`.
* ``"eager"`` — pure PyTorch matmul + masked softmax. Reference path,
  slow but exact.

Optional left sliding window: each query attends to at most
``sliding_window`` keys, with the current position counted toward the
window (HF / Mistral / Qwen / Gemma2 convention). Sliding window is
causal-only.

This module is the attention *op* — Q/K/V projection and RoPE are the
caller's responsibility. Q/K/V come in already-projected and the
attention output goes back out in the same layout.

Limitations
-----------
* The flashinfer ``plan()`` call is incompatible with ``torch.compile``
  / CUDA Graph capture; pick ``"sdpa"`` / ``"eager"`` for those callers.
* ``logits_soft_cap`` is honoured by ``"eager"`` and ``"flashinfer"``
  natively; with ``"sdpa"`` the call transparently falls back to eager
  since :func:`F.scaled_dot_product_attention` has no soft-cap parameter.
* The padded-batch path has no per-token pad mask — pad to a uniform
  length and rely on causal / SWA, or use the ragged path for varlen.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from phyai.layers.attention.utils import get_global_fi_workspace


_VALID_BACKENDS: tuple[str, ...] = ("flashinfer", "sdpa", "eager")


def _resolve_backend(name: str) -> str:
    canonical = name.lower().replace("_", "-")
    if canonical not in _VALID_BACKENDS:
        raise ValueError(
            f"Unknown attention backend {name!r}; expected one of "
            f"{_VALID_BACKENDS!r}."
        )
    return canonical


class NoStateAttention(nn.Module):
    """Prefill-only attention with selectable kernel backend.

    Parameters
    ----------
    num_heads:
        Number of query heads.
    head_dim:
        Per-head dimension. ``Q @ K^T`` is divided by ``sqrt(head_dim)``
        unless ``scale`` overrides it.
    num_kv_heads:
        Number of K/V heads. Defaults to ``num_heads`` (MHA). For GQA,
        must divide ``num_heads``.
    scale:
        Softmax scale. Defaults to ``1 / sqrt(head_dim)``.
    causal:
        Apply a (lower-triangular) causal mask. Required when
        ``sliding_window`` is set.
    sliding_window:
        Window size in tokens — current position counted in the window.
        Query at offset ``q_pos`` attends to keys
        ``[max(0, q_pos - W + 1), q_pos]``. ``None`` means full prefix.
    logits_soft_cap:
        If set, apply ``cap * tanh(logits / cap)`` to attention logits
        before softmax (Gemma2 / Grok / Gemini style).
    backend:
        ``"flashinfer"`` (default), ``"sdpa"``, or ``"eager"``.
    fi_workspace:
        Optional 1-D ``torch.uint8`` GPU tensor (recommended size
        ≥ 128 MiB; the global default is 128 MiB) to back the
        flashinfer split-k scratch. When given, the batched-ragged
        wrapper is built immediately at construction time. **Most
        callers should leave this ``None``**: by default every layer
        falls back to the process-global, per-device buffer (see
        :func:`get_global_fi_workspace`). Passing an explicit tensor is
        the escape hatch for callers who need a private scratch — for
        example a path that may run concurrently with another
        flashinfer-backed component on the same device. Ignored for
        non-flashinfer backends.

    Forward shape conventions
    -------------------------
    Two layouts are auto-detected by ``q.ndim``:

    * **Padded batch (4-D)**::

          q: (B, S_q,  H,    D)
          k: (B, S_kv, H_kv, D)
          v: (B, S_kv, H_kv, D)
          → out: (B, S_q, H, D)

      Every sequence in the batch shares the same length; per-token
      pad masking is not supported.

    * **Ragged / varlen (3-D)** — packed buffers plus indptrs::

          q: (N_q,  H,    D)
          k: (N_kv, H_kv, D)
          v: (N_kv, H_kv, D)
          cu_seqlens_q : int32 tensor, shape (B + 1,), starts at 0, monotonic
          cu_seqlens_kv: same; defaults to ``cu_seqlens_q``
          → out: (N_q, H, D)

    For "append" prefill where K/V is longer than Q, queries are aligned
    with the *trailing* keys (``q_pos[i] = i + (S_kv - S_q)``). This is
    the universal HF / vLLM / flashinfer convention.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        *,
        num_kv_heads: int | None = None,
        scale: float | None = None,
        causal: bool = True,
        sliding_window: int | None = None,
        logits_soft_cap: float | None = None,
        backend: str = "flashinfer",
        fi_workspace: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads={num_heads} must be a positive multiple of "
                f"num_kv_heads={num_kv_heads} for GQA."
            )
        if sliding_window is not None:
            if sliding_window <= 0:
                raise ValueError(
                    f"sliding_window must be positive, got {sliding_window}."
                )
            if not causal:
                raise ValueError("sliding_window requires causal=True.")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)
        self.causal = causal
        self.sliding_window = sliding_window
        self.logits_soft_cap = logits_soft_cap
        self.backend = _resolve_backend(backend)
        if self.backend == "flashinfer":
            # Fail at construction rather than first forward.
            try:
                import flashinfer.prefill  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "backend='flashinfer' but flashinfer is not installed; "
                    "either install flashinfer-python or pick "
                    "backend='sdpa'/'eager'."
                ) from e
        # Default workspace is the process-global, per-device buffer.
        # Construction stays cheap and CPU-only; the global tensor is
        # allocated lazily on the first ragged-batched forward, or
        # eagerly via :meth:`prepare`. Pass ``fi_workspace`` to wire in
        # a private scratch instead.
        self._fi_workspace: torch.Tensor | None = None
        self._fi_wrapper = None
        if self.backend == "flashinfer" and fi_workspace is not None:
            self._build_fi_wrapper(fi_workspace)

    # ------------------------------------------------------------------ #
    # Forward dispatch                                                   #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_kv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if q.ndim == 4:
            return self._forward_padded(q, k, v)
        if q.ndim == 3:
            if cu_seqlens_q is None:
                raise ValueError(
                    "ragged forward requires cu_seqlens_q (q has shape "
                    f"{tuple(q.shape)})."
                )
            cu_kv = cu_seqlens_kv if cu_seqlens_kv is not None else cu_seqlens_q
            return self._forward_ragged(q, k, v, cu_seqlens_q, cu_kv)
        raise ValueError(
            f"q must be 3-D (ragged) or 4-D (padded batch); got shape "
            f"{tuple(q.shape)}."
        )

    # ------------------------------------------------------------------ #
    # Padded batch                                                       #
    # ------------------------------------------------------------------ #

    def _forward_padded(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        if k.shape != v.shape:
            raise ValueError(
                f"k.shape={tuple(k.shape)} must equal v.shape={tuple(v.shape)}."
            )
        B, _, H_q, D = q.shape
        if H_q != self.num_heads or D != self.head_dim:
            raise ValueError(
                f"q heads/dim ({H_q}, {D}) does not match module "
                f"({self.num_heads}, {self.head_dim})."
            )
        if k.shape[0] != B or k.shape[2] != self.num_kv_heads or k.shape[3] != D:
            raise ValueError(
                f"k.shape={tuple(k.shape)} not compatible with q="
                f"{tuple(q.shape)} and num_kv_heads={self.num_kv_heads}."
            )
        if self.backend == "flashinfer":
            return self._padded_flashinfer(q, k, v)
        # sdpa / eager: convert to (B, H, S, D) layout, repeat KV for GQA.
        q_h = q.transpose(1, 2)
        k_h = self._maybe_repeat_kv(k.transpose(1, 2))
        v_h = self._maybe_repeat_kv(v.transpose(1, 2))
        if self.backend == "sdpa":
            out = self._sdpa(q_h, k_h, v_h)
        else:
            out = self._eager(q_h, k_h, v_h)
        return out.transpose(1, 2).contiguous()  # (B, S_q, H, D)

    def _padded_flashinfer(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        B, S_q, H, D = q.shape
        S_kv = k.shape[1]
        if B == 1:
            return self._flashinfer_single(q[0], k[0], v[0]).unsqueeze(0)
        device = q.device
        cu_q = torch.arange(0, (B + 1) * S_q, S_q, dtype=torch.int32, device=device)
        cu_kv = torch.arange(0, (B + 1) * S_kv, S_kv, dtype=torch.int32, device=device)
        out = self._flashinfer_batched(
            q.reshape(B * S_q, H, D),
            k.reshape(B * S_kv, self.num_kv_heads, D),
            v.reshape(B * S_kv, self.num_kv_heads, D),
            cu_q,
            cu_kv,
        )
        return out.reshape(B, S_q, H, D)

    # ------------------------------------------------------------------ #
    # Ragged                                                             #
    # ------------------------------------------------------------------ #

    def _forward_ragged(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_q: torch.Tensor,
        cu_kv: torch.Tensor,
    ) -> torch.Tensor:
        if k.shape != v.shape:
            raise ValueError(
                f"k.shape={tuple(k.shape)} must equal v.shape={tuple(v.shape)}."
            )
        _, H_q, D = q.shape
        _, H_kv, _ = k.shape
        if H_q != self.num_heads or D != self.head_dim or H_kv != self.num_kv_heads:
            raise ValueError(
                f"ragged input head/dim mismatch (q: {H_q}, {D}; k: {H_kv}); "
                f"expected ({self.num_heads}, {self.head_dim}) and "
                f"num_kv_heads={self.num_kv_heads}."
            )
        if self.backend == "flashinfer":
            B = cu_q.numel() - 1
            if B == 1:
                return self._flashinfer_single(q, k, v)
            return self._flashinfer_batched(q, k, v, cu_q, cu_kv)
        # eager / sdpa: per-sequence Python loop. Slow but correct;
        # flashinfer is the right pick when batched ragged perf matters.
        cu_q_list = cu_q.tolist()
        cu_kv_list = cu_kv.tolist()
        outs: list[torch.Tensor] = []
        for b in range(len(cu_q_list) - 1):
            qs, qe = cu_q_list[b], cu_q_list[b + 1]
            ks, ke = cu_kv_list[b], cu_kv_list[b + 1]
            qi = q[qs:qe].unsqueeze(0).transpose(1, 2)  # (1, H,    S_q,  D)
            ki = self._maybe_repeat_kv(k[ks:ke].unsqueeze(0).transpose(1, 2))
            vi = self._maybe_repeat_kv(v[ks:ke].unsqueeze(0).transpose(1, 2))
            if self.backend == "sdpa":
                oi = self._sdpa(qi, ki, vi)
            else:
                oi = self._eager(qi, ki, vi)
            outs.append(oi.transpose(1, 2).squeeze(0))  # (S_q, H, D)
        return torch.cat(outs, dim=0)

    # ------------------------------------------------------------------ #
    # SDPA / eager                                                       #
    # ------------------------------------------------------------------ #

    def _maybe_repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H_kv, S, D) → (B, H_q, S, D); identity for MHA.
        if self.num_heads == self.num_kv_heads:
            return x
        rep = self.num_heads // self.num_kv_heads
        return x.repeat_interleave(rep, dim=1)

    def _build_padded_mask(
        self, S_q: int, S_kv: int, device: torch.device
    ) -> torch.Tensor | None:
        # Bool mask (S_q, S_kv): True = attend. None when no masking is
        # needed (full non-causal). __init__ rejects sliding_window with
        # causal=False, so the only mask shapes here are causal /
        # causal+SWA.
        if not self.causal and self.sliding_window is None:
            return None
        i = torch.arange(S_q, device=device).unsqueeze(1)
        j = torch.arange(S_kv, device=device).unsqueeze(0)
        # Append-prefill alignment: q_pos[i] = i + (S_kv - S_q).
        q_pos = i + (S_kv - S_q)
        mask = q_pos >= j
        if self.sliding_window is not None:
            mask = mask & (q_pos - j < self.sliding_window)
        return mask

    def _sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.logits_soft_cap is not None:
            # SDPA has no soft-cap parameter; fall back to eager for this
            # call so the rest of the contract still holds.
            return self._eager(q, k, v)
        S_q = q.shape[-2]
        S_kv = k.shape[-2]
        if not self.causal and self.sliding_window is None:
            return F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        if self.causal and self.sliding_window is None and S_q == S_kv:
            # is_causal is only well-defined for square S_q == S_kv.
            return F.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=self.scale
            )
        mask = self._build_padded_mask(S_q, S_kv, q.device)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale)

    def _eager(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        S_q = q.shape[-2]
        S_kv = k.shape[-2]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if self.logits_soft_cap is not None:
            cap = self.logits_soft_cap
            attn = cap * torch.tanh(attn / cap)
        mask = self._build_padded_mask(S_q, S_kv, q.device)
        if mask is not None:
            attn = attn.masked_fill(~mask, float("-inf"))
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        return torch.matmul(attn, v)

    # ------------------------------------------------------------------ #
    # FlashInfer                                                         #
    # ------------------------------------------------------------------ #

    def _fi_window_left(self) -> int:
        # flashinfer's ``window_left`` is the max number of *previous*
        # keys visible (current position implicit), so a sliding_window
        # of W tokens — current included — maps to ``W - 1``.
        return -1 if self.sliding_window is None else self.sliding_window - 1

    def _build_fi_wrapper(self, workspace: torch.Tensor) -> None:
        from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

        if workspace.dtype != torch.uint8 or workspace.ndim != 1:
            raise ValueError(
                f"fi_workspace must be a 1-D uint8 tensor, got "
                f"shape={tuple(workspace.shape)}, dtype={workspace.dtype}."
            )
        self._fi_workspace = workspace
        self._fi_wrapper = BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD")

    def prepare(
        self,
        device: torch.device | str,
        *,
        workspace_bytes: int | None = None,
    ) -> "NoStateAttention":
        """Eagerly bind the flashinfer wrapper to the global scratch on ``device``.

        No-op for ``"sdpa"`` / ``"eager"`` backends, and a no-op when
        the wrapper has already been built (either by a previous call
        to this method, by passing ``fi_workspace`` to ``__init__``, or
        by a prior ragged batched forward).

        Returns ``self`` so it can chain after construction:

            attn = NoStateAttention(...).prepare("cuda:0")

        Triggers the global scratch allocation if it hasn't happened
        for ``device`` yet (see :func:`get_global_fi_workspace`).
        ``workspace_bytes`` lets the *first* caller for a given device
        pick a non-default size; once the global buffer for that device
        exists it is honored verbatim.
        """
        if self.backend == "flashinfer" and self._fi_wrapper is None:
            self._build_fi_wrapper(
                get_global_fi_workspace(device, workspace_bytes=workspace_bytes)
            )
        return self

    def _flashinfer_single(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        from flashinfer.prefill import single_prefill_with_kv_cache

        return single_prefill_with_kv_cache(
            q,
            k,
            v,
            causal=self.causal,
            kv_layout="NHD",
            sm_scale=self.scale,
            window_left=self._fi_window_left(),
            logits_soft_cap=self.logits_soft_cap,
        )

    def _ensure_fi_wrapper(self, device: torch.device):
        if self._fi_wrapper is None:
            self._build_fi_wrapper(get_global_fi_workspace(device))
        return self._fi_wrapper

    def _flashinfer_batched(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_q: torch.Tensor,
        cu_kv: torch.Tensor,
    ) -> torch.Tensor:
        wrapper = self._ensure_fi_wrapper(q.device)
        wrapper.plan(
            cu_q.to(torch.int32),
            cu_kv.to(torch.int32),
            num_qo_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim_qk=self.head_dim,
            causal=self.causal,
            sm_scale=self.scale,
            window_left=self._fi_window_left(),
            logits_soft_cap=self.logits_soft_cap,
            q_data_type=q.dtype,
            kv_data_type=k.dtype,
        )
        return wrapper.run(q, k, v)

    # ------------------------------------------------------------------ #

    def extra_repr(self) -> str:
        s = (
            f"num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, causal={self.causal}, "
            f"backend={self.backend!r}"
        )
        if self.sliding_window is not None:
            s += f", sliding_window={self.sliding_window}"
        if self.logits_soft_cap is not None:
            s += f", logits_soft_cap={self.logits_soft_cap}"
        return s


__all__ = ["NoStateAttention"]
