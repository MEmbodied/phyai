"""Static-allocation cache-pool-aware varlen attention for cuda-graph-friendly inference.

The attention layer here is the bridge between the
:class:`~phyai.cache.kv_cache_pool.KVCachePool` and the per-layer Q/K/V
projections in the model. It writes the freshly-projected K/V into the
pool at caller-specified slot indices, then computes attention reading
K/V back through one of two backends:

* ``"flashinfer-paged"`` (default, production) — runs a
  :class:`flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper` whose
  ``plan()`` was called by the caller (the runner) outside the captured
  CUDA graph. The wrapper is constructed with ``use_cuda_graph=True``
  and pre-allocated index buffers; each replay's ``plan()`` only
  ``.copy_()``-es new metadata into those buffers, leaving the captured
  ``run()`` kernels seeing the updated values.
* ``"sdpa"`` (CPU / CI fallback) — Python-level segment loop that
  gathers per-sample slot indices into contiguous K/V tensors and runs
  :func:`torch.nn.functional.scaled_dot_product_attention`. Slower; not
  graph-captureable; intended for tests without flashinfer.

Forward contract
----------------
:class:`StaticCachedAttention` is constructed with its ``layer_id``
baked in and is owned by a single transformer layer. Forward takes the
ragged ``(N, H_q, D)`` ``q`` plus ``(N, H_kv, D)`` ``k`` / ``v`` and a
single :class:`StaticCachedAttnCtx` packaging every per-call piece of
KV plumbing the runner builds once per forward pass:

* ``ctx.kv_pool`` — the :class:`KVCachePool` to scatter K/V into.
* ``ctx.write_indices`` — ``(N,)`` ``int64`` slot indices for the K/V
  scatter; padding rows write to a sentinel slot (typically index 0)
  and are never read by attention.
* ``ctx.attn_wrapper`` — the pre-planned flashinfer wrapper (paged
  path), or ``None`` to dispatch to sdpa.
* ``ctx.cu_seqlens_q`` / ``ctx.paged_kv_indptr`` / ``ctx.paged_kv_indices``
  — sdpa-only ragged metadata; ignored when ``attn_wrapper`` is set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from phyai.cache import KVCachePool


_VALID_BACKENDS: tuple[str, ...] = ("flashinfer-paged", "sdpa")


def _resolve_backend(name: str) -> str:
    canonical = name.lower().replace("_", "-")
    if canonical not in _VALID_BACKENDS:
        raise ValueError(
            f"Unknown StaticCachedAttention backend {name!r}; expected one of "
            f"{_VALID_BACKENDS!r}."
        )
    return canonical


@dataclass(frozen=True)
class StaticCachedAttnCtx:
    """Per-call KV plumbing shared across every layer of a single forward.

    Built once by the runner (or any caller) per forward pass and
    handed to each layer's :meth:`StaticCachedAttention.forward`. Every
    layer reads the same ctx but consults its own ``layer_id`` (baked
    into the attention module) when scattering K/V into ``kv_pool``.

    The ``write_indices`` tensor is per-token, not per-layer: every
    layer writes its (different) K/V values to the same row index
    inside its own layer's pool buffer.

    Fields
    ------
    kv_pool:
        Per-layer K/V buffer pool the layer scatters into.
    write_indices:
        ``(N,)`` ``int64`` — slot index per token. Padding rows point
        at a sentinel slot whose contents attention never reads.
    attn_wrapper:
        Pre-planned flashinfer ``BatchPrefillWithPagedKVCacheWrapper``
        for the paged backend, or ``None`` to dispatch to sdpa.
    cu_seqlens_q / paged_kv_indptr / paged_kv_indices:
        sdpa-only ragged metadata. Ignored when ``attn_wrapper`` is
        present.
    """

    kv_pool: KVCachePool
    write_indices: torch.Tensor
    attn_wrapper: Any | None = None
    cu_seqlens_q: torch.Tensor | None = None
    paged_kv_indptr: torch.Tensor | None = None
    paged_kv_indices: torch.Tensor | None = None


class StaticCachedAttention(nn.Module):
    """Static-allocation cache-pool-aware varlen attention with selectable backend.

    "Static" means the KV slots come from a one-shot
    :class:`~phyai.cache.static_cache.StaticCache` allocator (contiguous
    range, reset between requests, no eviction), as opposed to the
    radix-cache facades under :mod:`phyai.cache.radix_cache`. The
    attention module itself is allocator-agnostic — it just writes K/V
    at caller-supplied slot indices and reads them back through paged
    flashinfer or the sdpa fallback.

    Each instance is owned by a single transformer layer; ``layer_id``
    is baked in at construction so the per-forward call site reads
    cleanly: ``self.attn(q, k, v, ctx)``.

    Parameters
    ----------
    num_heads:
        Query head count.
    head_dim:
        Per-head dimension.
    layer_id:
        Index into the :class:`KVCachePool`'s per-layer K/V buffers.
        Required.
    num_kv_heads:
        K/V head count (defaults to ``num_heads`` for full MHA;
        ``num_heads // num_kv_heads`` is the GQA replication factor).
    scale:
        Softmax scale, defaults to ``1 / sqrt(head_dim)``.
    causal:
        Causal mask flag. Set ``False`` for bidirectional / encoder-style
        attention, ``True`` for autoregressive decoders.
    backend:
        ``"flashinfer-paged"`` (default) or ``"sdpa"``.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        *,
        layer_id: int,
        num_kv_heads: int | None = None,
        scale: float | None = None,
        causal: bool = False,
        backend: str = "flashinfer-paged",
    ) -> None:
        super().__init__()
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads={num_heads} must be a positive multiple of "
                f"num_kv_heads={num_kv_heads} for GQA."
            )
        if layer_id < 0:
            raise ValueError(f"layer_id must be non-negative, got {layer_id}.")
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.layer_id = int(layer_id)
        self.scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)
        self.causal = bool(causal)
        self.backend = _resolve_backend(backend)
        if self.backend == "flashinfer-paged":
            try:
                import flashinfer.prefill  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "backend='flashinfer-paged' but flashinfer is not "
                    "installed; either install flashinfer-python or pick "
                    "backend='sdpa'."
                ) from e

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: StaticCachedAttnCtx,
    ) -> torch.Tensor:
        """Write K/V to the pool and compute attention.

        Returns ``(N, H_q, D)`` — same row count as ``q``.
        """
        if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
            raise ValueError(
                f"q/k/v must be 3-D (N, H, D); got q={tuple(q.shape)}, "
                f"k={tuple(k.shape)}, v={tuple(v.shape)}."
            )
        if q.shape[-2] != self.num_heads or q.shape[-1] != self.head_dim:
            raise ValueError(
                f"q heads/dim ({q.shape[-2]}, {q.shape[-1]}) does not match "
                f"module ({self.num_heads}, {self.head_dim})."
            )
        if (
            k.shape[-2] != self.num_kv_heads
            or k.shape[-1] != self.head_dim
            or k.shape != v.shape
        ):
            raise ValueError(
                f"k/v shape mismatch: k={tuple(k.shape)}, v={tuple(v.shape)}, "
                f"expected ({q.shape[0]}, {self.num_kv_heads}, "
                f"{self.head_dim})."
            )
        if k.shape[0] != q.shape[0]:
            raise ValueError(f"k row count {k.shape[0]} != q row count {q.shape[0]}.")

        # 1. Scatter K/V into the cache pool. Captureable via index_put_.
        ctx.kv_pool.write_kv(self.layer_id, ctx.write_indices, k, v)

        # 2. Compute attention.
        if self.backend == "flashinfer-paged":
            if ctx.attn_wrapper is None:
                raise ValueError(
                    "backend='flashinfer-paged' requires ctx.attn_wrapper to "
                    "be set; the runner is responsible for constructing and "
                    "planning the wrapper outside the captured graph."
                )
            return self._forward_flashinfer_paged(q, ctx.kv_pool, ctx.attn_wrapper)
        return self._forward_sdpa(q, ctx)

    # ------------------------------------------------------------------ #
    # flashinfer-paged                                                   #
    # ------------------------------------------------------------------ #

    def _forward_flashinfer_paged(
        self,
        q: torch.Tensor,
        kv_pool: KVCachePool,
        attn_wrapper: Any,
    ) -> torch.Tensor:
        """Run the pre-planned wrapper against ``kv_pool``'s layer buffers.

        The wrapper has already been told the qo_indptr / paged_kv_*
        tensors for this batch; ``run`` only needs ``q`` and the
        ``(K, V)`` cache view. Both are tensors with stable storage
        across CUDA-graph replays.
        """
        # ``kv_buffer`` returns the pool's
        # ``(num_slots, page_size, num_kv_heads, head_dim)`` buffers
        # directly — no copy. flashinfer's wrapper reads through the
        # pre-planned paged_kv_indices.
        k_cache, v_cache = kv_pool.kv_buffer(self.layer_id)
        # The wrapper's run() can take either a stacked (N, 2, ...) tensor
        # or a (k_cache, v_cache) tuple. The tuple form keeps the code
        # symmetric with our pool layout.
        return attn_wrapper.run(q, (k_cache, v_cache))

    # ------------------------------------------------------------------ #
    # sdpa fallback                                                      #
    # ------------------------------------------------------------------ #

    def _forward_sdpa(
        self,
        q: torch.Tensor,
        ctx: StaticCachedAttnCtx,
    ) -> torch.Tensor:
        """Per-sample sdpa with gathered K/V — Python loop over the batch.

        Not captureable in a CUDA graph, not the production hot path.
        Used by CPU / CI tests and as a numerical reference for
        flashinfer.
        """
        if (
            ctx.cu_seqlens_q is None
            or ctx.paged_kv_indptr is None
            or ctx.paged_kv_indices is None
        ):
            raise ValueError(
                "backend='sdpa' requires ctx.cu_seqlens_q, ctx.paged_kv_indptr, "
                "and ctx.paged_kv_indices to identify each sample's query and "
                "KV ranges."
            )
        cu_q = ctx.cu_seqlens_q.to(torch.int64).tolist()
        cu_kv = ctx.paged_kv_indptr.to(torch.int64).tolist()
        out = torch.empty_like(q)
        rep = self.num_heads // self.num_kv_heads
        for b in range(len(cu_q) - 1):
            q_start, q_end = cu_q[b], cu_q[b + 1]
            kv_start, kv_end = cu_kv[b], cu_kv[b + 1]
            if q_end == q_start:
                continue
            slot_ids = ctx.paged_kv_indices[kv_start:kv_end].to(torch.int64)
            if slot_ids.numel() == 0:
                # No KV to attend to — define output as zeros (rare, only
                # valid if the matching sample is entirely empty, which
                # the scheduler should avoid).
                out[q_start:q_end] = 0
                continue
            k_seg, v_seg = ctx.kv_pool.gather_kv(self.layer_id, slot_ids)
            # (1, H, S_q, D) layouts for sdpa
            q_seg = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
            k_h = k_seg.transpose(0, 1).unsqueeze(0)
            v_h = v_seg.transpose(0, 1).unsqueeze(0)
            if rep > 1:
                k_h = k_h.repeat_interleave(rep, dim=1)
                v_h = v_h.repeat_interleave(rep, dim=1)
            attn = F.scaled_dot_product_attention(
                q_seg, k_h, v_h, is_causal=self.causal, scale=self.scale
            )  # (1, H, S_q, D)
            out[q_start:q_end] = attn.squeeze(0).transpose(0, 1)
        return out

    # ------------------------------------------------------------------ #

    def extra_repr(self) -> str:
        return (
            f"num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, layer_id={self.layer_id}, "
            f"causal={self.causal}, backend={self.backend!r}"
        )


__all__ = ["StaticCachedAttention", "StaticCachedAttnCtx"]
