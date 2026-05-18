"""Tests for :class:`phyai.layers.attention.StaticCachedAttention`.

The flashinfer-paged backend requires a planned wrapper, which the
runner constructs in Phase 5; tests here exercise the sdpa fallback
(graph-incompatible but exact reference) plus the cache-pool write
side-effect that both backends share. Numerical equivalence with
:class:`NoStateAttention` is the primary correctness check — both
should reduce to the same softmax for a single-sample batch.
"""

from __future__ import annotations

import pytest
import torch

from phyai.cache import KVCachePool
from phyai.layers.attention import (
    NoStateAttention,
    StaticCachedAttention,
    StaticCachedAttnCtx,
)


# --------------------------------------------------------------------- #
# Construction                                                          #
# --------------------------------------------------------------------- #


def test_cached_attention_sdpa_backend_constructs():
    attn = StaticCachedAttention(
        num_heads=4,
        head_dim=8,
        layer_id=0,
        num_kv_heads=4,
        backend="sdpa",
    )
    assert attn.backend == "sdpa"
    assert attn.num_heads == 4
    assert attn.num_kv_heads == 4
    assert attn.head_dim == 8
    assert attn.layer_id == 0


def test_cached_attention_rejects_invalid_backend():
    with pytest.raises(ValueError, match="Unknown StaticCachedAttention backend"):
        StaticCachedAttention(
            num_heads=4, head_dim=8, layer_id=0, backend="not-a-backend"
        )


def test_cached_attention_rejects_bad_gqa():
    with pytest.raises(ValueError, match="must be a positive multiple"):
        StaticCachedAttention(
            num_heads=4, head_dim=8, layer_id=0, num_kv_heads=3, backend="sdpa"
        )


def test_cached_attention_rejects_negative_layer_id():
    with pytest.raises(ValueError, match="layer_id must be non-negative"):
        StaticCachedAttention(num_heads=4, head_dim=8, layer_id=-1, backend="sdpa")


# --------------------------------------------------------------------- #
# write_kv side effect                                                  #
# --------------------------------------------------------------------- #


def test_forward_writes_k_v_to_pool_at_write_indices():
    """Both backends must scatter K/V into the pool before computing attention."""
    pool = KVCachePool(
        num_layers=2,
        num_slots=8,
        num_kv_heads=2,
        head_dim=4,
        dtype=torch.float32,
        device="cpu",
    )
    attn = StaticCachedAttention(
        num_heads=2, head_dim=4, layer_id=1, num_kv_heads=2, backend="sdpa"
    )

    N = 3
    q = torch.randn(N, 2, 4)
    k = torch.randn(N, 2, 4)
    v = torch.randn(N, 2, 4)
    ctx = StaticCachedAttnCtx(
        kv_pool=pool,
        write_indices=torch.tensor([1, 3, 5], dtype=torch.int64),
        cu_seqlens_q=torch.tensor([0, N], dtype=torch.int32),
        paged_kv_indptr=torch.tensor([0, N], dtype=torch.int32),
        paged_kv_indices=torch.tensor([1, 3, 5], dtype=torch.int32),
    )

    attn(q, k, v, ctx)

    # Layer 1 slots [1, 3, 5] now hold our K/V values; other slots zero.
    for src_row, slot in enumerate([1, 3, 5]):
        assert torch.equal(pool.k_buffer(1)[slot, 0], k[src_row])
        assert torch.equal(pool.v_buffer(1)[slot, 0], v[src_row])
    # Other slots untouched.
    for slot in [0, 2, 4, 6, 7]:
        assert torch.all(pool.k_buffer(1)[slot] == 0)
    # Layer 0 untouched.
    assert torch.all(pool.k_buffer(0) == 0)


# --------------------------------------------------------------------- #
# Numerical correctness vs NoStateAttention                             #
# --------------------------------------------------------------------- #


def test_sdpa_backend_matches_no_state_attention_single_sample():
    """For a single sample with all real tokens, StaticCachedAttention must
    produce the same output as NoStateAttention (both run sdpa over the
    same Q/K/V).
    """
    torch.manual_seed(0)
    H, H_kv, D = 4, 2, 8
    N = 6  # single sample, no padding

    q = torch.randn(N, H, D)
    k = torch.randn(N, H_kv, D)
    v = torch.randn(N, H_kv, D)

    # Reference: NoStateAttention non-causal
    ref = NoStateAttention(
        num_heads=H,
        head_dim=D,
        num_kv_heads=H_kv,
        causal=False,
        backend="sdpa",
    )
    cu_q_ragged = torch.tensor([0, N], dtype=torch.int32)
    ref_out = ref(q, k, v, cu_seqlens_q=cu_q_ragged, cu_seqlens_kv=cu_q_ragged)

    # Cached: write to pool at slots 0..N-1, attention reads them back.
    pool = KVCachePool(
        num_layers=1,
        num_slots=N,
        num_kv_heads=H_kv,
        head_dim=D,
        dtype=torch.float32,
        device="cpu",
    )
    attn = StaticCachedAttention(
        num_heads=H,
        head_dim=D,
        layer_id=0,
        num_kv_heads=H_kv,
        causal=False,
        backend="sdpa",
    )
    ctx = StaticCachedAttnCtx(
        kv_pool=pool,
        write_indices=torch.arange(N, dtype=torch.int64),
        cu_seqlens_q=torch.tensor([0, N], dtype=torch.int32),
        paged_kv_indptr=torch.tensor([0, N], dtype=torch.int32),
        paged_kv_indices=torch.arange(N, dtype=torch.int32),
    )
    out = attn(q, k, v, ctx)
    assert torch.allclose(out, ref_out, atol=1e-5)


def test_sdpa_backend_two_samples_disjoint_kv():
    """Per-sample sdpa keeps K/V isolated to each sample's slot range."""
    torch.manual_seed(1)
    H, D = 2, 8
    N0, N1 = 3, 5  # two samples, no padding

    q = torch.randn(N0 + N1, H, D)
    k = torch.randn(N0 + N1, H, D)
    v = torch.randn(N0 + N1, H, D)

    # Reference: NoStateAttention with cu_seqlens spanning both samples.
    ref = NoStateAttention(
        num_heads=H,
        head_dim=D,
        num_kv_heads=H,
        causal=False,
        backend="sdpa",
    )
    cu_q = torch.tensor([0, N0, N0 + N1], dtype=torch.int32)
    ref_out = ref(q, k, v, cu_seqlens_q=cu_q, cu_seqlens_kv=cu_q)

    # Cached: pool slots [0..N0) and [N0..N0+N1) per sample.
    pool = KVCachePool(
        num_layers=1,
        num_slots=N0 + N1,
        num_kv_heads=H,
        head_dim=D,
        dtype=torch.float32,
        device="cpu",
    )
    attn = StaticCachedAttention(
        num_heads=H,
        head_dim=D,
        layer_id=0,
        num_kv_heads=H,
        causal=False,
        backend="sdpa",
    )
    ctx = StaticCachedAttnCtx(
        kv_pool=pool,
        write_indices=torch.arange(N0 + N1, dtype=torch.int64),
        cu_seqlens_q=cu_q,
        paged_kv_indptr=cu_q,
        paged_kv_indices=torch.arange(N0 + N1, dtype=torch.int32),
    )
    out = attn(q, k, v, ctx)
    assert torch.allclose(out, ref_out, atol=1e-5)


def test_sdpa_backend_paged_indices_are_sample_local():
    """paged_kv_indices may point at non-contiguous slots (sentinel pattern).

    Sample 0 writes to slots [1, 2, 3] (skipping sentinel slot 0); the
    paged_kv_indices tells attention to read [1, 2, 3]. Output should
    match a vanilla single-sample attention over the same Q/K/V.
    """
    torch.manual_seed(2)
    H, D = 2, 4
    N = 3

    q = torch.randn(N, H, D)
    k = torch.randn(N, H, D)
    v = torch.randn(N, H, D)

    ref = NoStateAttention(
        num_heads=H, head_dim=D, num_kv_heads=H, causal=False, backend="sdpa"
    )
    ref_out = ref(q, k, v, cu_seqlens_q=torch.tensor([0, N], dtype=torch.int32))

    pool = KVCachePool(
        num_layers=1,
        num_slots=4,
        num_kv_heads=H,
        head_dim=D,
        dtype=torch.float32,
        device="cpu",
    )
    attn = StaticCachedAttention(
        num_heads=H,
        head_dim=D,
        layer_id=0,
        num_kv_heads=H,
        causal=False,
        backend="sdpa",
    )
    ctx = StaticCachedAttnCtx(
        kv_pool=pool,
        write_indices=torch.tensor([1, 2, 3], dtype=torch.int64),
        cu_seqlens_q=torch.tensor([0, N], dtype=torch.int32),
        paged_kv_indptr=torch.tensor([0, N], dtype=torch.int32),
        paged_kv_indices=torch.tensor([1, 2, 3], dtype=torch.int32),
    )
    out = attn(q, k, v, ctx)
    assert torch.allclose(out, ref_out, atol=1e-5)


# --------------------------------------------------------------------- #
# Validation                                                            #
# --------------------------------------------------------------------- #


def test_sdpa_backend_requires_indptr_args():
    pool = KVCachePool(
        num_layers=1,
        num_slots=4,
        num_kv_heads=1,
        head_dim=2,
        dtype=torch.float32,
        device="cpu",
    )
    attn = StaticCachedAttention(
        num_heads=1, head_dim=2, layer_id=0, num_kv_heads=1, backend="sdpa"
    )
    ctx = StaticCachedAttnCtx(
        kv_pool=pool,
        write_indices=torch.tensor([0], dtype=torch.int64),
    )
    with pytest.raises(ValueError, match="cu_seqlens_q"):
        attn(
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, 2),
            ctx,
        )
