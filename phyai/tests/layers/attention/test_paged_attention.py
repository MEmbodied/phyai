"""Tests for paged-KV attention layers — ``ARAttention`` and ``DiffusionAttention``.

The two stacks share the eager contiguous-slab reference path
(:class:`EagerARBackend` / :class:`EagerDiffusionBackend`) and the
flashinfer paged kernel. Tests are parametrized over both layer
classes — they should behave identically today, modulo class /
ctx / metadata type names.

Numerical equivalence with :class:`Attention` (the no-cache stack) is
the primary correctness check; both should reduce to the same softmax
for a single-sample batch.

Contiguity contract: the eager paged backends reject non-contiguous
``paged_kv_indices`` per sample. Tests cover both the contiguous
happy path and the rejection path.
"""

from __future__ import annotations

import pytest
import torch

from phyai.cache import KVCachePool
from phyai.layers.attention import (
    ARAttention,
    ARAttentionBackend,
    ARAttnCtx,
    ARAttnMetadata,
    Attention,
    AttnLayout,
    AttnMode,
    DiffusionAttention,
    DiffusionAttentionBackend,
    DiffusionAttnCtx,
    DiffusionAttnMetadata,
    get_ar_backend_factory,
    get_diffusion_backend_factory,
)


# --------------------------------------------------------------------- #
# Parametrization helpers                                               #
# --------------------------------------------------------------------- #


_PAGED_FLAVORS = ("ar", "diffusion")


def _layer_cls(flavor: str):
    return ARAttention if flavor == "ar" else DiffusionAttention


def _ctx_cls(flavor: str):
    return ARAttnCtx if flavor == "ar" else DiffusionAttnCtx


def _meta_cls(flavor: str):
    return ARAttnMetadata if flavor == "ar" else DiffusionAttnMetadata


def _factory_for(flavor: str, name: str):
    if flavor == "ar":
        return get_ar_backend_factory(name)
    return get_diffusion_backend_factory(name)


def _make_pool(num_slots: int, num_kv_heads: int, head_dim: int, num_layers: int = 1):
    return KVCachePool(
        num_layers=num_layers,
        num_slots=num_slots,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def _make_eager_ctx(
    flavor: str,
    pool: KVCachePool,
    *,
    cu_seqlens_q: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    write_indices: torch.Tensor,
    batch_size: int,
    num_query_tokens: int,
):
    """Build a flavor-specific ctx for the eager paged backend."""
    backend = _factory_for(flavor, "eager")(None)
    meta = _meta_cls(flavor)(
        mode=AttnMode.PREFILL,
        layout=AttnLayout.RAGGED_3D,
        batch_size=batch_size,
        num_query_tokens=num_query_tokens,
        cu_seqlens_q=cu_seqlens_q,
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices,
        paged_kv_last_page_len=paged_kv_last_page_len,
        write_indices=write_indices,
    )
    plan = backend.init_forward_metadata(meta)
    return _ctx_cls(flavor)(
        backend=backend,
        plan=plan,
        mode=AttnMode.PREFILL,
        layout=AttnLayout.RAGGED_3D,
        kv_pool=pool,
        write_indices=write_indices,
    )


# --------------------------------------------------------------------- #
# Construction                                                          #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("flavor", _PAGED_FLAVORS)
def test_paged_attention_eager_backend_constructs(flavor: str):
    cls = _layer_cls(flavor)
    attn = cls(
        num_heads=4,
        head_dim=8,
        layer_id=0,
        num_kv_heads=4,
        backend="eager",
    )
    assert attn.backend == "eager"
    assert attn.num_heads == 4
    assert attn.num_kv_heads == 4
    assert attn.head_dim == 8
    assert attn.layer_id == 0


@pytest.mark.parametrize("flavor", _PAGED_FLAVORS)
def test_paged_attention_rejects_sdpa_backend(flavor: str):
    """SDPA cannot serve the paged space — only registered in the
    no-cache stack, so layer construction must reject it."""
    cls = _layer_cls(flavor)
    with pytest.raises(ValueError, match="not registered"):
        cls(num_heads=4, head_dim=8, layer_id=0, backend="sdpa")


@pytest.mark.parametrize("flavor", _PAGED_FLAVORS)
def test_paged_attention_rejects_invalid_backend(flavor: str):
    cls = _layer_cls(flavor)
    with pytest.raises(ValueError, match="not registered"):
        cls(
            num_heads=4,
            head_dim=8,
            layer_id=0,
            backend="not-a-backend",
        )


@pytest.mark.parametrize("flavor", _PAGED_FLAVORS)
def test_paged_attention_rejects_bad_gqa(flavor: str):
    cls = _layer_cls(flavor)
    with pytest.raises(ValueError, match="must be a positive multiple"):
        cls(
            num_heads=4,
            head_dim=8,
            layer_id=0,
            num_kv_heads=3,
            backend="eager",
        )


@pytest.mark.parametrize("flavor", _PAGED_FLAVORS)
def test_paged_attention_rejects_negative_layer_id(flavor: str):
    cls = _layer_cls(flavor)
    with pytest.raises(ValueError, match="layer_id must be non-negative"):
        cls(num_heads=4, head_dim=8, layer_id=-1, backend="eager")


# --------------------------------------------------------------------- #
# write_kv side effect                                                  #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("flavor", _PAGED_FLAVORS)
def test_forward_writes_k_v_to_pool_at_write_indices(flavor: str):
    """Backend (not layer) scatters K/V into the pool before computing attention."""
    pool = _make_pool(num_slots=8, num_kv_heads=2, head_dim=4, num_layers=2)
    cls = _layer_cls(flavor)
    attn = cls(
        num_heads=2,
        head_dim=4,
        layer_id=1,
        num_kv_heads=2,
        backend="eager",
    )

    N = 3
    q = torch.randn(N, 2, 4)
    k = torch.randn(N, 2, 4)
    v = torch.randn(N, 2, 4)
    write_indices = torch.tensor([3, 4, 5], dtype=torch.int64)
    ctx = _make_eager_ctx(
        flavor,
        pool,
        cu_seqlens_q=torch.tensor([0, N], dtype=torch.int32),
        paged_kv_indptr=torch.tensor([0, N], dtype=torch.int32),
        paged_kv_indices=torch.tensor([3, 4, 5], dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor([1], dtype=torch.int32),
        write_indices=write_indices,
        batch_size=1,
        num_query_tokens=N,
    )
    attn(q, k, v, ctx)

    for src_row, slot in enumerate([3, 4, 5]):
        assert torch.equal(pool.k_buffer(1)[slot, 0], k[src_row])
        assert torch.equal(pool.v_buffer(1)[slot, 0], v[src_row])
    for slot in [0, 1, 2, 6, 7]:
        assert torch.all(pool.k_buffer(1)[slot] == 0)
    assert torch.all(pool.k_buffer(0) == 0)


# --------------------------------------------------------------------- #
# Numerical correctness vs Attention (the no-cache reference)           #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("flavor", _PAGED_FLAVORS)
def test_eager_backend_matches_attention_single_sample(flavor: str):
    """For a single sample with all real tokens, the paged layer must
    produce the same output as :class:`Attention` (both reduce to the
    same softmax over the same Q/K/V)."""
    torch.manual_seed(0)
    H, H_kv, D = 4, 2, 8
    N = 6

    q = torch.randn(N, H, D)
    k = torch.randn(N, H_kv, D)
    v = torch.randn(N, H_kv, D)

    ref = Attention(
        num_heads=H,
        head_dim=D,
        num_kv_heads=H_kv,
        causal=False,
        backend="eager",
    )
    cu_q_ragged = torch.tensor([0, N], dtype=torch.int32)
    ref_out = ref(q, k, v, cu_seqlens_q=cu_q_ragged, cu_seqlens_kv=cu_q_ragged)

    pool = _make_pool(num_slots=N, num_kv_heads=H_kv, head_dim=D)
    cls = _layer_cls(flavor)
    attn = cls(
        num_heads=H,
        head_dim=D,
        layer_id=0,
        num_kv_heads=H_kv,
        causal=False,
        backend="eager",
    )
    ctx = _make_eager_ctx(
        flavor,
        pool,
        cu_seqlens_q=torch.tensor([0, N], dtype=torch.int32),
        paged_kv_indptr=torch.tensor([0, N], dtype=torch.int32),
        paged_kv_indices=torch.arange(N, dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor([1], dtype=torch.int32),
        write_indices=torch.arange(N, dtype=torch.int64),
        batch_size=1,
        num_query_tokens=N,
    )
    out = attn(q, k, v, ctx)
    assert torch.allclose(out, ref_out, atol=1e-5)


@pytest.mark.parametrize("flavor", _PAGED_FLAVORS)
def test_eager_backend_two_samples_disjoint_kv(flavor: str):
    """Per-sample eager keeps K/V isolated to each sample's slot range."""
    torch.manual_seed(1)
    H, D = 2, 8
    N0, N1 = 3, 5

    q = torch.randn(N0 + N1, H, D)
    k = torch.randn(N0 + N1, H, D)
    v = torch.randn(N0 + N1, H, D)

    ref = Attention(
        num_heads=H,
        head_dim=D,
        num_kv_heads=H,
        causal=False,
        backend="eager",
    )
    cu_q = torch.tensor([0, N0, N0 + N1], dtype=torch.int32)
    ref_out = ref(q, k, v, cu_seqlens_q=cu_q, cu_seqlens_kv=cu_q)

    pool = _make_pool(num_slots=N0 + N1, num_kv_heads=H, head_dim=D)
    cls = _layer_cls(flavor)
    attn = cls(
        num_heads=H,
        head_dim=D,
        layer_id=0,
        num_kv_heads=H,
        causal=False,
        backend="eager",
    )
    ctx = _make_eager_ctx(
        flavor,
        pool,
        cu_seqlens_q=cu_q,
        paged_kv_indptr=cu_q,
        paged_kv_indices=torch.arange(N0 + N1, dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor([1, 1], dtype=torch.int32),
        write_indices=torch.arange(N0 + N1, dtype=torch.int64),
        batch_size=2,
        num_query_tokens=N0 + N1,
    )
    out = attn(q, k, v, ctx)
    assert torch.allclose(out, ref_out, atol=1e-5)


@pytest.mark.parametrize("flavor", _PAGED_FLAVORS)
def test_eager_backend_contiguous_offset_slab(flavor: str):
    """The contiguity contract is per-sample, not per-pool. Sample 0 may
    live at a non-zero slot base [1, 2, 3] as long as the indices are
    contiguous within that range."""
    torch.manual_seed(2)
    H, D = 2, 4
    N = 3

    q = torch.randn(N, H, D)
    k = torch.randn(N, H, D)
    v = torch.randn(N, H, D)

    ref = Attention(
        num_heads=H,
        head_dim=D,
        num_kv_heads=H,
        causal=False,
        backend="eager",
    )
    ref_out = ref(q, k, v, cu_seqlens_q=torch.tensor([0, N], dtype=torch.int32))

    pool = _make_pool(num_slots=4, num_kv_heads=H, head_dim=D)
    cls = _layer_cls(flavor)
    attn = cls(
        num_heads=H,
        head_dim=D,
        layer_id=0,
        num_kv_heads=H,
        causal=False,
        backend="eager",
    )
    ctx = _make_eager_ctx(
        flavor,
        pool,
        cu_seqlens_q=torch.tensor([0, N], dtype=torch.int32),
        paged_kv_indptr=torch.tensor([0, N], dtype=torch.int32),
        paged_kv_indices=torch.tensor([1, 2, 3], dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor([1], dtype=torch.int32),
        write_indices=torch.tensor([1, 2, 3], dtype=torch.int64),
        batch_size=1,
        num_query_tokens=N,
    )
    out = attn(q, k, v, ctx)
    assert torch.allclose(out, ref_out, atol=1e-5)


# --------------------------------------------------------------------- #
# Contiguity contract                                                   #
# --------------------------------------------------------------------- #


def test_diffusion_eager_backend_rejects_non_contiguous_indices():
    """The diffusion eager backend still requires per-sample contiguous
    paged_kv_indices (no radix consumer). The AR eager backend gathers
    instead — see test_ar_eager_backend_gathers_non_contiguous_indices."""
    backend = get_diffusion_backend_factory("eager")(None)
    meta = DiffusionAttnMetadata(
        mode=AttnMode.PREFILL,
        layout=AttnLayout.RAGGED_3D,
        batch_size=1,
        num_query_tokens=3,
        cu_seqlens_q=torch.tensor([0, 3], dtype=torch.int32),
        paged_kv_indptr=torch.tensor([0, 3], dtype=torch.int32),
        paged_kv_indices=torch.tensor([0, 2, 4], dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor([1], dtype=torch.int32),
        write_indices=torch.tensor([0, 2, 4], dtype=torch.int64),
    )
    with pytest.raises(ValueError, match="contiguous KV slots per sample"):
        backend.init_forward_metadata(meta)


# --------------------------------------------------------------------- #
# Validation                                                            #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("flavor", _PAGED_FLAVORS)
def test_eager_backend_requires_paged_metadata(flavor: str):
    """Plan-time validation: missing paged_kv_* fields raise."""
    backend = _factory_for(flavor, "eager")(None)
    with pytest.raises(ValueError, match="cu_seqlens_q"):
        backend.init_forward_metadata(
            _meta_cls(flavor)(
                mode=AttnMode.PREFILL,
                layout=AttnLayout.PADDED_4D,
                batch_size=1,
                num_query_tokens=1,
            )
        )


# --------------------------------------------------------------------- #
# Sanity: ar and diffusion backend classes are independent              #
# --------------------------------------------------------------------- #


def test_ar_and_diffusion_backends_are_independent_classes():
    """The two stacks have separate registries and separate backend
    classes — neither should accidentally alias the other."""
    ar_eager = get_ar_backend_factory("eager")
    diff_eager = get_diffusion_backend_factory("eager")
    assert ar_eager is not diff_eager
    assert isinstance(ar_eager(None), ARAttentionBackend)
    assert isinstance(diff_eager(None), DiffusionAttentionBackend)


def test_ar_eager_backend_gathers_non_contiguous_indices():
    """AR eager now gathers arbitrary (radix-reuse) slot layouts; the result
    matches a recompute over the same logical KV order."""
    torch.manual_seed(3)
    H, D, N = 2, 4, 4
    q = torch.randn(N, H, D)
    k = torch.randn(N, H, D)
    v = torch.randn(N, H, D)
    ref = Attention(
        num_heads=H, head_dim=D, num_kv_heads=H, causal=False, backend="eager"
    )
    ref_out = ref(q, k, v, cu_seqlens_q=torch.tensor([0, N], dtype=torch.int32))

    pool = _make_pool(num_slots=8, num_kv_heads=H, head_dim=D)
    attn = ARAttention(
        num_heads=H,
        head_dim=D,
        layer_id=0,
        num_kv_heads=H,
        causal=False,
        backend="eager",
    )
    slots = torch.tensor([1, 3, 5, 7], dtype=torch.int64)  # non-contiguous
    ctx = _make_eager_ctx(
        "ar",
        pool,
        cu_seqlens_q=torch.tensor([0, N], dtype=torch.int32),
        paged_kv_indptr=torch.tensor([0, N], dtype=torch.int32),
        paged_kv_indices=slots.to(torch.int32),
        paged_kv_last_page_len=torch.tensor([1], dtype=torch.int32),
        write_indices=slots,
        batch_size=1,
        num_query_tokens=N,
    )
    out = attn(q, k, v, ctx)
    assert torch.allclose(out, ref_out, atol=1e-5)
