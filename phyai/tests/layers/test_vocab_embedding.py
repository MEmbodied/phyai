"""VocabParallelEmbedding + ParallelLMHead integration tests.

ws=1 tests run via a mocked Mesh — :func:`phyai.parallel.all_reduce` and
:func:`phyai.parallel.all_gather` short-circuit when the axis size is 1,
so we can exercise construction, weight allocation, masked-lookup, and
forward without a real process group. Multi-rank correctness lives under
the existing multiprocess gloo harness and is out of scope here.

Vocab shard-bound math is exercised end-to-end through the loader's
``vocab(...)`` factory by invoking the param-attached ``weight_loader``
across mocked ``tp_rank`` / ``tp_size`` combinations.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import phyai.layers.linear as L
from phyai.layers.vocab_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
    pad_vocab_to,
)
from phyai.parallel.mesh import Mesh
from phyai.parallel.state import _meshes, register_mesh


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


def _make_fake_mesh(
    *,
    name: str = "model",
    sizes: dict[str, int] | None = None,
    ranks: dict[str, int] | None = None,
) -> Mesh:
    sizes = sizes or {"tp": 1}
    ranks = ranks or {}

    tm = MagicMock()
    tm.mesh_dim_names = tuple(sizes.keys())
    _names = tm.mesh_dim_names

    def _size(axis):
        if isinstance(axis, str):
            return sizes.get(axis, 1)
        return sizes.get(_names[axis], 1)

    tm.size.side_effect = _size
    tm.get_local_rank.side_effect = lambda axis: ranks.get(axis, 0)
    tm.get_group.side_effect = lambda axis: MagicMock(name=f"pg-{axis}")
    mesh = Mesh(tm, name=name)
    register_mesh(mesh)
    return mesh


@pytest.fixture
def fake_mesh():
    saved = dict(_meshes)
    try:
        yield _make_fake_mesh
    finally:
        _meshes.clear()
        _meshes.update(saved)
        L._reset_for_test()


def _init_dispatcher():
    """Init phyai.layers.linear without flashinfer / sample-spec validation."""
    return L.init(register_flashinfer=False, validate=False)


# --------------------------------------------------------------------------- #
# pad_vocab_to                                                                #
# --------------------------------------------------------------------------- #


def test_pad_vocab_already_aligned():
    assert pad_vocab_to(32000, tp_size=2, multiple=64) == 32000


def test_pad_vocab_rounds_up():
    # 151936 % 256 = 0 already (256 = 4*64), so identity. Pick a value that
    # actually triggers the round-up branch.
    assert pad_vocab_to(151700, tp_size=4, multiple=64) == 151808


def test_pad_vocab_respects_multiple():
    # FP8-style 128 alignment.
    assert pad_vocab_to(100, tp_size=2, multiple=128) == 256
    assert pad_vocab_to(257, tp_size=2, multiple=128) == 512


def test_pad_vocab_rejects_zero_tp():
    with pytest.raises(ValueError):
        pad_vocab_to(100, tp_size=0, multiple=64)


# --------------------------------------------------------------------------- #
# Vocab loader factory — covers all shard-bounds edge cases without needing a #
# constructed VocabParallelEmbedding (just an nn.Parameter with the right    #
# shape).                                                                      #
# --------------------------------------------------------------------------- #


def test_vocab_loader_tp1_full_copy(fake_mesh):
    fake_mesh(sizes={"tp": 1})
    from phyai.parallel.state import resolve_mesh
    from phyai.weights.shards import vocab

    V, D = 100, 16
    disk = torch.arange(V * D, dtype=torch.float32).reshape(V, D)
    p = nn.Parameter(torch.zeros(V, D), requires_grad=False)
    loader = vocab(axis="tp", mesh=resolve_mesh("model"))
    loader(p, disk, None)
    assert torch.equal(p.data, disk)


def test_vocab_loader_tp4_evenly_divisible(fake_mesh):
    from phyai.parallel.state import resolve_mesh
    from phyai.weights.shards import vocab

    V, D, tp = 256, 8, 4
    disk = torch.arange(V * D, dtype=torch.float32).reshape(V, D)
    per_rank = V // tp
    for rank in range(tp):
        fake_mesh(sizes={"tp": tp}, ranks={"tp": rank})
        p = nn.Parameter(torch.zeros(per_rank, D), requires_grad=False)
        loader = vocab(axis="tp", mesh=resolve_mesh("model"))
        loader(p, disk, None)
        assert torch.equal(p.data, disk.narrow(0, rank * per_rank, per_rank))
        _meshes.clear()


def test_vocab_loader_padding_overhang_zeros_tail(fake_mesh):
    from phyai.parallel.state import resolve_mesh
    from phyai.weights.shards import vocab

    V, V_padded, D, tp = 100, 128, 8, 4
    disk = torch.randn(V, D, dtype=torch.float32)
    per_rank = V_padded // tp  # 32

    # Rank 3: real rows 96..100 (4), padding 100..128 (28).
    fake_mesh(sizes={"tp": tp}, ranks={"tp": 3})
    p = nn.Parameter(torch.full((per_rank, D), 7.0), requires_grad=False)
    loader = vocab(axis="tp", mesh=resolve_mesh("model"))
    loader(p, disk, None)
    assert torch.equal(p.data[:4], disk.narrow(0, 96, 4))
    assert torch.all(p.data[4:] == 0)
    _meshes.clear()

    # Ranks 0..2 don't see padding.
    for rank in range(3):
        fake_mesh(sizes={"tp": tp}, ranks={"tp": rank})
        p = nn.Parameter(torch.zeros(per_rank, D), requires_grad=False)
        loader = vocab(axis="tp", mesh=resolve_mesh("model"))
        loader(p, disk, None)
        assert torch.equal(p.data, disk.narrow(0, rank * per_rank, per_rank))
        _meshes.clear()


def test_vocab_loader_pathological_all_padding_rank(fake_mesh):
    from phyai.parallel.state import resolve_mesh
    from phyai.weights.shards import vocab

    V, V_padded, D, tp = 20, 128, 4, 4
    disk = torch.randn(V, D, dtype=torch.float32)
    per_rank = V_padded // tp  # 32
    # Rank 1: shard_start=32 > V=20 -> entirely padding.
    fake_mesh(sizes={"tp": tp}, ranks={"tp": 1})
    p = nn.Parameter(torch.full((per_rank, D), 9.0), requires_grad=False)
    loader = vocab(axis="tp", mesh=resolve_mesh("model"))
    loader(p, disk, None)
    assert torch.all(p.data == 0)


# --------------------------------------------------------------------------- #
# VocabParallelEmbedding — construction                                       #
# --------------------------------------------------------------------------- #


def test_embedding_tp1_construct_attrs(fake_mesh):
    fake_mesh(sizes={"tp": 1})
    layer = VocabParallelEmbedding(
        num_embeddings=100,
        embedding_dim=16,
        params_dtype=torch.float32,
        prefix="embed_tokens",
    )
    assert layer.num_embeddings == 100
    assert layer.num_embeddings_padded == 128  # 100 -> 128 (multiple of 64)
    assert layer.num_embeddings_per_partition == 128
    assert layer.shard_start == 0
    assert layer.shard_end == 100  # clamped to V_real
    assert layer.weight.shape == (128, 16)
    assert layer.weight.dtype == torch.float32
    # The param-attached HF mapping.
    assert layer.weight.hf_keys == [("embed_tokens.weight", None)]
    assert callable(layer.weight.weight_loader)


def test_embedding_tp4_shard_bounds_per_rank(fake_mesh):
    """Across ranks, shard_start/end tile [0, V_padded) and clamp to V."""
    V, D, tp = 151700, 32, 4
    expected_padded = pad_vocab_to(V, tp, multiple=64)
    per_rank = expected_padded // tp
    boundaries = []
    for rank in range(tp):
        fake_mesh(sizes={"tp": tp}, ranks={"tp": rank})
        layer = VocabParallelEmbedding(
            num_embeddings=V, embedding_dim=D, params_dtype=torch.float32
        )
        assert layer.num_embeddings_padded == expected_padded
        assert layer.weight.shape == (per_rank, D)
        boundaries.append((layer.shard_start, layer.shard_end))
        # Clean up so the next iteration's fake_mesh() takes effect.
        _meshes.clear()

    # Check tile coverage and clamping.
    assert boundaries[0] == (0, per_rank)
    assert boundaries[1] == (per_rank, 2 * per_rank)
    assert boundaries[2] == (2 * per_rank, 3 * per_rank)
    # Rank 3: starts at 3 * per_rank but ends at V (clamped, padding past V_real).
    assert boundaries[3][0] == 3 * per_rank
    assert boundaries[3][1] == V


def test_embedding_rejects_unimplemented_layouts(fake_mesh):
    fake_mesh(sizes={"tp": 1})
    with pytest.raises(NotImplementedError, match="vocab_parallel"):
        VocabParallelEmbedding(
            num_embeddings=100, embedding_dim=16, layout="hidden_parallel"
        )


def test_embedding_rejects_invalid_sizes(fake_mesh):
    fake_mesh(sizes={"tp": 1})
    with pytest.raises(ValueError, match="num_embeddings must be positive"):
        VocabParallelEmbedding(num_embeddings=0, embedding_dim=16)
    with pytest.raises(ValueError, match="embedding_dim must be positive"):
        VocabParallelEmbedding(num_embeddings=100, embedding_dim=0)


# --------------------------------------------------------------------------- #
# VocabParallelEmbedding — forward parity                                     #
# --------------------------------------------------------------------------- #


def test_embedding_tp1_forward_matches_nn_embedding(fake_mesh):
    fake_mesh(sizes={"tp": 1})
    V, D = 64, 16
    layer = VocabParallelEmbedding(
        num_embeddings=V, embedding_dim=D, params_dtype=torch.float32
    )
    nn.init.normal_(layer.weight, std=0.05)
    # The padding rows (V..V_padded) must be zero so they don't affect any
    # in-range gather. The loader writes them; here we initialise manually.
    layer.weight.data[V:].zero_()

    ids = torch.randint(0, V, (4, 8), dtype=torch.int64)
    out = layer(ids)

    # Reference: plain F.embedding using only the first V rows.
    expected = F.embedding(ids, layer.weight[:V])
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


def test_embedding_tp1_zero_input(fake_mesh):
    fake_mesh(sizes={"tp": 1})
    layer = VocabParallelEmbedding(
        num_embeddings=32, embedding_dim=8, params_dtype=torch.float32
    )
    nn.init.normal_(layer.weight, std=0.05)
    layer.weight.data[32:].zero_()
    ids = torch.empty((0,), dtype=torch.int64)
    out = layer(ids)
    assert out.shape == (0, 8)


def test_embedding_tp1_3d_input_preserves_shape(fake_mesh):
    fake_mesh(sizes={"tp": 1})
    layer = VocabParallelEmbedding(
        num_embeddings=64, embedding_dim=12, params_dtype=torch.float32
    )
    nn.init.normal_(layer.weight, std=0.05)
    layer.weight.data[64:].zero_()
    ids = torch.randint(0, 64, (2, 3, 4), dtype=torch.int64)
    out = layer(ids)
    assert out.shape == (2, 3, 4, 12)
    expected = F.embedding(ids, layer.weight[:64])
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


# --------------------------------------------------------------------------- #
# ParallelLMHead — construction                                               #
# --------------------------------------------------------------------------- #


def test_lmhead_tp1_construct_attrs(fake_mesh):
    fake_mesh(sizes={"tp": 1})
    _init_dispatcher()
    head = ParallelLMHead(
        embedding_dim=16,
        num_embeddings=100,
        params_dtype=torch.float32,
        prefix="lm_head",
    )
    assert head.num_embeddings == 100
    assert head.num_embeddings_padded == 128
    assert head.num_embeddings_per_partition == 128
    assert head.weight.shape == (128, 16)
    assert head.input_size_per_partition == 16
    assert head.output_size_per_partition == 128
    assert head.bias is None
    # The param-attached HF mapping for the un-tied head.
    assert head.weight.hf_keys == [("lm_head.weight", None)]


def test_lmhead_rejects_bias(fake_mesh):
    fake_mesh(sizes={"tp": 1})
    with pytest.raises(NotImplementedError, match="bias"):
        ParallelLMHead(embedding_dim=16, num_embeddings=100, bias=True)


def test_lmhead_tied_weight_shares_parameter(fake_mesh):
    fake_mesh(sizes={"tp": 1})
    _init_dispatcher()
    embed = VocabParallelEmbedding(
        num_embeddings=100, embedding_dim=16, params_dtype=torch.float32
    )
    head = ParallelLMHead(
        embedding_dim=16,
        num_embeddings=100,
        tied_weight=embed.weight,
        params_dtype=torch.float32,
    )
    # Same Parameter object — not just same shape / values.
    assert head.weight is embed.weight
    assert head.logical_widths == [128]


def test_lmhead_tied_weight_rejects_shape_mismatch(fake_mesh):
    fake_mesh(sizes={"tp": 1})
    bogus = nn.Parameter(torch.empty(64, 16), requires_grad=False)
    with pytest.raises(ValueError, match="shape"):
        ParallelLMHead(
            embedding_dim=16,
            num_embeddings=100,
            tied_weight=bogus,
            params_dtype=torch.float32,
        )


# --------------------------------------------------------------------------- #
# ParallelLMHead — forward parity                                             #
# --------------------------------------------------------------------------- #


def test_lmhead_tp1_forward_matches_F_linear(fake_mesh):
    fake_mesh(sizes={"tp": 1})
    _init_dispatcher()
    head = ParallelLMHead(
        embedding_dim=32,
        num_embeddings=100,
        params_dtype=torch.bfloat16,
    )
    nn.init.normal_(head.weight, std=0.02)

    x = torch.randn(4, 32, dtype=torch.bfloat16)
    y = head(x)
    # Reference: x @ weight.T (bias is None).
    ref = F.linear(x, head.weight)
    torch.testing.assert_close(y, ref, atol=0, rtol=0)
    # Output shape matches per-rank V (= padded V at tp=1).
    assert y.shape == (4, 128)


def test_lmhead_tied_forward_uses_shared_weight(fake_mesh):
    fake_mesh(sizes={"tp": 1})
    _init_dispatcher()
    embed = VocabParallelEmbedding(
        num_embeddings=64, embedding_dim=16, params_dtype=torch.bfloat16
    )
    nn.init.normal_(embed.weight, std=0.02)
    head = ParallelLMHead(
        embedding_dim=16,
        num_embeddings=64,
        tied_weight=embed.weight,
        params_dtype=torch.bfloat16,
    )

    # Modifying embed.weight must affect head.weight (shared Parameter).
    embed.weight.data.fill_(0.5)
    assert torch.all(head.weight.data == 0.5)

    x = torch.randn(2, 16, dtype=torch.bfloat16)
    y = head(x)
    ref = F.linear(x, embed.weight)
    torch.testing.assert_close(y, ref, atol=0, rtol=0)


# --------------------------------------------------------------------------- #
# End-to-end: embedding -> linear-style -> tied lm head, padding zeroing       #
# --------------------------------------------------------------------------- #


def test_padding_logits_are_zero_after_load(fake_mesh):
    """The whole point of zero-fill padding: out-of-vocab logits stay 0."""
    fake_mesh(sizes={"tp": 1})
    _init_dispatcher()
    V, D = 100, 8
    head = ParallelLMHead(
        embedding_dim=D,
        num_embeddings=V,
        params_dtype=torch.float32,
        prefix="lm_head",
    )

    # Simulate a checkpoint load via the param-attached weight_loader.
    disk_w = torch.randn(V, D, dtype=torch.float32)
    head.weight.weight_loader(head.weight, disk_w, None)

    # First V rows match disk; last V_padded - V are zero.
    assert torch.equal(head.weight.data[:V], disk_w)
    assert torch.all(head.weight.data[V:] == 0)

    # Forward: any column ≥ V should produce exactly-zero logits regardless of x.
    x = torch.randn(7, D, dtype=torch.float32)
    logits = head(x)
    assert torch.all(logits[:, V:] == 0)


# --------------------------------------------------------------------------- #
# VocabParallelEmbedding — embed_scale (Gemma / PaliGemma scaled embeddings) #
# --------------------------------------------------------------------------- #


def test_embedding_scale_default_no_buffer(fake_mesh):
    """Default embed_scale=1.0 should not allocate the scale buffer."""
    fake_mesh(sizes={"tp": 1})
    layer = VocabParallelEmbedding(
        num_embeddings=64, embedding_dim=16, params_dtype=torch.float32
    )
    assert layer.embed_scale == 1.0
    assert not hasattr(layer, "_embed_scale_t")


def test_embedding_scale_non_default_registers_buffer(fake_mesh):
    fake_mesh(sizes={"tp": 1})
    D = 16
    layer = VocabParallelEmbedding(
        num_embeddings=64,
        embedding_dim=D,
        embed_scale=D**0.5,
        params_dtype=torch.float32,
    )
    assert layer.embed_scale == D**0.5
    assert hasattr(layer, "_embed_scale_t")
    assert layer._embed_scale_t.dtype == torch.float32


def test_embedding_scale_invalid_raises(fake_mesh):
    fake_mesh(sizes={"tp": 1})
    with pytest.raises(ValueError, match="embed_scale"):
        VocabParallelEmbedding(num_embeddings=64, embedding_dim=16, embed_scale=0.0)
    with pytest.raises(ValueError, match="embed_scale"):
        VocabParallelEmbedding(num_embeddings=64, embedding_dim=16, embed_scale=-1.0)


def test_embedding_scale_forward_matches_unscaled_times_scale(fake_mesh):
    """Output of the scaled embedding equals output of un-scaled * scale."""
    fake_mesh(sizes={"tp": 1})
    V, D = 64, 16
    scale = D**0.5
    plain = VocabParallelEmbedding(
        num_embeddings=V, embedding_dim=D, params_dtype=torch.float32
    )
    scaled = VocabParallelEmbedding(
        num_embeddings=V,
        embedding_dim=D,
        embed_scale=scale,
        params_dtype=torch.float32,
    )
    nn.init.normal_(plain.weight, std=0.05)
    plain.weight.data[V:].zero_()
    # Mirror the same weight onto the scaled layer.
    scaled.weight.data.copy_(plain.weight.data)

    ids = torch.randint(0, V, (4, 8), dtype=torch.int64)
    out_plain = plain(ids)
    out_scaled = scaled(ids)
    torch.testing.assert_close(out_scaled, out_plain * scale, atol=0, rtol=0)


def test_embedding_scale_dtype_matches_input(fake_mesh):
    """Scale is fp32 internally but result keeps input dtype (Gemma cast pattern)."""
    fake_mesh(sizes={"tp": 1})
    V, D = 32, 16
    layer = VocabParallelEmbedding(
        num_embeddings=V,
        embedding_dim=D,
        embed_scale=D**0.5,
        params_dtype=torch.bfloat16,
    )
    nn.init.normal_(layer.weight, std=0.05)
    layer.weight.data[V:].zero_()
    ids = torch.randint(0, V, (4,), dtype=torch.int64)
    out = layer(ids)
    assert out.dtype == torch.bfloat16


def test_embedding_scale_does_not_mutate_weight(fake_mesh):
    """Forward-time scale must not mutate the stored weight (this is the whole
    reason we picked 'A': a tied lm_head must keep seeing the un-scaled weight).
    """
    fake_mesh(sizes={"tp": 1})
    _init_dispatcher()
    V, D = 64, 16
    embed = VocabParallelEmbedding(
        num_embeddings=V,
        embedding_dim=D,
        embed_scale=D**0.5,
        params_dtype=torch.float32,
    )
    nn.init.normal_(embed.weight, std=0.02)
    embed.weight.data[V:].zero_()
    weight_before = embed.weight.data.clone()

    ids = torch.randint(0, V, (3, 5), dtype=torch.int64)
    _ = embed(ids)

    torch.testing.assert_close(embed.weight.data, weight_before, atol=0, rtol=0)


def test_tied_lmhead_sees_unscaled_weight(fake_mesh):
    """The point of forward-time scale: tied lm_head logits are NOT scaled.

    Mirrors HF Gemma semantics — embedding output is multiplied by sqrt(D),
    but the tied lm_head produces ``x @ weight.T`` with the un-scaled weight.
    """
    fake_mesh(sizes={"tp": 1})
    _init_dispatcher()
    V, D = 64, 16
    scale = D**0.5
    embed = VocabParallelEmbedding(
        num_embeddings=V,
        embedding_dim=D,
        embed_scale=scale,
        params_dtype=torch.float32,
    )
    nn.init.normal_(embed.weight, std=0.02)
    embed.weight.data[V:].zero_()
    head = ParallelLMHead(
        embedding_dim=D,
        num_embeddings=V,
        tied_weight=embed.weight,
        params_dtype=torch.float32,
    )

    x = torch.randn(2, D, dtype=torch.float32)
    y = head(x)
    ref = F.linear(x, embed.weight)  # un-scaled
    ref_scaled = F.linear(x, embed.weight * scale)  # would be wrong
    torch.testing.assert_close(y, ref, atol=0, rtol=0)
    # Sanity: scaled reference is meaningfully different.
    assert not torch.allclose(y, ref_scaled, atol=1e-3)


def test_embedding_scale_extra_repr(fake_mesh):
    fake_mesh(sizes={"tp": 1})
    layer = VocabParallelEmbedding(num_embeddings=64, embedding_dim=16, embed_scale=4.0)
    s = repr(layer)
    assert "embed_scale=4.0" in s
    plain = VocabParallelEmbedding(num_embeddings=64, embedding_dim=16)
    assert "embed_scale" not in repr(plain)
