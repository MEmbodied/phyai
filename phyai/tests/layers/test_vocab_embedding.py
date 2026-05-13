"""VocabParallelEmbedding + ParallelLMHead integration tests.

ws=1 tests run via a mocked Mesh — :func:`phyai.parallel.all_reduce` and
:func:`phyai.parallel.all_gather` short-circuit when the axis size is 1,
so we can exercise construction, weight allocation, masked-lookup, and
forward without a real process group. Multi-rank correctness lives under
the existing multiprocess gloo harness and is out of scope here.

We also exercise the loader directly across mocked ``tp_rank`` /
``tp_size`` combinations — that's enough to validate every shard-bounds
edge case (padding overhang, all-padding rank, V-evenly-divisible).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import phyai.layers.linear as L
from phyai.layers.loaders import VocabShardLoader
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
# VocabShardLoader                                                            #
# --------------------------------------------------------------------------- #


def test_loader_tp1_full_copy():
    """tp=1 → loader copies the entire disk tensor verbatim."""
    V, D = 100, 16
    loader = VocabShardLoader(
        num_embeddings=V, num_embeddings_padded=V, tp_rank=0, tp_size=1
    )
    disk = torch.arange(V * D, dtype=torch.float32).reshape(V, D)
    param = nn.Parameter(torch.empty(V, D), requires_grad=False)
    loader.load_weight(param, disk)
    assert torch.equal(param.data, disk)


def test_loader_tp4_evenly_divisible():
    """V evenly divides tp_size: each rank gets a contiguous chunk."""
    V, D, tp = 256, 8, 4
    disk = torch.arange(V * D, dtype=torch.float32).reshape(V, D)
    per_rank = V // tp
    for rank in range(tp):
        loader = VocabShardLoader(
            num_embeddings=V,
            num_embeddings_padded=V,
            tp_rank=rank,
            tp_size=tp,
        )
        param = nn.Parameter(torch.empty(per_rank, D), requires_grad=False)
        loader.load_weight(param, disk)
        expected = disk.narrow(0, rank * per_rank, per_rank)
        assert torch.equal(param.data, expected)


def test_loader_padding_overhang_zeros_tail():
    """V=100, V_padded=128, tp=4 → rank 3 holds rows [96, 128); 28 are real, 4 are pad."""
    V, V_padded, D, tp = 100, 128, 8, 4
    disk = torch.randn(V, D, dtype=torch.float32)
    per_rank = V_padded // tp  # 32

    # Rank 3: real rows 96..100 (4 rows), padding 100..128 (28 rows).
    loader = VocabShardLoader(
        num_embeddings=V,
        num_embeddings_padded=V_padded,
        tp_rank=3,
        tp_size=tp,
    )
    param = nn.Parameter(torch.empty(per_rank, D), requires_grad=False)
    # Pre-fill with non-zero garbage to verify the loader actually zeros the tail.
    param.data.fill_(7.0)
    loader.load_weight(param, disk)

    # First 4 rows = disk[96:100]
    assert torch.equal(param.data[:4], disk.narrow(0, 96, 4))
    # Remaining 28 rows = exactly zero
    assert torch.all(param.data[4:] == 0)

    # Rank 0..2 don't see any padding.
    for rank in range(3):
        loader_r = VocabShardLoader(
            num_embeddings=V,
            num_embeddings_padded=V_padded,
            tp_rank=rank,
            tp_size=tp,
        )
        p = nn.Parameter(torch.empty(per_rank, D), requires_grad=False)
        loader_r.load_weight(p, disk)
        assert torch.equal(p.data, disk.narrow(0, rank * per_rank, per_rank))


def test_loader_pathological_all_padding_rank():
    """V=20, V_padded=128, tp=4 → rank 1 onwards holds only padding."""
    V, V_padded, D, tp = 20, 128, 4, 4
    disk = torch.randn(V, D, dtype=torch.float32)
    per_rank = V_padded // tp  # 32
    # Rank 1: shard_start=32 > V=20 → entirely padding → entirely zeros.
    loader = VocabShardLoader(
        num_embeddings=V,
        num_embeddings_padded=V_padded,
        tp_rank=1,
        tp_size=tp,
    )
    param = nn.Parameter(torch.empty(per_rank, D), requires_grad=False)
    param.data.fill_(9.0)
    loader.load_weight(param, disk)
    assert torch.all(param.data == 0)


def test_loader_handles_1d_bias():
    """1-D bias along the V dim flows through the same logic."""
    V, V_padded, tp = 100, 128, 4
    disk_bias = torch.arange(V, dtype=torch.float32)
    per_rank = V_padded // tp

    # Rank 3 has 4 real bias entries + 28 zeros.
    loader = VocabShardLoader(
        num_embeddings=V,
        num_embeddings_padded=V_padded,
        tp_rank=3,
        tp_size=tp,
    )
    param = nn.Parameter(torch.empty(per_rank), requires_grad=False)
    param.data.fill_(99.0)
    loader.load_weight(param, disk_bias)
    assert torch.equal(param.data[:4], disk_bias.narrow(0, 96, 4))
    assert torch.all(param.data[4:] == 0)


def test_loader_rejects_wrong_shape():
    loader = VocabShardLoader(
        num_embeddings=100, num_embeddings_padded=128, tp_rank=0, tp_size=4
    )
    param = nn.Parameter(torch.empty(32, 8), requires_grad=False)
    # Wrong V on disk:
    with pytest.raises(ValueError, match="loaded.shape\\[0\\]"):
        loader.load_weight(param, torch.empty(50, 8))
    # Trailing dim mismatch:
    with pytest.raises(ValueError, match="trailing dims"):
        loader.load_weight(param, torch.empty(100, 16))


def test_loader_constructor_validates_inputs():
    with pytest.raises(ValueError, match="not divisible"):
        VocabShardLoader(
            num_embeddings=100,
            num_embeddings_padded=129,  # not divisible by tp=4
            tp_rank=0,
            tp_size=4,
        )
    with pytest.raises(ValueError, match="< num_embeddings"):
        VocabShardLoader(
            num_embeddings=200,
            num_embeddings_padded=128,
            tp_rank=0,
            tp_size=4,
        )


# --------------------------------------------------------------------------- #
# VocabParallelEmbedding — construction                                       #
# --------------------------------------------------------------------------- #


def test_embedding_tp1_construct_attrs(fake_mesh):
    fake_mesh(sizes={"tp": 1})
    layer = VocabParallelEmbedding(
        num_embeddings=100,
        embedding_dim=16,
        params_dtype=torch.float32,
    )
    assert layer.num_embeddings == 100
    assert layer.num_embeddings_padded == 128  # 100 → 128 (multiple of 64)
    assert layer.num_embeddings_per_partition == 128
    assert layer.shard_start == 0
    assert layer.shard_end == 100  # clamped to V_real
    assert layer.weight.shape == (128, 16)
    assert layer.weight.dtype == torch.float32
    # Loader is attached for checkpoint loading.
    assert isinstance(layer.weight.loader, VocabShardLoader)


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
    )
    assert head.num_embeddings == 100
    assert head.num_embeddings_padded == 128
    assert head.num_embeddings_per_partition == 128
    assert head.weight.shape == (128, 16)
    assert head.input_size_per_partition == 16
    assert head.output_size_per_partition == 128
    assert head.bias is None
    assert isinstance(head.weight.loader, VocabShardLoader)


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
# End-to-end: embedding → linear-style → tied lm head, padding zeroing       #
# --------------------------------------------------------------------------- #


def test_padding_logits_are_zero_after_load(fake_mesh):
    """The whole point of zero-fill padding: out-of-vocab logits stay 0."""
    fake_mesh(sizes={"tp": 1})
    _init_dispatcher()
    V, D = 100, 8
    head = ParallelLMHead(embedding_dim=D, num_embeddings=V, params_dtype=torch.float32)

    # Simulate a checkpoint load via the attached loader.
    disk_w = torch.randn(V, D, dtype=torch.float32)
    head.weight.loader.load_weight(head.weight, disk_w)

    # First V rows match disk; last V_padded - V are zero.
    assert torch.equal(head.weight.data[:V], disk_w)
    assert torch.all(head.weight.data[V:] == 0)

    # Forward: any column ≥ V should produce exactly-zero logits regardless of x.
    x = torch.randn(7, D, dtype=torch.float32)
    logits = head(x)
    assert torch.all(logits[:, V:] == 0)
