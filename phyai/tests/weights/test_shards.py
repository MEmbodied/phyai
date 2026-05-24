"""Unit tests for the shard-loader factories in phyai.weights.shards."""

from __future__ import annotations

import torch
import torch.nn as nn

from phyai.parallel.state import resolve_mesh
from phyai.weights.shards import _Leg, fused, replicated, sharded, vocab


def test_replicated_full_copy():
    p = nn.Parameter(torch.zeros(4, 8), requires_grad=False)
    src = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    replicated()(p, src, None)
    torch.testing.assert_close(p.data, src)


def test_replicated_scalar_to_singleton():
    p = nn.Parameter(torch.zeros(1), requires_grad=False)
    src = torch.tensor(3.5)  # 0-D
    replicated()(p, src, None)
    assert p.data.item() == 3.5


def test_sharded_dim0_tp4_rank2(fake_mesh):
    fake_mesh(sizes={"tp": 4}, ranks={"tp": 2})
    mesh = resolve_mesh("model")
    src = torch.arange(32 * 8, dtype=torch.float32).reshape(32, 8)
    p = nn.Parameter(torch.zeros(8, 8), requires_grad=False)
    sharded(dim=0, axis="tp", mesh=mesh)(p, src, None)
    torch.testing.assert_close(p.data, src.narrow(0, 16, 8))


def test_sharded_dim1_tp2_rank0(fake_mesh):
    fake_mesh(sizes={"tp": 2}, ranks={"tp": 0})
    mesh = resolve_mesh("model")
    src = torch.arange(4 * 16, dtype=torch.float32).reshape(4, 16)
    p = nn.Parameter(torch.zeros(4, 8), requires_grad=False)
    sharded(dim=1, axis="tp", mesh=mesh)(p, src, None)
    torch.testing.assert_close(p.data, src.narrow(1, 0, 8))


def test_sharded_replicate_gqa_share_slot(fake_mesh):
    """Two ranks with replicate=2 should read the SAME source slot."""
    src = torch.arange(16 * 4, dtype=torch.float32).reshape(16, 4)

    fake_mesh(sizes={"tp": 4}, ranks={"tp": 0})
    mesh = resolve_mesh("model")
    p0 = nn.Parameter(torch.zeros(8, 4), requires_grad=False)
    sharded(dim=0, axis="tp", mesh=mesh, replicate=2)(p0, src, None)

    from phyai.parallel.state import _meshes

    _meshes.clear()
    fake_mesh(sizes={"tp": 4}, ranks={"tp": 1})
    mesh = resolve_mesh("model")
    p1 = nn.Parameter(torch.zeros(8, 4), requires_grad=False)
    sharded(dim=0, axis="tp", mesh=mesh, replicate=2)(p1, src, None)

    # Ranks 0 and 1 both have rank // 2 = 0 → same slot.
    torch.testing.assert_close(p0.data, p1.data)


def test_fused_qkv_layout(fake_mesh):
    """Fused QKV at tp=1: q/k/v each go to their fuse offsets."""
    fake_mesh(sizes={"tp": 1})
    mesh = resolve_mesh("model")
    legs = {
        "q": _Leg(offset=0, size=4, dim=0, axis="tp", replicate=1),
        "k": _Leg(offset=4, size=2, dim=0, axis="tp", replicate=1),
        "v": _Leg(offset=6, size=2, dim=0, axis="tp", replicate=1),
    }
    loader = fused(fuse_dim=0, legs=legs, mesh=mesh)
    p = nn.Parameter(torch.zeros(8, 3), requires_grad=False)

    src_q = torch.full((4, 3), 1.0)
    src_k = torch.full((2, 3), 2.0)
    src_v = torch.full((2, 3), 3.0)
    loader(p, src_q, "q")
    loader(p, src_k, "k")
    loader(p, src_v, "v")

    assert torch.all(p.data[0:4] == 1.0)
    assert torch.all(p.data[4:6] == 2.0)
    assert torch.all(p.data[6:8] == 3.0)


def test_vocab_padding_zeros_trailing_rank(fake_mesh):
    fake_mesh(sizes={"tp": 4}, ranks={"tp": 3})
    mesh = resolve_mesh("model")
    V, V_padded, D = 100, 128, 4
    per_rank = V_padded // 4  # 32
    src = torch.arange(V * D, dtype=torch.float32).reshape(V, D)
    p = nn.Parameter(torch.full((per_rank, D), 7.0), requires_grad=False)
    vocab(axis="tp", mesh=mesh)(p, src, None)
    # Rows 96..100 from src → first 4 rows of param.
    torch.testing.assert_close(p.data[:4], src.narrow(0, 96, 4))
    # Rest zero-filled.
    assert torch.all(p.data[4:] == 0)


def test_vocab_full_padding_rank(fake_mesh):
    fake_mesh(sizes={"tp": 4}, ranks={"tp": 1})
    mesh = resolve_mesh("model")
    V, V_padded, D = 20, 128, 2
    per_rank = V_padded // 4  # 32
    src = torch.randn(V, D)
    # Rank 1 starts at row 32 > V=20 → all padding.
    p = nn.Parameter(torch.full((per_rank, D), 9.0), requires_grad=False)
    vocab(axis="tp", mesh=mesh)(p, src, None)
    assert torch.all(p.data == 0)
