"""TP scale-sharding loaders: fake-mesh unit tests (CPU)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from phyai.layers.linear.layers import LinearBase
from phyai.weights.shards import _Leg, _ScaleLeg, scale_fused, scale_sharded


class FakeMesh:
    """Minimal mesh: scale loaders only call axis_size / axis_local_rank."""

    def __init__(self, size: int, rank: int) -> None:
        self._size = size
        self._rank = rank

    def axis_size(self, axis: str = "tp") -> int:
        return self._size

    def axis_local_rank(self, axis: str = "tp") -> int:
        return self._rank


def _param(shape) -> nn.Parameter:
    return nn.Parameter(torch.zeros(*shape), requires_grad=False)


# --------------------------------------------------------------------------- #
# scale_sharded — column family (shard N/output)
# --------------------------------------------------------------------------- #


def test_column_channel_shards_dim0():
    mesh = FakeMesh(size=2, rank=1)
    loaded = torch.arange(8, dtype=torch.float32).reshape(8, 1)
    p = _param((4, 1))
    scale_sharded(extra_attrs={"output_dim": 0, "scale_type": "channel"}, mesh=mesh)(
        p, loaded, None
    )
    torch.testing.assert_close(p.data, loaded[4:8])


def test_column_group_shards_dim0_only():
    # group scale (N, K//g): column-parallel shards N (dim0), not K.
    mesh = FakeMesh(size=2, rank=0)
    loaded = torch.arange(8 * 3, dtype=torch.float32).reshape(8, 3)
    p = _param((4, 3))
    scale_sharded(
        extra_attrs={"output_dim": 0, "input_dim": 1, "scale_type": "group"}, mesh=mesh
    )(p, loaded, None)
    torch.testing.assert_close(p.data, loaded[0:4, :])


def test_column_packed_zero_point_shards_packed_dim0():
    # packed int32 zp dim0 = N*bits//32; narrowing the packed dim is contiguous.
    mesh = FakeMesh(size=2, rank=1)
    loaded = torch.arange(8 * 2, dtype=torch.int32).reshape(8, 2)
    p = _param((4, 2)).to(torch.int32)
    scale_sharded(
        extra_attrs={
            "output_dim": 0,
            "input_dim": 1,
            "packed_dim": 0,
            "packed_factor": 8,
        },
        mesh=mesh,
    )(p, loaded, None)
    torch.testing.assert_close(p.data, loaded[4:8, :])


# --------------------------------------------------------------------------- #
# scale_sharded — row family (shard K/input; per-channel replicates)
# --------------------------------------------------------------------------- #


def test_row_channel_replicates():
    # per-channel scale (N,1) has no K axis → replicated on row-parallel.
    mesh = FakeMesh(size=2, rank=1)
    loaded = torch.arange(6, dtype=torch.float32).reshape(6, 1)
    p = _param((6, 1))
    scale_sharded(
        extra_attrs={"output_dim": 0, "scale_type": "channel"},
        mesh=mesh,
        row_parallel=True,
    )(p, loaded, None)
    torch.testing.assert_close(p.data, loaded)


def test_row_group_shards_dim1():
    mesh = FakeMesh(size=2, rank=1)
    loaded = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4)
    p = _param((6, 2))
    scale_sharded(
        extra_attrs={"output_dim": 0, "input_dim": 1, "scale_type": "group"},
        mesh=mesh,
        row_parallel=True,
    )(p, loaded, None)
    torch.testing.assert_close(p.data, loaded[:, 2:4])


def test_global_scale_replicates_both_families():
    mesh = FakeMesh(size=4, rank=2)
    loaded = torch.tensor([3.5])
    for rp in (False, True):
        p = _param((1,))
        scale_sharded(extra_attrs={"scale_type": "tensor"}, mesh=mesh, row_parallel=rp)(
            p, loaded, None
        )
        torch.testing.assert_close(p.data, loaded)


def test_scale_sharded_non_divisible_raises():
    mesh = FakeMesh(size=3, rank=0)
    loaded = torch.zeros(8, 1)
    with pytest.raises(ValueError, match="not divisible"):
        scale_sharded(extra_attrs={"output_dim": 0}, mesh=mesh)(
            _param((2, 1)), loaded, None
        )


# --------------------------------------------------------------------------- #
# scale_fused — Merged / QKV (channel), incl. GQA replicate
# --------------------------------------------------------------------------- #


def test_scale_fused_merged_two_legs_channel():
    mesh = FakeMesh(size=2, rank=1)
    n0, n1 = 4, 6
    total = n0 + n1
    param = _param((total, 1))
    legs = {
        0: _ScaleLeg(weight_offset=0, weight_size=n0, total_weight=total),
        1: _ScaleLeg(weight_offset=n0, weight_size=n1, total_weight=total),
    }
    loader = scale_fused(fuse_dim=0, legs=legs, mesh=mesh)
    loaded0 = torch.arange(2 * n0, dtype=torch.float32).reshape(2 * n0, 1)
    loaded1 = 100 + torch.arange(2 * n1, dtype=torch.float32).reshape(2 * n1, 1)
    loader(param, loaded0, 0)
    loader(param, loaded1, 1)
    torch.testing.assert_close(param.data[0:n0], loaded0[n0 : 2 * n0])  # rank1 slice
    torch.testing.assert_close(param.data[n0:total], loaded1[n1 : 2 * n1])


def test_scale_fused_qkv_gqa_kv_replicated():
    # world=2, K/V replicate=2 => world_eff=1 => k/v read the full (replicated) scale.
    mesh = FakeMesh(size=2, rank=1)
    q, kv = 8, 2
    total = q + 2 * kv
    param = _param((total, 1))
    legs = {
        "q": _ScaleLeg(weight_offset=0, weight_size=q, total_weight=total, replicate=1),
        "k": _ScaleLeg(
            weight_offset=q, weight_size=kv, total_weight=total, replicate=2
        ),
        "v": _ScaleLeg(
            weight_offset=q + kv, weight_size=kv, total_weight=total, replicate=2
        ),
    }
    loader = scale_fused(fuse_dim=0, legs=legs, mesh=mesh)
    q_loaded = torch.arange(2 * q, dtype=torch.float32).reshape(2 * q, 1)
    k_loaded = 100 + torch.arange(kv, dtype=torch.float32).reshape(kv, 1)
    v_loaded = 200 + torch.arange(kv, dtype=torch.float32).reshape(kv, 1)
    loader(param, q_loaded, "q")
    loader(param, k_loaded, "k")
    loader(param, v_loaded, "v")
    torch.testing.assert_close(param.data[0:q], q_loaded[q : 2 * q])  # rank1 q shard
    torch.testing.assert_close(param.data[q : q + kv], k_loaded)  # full (replicated)
    torch.testing.assert_close(param.data[q + kv : total], v_loaded)


# --------------------------------------------------------------------------- #
# _attach_optional_scales loader dispatch
# --------------------------------------------------------------------------- #


def _layer_with_scale(shape, extra):
    layer = nn.Module()
    p = nn.Parameter(torch.zeros(*shape), requires_grad=False)
    p._humming_attrs = extra
    layer.weight_scale = p
    return layer


def test_attach_column_uses_scale_sharded():
    mesh = FakeMesh(size=2, rank=1)
    layer = _layer_with_scale((4, 1), {"output_dim": 0, "scale_type": "channel"})
    LinearBase._attach_optional_scales(layer, "m.proj", kind="column", mesh=mesh)
    assert layer.weight_scale.optional is True
    assert layer.weight_scale.hf_keys == [("m.proj.weight_scale", None)]
    loaded = torch.arange(8, dtype=torch.float32).reshape(8, 1)
    layer.weight_scale.weight_loader(layer.weight_scale, loaded, None)
    torch.testing.assert_close(layer.weight_scale.data, loaded[4:8])


def test_attach_non_humming_scale_replicates():
    # fp8/nvfp4 scales carry no _humming_attrs → replicated (back-compat).
    mesh = FakeMesh(size=2, rank=1)
    layer = nn.Module()
    layer.weight_scale = nn.Parameter(torch.zeros(8), requires_grad=False)
    LinearBase._attach_optional_scales(layer, "m.proj", kind="column", mesh=mesh)
    loaded = torch.arange(8, dtype=torch.float32)
    layer.weight_scale.weight_loader(layer.weight_scale, loaded, None)
    torch.testing.assert_close(layer.weight_scale.data, loaded)  # full copy


def test_attach_merged_builds_per_leg_keys_and_fused_loader():
    mesh = FakeMesh(size=1, rank=0)
    layer = _layer_with_scale((10, 1), {"output_dim": 0, "scale_type": "channel"})
    fused_legs = {
        0: ("m.gate_proj", _Leg(offset=0, size=4, dim=0)),
        1: ("m.up_proj", _Leg(offset=4, size=6, dim=0)),
    }
    LinearBase._attach_optional_scales(
        layer, kind="merged", mesh=mesh, fused_legs=fused_legs, total_weight=10
    )
    assert set(layer.weight_scale.hf_keys) == {
        ("m.gate_proj.weight_scale", 0),
        ("m.up_proj.weight_scale", 1),
    }
    g = torch.arange(4, dtype=torch.float32).reshape(4, 1)
    u = 100 + torch.arange(6, dtype=torch.float32).reshape(6, 1)
    layer.weight_scale.weight_loader(layer.weight_scale, g, 0)
    layer.weight_scale.weight_loader(layer.weight_scale, u, 1)
    torch.testing.assert_close(layer.weight_scale.data[0:4], g)
    torch.testing.assert_close(layer.weight_scale.data[4:10], u)
