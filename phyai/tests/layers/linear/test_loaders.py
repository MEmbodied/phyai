"""Loader tests — column/row/QKV shard slicing."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from phyai.layers.loaders import (
    ColumnShardLoader,
    QKVShardLoader,
    ReplicatedLoader,
    RowShardLoader,
)


def _param(shape, dtype=torch.float32) -> nn.Parameter:
    return nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)


# ---------------------------------------------------------------------------
# ColumnShardLoader
# ---------------------------------------------------------------------------


def test_column_load_full_ws1():
    loader = ColumnShardLoader(
        output_partition_sizes=[8],
        tp_rank=0,
        tp_size=1,
    )
    param = _param((8, 4))
    full = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    loader.load_weight(param, full)
    assert torch.equal(param.data, full)


def test_column_load_full_ws2_rank1():
    """Rank 1 of tp=2 should take the second half of the output rows."""
    loader = ColumnShardLoader(
        output_partition_sizes=[4],
        tp_rank=1,
        tp_size=2,
    )
    param = _param((4, 4))
    full = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    loader.load_weight(param, full)
    expected = full.narrow(0, 4, 4)
    assert torch.equal(param.data, expected)


def test_column_load_full_shape_mismatch():
    loader = ColumnShardLoader(
        output_partition_sizes=[4],
        tp_rank=0,
        tp_size=2,
    )
    param = _param((4, 4))
    with pytest.raises(ValueError, match="global_out"):
        loader.load_weight(param, torch.zeros(10, 4))


def test_column_load_shard_into_fused_param():
    """Merged Linear: shard_id=0 writes into [0:4), shard_id=1 writes into [4:6)."""
    loader = ColumnShardLoader(
        output_partition_sizes=[4, 2],
        tp_rank=0,
        tp_size=2,
    )
    param = _param((6, 4))
    # shard 0: global width 8, this rank wants rows 0..4
    disk0 = torch.full((8, 4), 1.0)
    loader.load_weight(param, disk0, shard_id=0)
    assert torch.all(param.data[:4] == 1.0)
    assert torch.all(param.data[4:] == 0.0)

    # shard 1: global width 4, this rank wants rows 0..2
    disk1 = torch.full((4, 4), 7.0)
    loader.load_weight(param, disk1, shard_id=1)
    assert torch.all(param.data[4:6] == 7.0)


def test_column_load_shard_rank_isolation():
    """tp_rank=1 should read the second half of each disk shard."""
    loader = ColumnShardLoader(
        output_partition_sizes=[4],
        tp_rank=1,
        tp_size=2,
    )
    param = _param((4, 4))
    disk = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    loader.load_weight(param, disk, shard_id=0)
    assert torch.equal(param.data, disk.narrow(0, 4, 4))


def test_column_load_shard_out_of_range():
    loader = ColumnShardLoader(
        output_partition_sizes=[4],
        tp_rank=0,
        tp_size=1,
    )
    param = _param((4, 4))
    with pytest.raises(IndexError):
        loader.load_weight(param, torch.zeros(4, 4), shard_id=5)


# ---------------------------------------------------------------------------
# RowShardLoader
# ---------------------------------------------------------------------------


def test_row_load_full_ws1():
    loader = RowShardLoader(tp_rank=0, tp_size=1)
    param = _param((4, 8))
    full = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    loader.load_weight(param, full)
    assert torch.equal(param.data, full)


def test_row_load_full_ws2_rank0():
    loader = RowShardLoader(tp_rank=0, tp_size=2)
    param = _param((4, 4))
    full = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    loader.load_weight(param, full)
    assert torch.equal(param.data, full.narrow(1, 0, 4))


def test_row_load_full_ws2_rank1():
    loader = RowShardLoader(tp_rank=1, tp_size=2)
    param = _param((4, 4))
    full = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    loader.load_weight(param, full)
    assert torch.equal(param.data, full.narrow(1, 4, 4))


def test_row_load_full_indivisible():
    loader = RowShardLoader(tp_rank=0, tp_size=2)
    param = _param((4, 3))
    with pytest.raises(ValueError, match="not divisible"):
        loader.load_weight(param, torch.zeros(4, 7))


# ---------------------------------------------------------------------------
# QKVShardLoader
# ---------------------------------------------------------------------------


def test_qkv_loader_shard_offsets_no_gqa():
    loader = QKVShardLoader(
        q_size=8,
        kv_size=8,
        num_kv_replicas=1,
        tp_rank=0,
        tp_size=1,
    )
    param = _param((24, 4))  # q(8) + k(8) + v(8)
    q_full = torch.full((8, 4), 1.0)
    k_full = torch.full((8, 4), 2.0)
    v_full = torch.full((8, 4), 3.0)

    loader.load_weight(param, q_full, "q")
    loader.load_weight(param, k_full, "k")
    loader.load_weight(param, v_full, "v")

    assert torch.all(param.data[0:8] == 1.0)
    assert torch.all(param.data[8:16] == 2.0)
    assert torch.all(param.data[16:24] == 3.0)


def test_qkv_loader_gqa_replicates_kv():
    """tp_size=4, num_kv_heads=2 → num_kv_replicas=2. Disk K/V is half width."""
    loader = QKVShardLoader(
        q_size=2,
        kv_size=1,
        num_kv_replicas=2,
        tp_rank=0,
        tp_size=4,
    )
    # q is 2 per rank × 4 = 8 global; kv is 1 per rank × 4 = 4 per-rank sum,
    # but on disk kv is only 4 // 2 = 2 wide.
    param = _param((4, 4))  # 2 + 1 + 1
    q_full = torch.full((8, 4), 0.5)
    kv_full = torch.full((2, 4), 9.0)
    loader.load_weight(param, q_full, "q")
    loader.load_weight(param, kv_full, "k")
    loader.load_weight(param, kv_full, "v")

    # rank 0 → q rows 0..2
    assert torch.all(param.data[0:2] == 0.5)
    # rank 0 → kv_rank = 0//2 = 0, narrow(0, 0*1, 1) → row 0 of disk
    assert torch.all(param.data[2:3] == 9.0)
    assert torch.all(param.data[3:4] == 9.0)


def test_qkv_loader_rejects_bad_shard_id():
    loader = QKVShardLoader(
        q_size=4,
        kv_size=4,
        num_kv_replicas=1,
        tp_rank=0,
        tp_size=1,
    )
    param = _param((12, 4))
    with pytest.raises(ValueError, match="q/k/v"):
        loader.load_weight(param, torch.zeros(4, 4), "x")


def test_qkv_loader_rejects_bad_num_replicas():
    with pytest.raises(ValueError, match="num_kv_replicas"):
        QKVShardLoader(
            q_size=4,
            kv_size=4,
            num_kv_replicas=0,
            tp_rank=0,
            tp_size=1,
        )


def test_qkv_loader_q_shape_mismatch():
    loader = QKVShardLoader(
        q_size=4,
        kv_size=4,
        num_kv_replicas=1,
        tp_rank=0,
        tp_size=1,
    )
    param = _param((12, 4))
    with pytest.raises(ValueError, match="natural_width"):
        loader.load_weight(param, torch.zeros(8, 4), "q")


# ---------------------------------------------------------------------------
# ReplicatedLoader
# ---------------------------------------------------------------------------


def test_replicated_load_full_copies_2d():
    loader = ReplicatedLoader()
    param = _param((4, 8))
    src = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    loader.load_weight(param, src)
    assert torch.equal(param.data, src)


def test_replicated_load_full_copies_1d_bias():
    loader = ReplicatedLoader()
    param = _param((16,))
    src = torch.arange(16, dtype=torch.float32)
    loader.load_weight(param, src)
    assert torch.equal(param.data, src)


def test_replicated_load_full_scalar_broadcast():
    """A 1-element saved scalar fills a 1-element parameter."""
    loader = ReplicatedLoader()
    param = _param((1,))
    loader.load_weight(param, torch.tensor([3.5]))
    assert param.item() == 3.5


def test_replicated_load_full_shape_mismatch():
    loader = ReplicatedLoader()
    param = _param((8,))
    with pytest.raises(ValueError, match="shape"):
        loader.load_weight(param, torch.zeros(4))
