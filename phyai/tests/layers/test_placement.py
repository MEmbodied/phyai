"""Tests for :mod:`phyai.layers.placement` — declarative HF→phyai loader.

No GPU needed; placements are pure data and ``apply_placements`` does
narrows + copies on CPU tensors.
"""

from __future__ import annotations

import pytest
import torch

from phyai.layers.placement import (
    CopyPlacement,
    Slice1D,
    ZeroPlacement,
    apply_placements,
    split_prefix,
)


def _dest(shape, dtype=torch.float32) -> torch.Tensor:
    return torch.zeros(shape, dtype=dtype)


# ---------------------------------------------------------------------------
# Slice1D / split_prefix
# ---------------------------------------------------------------------------


def test_split_prefix_dotted():
    assert split_prefix("model.layers.3.mlp.gate_up_proj") == (
        "model.layers.3.mlp",
        "gate_up_proj",
    )


def test_split_prefix_single_token():
    assert split_prefix("embed_tokens") == ("", "embed_tokens")


def test_split_prefix_empty_rejected():
    with pytest.raises(ValueError, match="non-empty prefix"):
        split_prefix("")


# ---------------------------------------------------------------------------
# apply_placements — replicated copy
# ---------------------------------------------------------------------------


def test_replicated_copy_full_tensor():
    src = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    dst = _dest((6, 4))
    placements = [CopyPlacement(hf_key="foo.weight", phyai_key="foo.weight")]
    apply_placements(placements, lambda k: src, {"foo.weight": dst})
    assert torch.equal(dst, src)


# ---------------------------------------------------------------------------
# apply_placements — column-parallel single-slot (sharded along dim 0)
# ---------------------------------------------------------------------------


def test_column_parallel_rank1_of_2():
    src = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    dst = _dest((4, 4))  # per-rank shape
    per_rank = 4
    rank = 1
    placements = [
        CopyPlacement(
            hf_key="proj.weight",
            phyai_key="proj.weight",
            src_slices=(Slice1D(0, rank * per_rank, per_rank),),
        )
    ]
    apply_placements(placements, lambda k: src, {"proj.weight": dst})
    assert torch.equal(dst, src.narrow(0, 4, 4))


# ---------------------------------------------------------------------------
# apply_placements — fused gate/up
# ---------------------------------------------------------------------------


def test_fused_gate_up_tp1():
    """Single-rank fused load: gate goes into [0:I) of dst, up into [I:2I)."""
    intermediate = 8
    hidden = 4
    gate = torch.arange(intermediate * hidden, dtype=torch.float32).reshape(
        intermediate, hidden
    )
    up = (
        torch.arange(intermediate * hidden, dtype=torch.float32).reshape(
            intermediate, hidden
        )
        + 1000.0
    )
    dst = _dest((2 * intermediate, hidden))

    placements = [
        CopyPlacement(
            hf_key="mlp.gate_proj.weight",
            phyai_key="mlp.gate_up_proj.weight",
            src_slices=(Slice1D(0, 0, intermediate),),
            dst_slices=(Slice1D(0, 0, intermediate),),
        ),
        CopyPlacement(
            hf_key="mlp.up_proj.weight",
            phyai_key="mlp.gate_up_proj.weight",
            src_slices=(Slice1D(0, 0, intermediate),),
            dst_slices=(Slice1D(0, intermediate, intermediate),),
        ),
    ]
    apply_placements(
        placements,
        {"mlp.gate_proj.weight": gate, "mlp.up_proj.weight": up}.__getitem__,
        {"mlp.gate_up_proj.weight": dst},
    )
    assert torch.equal(dst.narrow(0, 0, intermediate), gate)
    assert torch.equal(dst.narrow(0, intermediate, intermediate), up)


def test_fused_gate_up_tp2_rank0():
    """Per-rank gate/up at tp=2, rank=0 takes first half of each src."""
    intermediate_global = 8
    intermediate_per_rank = 4
    hidden = 4
    gate = torch.arange(intermediate_global * hidden, dtype=torch.float32).reshape(
        intermediate_global, hidden
    )
    up = (
        torch.arange(intermediate_global * hidden, dtype=torch.float32).reshape(
            intermediate_global, hidden
        )
        + 1000.0
    )
    dst = _dest((2 * intermediate_per_rank, hidden))
    rank = 0

    placements = [
        CopyPlacement(
            hf_key="mlp.gate_proj.weight",
            phyai_key="mlp.gate_up_proj.weight",
            src_slices=(
                Slice1D(0, rank * intermediate_per_rank, intermediate_per_rank),
            ),
            dst_slices=(Slice1D(0, 0, intermediate_per_rank),),
        ),
        CopyPlacement(
            hf_key="mlp.up_proj.weight",
            phyai_key="mlp.gate_up_proj.weight",
            src_slices=(
                Slice1D(0, rank * intermediate_per_rank, intermediate_per_rank),
            ),
            dst_slices=(Slice1D(0, intermediate_per_rank, intermediate_per_rank),),
        ),
    ]
    apply_placements(
        placements,
        {"mlp.gate_proj.weight": gate, "mlp.up_proj.weight": up}.__getitem__,
        {"mlp.gate_up_proj.weight": dst},
    )
    assert torch.equal(
        dst.narrow(0, 0, intermediate_per_rank),
        gate.narrow(0, 0, intermediate_per_rank),
    )
    assert torch.equal(
        dst.narrow(0, intermediate_per_rank, intermediate_per_rank),
        up.narrow(0, 0, intermediate_per_rank),
    )


# ---------------------------------------------------------------------------
# apply_placements — QKV with GQA replica_factor=2
# ---------------------------------------------------------------------------


def test_qkv_gqa_replicas_share_kv_slot():
    """At tp=4 with num_kv_heads=2 (replica_factor=2), ranks 0/1 share the
    same K slice; ranks 2/3 share the other K slice. Q is sharded normally
    across all 4 ranks."""
    head_dim = 8
    num_q_heads = 8
    num_kv_heads = 2
    tp_size = 4
    replica_factor = tp_size // num_kv_heads  # 2

    q_size_per_rank = num_q_heads * head_dim // tp_size  # 16
    kv_size_per_rank = num_kv_heads * head_dim * replica_factor // tp_size  # 8
    # Note: per-rank kv width includes the GQA replica → 8 == one head's width

    hidden = 16
    q_weight = torch.randn(num_q_heads * head_dim, hidden)
    k_weight = torch.randn(num_kv_heads * head_dim, hidden)
    v_weight = torch.randn(num_kv_heads * head_dim, hidden)

    def placements_for_rank(tp_rank: int):
        out: list = []
        offset = 0
        # Q
        out.append(
            CopyPlacement(
                hf_key="q_proj.weight",
                phyai_key="qkv.weight",
                src_slices=(Slice1D(0, tp_rank * q_size_per_rank, q_size_per_rank),),
                dst_slices=(Slice1D(0, offset, q_size_per_rank),),
            )
        )
        offset += q_size_per_rank
        # K, V — replica_factor floor-div
        slot_rank = tp_rank // replica_factor
        for hf_name in ("k_proj.weight", "v_proj.weight"):
            out.append(
                CopyPlacement(
                    hf_key=hf_name,
                    phyai_key="qkv.weight",
                    src_slices=(
                        Slice1D(0, slot_rank * kv_size_per_rank, kv_size_per_rank),
                    ),
                    dst_slices=(Slice1D(0, offset, kv_size_per_rank),),
                )
            )
            offset += kv_size_per_rank
        return out

    rank0_dst = torch.zeros(q_size_per_rank + 2 * kv_size_per_rank, hidden)
    rank1_dst = torch.zeros(q_size_per_rank + 2 * kv_size_per_rank, hidden)

    src_state = {
        "q_proj.weight": q_weight,
        "k_proj.weight": k_weight,
        "v_proj.weight": v_weight,
    }
    apply_placements(
        placements_for_rank(0), src_state.__getitem__, {"qkv.weight": rank0_dst}
    )
    apply_placements(
        placements_for_rank(1), src_state.__getitem__, {"qkv.weight": rank1_dst}
    )

    # Ranks 0 and 1 see different Q slots but the SAME K and V slots.
    q0 = rank0_dst.narrow(0, 0, q_size_per_rank)
    q1 = rank1_dst.narrow(0, 0, q_size_per_rank)
    assert not torch.equal(q0, q1)

    k0 = rank0_dst.narrow(0, q_size_per_rank, kv_size_per_rank)
    k1 = rank1_dst.narrow(0, q_size_per_rank, kv_size_per_rank)
    assert torch.equal(k0, k1)

    v0 = rank0_dst.narrow(0, q_size_per_rank + kv_size_per_rank, kv_size_per_rank)
    v1 = rank1_dst.narrow(0, q_size_per_rank + kv_size_per_rank, kv_size_per_rank)
    assert torch.equal(v0, v1)


# ---------------------------------------------------------------------------
# apply_placements — vocab padding on the trailing rank
# ---------------------------------------------------------------------------


def test_vocab_padding_trailing_rank():
    """Last rank: real rows fewer than per_rank — overhang is zeroed."""
    V_real = 13
    V_padded = 16
    tp_size = 4
    per_rank = V_padded // tp_size  # 4
    rank = 3
    start = rank * per_rank  # 12
    end = start + per_rank  # 16
    real_start = min(start, V_real)  # 12
    real_end = min(end, V_real)  # 13
    n_real = real_end - real_start  # 1

    embed_dim = 6
    hf_weight = torch.arange(V_real * embed_dim, dtype=torch.float32).reshape(
        V_real, embed_dim
    )
    dst = torch.full((per_rank, embed_dim), 99.0)  # poison

    placements = [
        CopyPlacement(
            hf_key="embed.weight",
            phyai_key="embed.weight",
            src_slices=(Slice1D(0, real_start, n_real),),
            dst_slices=(Slice1D(0, 0, n_real),),
        ),
        ZeroPlacement(
            phyai_key="embed.weight",
            dst_slices=(Slice1D(0, n_real, per_rank - n_real),),
        ),
    ]
    apply_placements(placements, lambda k: hf_weight, {"embed.weight": dst})

    # First row is the real V[12]; remaining 3 rows are zero.
    assert torch.equal(dst.narrow(0, 0, 1), hf_weight.narrow(0, 12, 1))
    assert torch.equal(
        dst.narrow(0, 1, 3),
        torch.zeros(3, embed_dim),
    )


# ---------------------------------------------------------------------------
# apply_placements — error / fast-path edges
# ---------------------------------------------------------------------------


def test_shape_mismatch_raises():
    src = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    dst = _dest((4, 4))
    placements = [CopyPlacement(hf_key="x", phyai_key="x")]
    with pytest.raises(ValueError, match="placement shape mismatch"):
        apply_placements(placements, lambda k: src, {"x": dst})


def test_scalar_fast_path():
    """``shape=()`` HF tensor lands on 1-elem param via fill_."""
    src = torch.tensor(7.5)  # 0-d
    dst = torch.zeros(1)
    placements = [CopyPlacement(hf_key="scale", phyai_key="scale")]
    apply_placements(placements, lambda k: src, {"scale": dst})
    assert dst.item() == pytest.approx(7.5)


def test_chained_narrows_apply_left_to_right():
    """Two src slices applied in sequence equal a single combined slice."""
    src = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    dst = _dest((2, 4))
    placements = [
        CopyPlacement(
            hf_key="x",
            phyai_key="x",
            src_slices=(Slice1D(0, 2, 4), Slice1D(0, 1, 2)),  # rows 3..4
            dst_slices=(
                Slice1D(1, 0, 4),
            ),  # only first 4 cols of dst — but dst is 2x4 so this is the whole row
        ),
    ]
    # Rebuild dst to match — combined narrow gives shape (2, 8); src_slices last
    # narrow leaves 2 rows; we want full 8 cols. Adjust dst.
    dst = _dest((2, 8))
    placements = [
        CopyPlacement(
            hf_key="x",
            phyai_key="x",
            src_slices=(Slice1D(0, 2, 4), Slice1D(0, 1, 2)),
        ),
    ]
    apply_placements(placements, lambda k: src, {"x": dst})
    # Rows 2..6 first, then within that rows 1..3 → original rows 3..4.
    assert torch.equal(dst, src.narrow(0, 3, 2))
