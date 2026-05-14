"""Linear layer integration tests.

Most tests run at ws=1 via a mocked Mesh — the parallel ops bail out
without touching the dispatcher, so we can exercise layer construction,
weight allocation, and forward() on CPU. A couple of multi-rank tests
spin up gloo workers to verify the collective glue really fires at ws>1.
"""

from __future__ import annotations

import os
import socket
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

import phyai.layers.linear as L
from phyai.layers.placement import (
    CopyPlacement,
    Slice1D,
    apply_placements,
)


# ---------------------------------------------------------------------------
# ws=1 fixture-driven tests
# ---------------------------------------------------------------------------


def _init_default_dispatcher():
    """Re-init the dispatcher without flashinfer so tests don't depend on it."""
    return L.init(register_flashinfer=False, validate=False)


def test_replicated_linear_bf16_matches_F_linear(fake_mesh):
    fake_mesh(name="model")
    _init_default_dispatcher()
    layer = L.ReplicatedLinear(
        in_features=32,
        out_features=16,
        bias=True,
        params_dtype=torch.bfloat16,
    )
    nn.init.normal_(layer.weight, std=0.05)
    nn.init.normal_(layer.bias, std=0.05)

    x = torch.randn(4, 32, dtype=torch.bfloat16)
    y, bias_out = layer(x)
    assert bias_out is None
    ref = F.linear(x, layer.weight, layer.bias)
    torch.testing.assert_close(y, ref, atol=0, rtol=0)


def test_replicated_linear_skip_bias_add_returns_bias(fake_mesh):
    fake_mesh()
    _init_default_dispatcher()
    layer = L.ReplicatedLinear(
        in_features=16,
        out_features=8,
        bias=True,
        skip_bias_add=True,
        params_dtype=torch.bfloat16,
    )
    nn.init.normal_(layer.weight, std=0.05)
    nn.init.normal_(layer.bias, std=0.05)

    x = torch.randn(2, 16, dtype=torch.bfloat16)
    y, bias_out = layer(x)
    assert bias_out is layer.bias
    # y should NOT include bias (skip_bias_add=True).
    ref = F.linear(x, layer.weight, None)
    torch.testing.assert_close(y, ref, atol=0, rtol=0)


def test_column_parallel_ws1_matches_F_linear(fake_mesh):
    fake_mesh(sizes={"tp": 1})
    _init_default_dispatcher()
    layer = L.ColumnParallelLinear(
        in_features=32,
        out_features=16,
        axis="tp",
        bias=False,
        params_dtype=torch.bfloat16,
    )
    nn.init.normal_(layer.weight, std=0.05)

    x = torch.randn(5, 32, dtype=torch.bfloat16)
    y, _ = layer(x)
    ref = F.linear(x, layer.weight)
    torch.testing.assert_close(y, ref, atol=0, rtol=0)
    # With ws=1 and gather_output default False, shape is the per-rank output.
    assert y.shape == (5, 16)


def test_column_parallel_rejects_indivisible_split(fake_mesh):
    fake_mesh(sizes={"tp": 4})
    _init_default_dispatcher()
    with pytest.raises(ValueError, match="not divisible"):
        L.ColumnParallelLinear(
            in_features=8,
            out_features=30,  # 30 % 4 != 0
            axis="tp",
            bias=False,
        )


def test_row_parallel_ws1_matches_F_linear(fake_mesh):
    fake_mesh(sizes={"tp": 1})
    _init_default_dispatcher()
    layer = L.RowParallelLinear(
        in_features=32,
        out_features=16,
        axis="tp",
        bias=False,
        params_dtype=torch.bfloat16,
    )
    nn.init.normal_(layer.weight, std=0.05)

    x = torch.randn(3, 32, dtype=torch.bfloat16)
    y, _ = layer(x)
    ref = F.linear(x, layer.weight)
    torch.testing.assert_close(y, ref, atol=0, rtol=0)


def test_row_parallel_bias_only_on_rank0(fake_mesh):
    """At ws=1 the bias IS added (rank==0)."""
    fake_mesh(sizes={"tp": 1})
    _init_default_dispatcher()
    layer = L.RowParallelLinear(
        in_features=16,
        out_features=8,
        axis="tp",
        bias=True,
        params_dtype=torch.bfloat16,
    )
    layer.weight.data.zero_()
    layer.bias.data.fill_(3.0)

    x = torch.zeros(2, 16, dtype=torch.bfloat16)
    y, _ = layer(x)
    assert torch.all(y == 3.0)


def test_row_parallel_rejects_indivisible_in(fake_mesh):
    fake_mesh(sizes={"tp": 4})
    _init_default_dispatcher()
    with pytest.raises(ValueError, match="not divisible"):
        L.RowParallelLinear(in_features=30, out_features=16, axis="tp")


# ---------------------------------------------------------------------------
# MergedColumnParallelLinear
# ---------------------------------------------------------------------------


def test_merged_column_construct_fused_weight(fake_mesh):
    fake_mesh(sizes={"tp": 1})
    _init_default_dispatcher()
    layer = L.MergedColumnParallelLinear(
        in_features=32,
        output_sizes=[16, 16],  # gate/up style
        axis="tp",
        bias=False,
        params_dtype=torch.bfloat16,
    )
    # Weight is fused: (gate_size + up_size, in)
    assert layer.weight.shape == (32, 32)
    assert layer.output_partition_sizes == [16, 16]
    assert layer.output_sizes_global == [16, 16]


def test_merged_column_placements_two_shards(fake_mesh):
    fake_mesh(sizes={"tp": 2})
    _init_default_dispatcher()
    layer = L.MergedColumnParallelLinear(
        in_features=16,
        output_sizes=[16, 16],
        axis="tp",
        bias=False,
        params_dtype=torch.bfloat16,
        prefix="model.layers.0.mlp.gate_up_proj",
    )
    # Per-rank partition sizes.
    assert layer.output_partition_sizes == [8, 8]
    assert layer.weight.shape == (16, 16)

    plans = layer.placements()
    # Two CopyPlacements (gate, up) — bias=False so no extras.
    assert len(plans) == 2

    # gate_proj.weight → first half of dst, src rows 0..8 (rank 0).
    assert plans[0] == CopyPlacement(
        hf_key="model.layers.0.mlp.gate_proj.weight",
        phyai_key="model.layers.0.mlp.gate_up_proj.weight",
        src_slices=(Slice1D(0, 0, 8),),
        dst_slices=(Slice1D(0, 0, 8),),
    )
    # up_proj.weight → second half of dst.
    assert plans[1] == CopyPlacement(
        hf_key="model.layers.0.mlp.up_proj.weight",
        phyai_key="model.layers.0.mlp.gate_up_proj.weight",
        src_slices=(Slice1D(0, 0, 8),),
        dst_slices=(Slice1D(0, 8, 8),),
    )

    # Apply: feed disk gate/up tensors, check fused weight is laid out
    # as [gate_rank0; up_rank0].
    disk_gate = torch.full((16, 16), 1.0, dtype=torch.bfloat16)
    disk_up = torch.full((16, 16), 2.0, dtype=torch.bfloat16)
    apply_placements(
        plans,
        {
            "model.layers.0.mlp.gate_proj.weight": disk_gate,
            "model.layers.0.mlp.up_proj.weight": disk_up,
        }.__getitem__,
        {"model.layers.0.mlp.gate_up_proj.weight": layer.weight.data},
    )
    assert torch.all(layer.weight.data[0:8] == 1.0)
    assert torch.all(layer.weight.data[8:16] == 2.0)


def test_merged_column_placements_with_custom_hf_names(fake_mesh):
    fake_mesh(sizes={"tp": 1})
    _init_default_dispatcher()
    layer = L.MergedColumnParallelLinear(
        in_features=16,
        output_sizes=[8, 8],
        axis="tp",
        bias=False,
        prefix="block.mlp.w12",
    )
    plans = layer.placements(hf_names=("w_gate", "w_up"))
    assert plans[0].hf_key == "block.mlp.w_gate.weight"
    assert plans[1].hf_key == "block.mlp.w_up.weight"


# ---------------------------------------------------------------------------
# QKVParallelLinear
# ---------------------------------------------------------------------------


def test_qkv_linear_no_gqa_shapes(fake_mesh):
    fake_mesh(sizes={"tp": 1})
    _init_default_dispatcher()
    layer = L.QKVParallelLinear(
        hidden_size=32,
        head_dim=8,
        num_heads=4,
        num_kv_heads=4,
        axis="tp",
        bias=False,
        params_dtype=torch.bfloat16,
    )
    # out = q (4*8) + k (4*8) + v (4*8) = 96
    assert layer.weight.shape == (96, 32)
    assert layer.num_kv_replicas == 1


def test_qkv_linear_gqa_replicates_kv(fake_mesh):
    """tp_size=4 with 2 KV heads → each rank gets the whole KV replicated twice."""
    fake_mesh(sizes={"tp": 4})
    _init_default_dispatcher()
    layer = L.QKVParallelLinear(
        hidden_size=32,
        head_dim=8,
        num_heads=8,
        num_kv_heads=2,
        axis="tp",
        bias=False,
        params_dtype=torch.bfloat16,
    )
    # Effective kv_heads = tp_size (2 kv heads replicated 4//2=2 times).
    # q_size = 8 * 8 = 64, kv_size = 4 * 8 = 32.
    # per-rank: q = 64/4 = 16, kv = 32/4 = 8 each.
    assert layer.num_kv_replicas == 2
    assert layer.output_partition_sizes == [16, 8, 8]
    assert layer.weight.shape == (32, 32)


def test_qkv_linear_rejects_nonmultiple_tp(fake_mesh):
    fake_mesh(sizes={"tp": 3})
    _init_default_dispatcher()
    with pytest.raises(ValueError, match="multiple of num_kv_heads"):
        L.QKVParallelLinear(
            hidden_size=32,
            head_dim=8,
            num_heads=6,
            num_kv_heads=2,
            axis="tp",
            bias=False,
        )


def test_qkv_linear_placements_q_k_v_shards(fake_mesh):
    fake_mesh(sizes={"tp": 1})
    _init_default_dispatcher()
    layer = L.QKVParallelLinear(
        hidden_size=16,
        head_dim=4,
        num_heads=2,
        num_kv_heads=2,
        axis="tp",
        bias=False,
        params_dtype=torch.bfloat16,
        prefix="model.layers.0.self_attn.qkv_proj",
    )

    plans = layer.placements()
    # Q, K, V — bias=False so 3 placements.
    assert len(plans) == 3
    assert [p.hf_key for p in plans] == [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
    ]
    # All write into qkv_proj.weight at offsets 0 / 8 / 16 (each is 8 rows).
    assert plans[0].dst_slices == (Slice1D(0, 0, 8),)
    assert plans[1].dst_slices == (Slice1D(0, 8, 8),)
    assert plans[2].dst_slices == (Slice1D(0, 16, 8),)

    disk_q = torch.full((2 * 4, 16), 1.0, dtype=torch.bfloat16)
    disk_k = torch.full((2 * 4, 16), 2.0, dtype=torch.bfloat16)
    disk_v = torch.full((2 * 4, 16), 3.0, dtype=torch.bfloat16)
    apply_placements(
        plans,
        {
            "model.layers.0.self_attn.q_proj.weight": disk_q,
            "model.layers.0.self_attn.k_proj.weight": disk_k,
            "model.layers.0.self_attn.v_proj.weight": disk_v,
        }.__getitem__,
        {"model.layers.0.self_attn.qkv_proj.weight": layer.weight.data},
    )
    # [q | k | v] in fused param
    assert torch.all(layer.weight.data[0:8] == 1.0)
    assert torch.all(layer.weight.data[8:16] == 2.0)
    assert torch.all(layer.weight.data[16:24] == 3.0)


def test_qkv_linear_placements_with_custom_hf_names(fake_mesh):
    fake_mesh(sizes={"tp": 1})
    _init_default_dispatcher()
    layer = L.QKVParallelLinear(
        hidden_size=16,
        head_dim=4,
        num_heads=2,
        num_kv_heads=2,
        axis="tp",
        bias=False,
        prefix="block.attn.qkv_proj",
    )
    plans = layer.placements(hf_names={"q": "query", "k": "key", "v": "value"})
    assert plans[0].hf_key == "block.attn.query.weight"
    assert plans[1].hf_key == "block.attn.key.weight"
    assert plans[2].hf_key == "block.attn.value.weight"


def test_qkv_linear_placements_gqa_replica_share_kv(fake_mesh):
    """tp=4 with 2 KV heads: ranks 0/1 share the same K/V slot, 2/3 share the other."""
    fake_mesh(sizes={"tp": 4}, ranks={"tp": 0})
    _init_default_dispatcher()
    layer_r0 = L.QKVParallelLinear(
        hidden_size=32,
        head_dim=8,
        num_heads=8,
        num_kv_heads=2,
        axis="tp",
        bias=False,
        prefix="block.qkv_proj",
    )
    plans_r0 = layer_r0.placements()
    # K placement at rank 0 — slot_rank = 0 // 2 = 0 → src_slice starts at 0.
    k_r0 = plans_r0[1]  # plans = [q, k, v]
    assert k_r0.src_slices == (Slice1D(0, 0, 8),)

    fake_mesh(sizes={"tp": 4}, ranks={"tp": 1})
    layer_r1 = L.QKVParallelLinear(
        hidden_size=32,
        head_dim=8,
        num_heads=8,
        num_kv_heads=2,
        axis="tp",
        bias=False,
        prefix="block.qkv_proj",
    )
    plans_r1 = layer_r1.placements()
    # K at rank 1 — slot_rank = 1 // 2 = 0 → SAME src_slice as rank 0.
    k_r1 = plans_r1[1]
    assert k_r1.src_slices == k_r0.src_slices

    fake_mesh(sizes={"tp": 4}, ranks={"tp": 2})
    layer_r2 = L.QKVParallelLinear(
        hidden_size=32,
        head_dim=8,
        num_heads=8,
        num_kv_heads=2,
        axis="tp",
        bias=False,
        prefix="block.qkv_proj",
    )
    plans_r2 = layer_r2.placements()
    # K at rank 2 — slot_rank = 2 // 2 = 1 → src_slice starts at 8.
    k_r2 = plans_r2[1]
    assert k_r2.src_slices == (Slice1D(0, 8, 8),)


# ---------------------------------------------------------------------------
# Replicated / Column / Row placements smoke tests
# ---------------------------------------------------------------------------


def test_replicated_linear_placements(fake_mesh):
    fake_mesh()
    _init_default_dispatcher()
    layer = L.ReplicatedLinear(
        in_features=4, out_features=8, bias=True, prefix="block.fc"
    )
    plans = layer.placements()
    assert plans[0] == CopyPlacement(
        hf_key="block.fc.weight", phyai_key="block.fc.weight"
    )
    assert plans[1] == CopyPlacement(hf_key="block.fc.bias", phyai_key="block.fc.bias")


def test_column_parallel_placements_tp2_rank1(fake_mesh):
    fake_mesh(sizes={"tp": 2}, ranks={"tp": 1})
    _init_default_dispatcher()
    layer = L.ColumnParallelLinear(
        in_features=16,
        out_features=32,
        axis="tp",
        bias=False,
        prefix="block.fc",
    )
    plans = layer.placements()
    # per_rank = 16; rank=1 → src_slice starts at 16.
    assert plans[0].src_slices == (Slice1D(0, 16, 16),)
    assert plans[0].hf_key == "block.fc.weight"


def test_row_parallel_placements_slice_dim1(fake_mesh):
    fake_mesh(sizes={"tp": 4}, ranks={"tp": 2})
    _init_default_dispatcher()
    layer = L.RowParallelLinear(
        in_features=64,
        out_features=16,
        axis="tp",
        bias=True,
        prefix="block.out_proj",
    )
    plans = layer.placements()
    # weight: narrow on dim 1 at rank * 16.
    assert plans[0].src_slices == (Slice1D(1, 32, 16),)
    # bias: replicated, no slice.
    assert plans[1].src_slices == ()


def test_empty_prefix_rejected(fake_mesh):
    fake_mesh()
    _init_default_dispatcher()
    layer = L.ReplicatedLinear(in_features=4, out_features=8, prefix="")
    with pytest.raises(ValueError, match="non-empty prefix"):
        layer.placements()


# ---------------------------------------------------------------------------
# init() + force-env integration
# ---------------------------------------------------------------------------


def test_init_registers_torch_fallback_and_validates(fake_mesh):
    fake_mesh()
    d = L.init(register_flashinfer=False, validate=True, sample_specs=["bf16"])
    assert d is L.get_linear_dispatcher()
    # Picks TorchKernel for bf16.
    k = d.select(
        spec_id="bf16",
        M=8,
        N=64,
        K=64,
        in_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
    )
    assert k.name == "torch"


def test_init_force_env_overrides(fake_mesh, monkeypatch):
    monkeypatch.setenv("PHYAI_FORCE_LINEAR_KERNEL", "torch")
    fake_mesh()
    d = L.init(register_flashinfer=True, validate=False)
    # Even if flashinfer preferred for bf16 prefill, force→torch.
    k = d.select(
        spec_id="bf16",
        M=1024,
        N=64,
        K=64,
        in_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
    )
    assert k.name == "torch"


# ---------------------------------------------------------------------------
# Real ws>1 via gloo — confirms collective wiring
# ---------------------------------------------------------------------------


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _gloo_worker(rank, world_size, port, test_fn, err_queue):
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        try:
            test_fn(rank, world_size)
        finally:
            try:
                dist.destroy_process_group()
            except Exception:
                pass
    except BaseException as e:
        err_queue.put((rank, repr(e), traceback.format_exc()))


def _run_gloo(test_fn, *, world_size: int, timeout_s: float = 30.0) -> None:
    # Use ``fork`` instead of ``spawn`` so the workers inherit the parent's
    # already-imported modules — the test module isn't on ``sys.path`` under
    # pytest ``--import-mode=importlib`` and spawn would fail unpickling.
    # Fork is safe here because the gloo backend doesn't touch CUDA.
    ctx = mp.get_context("fork")
    err_queue = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(
            target=_gloo_worker,
            args=(r, world_size, port, test_fn, err_queue),
        )
        for r in range(world_size)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=timeout_s)
        if p.is_alive():
            p.terminate()
            p.join()
            raise TimeoutError("gloo worker hung")
    errors = []
    while not err_queue.empty():
        errors.append(err_queue.get_nowait())
    if errors:
        r, e, tb = errors[0]
        raise AssertionError(f"worker rank={r} failed: {e}\n{tb}")
    for i, p in enumerate(procs):
        if p.exitcode != 0:
            raise AssertionError(f"gloo worker rank={i} exited with code {p.exitcode}")


def _w_column_tp2_column_row_equiv(rank, world_size):
    """Column-then-Row at tp=2 should produce correct per-rank outputs.

    We disable the final all_reduce (``reduce_results=False``) because the
    collective layer has a pre-existing torch 2.10 compatibility issue
    that is orthogonal to the Linear tests. Summing the per-rank partials
    across ranks (done in the parent via ``err_queue``) reproduces the
    un-sharded F.linear ∘ F.linear reference.
    """
    import phyai.parallel as P
    import phyai.layers.linear as L
    from phyai.layers.placement import apply_placements

    torch.manual_seed(0)
    P.init(layout=(world_size,), mesh_dim_names=("tp",), device="cpu", backend="gloo")
    L.init(register_flashinfer=False, validate=False)

    hidden = 16
    inter = 32
    W1 = torch.randn(inter, hidden, dtype=torch.float32) * 0.1
    W2 = torch.randn(hidden, inter, dtype=torch.float32) * 0.1
    x = torch.randn(4, hidden, dtype=torch.float32) * 0.1

    col = L.ColumnParallelLinear(
        in_features=hidden,
        out_features=inter,
        axis="tp",
        bias=False,
        params_dtype=torch.float32,
        prefix="col",
    )
    row = L.RowParallelLinear(
        in_features=inter,
        out_features=hidden,
        axis="tp",
        bias=False,
        reduce_results=False,
        params_dtype=torch.float32,
        prefix="row",
    )
    apply_placements(
        col.placements(),
        {"col.weight": W1}.__getitem__,
        {"col.weight": col.weight.data},
    )
    apply_placements(
        row.placements(),
        {"row.weight": W2}.__getitem__,
        {"row.weight": row.weight.data},
    )

    y_col, _ = col(x)
    assert y_col.shape == (
        4,
        inter // world_size,
    ), f"rank={rank} got y_col.shape={y_col.shape}"
    # y_col per rank is x @ W1_this_rank_cols.T — verify.
    start = rank * (inter // world_size)
    end = start + (inter // world_size)
    expected_col = F.linear(x, W1[start:end, :])
    torch.testing.assert_close(y_col, expected_col, atol=1e-6, rtol=1e-6)

    # Row forward without reduce returns per-rank partial sum.
    y_row_partial, _ = row(y_col)
    assert y_row_partial.shape == (4, hidden)
    # Each rank's partial = y_col @ W2_this_rank_cols.T using W2 row-sliced.
    W2_rank = W2[:, start:end]
    expected_partial = F.linear(y_col, W2_rank)
    torch.testing.assert_close(
        y_row_partial,
        expected_partial,
        atol=1e-6,
        rtol=1e-6,
    )


def test_column_then_row_tp2_numerical_equivalence():
    _run_gloo(_w_column_tp2_column_row_equiv, world_size=2)
