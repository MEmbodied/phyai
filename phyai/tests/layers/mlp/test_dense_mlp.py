"""DenseMLP — forward parity, attached-loader smoke tests, weight-load equivalence."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import phyai.layers.linear as L
from phyai.layers.mlp import DenseMLP
from phyai.layers.quant import Fp8Spec, Granularity
from phyai.parallel.mesh import Mesh
from phyai.parallel.state import _meshes, register_mesh


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fake_mesh(
    *, sizes: dict[str, int] | None = None, ranks: dict[str, int] | None = None
) -> Mesh:
    sizes = sizes or {"tp": 1}
    ranks = ranks or {}
    tm = MagicMock()
    tm.mesh_dim_names = tuple(sizes.keys())
    _names = tm.mesh_dim_names
    tm.size.side_effect = lambda axis: sizes.get(
        axis if isinstance(axis, str) else _names[axis], 1
    )
    tm.get_local_rank.side_effect = lambda axis: ranks.get(axis, 0)
    tm.get_group.side_effect = lambda axis: MagicMock(name=f"pg-{axis}")
    mesh = Mesh(tm, name="model")
    register_mesh(mesh)
    return mesh


@pytest.fixture
def fake_mesh():
    saved = dict(_meshes)
    try:
        yield _fake_mesh
    finally:
        _meshes.clear()
        _meshes.update(saved)
        L._reset_for_test()


def _init_dispatcher():
    return L.init(register_flashinfer=False, validate=False)


# ---------------------------------------------------------------------------
# Activation alias normalisation
# ---------------------------------------------------------------------------


def test_activation_aliases_resolve_consistently(fake_mesh):
    fake_mesh()
    _init_dispatcher()
    aliases = ("gelu_tanh", "gelu_pytorch_tanh", "gelu-tanh", "gelu_new")
    for alias in aliases:
        m = DenseMLP(
            hidden_size=8,
            intermediate_size=16,
            activation=alias,
            gated=True,
            prefix="block.mlp",
        )
        assert m.activation == "gelu_tanh"


def test_gated_silu_construct_default(fake_mesh):
    fake_mesh()
    _init_dispatcher()
    m = DenseMLP(
        hidden_size=8,
        intermediate_size=16,
        activation="silu",
        gated=True,
        prefix="block.mlp",
    )
    assert m.activation == "silu"
    assert m.gated is True
    assert hasattr(m, "gate_up_proj")
    assert hasattr(m, "down_proj")
    assert m.gate_up_proj.weight.shape == (32, 8)  # gate + up = 2 * 16
    assert m.down_proj.weight.shape == (8, 16)


def test_plain_construct_with_bias(fake_mesh):
    fake_mesh()
    _init_dispatcher()
    m = DenseMLP(
        hidden_size=8,
        intermediate_size=24,
        activation="gelu_tanh",
        gated=False,
        bias=True,
        prefix="block.mlp",
    )
    assert m.gated is False
    assert hasattr(m, "fc1")
    assert hasattr(m, "fc2")
    assert m.fc1.weight.shape == (24, 8)
    assert m.fc2.weight.shape == (8, 24)
    assert m.fc1.bias is not None
    assert m.fc2.bias is not None


def test_plain_relu2_constructs_cosmos3_edge_layout(fake_mesh):
    fake_mesh()
    _init_dispatcher()
    m = DenseMLP(
        hidden_size=8,
        intermediate_size=24,
        activation="relu2",
        gated=False,
        plain_hf_legs=("up_proj", "down_proj"),
        prefix="model.layers.0.mlp",
    )
    assert m.fc1.weight.hf_keys == [("model.layers.0.mlp.up_proj.weight", None)]
    assert m.fc2.weight.hf_keys == [("model.layers.0.mlp.down_proj.weight", None)]


def test_plain_silu_rejected(fake_mesh):
    fake_mesh()
    _init_dispatcher()
    with pytest.raises(ValueError, match="non-gated SiLU"):
        DenseMLP(
            hidden_size=8,
            intermediate_size=16,
            activation="silu",
            gated=False,
            prefix="block.mlp",
        )


def test_gelu_tanh_block_fp8_enables_fused_activation_quant(fake_mesh, monkeypatch):
    fake_mesh()
    _init_dispatcher()
    monkeypatch.setattr(
        "phyai.layers.mlp.dense_mlp._resolve_gated_act", lambda _name: lambda x: x
    )
    block_fp8 = Fp8Spec(
        granularity=Granularity.BLOCK,
        block_shape=(128, 128),
    )

    fused = DenseMLP(
        hidden_size=128,
        intermediate_size=256,
        activation="gelu_tanh",
        gated=True,
        spec_out=block_fp8,
        prefix="mlp",
    )
    silu = DenseMLP(
        hidden_size=128,
        intermediate_size=256,
        activation="silu",
        gated=True,
        spec_out=block_fp8,
        prefix="mlp",
    )

    assert fused._use_fused_gelu_tanh_fp8 is True
    assert silu._use_fused_gelu_tanh_fp8 is False


def test_unknown_activation_rejected(fake_mesh):
    fake_mesh()
    _init_dispatcher()
    with pytest.raises(ValueError, match="Unsupported"):
        DenseMLP(
            hidden_size=8,
            intermediate_size=16,
            activation="relu",
            gated=True,
            prefix="block.mlp",
        )


# ---------------------------------------------------------------------------
# Attached-loader smoke tests
# ---------------------------------------------------------------------------


def test_attach_gated_default_names(fake_mesh):
    fake_mesh()
    _init_dispatcher()
    m = DenseMLP(
        hidden_size=8,
        intermediate_size=16,
        activation="silu",
        gated=True,
        bias=False,
        prefix="model.layers.3.mlp",
    )
    # gate_up_proj is a fused MergedColumn with two HF source legs.
    assert m.gate_up_proj.weight.hf_keys == [
        ("model.layers.3.mlp.gate_proj.weight", 0),
        ("model.layers.3.mlp.up_proj.weight", 1),
    ]
    # down_proj is row-parallel; one source.
    assert m.down_proj.weight.hf_keys == [("model.layers.3.mlp.down_proj.weight", None)]


def test_attach_gated_custom_legs(fake_mesh):
    fake_mesh()
    _init_dispatcher()
    m = DenseMLP(
        hidden_size=8,
        intermediate_size=16,
        activation="silu",
        gated=True,
        gated_hf_legs=("w_gate", "w_up"),
        prefix="block.mlp",
    )
    assert m.gate_up_proj.weight.hf_keys == [
        ("block.mlp.w_gate.weight", 0),
        ("block.mlp.w_up.weight", 1),
    ]


def test_attach_plain_path(fake_mesh):
    fake_mesh()
    _init_dispatcher()
    m = DenseMLP(
        hidden_size=8,
        intermediate_size=24,
        activation="gelu_tanh",
        gated=False,
        bias=True,
        prefix="vision.encoder.layers.0.mlp",
    )
    assert m.fc1.weight.hf_keys == [("vision.encoder.layers.0.mlp.fc1.weight", None)]
    assert m.fc1.bias.hf_keys == [("vision.encoder.layers.0.mlp.fc1.bias", None)]
    assert m.fc2.weight.hf_keys == [("vision.encoder.layers.0.mlp.fc2.weight", None)]
    assert m.fc2.bias.hf_keys == [("vision.encoder.layers.0.mlp.fc2.bias", None)]


# ---------------------------------------------------------------------------
# Forward parity (CUDA — flashinfer fused kernels)
# ---------------------------------------------------------------------------


cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="flashinfer act_and_mul needs CUDA"
)


def _load_gated(m: DenseMLP, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor):
    """Load three HF-style tensors via the param-attached weight_loader."""
    m.gate_up_proj.weight.weight_loader(m.gate_up_proj.weight, gate, 0)
    m.gate_up_proj.weight.weight_loader(m.gate_up_proj.weight, up, 1)
    m.down_proj.weight.weight_loader(m.down_proj.weight, down, None)


@cuda_only
def test_forward_gated_silu_matches_torch_reference(fake_mesh):
    fake_mesh()
    _init_dispatcher()
    H, I = 64, 256
    m = DenseMLP(
        hidden_size=H,
        intermediate_size=I,
        activation="silu",
        gated=True,
        params_dtype=torch.bfloat16,
        prefix="mlp",
    ).cuda()

    torch.manual_seed(0)
    gate = (torch.randn(I, H) * 0.02).to(torch.bfloat16).cuda()
    up = (torch.randn(I, H) * 0.02).to(torch.bfloat16).cuda()
    down = (torch.randn(H, I) * 0.02).to(torch.bfloat16).cuda()
    _load_gated(m, gate, up, down)

    x = (torch.randn(8, H) * 0.1).to(torch.bfloat16).cuda()
    y = m(x)
    ref = F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)
    torch.testing.assert_close(y, ref, atol=2e-2, rtol=2e-2)


@cuda_only
def test_forward_gated_gelu_tanh_matches_torch_reference(fake_mesh):
    fake_mesh()
    _init_dispatcher()
    H, I = 64, 256
    m = DenseMLP(
        hidden_size=H,
        intermediate_size=I,
        activation="gelu_tanh",
        gated=True,
        params_dtype=torch.bfloat16,
        prefix="mlp",
    ).cuda()

    torch.manual_seed(1)
    gate = (torch.randn(I, H) * 0.02).to(torch.bfloat16).cuda()
    up = (torch.randn(I, H) * 0.02).to(torch.bfloat16).cuda()
    down = (torch.randn(H, I) * 0.02).to(torch.bfloat16).cuda()
    _load_gated(m, gate, up, down)

    x = (torch.randn(8, H) * 0.1).to(torch.bfloat16).cuda()
    y = m(x)
    ref = F.linear(
        F.gelu(F.linear(x, gate), approximate="tanh") * F.linear(x, up), down
    )
    torch.testing.assert_close(y, ref, atol=2e-2, rtol=2e-2)


def test_forward_plain_gelu_tanh_matches_torch_reference(fake_mesh):
    fake_mesh()
    _init_dispatcher()
    # Plain path uses F.gelu — works on CPU, no flashinfer needed.
    H, I = 32, 96
    m = DenseMLP(
        hidden_size=H,
        intermediate_size=I,
        activation="gelu_tanh",
        gated=False,
        bias=True,
        params_dtype=torch.bfloat16,
        prefix="vit.mlp",
    )
    torch.manual_seed(2)
    nn.init.normal_(m.fc1.weight, std=0.02)
    nn.init.normal_(m.fc1.bias, std=0.02)
    nn.init.normal_(m.fc2.weight, std=0.02)
    nn.init.normal_(m.fc2.bias, std=0.02)

    x = (torch.randn(4, H) * 0.1).to(torch.bfloat16)
    y = m(x)
    h = F.gelu(F.linear(x, m.fc1.weight, m.fc1.bias), approximate="tanh")
    ref = F.linear(h, m.fc2.weight, m.fc2.bias)
    torch.testing.assert_close(y, ref, atol=0, rtol=0)


def test_forward_plain_relu2_matches_torch_reference(fake_mesh):
    fake_mesh()
    _init_dispatcher()
    m = DenseMLP(
        hidden_size=32,
        intermediate_size=96,
        activation="relu2",
        gated=False,
        params_dtype=torch.bfloat16,
        plain_hf_legs=("up_proj", "down_proj"),
        prefix="mlp",
    )
    torch.manual_seed(4)
    nn.init.normal_(m.fc1.weight, std=0.02)
    nn.init.normal_(m.fc2.weight, std=0.02)

    x = (torch.randn(4, 32) * 0.1).to(torch.bfloat16)
    y = m(x)
    ref = F.linear(F.relu(F.linear(x, m.fc1.weight)).square(), m.fc2.weight)
    torch.testing.assert_close(y, ref, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Fused-vs-split equivalence
# ---------------------------------------------------------------------------


@cuda_only
def test_fused_vs_split_load_produces_identical_output(fake_mesh):
    """Loading via [gate, up] split == pre-concatenated fused weight."""
    fake_mesh()
    _init_dispatcher()
    H, I = 32, 64
    torch.manual_seed(3)
    gate = (torch.randn(I, H) * 0.02).to(torch.bfloat16).cuda()
    up = (torch.randn(I, H) * 0.02).to(torch.bfloat16).cuda()
    down = (torch.randn(H, I) * 0.02).to(torch.bfloat16).cuda()
    x = (torch.randn(4, H) * 0.1).to(torch.bfloat16).cuda()

    # Path A: official placements (split gate/up).
    m_a = DenseMLP(
        hidden_size=H,
        intermediate_size=I,
        activation="silu",
        gated=True,
        params_dtype=torch.bfloat16,
        prefix="mlp",
    ).cuda()
    _load_gated(m_a, gate, up, down)
    y_a = m_a(x)

    # Path B: hand-stamp the fused [gate; up] weight directly.
    m_b = DenseMLP(
        hidden_size=H,
        intermediate_size=I,
        activation="silu",
        gated=True,
        params_dtype=torch.bfloat16,
        prefix="mlp",
    ).cuda()
    fused = torch.cat([gate, up], dim=0)
    m_b.gate_up_proj.weight.data.copy_(fused)
    m_b.down_proj.weight.data.copy_(down)
    y_b = m_b(x)

    torch.testing.assert_close(y_a, y_b, atol=0, rtol=0)
