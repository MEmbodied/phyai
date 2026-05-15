"""LayerNorm — backend parity, F.layer_norm reference, placement loading."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from phyai.layers.layer_norm import LayerNorm
from phyai.layers.placement import CopyPlacement, apply_placements


cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="phyai LayerNorm backends are CUDA-only",
)


# --------------------------------------------------------------------------- #
# Construction-time validation                                                #
# --------------------------------------------------------------------------- #


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown norm backend"):
        LayerNorm(64, backend="banana")


def test_zero_hidden_size_raises():
    with pytest.raises(ValueError, match="hidden_size"):
        LayerNorm(0, backend="phyai-kernel")


def test_extra_repr_contains_key_fields():
    m = LayerNorm(128, eps=1e-6, backend="phyai-kernel", bias=False)
    s = repr(m)
    assert "128" in s
    assert "eps=1e-06" in s
    assert "bias=False" in s
    assert "backend='phyai-kernel'" in s


def test_no_bias_does_not_register_parameter():
    m = LayerNorm(64, backend="phyai-kernel", bias=False)
    assert m.bias is None
    assert not m.has_bias
    assert "bias" not in dict(m.named_parameters())


# --------------------------------------------------------------------------- #
# Placements                                                                  #
# --------------------------------------------------------------------------- #


def test_placements_with_bias():
    m = LayerNorm(64, backend="phyai-kernel", prefix="vision.encoder.layer_norm1")
    pls = m.placements()
    assert len(pls) == 2
    assert all(isinstance(p, CopyPlacement) for p in pls)
    keys = {(p.hf_key, p.phyai_key) for p in pls}
    assert (
        "vision.encoder.layer_norm1.weight",
        "vision.encoder.layer_norm1.weight",
    ) in keys
    assert (
        "vision.encoder.layer_norm1.bias",
        "vision.encoder.layer_norm1.bias",
    ) in keys


def test_placements_without_bias():
    m = LayerNorm(64, backend="phyai-kernel", bias=False, prefix="text.norm")
    pls = m.placements()
    assert len(pls) == 1
    assert pls[0].hf_key == "text.norm.weight"


def test_placements_load_weight_and_bias():
    """End-to-end: load HF tensors via the placement protocol."""
    D = 32
    m = LayerNorm(D, backend="phyai-kernel", prefix="ln")
    src_w = torch.randn(D)
    src_b = torch.randn(D)
    apply_placements(
        m.placements(),
        {"ln.weight": src_w, "ln.bias": src_b}.__getitem__,
        {"ln.weight": m.weight.data, "ln.bias": m.bias.data},
    )
    torch.testing.assert_close(m.weight.data, src_w)
    torch.testing.assert_close(m.bias.data, src_b)


# --------------------------------------------------------------------------- #
# Forward correctness                                                         #
# --------------------------------------------------------------------------- #


def _ref_layer_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, eps: float
) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), weight=weight, bias=bias, eps=eps)


@cuda_only
@pytest.mark.parametrize("backend", ["flashinfer", "phyai-kernel"])
@pytest.mark.parametrize("with_bias", [True, False])
def test_forward_matches_torch_reference_bf16(backend, with_bias):
    torch.manual_seed(0)
    D = 1152  # PaliGemma SigLIP hidden
    m = LayerNorm(
        D, eps=1e-6, backend=backend, bias=with_bias, dtype=torch.bfloat16
    ).cuda()

    src_w = (torch.randn(D) * 0.05 + 1.0).to(torch.bfloat16).cuda()
    m.weight.data.copy_(src_w)
    if with_bias:
        src_b = (torch.randn(D) * 0.02).to(torch.bfloat16).cuda()
        m.bias.data.copy_(src_b)
    else:
        src_b = None

    x = (torch.randn(8, 16, D) * 0.5).to(torch.bfloat16).cuda()
    y = m(x)
    ref = _ref_layer_norm(x, src_w, src_b, 1e-6)
    torch.testing.assert_close(y, ref, atol=2e-2, rtol=2e-2)


@cuda_only
def test_flashinfer_matches_phyai_kernel():
    """Both backends must produce numerically equivalent output."""
    torch.manual_seed(1)
    D = 768
    src_w = (torch.randn(D) * 0.05 + 1.0).to(torch.bfloat16).cuda()
    src_b = (torch.randn(D) * 0.02).to(torch.bfloat16).cuda()

    m_fi = LayerNorm(D, eps=1e-5, backend="flashinfer", dtype=torch.bfloat16).cuda()
    m_pk = LayerNorm(D, eps=1e-5, backend="phyai-kernel", dtype=torch.bfloat16).cuda()
    m_fi.weight.data.copy_(src_w)
    m_pk.weight.data.copy_(src_w)
    m_fi.bias.data.copy_(src_b)
    m_pk.bias.data.copy_(src_b)

    x = (torch.randn(4, 32, D) * 0.5).to(torch.bfloat16).cuda()
    y_fi = m_fi(x)
    y_pk = m_pk(x)
    torch.testing.assert_close(y_fi, y_pk, atol=2e-2, rtol=2e-2)


@cuda_only
def test_phyai_kernel_higher_rank_input():
    """4-D input flattens to (N, D) and reshapes back."""
    torch.manual_seed(2)
    B, S, H, D = 2, 4, 3, 256
    m = LayerNorm(D, backend="phyai-kernel", dtype=torch.bfloat16).cuda()
    x = (torch.randn(B, S, H, D) * 0.3).to(torch.bfloat16).cuda()
    y = m(x)
    assert y.shape == (B, S, H, D)
    ref = _ref_layer_norm(x, m.weight.data, m.bias.data, m.variance_epsilon)
    torch.testing.assert_close(y, ref, atol=2e-2, rtol=2e-2)


@cuda_only
@pytest.mark.parametrize("backend", ["flashinfer", "phyai-kernel"])
def test_no_bias_path_matches_reference(backend):
    """``bias=False`` must give identical output to F.layer_norm(..., bias=None)."""
    torch.manual_seed(3)
    D = 384
    m = LayerNorm(D, backend=backend, bias=False, dtype=torch.bfloat16).cuda()
    src_w = (torch.randn(D) * 0.05 + 1.0).to(torch.bfloat16).cuda()
    m.weight.data.copy_(src_w)

    x = (torch.randn(16, D) * 0.4).to(torch.bfloat16).cuda()
    y = m(x)
    ref = _ref_layer_norm(x, src_w, None, m.variance_epsilon)
    torch.testing.assert_close(y, ref, atol=2e-2, rtol=2e-2)


@cuda_only
def test_phyai_kernel_fp32_path():
    """fp32 path on phyai-kernel: tighter tolerance."""
    torch.manual_seed(4)
    D = 1024
    m = LayerNorm(D, backend="phyai-kernel", dtype=torch.float32).cuda()
    src_w = (torch.randn(D) * 0.05 + 1.0).cuda()
    src_b = (torch.randn(D) * 0.02).cuda()
    m.weight.data.copy_(src_w)
    m.bias.data.copy_(src_b)

    x = (torch.randn(8, D) * 0.5).cuda()
    y = m(x)
    ref = _ref_layer_norm(x, src_w, src_b, m.variance_epsilon)
    torch.testing.assert_close(y, ref, atol=1e-5, rtol=1e-5)
