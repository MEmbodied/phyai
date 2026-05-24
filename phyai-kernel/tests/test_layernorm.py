"""Numerical-equivalence tests for the Triton LayerNorm kernel.

Validates :func:`phyai_kernel.triton.layernorm` against
:func:`torch.nn.functional.layer_norm` across the dtype x hidden_size
matrix relevant to SigLIP / BERT / ViT.
"""

from __future__ import annotations

import pytest
import torch

import phyai_kernel
import phyai_kernel.triton.layer_norm as triton_ln_mod


if not torch.cuda.is_available():
    pytest.skip(
        "CUDA is required for phyai-kernel Triton tests", allow_module_level=True
    )


# --------------------------------------------------------------------------- #
# Reference                                                                    #
# --------------------------------------------------------------------------- #


def _ref_layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    return torch.nn.functional.layer_norm(
        x, normalized_shape=(x.shape[-1],), weight=weight, bias=bias, eps=eps
    )


# --------------------------------------------------------------------------- #
# Public-API smoke                                                             #
# --------------------------------------------------------------------------- #


def test_module_exposes_layernorm():
    assert phyai_kernel.layernorm is triton_ln_mod.layernorm


# --------------------------------------------------------------------------- #
# Shape x dtype x bias matrix                                                  #
# --------------------------------------------------------------------------- #


_HIDDEN_SIZES = [
    384,  # SigLIP-tiny
    768,  # ViT-base / BERT-base
    1024,  # ViT-large
    1152,  # PaliGemma SigLIP
    2048,
    3584,  # awkward, non-power-of-two
    4096,
    8192,  # boundary of single-block path
    12288,  # forces the two-pass kernel
    16384,
]

_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


@pytest.mark.parametrize("hidden_size", _HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("with_bias", [True, False])
def test_layernorm_matches_reference(hidden_size, dtype, with_bias):
    torch.manual_seed(0)
    n_rows = 17  # awkward to exercise masked tail
    x = (torch.randn(n_rows, hidden_size, device="cuda") * 0.5).to(dtype)
    # SigLIP stores weight/bias in the activation dtype; mirror that.
    weight = (torch.randn(hidden_size, device="cuda") * 0.1 + 1.0).to(dtype)
    bias = (
        (torch.randn(hidden_size, device="cuda") * 0.02).to(dtype)
        if with_bias
        else None
    )
    eps = 1e-5

    out = phyai_kernel.layernorm(x, weight, bias, eps)
    ref = _ref_layernorm(x, weight, bias, eps)

    if dtype == torch.float32:
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    else:
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


# --------------------------------------------------------------------------- #
# Higher-rank input (B, S, D) flattens correctly                              #
# --------------------------------------------------------------------------- #


def test_layernorm_3d_input():
    torch.manual_seed(1)
    B, S, D = 2, 8, 1152
    x = (torch.randn(B, S, D, device="cuda") * 0.5).to(torch.bfloat16)
    weight = (torch.randn(D, device="cuda") * 0.1 + 1.0).to(torch.bfloat16)
    bias = (torch.randn(D, device="cuda") * 0.02).to(torch.bfloat16)

    out = phyai_kernel.layernorm(x, weight, bias, 1e-5)
    ref = _ref_layernorm(x, weight, bias, 1e-5)
    assert out.shape == (B, S, D)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


# --------------------------------------------------------------------------- #
# `out` argument writes into the user-provided buffer                         #
# --------------------------------------------------------------------------- #


def test_layernorm_out_argument():
    torch.manual_seed(2)
    n_rows, D = 4, 768
    x = (torch.randn(n_rows, D, device="cuda") * 0.5).to(torch.bfloat16)
    weight = torch.ones(D, device="cuda").to(torch.bfloat16)
    out = torch.empty_like(x)
    returned = phyai_kernel.layernorm(x, weight, None, 1e-5, out=out)
    assert returned.data_ptr() == out.data_ptr()
    ref = _ref_layernorm(x, weight, None, 1e-5)
    torch.testing.assert_close(returned, ref, atol=2e-2, rtol=2e-2)


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #


def test_cpu_input_raises():
    x = torch.randn(2, 64)
    w = torch.randn(64)
    with pytest.raises(RuntimeError, match="must live on CUDA"):
        phyai_kernel.layernorm(x, w)


def test_weight_shape_mismatch_raises():
    x = torch.randn(2, 64, device="cuda").to(torch.bfloat16)
    w = torch.randn(32, device="cuda").to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="weight"):
        phyai_kernel.layernorm(x, w)


def test_bias_shape_mismatch_raises():
    x = torch.randn(2, 64, device="cuda").to(torch.bfloat16)
    w = torch.randn(64, device="cuda").to(torch.bfloat16)
    b = torch.randn(32, device="cuda").to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="bias"):
        phyai_kernel.layernorm(x, w, b)


def test_out_shape_mismatch_raises():
    x = torch.randn(2, 64, device="cuda").to(torch.bfloat16)
    w = torch.randn(64, device="cuda").to(torch.bfloat16)
    out = torch.empty(2, 32, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="`out` must match"):
        phyai_kernel.layernorm(x, w, None, 1e-5, out=out)
