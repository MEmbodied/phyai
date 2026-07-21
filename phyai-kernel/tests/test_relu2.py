"""Correctness checks for the fused ReLU-squared activation."""

from __future__ import annotations

import pytest
import torch

import phyai_kernel


if not torch.cuda.is_available():
    pytest.skip("CUDA is required for Triton tests", allow_module_level=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(17,), (53, 9216), (1966, 9216)])
def test_relu2_matches_torch(dtype: torch.dtype, shape: tuple[int, ...]) -> None:
    torch.manual_seed(sum(shape))
    input = torch.randn(shape, device="cuda", dtype=dtype).contiguous()

    expected = torch.relu(input).square()
    actual = phyai_kernel.relu2(input)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_relu2_preserves_shape_stride_and_dtype() -> None:
    input = torch.randn(7, 11, device="cuda", dtype=torch.bfloat16)
    output = phyai_kernel.relu2(input)

    assert output.shape == input.shape
    assert output.stride() == input.stride()
    assert output.dtype == input.dtype


def test_relu2_rejects_unsupported_inputs() -> None:
    with pytest.raises(RuntimeError, match="CUDA"):
        phyai_kernel.relu2(torch.randn(4))
    with pytest.raises(TypeError, match="floating"):
        phyai_kernel.relu2(torch.ones(4, device="cuda", dtype=torch.int32))
    with pytest.raises(ValueError, match="contiguous"):
        phyai_kernel.relu2(torch.randn(4, 8, device="cuda").t())
