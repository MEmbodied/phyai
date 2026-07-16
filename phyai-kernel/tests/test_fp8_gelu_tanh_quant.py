"""Tests for fused GeGLU-tanh + dynamic group-128 FP8 quantization."""

from __future__ import annotations

import pytest
import torch

import phyai_kernel


if not torch.cuda.is_available():
    pytest.skip(
        "CUDA is required for phyai-kernel Triton tests", allow_module_level=True
    )


def _split_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flashinfer_activation = pytest.importorskip("flashinfer.activation")
    activated = flashinfer_activation.gelu_tanh_and_mul(x)
    return phyai_kernel.fp8_quantize_per_group(activated, 128)


@pytest.mark.parametrize(
    ("tokens", "hidden"),
    [
        (7, 256),
        (50, 4096),
        (16, 16384),
    ],
)
def test_matches_flashinfer_split_path_bitwise(tokens: int, hidden: int):
    torch.manual_seed(tokens + hidden)
    x = (
        torch.randn(tokens, 2 * hidden, device="cuda", dtype=torch.bfloat16) * 0.5
    ).contiguous()

    expected_q, expected_scale = _split_reference(x)
    actual_q, actual_scale = phyai_kernel.gelu_tanh_and_mul_fp8_group128(x)

    assert actual_q.shape == (tokens, hidden)
    assert actual_q.dtype is torch.float8_e4m3fn
    assert actual_scale.shape == (tokens, hidden // 128)
    assert actual_scale.dtype is torch.float32
    assert torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(actual_scale, expected_scale)


def test_column_major_scale_layout_matches_row_major_values():
    torch.manual_seed(17)
    x = torch.randn(50, 8192, device="cuda", dtype=torch.bfloat16).contiguous()

    expected_q, expected_scale = phyai_kernel.gelu_tanh_and_mul_fp8_group128(x)
    actual_q, actual_scale = phyai_kernel.gelu_tanh_and_mul_fp8_group128(
        x, column_major_scales=True
    )

    assert actual_scale.shape == (50, 32)
    assert actual_scale.stride() == (1, 52)
    assert not actual_scale.is_contiguous()
    torch.testing.assert_close(actual_scale, expected_scale, rtol=3e-7, atol=0)
    actual_dequant = actual_q.float() * actual_scale.repeat_interleave(128, dim=1)
    expected_dequant = expected_q.float() * expected_scale.repeat_interleave(128, dim=1)
    similarity = torch.nn.functional.cosine_similarity(
        actual_dequant.reshape(1, -1), expected_dequant.reshape(1, -1)
    )
    assert similarity.item() > 0.99999


def test_cuda_graph_replay_matches_eager():
    torch.manual_seed(11)
    x = torch.randn(50, 8192, device="cuda", dtype=torch.bfloat16).contiguous()

    expected_q, expected_scale = phyai_kernel.gelu_tanh_and_mul_fp8_group128(
        x, column_major_scales=True
    )
    for _ in range(3):
        phyai_kernel.gelu_tanh_and_mul_fp8_group128(x, column_major_scales=True)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_q, graph_scale = phyai_kernel.gelu_tanh_and_mul_fp8_group128(
            x, column_major_scales=True
        )
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(graph_q.view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(graph_scale, expected_scale)
    assert graph_scale.stride() == (1, 52)


def test_rejects_non_bf16_and_misaligned_hidden():
    with pytest.raises(TypeError, match="bfloat16"):
        phyai_kernel.gelu_tanh_and_mul_fp8_group128(
            torch.randn(2, 256, device="cuda", dtype=torch.float16)
        )
    with pytest.raises(ValueError, match="divisible by 128"):
        phyai_kernel.gelu_tanh_and_mul_fp8_group128(
            torch.randn(2, 384, device="cuda", dtype=torch.bfloat16)
        )
