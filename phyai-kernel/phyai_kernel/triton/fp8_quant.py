"""Dynamic E4M3 activation quantization implemented entirely with Triton.

The grouped kernel follows the register-resident, one-read design used by
SGLang and vLLM activation quantizers. Multiple adjacent groups share one
Triton program, and a small autotune set chooses the launch geometry on each
GPU/shape. This keeps the same implementation portable across Hopper and the
SM100/103/110/120/121 FlashInfer block-GEMM targets.

All returned scales are dequantization factors: ``x ~= x_q.float() * scale``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


FP8_E4M3_MAX = 448.0
FP8_MIN_SCALE_EPS = 1e-12
_MAX_ROW_WIDTH = 65536


@triton.autotune(
    configs=[
        triton.Config({"GROUPS_PER_PROGRAM": 4}, num_warps=1),
        triton.Config({"GROUPS_PER_PROGRAM": 8}, num_warps=1),
        triton.Config({"GROUPS_PER_PROGRAM": 16}, num_warps=2),
        triton.Config({"GROUPS_PER_PROGRAM": 16}, num_warps=4),
    ],
    key=["n_groups"],
)
@triton.jit
def _fp8_group_quant_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    n_groups,
    GROUPS_PER_PROGRAM: tl.constexpr,
):
    group_base = tl.program_id(0).to(tl.int64) * GROUPS_PER_PROGRAM
    group_ids = group_base + tl.arange(0, GROUPS_PER_PROGRAM)
    cols = tl.arange(0, 128)
    offsets = group_ids[:, None] * 128 + cols[None, :]
    mask = group_ids[:, None] < n_groups

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-12)
    scale = amax * (1.0 / 448.0)
    q = tl.clamp(x / scale[:, None], -448.0, 448.0)

    tl.store(q_ptr + offsets, q, mask=mask)
    tl.store(scale_ptr + group_ids, scale, mask=group_ids < n_groups)


@triton.jit
def _fp8_token_quant_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    K,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < K
    offsets = row * K + cols

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x)), 1e-12)
    scale = amax * (1.0 / 448.0)
    q = tl.clamp(x / scale, -448.0, 448.0)

    tl.store(q_ptr + offsets, q, mask=mask)
    tl.store(scale_ptr + row, scale)


@triton.jit
def _fp8_tensor_amax_kernel(
    x_ptr,
    scale_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets, mask=offsets < n_elements, other=0.0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x)), 1e-12) * (1.0 / 448.0)
    tl.atomic_max(scale_ptr, scale)


@triton.jit
def _fp8_tensor_quant_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr)
    q = tl.clamp(x / scale, -448.0, 448.0)
    tl.store(q_ptr + offsets, q, mask=mask)


@triton.jit
def _fp8_requantize_kernel(
    x_ptr,
    q_ptr,
    ratio_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    ratio = tl.load(ratio_ptr).to(tl.float32)
    q = tl.clamp(x * ratio, -448.0, 448.0)
    tl.store(q_ptr + offsets, q, mask=mask)


@triton.jit
def _fp8_weight_block_quant_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    K,
    N_BLOCKS_K,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    block_n = tl.program_id(0).to(tl.int64)
    block_k = tl.program_id(1).to(tl.int64)
    rows = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    cols = block_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offsets = rows[:, None] * K + cols[None, :]

    x = tl.load(x_ptr + offsets).to(tl.float32)
    amax = tl.maximum(tl.max(tl.max(tl.abs(x), axis=1), axis=0), 1e-12)
    scale = amax * (1.0 / 448.0)
    q = tl.clamp(x / scale, -448.0, 448.0)

    tl.store(q_ptr + offsets, q)
    tl.store(scale_ptr + block_n * N_BLOCKS_K + block_k, scale)


def _check_input(x: torch.Tensor) -> None:
    if not x.is_cuda:
        raise RuntimeError("phyai_kernel FP8 activation quantization requires CUDA")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            "phyai_kernel FP8 activation quantization requires fp16, bf16, "
            f"or fp32 input, got {x.dtype}"
        )
    if not x.is_contiguous():
        raise ValueError(
            "phyai_kernel FP8 activation quantization requires contiguous input"
        )
    if x.numel() == 0:
        raise ValueError(
            "phyai_kernel FP8 activation quantization requires non-empty input"
        )


def fp8_quantize_per_tensor(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamically quantize all of ``x`` with one FP32 dequantization scale."""
    _check_input(x)
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.zeros(1, dtype=torch.float32, device=x.device)
    block_size = 8192
    grid = (triton.cdiv(x.numel(), block_size),)
    _fp8_tensor_amax_kernel[grid](
        x,
        scale,
        x.numel(),
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    _fp8_tensor_quant_kernel[grid](
        x,
        q,
        scale,
        x.numel(),
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return q, scale


def fp8_quantize_per_tensor_with_scale(
    x: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x`` with a preloaded FP32 dequantization scale.

    This is the static-activation path used by serialized ModelOpt FP8.  The
    arithmetic stays in one Triton kernel and does not synchronize the scale
    through ``Tensor.item()``.
    """
    _check_input(x)
    if not scale.is_cuda or scale.device != x.device:
        raise ValueError("static FP8 scale must be on the same CUDA device as input")
    if scale.dtype is not torch.float32 or scale.numel() != 1:
        raise ValueError("static FP8 scale must be one FP32 value")
    if not scale.is_contiguous():
        raise ValueError("static FP8 scale must be contiguous")

    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    block_size = 8192
    grid = (triton.cdiv(x.numel(), block_size),)
    _fp8_tensor_quant_kernel[grid](
        x,
        q,
        scale,
        x.numel(),
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return q, scale


def fp8_requantize_with_scale_ratio(
    x: torch.Tensor,
    ratio: torch.Tensor,
) -> torch.Tensor:
    """Requantize FP8 values after changing their dequantization scale."""
    if not x.is_cuda or x.dtype is not torch.float8_e4m3fn:
        raise TypeError("FP8 requantization requires a CUDA float8_e4m3fn tensor")
    if not x.is_contiguous():
        raise ValueError("FP8 requantization requires contiguous input")
    if (
        not ratio.is_cuda
        or ratio.device != x.device
        or ratio.dtype is not torch.float32
        or ratio.numel() != 1
    ):
        raise ValueError(
            "FP8 requantization ratio must be one FP32 value on input device"
        )
    q = torch.empty_like(x)
    block_size = 8192
    _fp8_requantize_kernel[(triton.cdiv(x.numel(), block_size),)](
        x,
        q,
        ratio,
        x.numel(),
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return q


def fp8_quantize_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamically quantize each row along the last dimension of ``x``."""
    _check_input(x)
    if x.ndim == 0:
        raise ValueError("per-token FP8 quantization requires at least one dimension")
    K = x.shape[-1]
    if K > _MAX_ROW_WIDTH:
        raise ValueError(
            f"per-token FP8 quantization supports K <= {_MAX_ROW_WIDTH}, got {K}"
        )
    M = x.numel() // K
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.empty((*x.shape[:-1], 1), dtype=torch.float32, device=x.device)
    block_size = triton.next_power_of_2(K)
    num_warps = min(8, max(1, block_size // 256))
    _fp8_token_quant_kernel[(M,)](
        x,
        q,
        scale,
        K,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return q, scale


def fp8_quantize_per_group(
    x: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamically quantize contiguous last-dimension groups of ``x``.

    The production block-FP8 path uses ``group_size=128`` and returns scales
    with logical shape ``(M, K // 128)`` where all leading input dimensions are
    flattened into ``M``.
    """
    _check_input(x)
    if x.ndim == 0:
        raise ValueError("grouped FP8 quantization requires at least one dimension")
    if group_size != 128:
        raise ValueError(
            f"phyai_kernel grouped FP8 quantization supports group_size=128, got {group_size}"
        )
    K = x.shape[-1]
    if K % group_size != 0:
        raise ValueError(
            f"FP8 grouped activation K={K} is not divisible by {group_size}"
        )

    M = x.numel() // K
    n_groups = x.numel() // group_size
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.empty(
        (M, K // group_size),
        dtype=torch.float32,
        device=x.device,
    )

    def grid(meta):
        return (triton.cdiv(n_groups, meta["GROUPS_PER_PROGRAM"]),)

    _fp8_group_quant_kernel[grid](x, q, scale, n_groups)
    return q, scale


def fp8_quantize_weight_per_block(
    weight: torch.Tensor,
    block_shape: tuple[int, int] = (128, 128),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a contiguous ``(N, K)`` weight with one scale per 2-D block."""
    _check_input(weight)
    if weight.ndim != 2:
        raise ValueError(
            f"block FP8 weight quantization requires a 2-D tensor, got {weight.ndim}D"
        )
    if block_shape != (128, 128):
        raise ValueError(
            "phyai_kernel block FP8 weight quantization supports block_shape=(128, 128)"
        )
    N, K = weight.shape
    block_n, block_k = block_shape
    if N % block_n or K % block_k:
        raise ValueError(
            f"block FP8 weight shape {(N, K)} must be divisible by {block_shape}"
        )

    q = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    scale = torch.empty(
        (N // block_n, K // block_k),
        dtype=torch.float32,
        device=weight.device,
    )
    _fp8_weight_block_quant_kernel[(N // block_n, K // block_k)](
        weight,
        q,
        scale,
        K,
        K // block_k,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=8,
    )
    return q, scale


__all__ = [
    "FP8_E4M3_MAX",
    "FP8_MIN_SCALE_EPS",
    "fp8_requantize_with_scale_ratio",
    "fp8_quantize_per_group",
    "fp8_quantize_per_tensor",
    "fp8_quantize_per_tensor_with_scale",
    "fp8_quantize_per_token",
    "fp8_quantize_weight_per_block",
]
