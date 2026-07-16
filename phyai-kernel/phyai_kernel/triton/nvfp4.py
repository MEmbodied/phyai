"""Triton epilogue for online per-token NVFP4 dense GEMM.

FlashInfer's per-token NVFP4 quantizer returns one FP32 decode scale for each
input row.  Dense ``mm_fp4`` currently accepts only a scalar ``alpha``, so the
activation decode scale has to be applied to the GEMM result.  This kernel
does that in place and folds the optional linear bias into the same launch.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _nvfp4_scale_output_kernel(
    output_ptr,
    scale_ptr,
    bias_ptr,
    n_cols,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1).to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    offsets = row * n_cols + cols

    output = tl.load(output_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    row_scale = tl.load(scale_ptr + row).to(tl.float32)
    output *= row_scale
    if HAS_BIAS:
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        output += bias
    tl.store(output_ptr + offsets, output, mask=mask)


def _check_scale_output_inputs(
    output: torch.Tensor,
    per_token_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> None:
    if output.ndim != 2:
        raise ValueError(f"NVFP4 output must be 2-D, got shape={tuple(output.shape)}")
    if not output.is_cuda or not per_token_scale.is_cuda:
        raise RuntimeError("NVFP4 output scaling requires CUDA tensors")
    if output.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(
            f"NVFP4 output scaling requires fp16 or bf16 output, got {output.dtype}"
        )
    if per_token_scale.dtype != torch.float32:
        raise TypeError(
            f"NVFP4 per-token scale must be float32, got {per_token_scale.dtype}"
        )
    if output.device != per_token_scale.device:
        raise ValueError("NVFP4 output and per-token scale must use the same device")
    if not output.is_contiguous() or not per_token_scale.is_contiguous():
        raise ValueError("NVFP4 output and per-token scale must be contiguous")
    if per_token_scale.numel() != output.shape[0]:
        raise ValueError(
            "NVFP4 per-token scale must contain one value per output row, "
            f"got {per_token_scale.numel()} values for {output.shape[0]} rows"
        )
    if bias is None:
        return
    if not bias.is_cuda or bias.device != output.device:
        raise ValueError("NVFP4 bias must be on the output CUDA device")
    if bias.ndim != 1 or bias.numel() != output.shape[1]:
        raise ValueError(
            f"NVFP4 bias must have shape ({output.shape[1]},), got {tuple(bias.shape)}"
        )
    if bias.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"NVFP4 bias must be floating point, got {bias.dtype}")
    if not bias.is_contiguous():
        raise ValueError("NVFP4 bias must be contiguous")


def nvfp4_scale_output(
    output: torch.Tensor,
    per_token_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a per-row decode scale and optional bias to ``output`` in place."""
    _check_scale_output_inputs(output, per_token_scale, bias)
    n_rows, n_cols = output.shape
    if n_rows == 0 or n_cols == 0:
        return output

    block_n = min(256, triton.next_power_of_2(n_cols))
    num_warps = 4 if block_n >= 128 else 1
    bias_ptr = bias if bias is not None else output
    _nvfp4_scale_output_kernel[(n_rows, triton.cdiv(n_cols, block_n))](
        output,
        per_token_scale,
        bias_ptr,
        n_cols,
        HAS_BIAS=bias is not None,
        BLOCK_N=block_n,
        num_warps=num_warps,
    )
    return output


__all__ = ["nvfp4_scale_output"]
