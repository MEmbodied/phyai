"""Fused ReLU-squared activation for Cosmos3-Edge dense MLPs."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _relu2_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    values = tl.maximum(values, 0.0)
    tl.store(output_ptr + offsets, values * values, mask=mask)


def relu2(input: torch.Tensor) -> torch.Tensor:
    """Return ``relu(input).square()`` using one Triton launch."""
    if not input.is_cuda:
        raise RuntimeError("phyai_kernel.relu2 requires a CUDA tensor")
    if not input.is_floating_point():
        raise TypeError(
            f"phyai_kernel.relu2 requires floating input, got {input.dtype}"
        )
    if not input.is_contiguous():
        raise ValueError("phyai_kernel.relu2 requires contiguous input")

    output = torch.empty_like(input)
    n_elements = input.numel()
    if n_elements == 0:
        return output

    _relu2_kernel[(triton.cdiv(n_elements, 1024),)](
        input,
        output,
        n_elements,
        BLOCK_SIZE=1024,
        num_warps=4,
    )
    return output


__all__ = ["relu2"]
