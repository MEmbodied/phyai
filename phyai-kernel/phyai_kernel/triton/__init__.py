"""phyai-kernel Triton kernels (pure-Python, no tvm-ffi build)."""

from phyai_kernel.triton.ada_rms_norm import adarmsnorm
from phyai_kernel.triton.fp8_quant import (
    fp8_requantize_with_scale_ratio,
    fp8_quantize_per_group,
    fp8_quantize_per_tensor,
    fp8_quantize_per_tensor_with_scale,
    fp8_quantize_per_token,
    fp8_quantize_weight_per_block,
    gelu_tanh_and_mul_fp8_group128,
)
from phyai_kernel.triton.layer_norm import layernorm
from phyai_kernel.triton.masked_embedding import masked_embedding_lookup
from phyai_kernel.triton.nvfp4 import nvfp4_scale_output
from phyai_kernel.triton.paged_kv_indices import create_paged_kv_indices
from phyai_kernel.triton.rms_norm import (
    fused_add_rmsnorm,
    gemma_fused_add_rmsnorm,
    gemma_rmsnorm,
    rmsnorm,
    rmsnorm_hf,
)

__all__ = [
    "adarmsnorm",
    "create_paged_kv_indices",
    "fused_add_rmsnorm",
    "fp8_requantize_with_scale_ratio",
    "fp8_quantize_per_group",
    "fp8_quantize_per_tensor",
    "fp8_quantize_per_tensor_with_scale",
    "fp8_quantize_per_token",
    "fp8_quantize_weight_per_block",
    "gelu_tanh_and_mul_fp8_group128",
    "gemma_fused_add_rmsnorm",
    "gemma_rmsnorm",
    "layernorm",
    "masked_embedding_lookup",
    "nvfp4_scale_output",
    "rmsnorm",
    "rmsnorm_hf",
]
