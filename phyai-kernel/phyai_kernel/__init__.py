"""phyai-kernel — JIT-compiled CPU/CUDA kernels for phyai via tvm-ffi."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from phyai_kernel import jit_utils
from phyai_kernel.jit_utils import jit
from phyai_kernel.triton import (
    adarmsnorm,
    create_paged_kv_indices,
    fp8_quantize_per_group,
    fp8_quantize_per_tensor,
    fp8_quantize_per_tensor_with_scale,
    fp8_quantize_per_token,
    fp8_quantize_weight_per_block,
    fp8_requantize_with_scale_ratio,
    fused_add_rmsnorm,
    gelu_tanh_and_mul_fp8_group128,
    gemma_fused_add_rmsnorm,
    gemma_rmsnorm,
    layernorm,
    masked_embedding_lookup,
    nvfp4_scale_output,
    rmsnorm,
    rmsnorm_hf,
)

try:
    __version__ = _pkg_version("phyai-kernel")
except PackageNotFoundError:  # raw source tree, not installed
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    "adarmsnorm",
    "create_paged_kv_indices",
    "fused_add_rmsnorm",
    "fp8_quantize_per_group",
    "fp8_quantize_per_tensor",
    "fp8_quantize_per_tensor_with_scale",
    "fp8_quantize_per_token",
    "fp8_quantize_weight_per_block",
    "fp8_requantize_with_scale_ratio",
    "gelu_tanh_and_mul_fp8_group128",
    "gemma_fused_add_rmsnorm",
    "gemma_rmsnorm",
    "jit",
    "jit_utils",
    "layernorm",
    "masked_embedding_lookup",
    "nvfp4_scale_output",
    "rmsnorm",
    "rmsnorm_hf",
]
