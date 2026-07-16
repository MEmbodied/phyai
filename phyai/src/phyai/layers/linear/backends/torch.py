"""TorchKernel — PyTorch-native BF16 and explicit reference paths.

FP8 is intentionally absent from this production registry entry. A
``torch._scaled_mm`` or dequantized implementation is useful for isolated
correctness tests, but silently selecting it in a model run hides missing
FlashInfer capability and makes CUDA-graph performance unpredictable.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from phyai.layers.linear.backend import KernelProbe
from phyai.layers.linear.registry import register_linear_kernel


_E2M1_VALUES = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=torch.float32,
)


def _unpack_e2m1(weight: torch.Tensor) -> torch.Tensor:
    """Unpack ``(N, K//2)`` E2M1 bytes into a float32 ``(N, K)`` tensor."""
    packed = weight.view(torch.uint8)
    codes = torch.empty(
        packed.shape[0],
        packed.shape[1] * 2,
        dtype=torch.uint8,
        device=packed.device,
    )
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = (packed >> 4) & 0x0F
    lut = _E2M1_VALUES.to(packed.device)
    return lut[codes.long()]


def _linear_nvfp4_scale(
    scale: torch.Tensor,
    weight_shape: tuple[int, ...],
) -> torch.Tensor:
    """Return the logical ``(N, K//16)`` scale view from a padded scale tensor."""
    N, K_half = weight_shape
    k_blocks = (K_half * 2) // 16
    return scale[:N, :k_blocks]


def _dequant_nvfp4_weight(layer: torch.nn.Module) -> torch.Tensor:
    """Dequantise packed NVFP4 weight to float32 for the reference path."""
    fp4 = _unpack_e2m1(layer.weight)
    scale = _linear_nvfp4_scale(layer.weight_scale, tuple(layer.weight.shape)).float()
    scale = scale.repeat_interleave(16, dim=1)
    global_scale = layer.weight_global_scale.float().reshape(())
    return fp4 * scale * global_scale


@register_linear_kernel()
class TorchKernel:
    """BF16 plus the non-FP8 NVFP4 reference path."""

    name = "torch"

    def supports_capture(self) -> bool:
        return True

    def can_handle(self, probe: KernelProbe) -> bool:
        if probe.spec_id == "bf16":
            return True
        if probe.spec_id == "nvfp4_block_16_linear":
            return True
        return False

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        spec = layer.spec
        if spec.spec_id == "bf16":
            return F.linear(x, layer.weight, bias)

        if spec.spec_id == "nvfp4_block_16_linear":
            return self._nvfp4_reference(layer, x, bias)

        raise RuntimeError(f"TorchKernel got unhandled spec_id={spec.spec_id!r}")

    def _nvfp4_reference(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Reference dequant + F.linear path. Correct but not fast."""
        w = _dequant_nvfp4_weight(layer).to(x.dtype)
        return F.linear(x, w, bias)
