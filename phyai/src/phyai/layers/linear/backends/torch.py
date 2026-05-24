"""TorchKernel — PyTorch-native fallback (``F.linear`` / ``_scaled_mm``).

Always present (registered last when iterating the decorator-gathered
list in :func:`phyai.layers.linear.init`) so ``validate()`` can find a
candidate for every probed spec. The fp8 paths require sm≥89; block-fp8
falls back to a dequant + ``F.linear`` reference path that is correct
but slow.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from phyai.layers.linear.backend import Granularity, KernelProbe
from phyai.layers.linear.registry import register_linear_kernel


_FP8_E4M3_AMAX = 448.0


def _expand_block_scale(
    scale: torch.Tensor,
    weight_shape: tuple[int, ...],
    block_shape: tuple[int, int],
) -> torch.Tensor:
    """Expand ``(N//bn, K//bk)`` block scales to full ``(N, K)``."""
    bn, bk = block_shape
    N, K = weight_shape
    expanded = scale.repeat_interleave(bn, dim=0).repeat_interleave(bk, dim=1)
    return expanded[:N, :K]


@register_linear_kernel()
class TorchKernel:
    """F.linear + torch._scaled_mm paths. Always registered as a fallback."""

    name = "torch"

    def supports_capture(self) -> bool:
        return True

    def can_handle(self, probe: KernelProbe) -> bool:
        if probe.spec_id == "bf16":
            return True
        if probe.spec_id.startswith("fp8_"):
            # torch._scaled_mm requires sm89+ and K dim divisible by 16.
            if probe.sm < 89:
                return False
            if probe.spec_id.startswith("fp8_block_"):
                # Dequant reference path — works everywhere the dtype exists.
                return True
            if probe.K % 16 != 0 or probe.N % 16 != 0:
                return False
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

        if spec.spec_id == "fp8_per_tensor":
            return self._fp8_per_tensor(layer, x, bias)
        if spec.spec_id == "fp8_per_channel":
            return self._fp8_per_channel(layer, x, bias)
        if spec.spec_id.startswith("fp8_block_"):
            return self._fp8_block(layer, x, bias)

        raise RuntimeError(f"TorchKernel got unhandled spec_id={spec.spec_id!r}")

    # ------------------------------------------------------------------
    # fp8 paths — prefer spec.quantize_activation for uniformity
    # ------------------------------------------------------------------

    def _fp8_per_tensor(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        spec = layer.spec
        act = spec.quantize_activation(x, layer)
        # _scaled_mm wants (M, K) x (K, N). We stored weight as (N, K)
        # row-major; ``.t()`` gives the column-major (K, N) view that
        # cuBLASLt requires.
        K = act.x.shape[-1]
        x_2d = act.x.reshape(-1, K)
        # Per-tensor weight scale after process_after_loading is
        # per-channel shape (N,); broadcast it as (1, N).
        w_scale = layer.weight_scale
        if w_scale.ndim == 1:
            w_scale = w_scale.reshape(1, -1)
        # Static per-tensor input scale is scalar (1,).
        a_scale = layer.input_scale
        if a_scale.ndim == 1 and a_scale.numel() == 1:
            a_scale = a_scale.reshape(1, 1)
        out = torch._scaled_mm(
            x_2d,
            layer.weight.t(),
            scale_a=a_scale,
            scale_b=w_scale,
            bias=bias,
            out_dtype=x.dtype,
        )
        return out.reshape(*x.shape[:-1], -1)

    def _fp8_per_channel(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        spec = layer.spec
        act = spec.quantize_activation(x, layer)
        K = act.x.shape[-1]
        x_2d = act.x.reshape(-1, K)
        # act.x_scale has the same leading shape as x (per-token scalar per row).
        a_scale = act.x_scale.reshape(-1, 1)
        b_scale = layer.weight_scale.reshape(1, -1)
        out = torch._scaled_mm(
            x_2d,
            layer.weight.t(),
            scale_a=a_scale,
            scale_b=b_scale,
            bias=bias,
            out_dtype=x.dtype,
        )
        return out.reshape(*x.shape[:-1], -1)

    def _fp8_block(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Reference dequant + F.linear path. Correct but not fast —
        flashinfer's ``gemm_fp8_nt_groupwise`` is the intended winner."""
        spec = layer.spec
        assert spec.block_shape is not None
        w = layer.weight.to(x.dtype) * _expand_block_scale(
            layer.weight_scale,
            tuple(layer.weight.shape),
            spec.block_shape,
        ).to(x.dtype)
        return F.linear(x, w, bias)
