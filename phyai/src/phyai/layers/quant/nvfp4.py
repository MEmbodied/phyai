"""Nvfp4Spec — packed E2M1 FP4 weight with FP8 block scales.

NVFP4 stores two E2M1 values per byte and uses one FP8-E4M3 scale for
every 16 values along the K dimension. The logical weight is still the
linear ``(N, K)`` matrix, but the stored parameter is ``(N, K // 2)``.

Two scale layouts are useful:

* ``"linear"`` — test/reference layout, ``(N, K // 16)``.
* ``"128x4"`` — FlashInfer / Blackwell GEMM layout, padded to
  ``(ceil(N, 128), ceil(K // 16, 4))``.

The activation path is handled by the backend because FlashInfer's FP4
GEMM needs both the per-block activation scales and a separate global
scale scalar, which does not fit the current :class:`ActivationView`
shape used by FP8.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch
import torch.nn as nn

from phyai.layers.quant.base import AllocationRequest


_NVFP4_BLOCK_SIZE = 16
_FP8_E4M3_AMAX = 448.0
_NVFP4_MAX = 6.0
_E2M1_THRESHOLDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def flashinfer_nvfp4_e4m3_max() -> float:
    """Return FlashInfer's active NVFP4 E4M3 limit, including 4-over-6."""
    try:
        from flashinfer.quantization.nvfp4_quantization_utils import (
            current_nvfp4_4over6_config,
            nvfp4_e4m3_max,
        )
    except (AttributeError, ImportError):
        return _FP8_E4M3_AMAX
    return nvfp4_e4m3_max(current_nvfp4_4over6_config())


def _scale_shape(
    out_per_rank: int,
    in_per_rank: int,
    scale_layout: Literal["linear", "128x4"],
) -> tuple[int, int]:
    k_blocks = in_per_rank // _NVFP4_BLOCK_SIZE
    if scale_layout == "linear":
        return out_per_rank, k_blocks
    if scale_layout == "128x4":
        return _round_up(out_per_rank, 128), _round_up(k_blocks, 4)
    raise ValueError(
        f"Nvfp4Spec scale_layout must be 'linear' or '128x4', got {scale_layout!r}"
    )


def _per_tensor_amax_to_scale(amax: torch.Tensor) -> torch.Tensor:
    """Convert tensor amax to the TorchAO-style NVFP4 per-tensor scale."""
    return amax.float() / (_FP8_E4M3_AMAX * _NVFP4_MAX)


def _quantize_e2m1(data: torch.Tensor) -> torch.Tensor:
    thresholds = torch.tensor(_E2M1_THRESHOLDS, dtype=torch.float32, device=data.device)
    mag = torch.bucketize(data.abs(), thresholds).to(torch.uint8)
    sign = (data < 0).to(torch.uint8) << 3
    codes = (mag | sign).contiguous()
    return codes[:, 0::2] | (codes[:, 1::2] << 4)


def _quantize_nvfp4_linear(
    weight: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    per_tensor_scale = _per_tensor_amax_to_scale(weight.abs().amax()).clamp_min(1e-12)
    src = weight.float().reshape(weight.shape[0], -1, block_size)
    block_scale = src.abs().amax(dim=-1) / _NVFP4_MAX
    block_scale = (block_scale / per_tensor_scale).clamp(
        torch.finfo(torch.float8_e4m3fn).tiny,
        _FP8_E4M3_AMAX,
    )
    block_scale_fp8 = block_scale.to(torch.float8_e4m3fn)
    reciprocal = (1.0 / per_tensor_scale) / block_scale_fp8.float()
    scaled = (src * reciprocal.unsqueeze(-1)).clamp(-_NVFP4_MAX, _NVFP4_MAX)
    packed = _quantize_e2m1(scaled.reshape(weight.shape))
    return packed, block_scale_fp8, per_tensor_scale.reshape(1)


@dataclass
class Nvfp4Spec:
    """NVFP4 Linear weight spec.

    ``scale_layout="128x4"`` is the production layout consumed by
    FlashInfer's ``mm_fp4`` backend. ``"linear"`` is kept for the Torch
    reference path and small CPU tests.
    """

    scale_layout: Literal["linear", "128x4"] = "128x4"
    weight_dtype: torch.dtype = torch.uint8
    block_size: int = _NVFP4_BLOCK_SIZE
    raw_config: dict | None = None
    input_raw_config: dict | None = None
    online: bool = False

    def __post_init__(self) -> None:
        if self.block_size != _NVFP4_BLOCK_SIZE:
            raise ValueError("Nvfp4Spec only supports block_size=16")
        if self.scale_layout not in ("linear", "128x4"):
            raise ValueError(
                f"Nvfp4Spec scale_layout must be 'linear' or '128x4', "
                f"got {self.scale_layout!r}"
            )
        if self.raw_config is not None:
            quant_method = str(self.raw_config.get("quant_method", "")).lower()
            if quant_method == "modelopt":
                quant_algo = str(self.raw_config.get("quant_algo", "")).upper()
                if quant_algo not in ("NVFP4", "FP4"):
                    raise ValueError(
                        "Nvfp4Spec ModelOpt source requires quant_algo='NVFP4'"
                    )
                if int(self.raw_config.get("group_size", 16) or 16) != 16:
                    raise ValueError("Nvfp4Spec ModelOpt source requires group_size=16")
            elif quant_method != "compressed-tensors":
                raise ValueError(
                    "Nvfp4Spec raw_config only supports compressed-tensors or ModelOpt"
                )
            elif self.raw_config.get("format") != "nvfp4-pack-quantized":
                raise ValueError(
                    "Nvfp4Spec compressed-tensors source must use "
                    "format='nvfp4-pack-quantized'"
                )
            if quant_method == "compressed-tensors":
                expected = {
                    "type": "float",
                    "num_bits": 4,
                    "strategy": "tensor_group",
                    "group_size": 16,
                    "symmetric": True,
                }
                for name, value in expected.items():
                    if self.raw_config.get(name) != value:
                        raise ValueError(
                            "Nvfp4Spec compressed-tensors source requires "
                            f"{name}={value!r}, got {self.raw_config.get(name)!r}"
                        )
            if not self.online and self.scale_layout != "128x4":
                raise ValueError(
                    "serialized NVFP4 must be converted to the FlashInfer "
                    "128x4 scale layout"
                )

    @property
    def spec_id(self) -> str:
        return f"nvfp4_block_{self.block_size}_{self.scale_layout}"

    def allocate(self, layer: nn.Module, request: AllocationRequest) -> None:
        if len(request.weight_shape) != 2:
            raise ValueError(
                f"Nvfp4Spec.allocate expects a 2-D weight_shape (N, K), "
                f"got {request.weight_shape!r}"
            )
        out_per_rank, in_per_rank = request.weight_shape
        if in_per_rank % self.block_size != 0:
            raise ValueError(
                f"Nvfp4Spec: in_per_rank={in_per_rank} not divisible by "
                f"block_size={self.block_size}"
            )
        if self.scale_layout == "128x4" and (
            out_per_rank % 32 != 0 or in_per_rank % 32 != 0
        ):
            raise ValueError(
                "Nvfp4Spec(scale_layout='128x4') requires out_features and "
                "in_features divisible by 32 for FlashInfer FP4 GEMM, got "
                f"({out_per_rank}, {in_per_rank})"
            )
        device = request.device

        layer.weight = nn.Parameter(
            torch.empty(
                out_per_rank,
                in_per_rank // 2,
                dtype=self.weight_dtype,
                device=device,
            ),
            requires_grad=False,
        )
        source_method = str((self.raw_config or {}).get("quant_method", "")).lower()
        serialized_source = self.raw_config is not None and not self.online
        if serialized_source:
            layer.weight._quant_source_name = (
                "weight_packed" if source_method == "compressed-tensors" else "weight"
            )
            layer.weight._quant_attrs = {"output_dim": 0, "input_dim": 1}
            scale_shape = (out_per_rank, in_per_rank // self.block_size)
        else:
            layer.weight._quant_source_name = "weight"
            scale_shape = _scale_shape(
                out_per_rank,
                in_per_rank,
                self.scale_layout,
            )
        layer.weight._accepted_source_dtypes = (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.uint8,
        )
        layer.weight_scale = nn.Parameter(
            torch.full(
                scale_shape,
                float("nan"),
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
            requires_grad=False,
        )
        layer.weight_scale._quant_source_name = "weight_scale"
        layer.weight_scale._quant_attrs = {
            "output_dim": 0,
            "input_dim": 1,
            "scale_type": "group",
        }
        global_scale_count = len(request.logical_widths) if serialized_source else 1
        layer.weight_global_scale = nn.Parameter(
            torch.full(
                (global_scale_count,),
                float("nan"),
                dtype=torch.float32,
                device=device,
            ),
            requires_grad=False,
        )
        layer.weight_global_scale._quant_source_name = (
            "weight_scale_2" if source_method == "modelopt" else "weight_global_scale"
        )
        layer.weight_global_scale._quant_attrs = {
            "scale_type": "tensor",
            "stacked_scalar": global_scale_count > 1,
        }
        if serialized_source:
            input_scale_source = (
                "input_scale" if source_method == "modelopt" else "input_global_scale"
            )
            layer.input_global_scale = nn.Parameter(
                torch.ones(
                    (len(request.logical_widths),),
                    dtype=torch.float32,
                    device=device,
                ),
                requires_grad=False,
            )
            layer.input_global_scale._quant_source_name = input_scale_source
            layer.input_global_scale._quant_attrs = {
                "scale_type": "tensor",
                "stacked_scalar": len(request.logical_widths) > 1,
            }
            layer.input_global_scale._quant_optional = True
        layer._nvfp4_pending_weight = None
        layer.logical_widths = list(request.logical_widths)
        layer._nvfp4_in_per_rank = in_per_rank
        layer._nvfp4_out_per_rank = out_per_rank
        layer._nvfp4_source_method = source_method if serialized_source else ""
        layer._nvfp4_serialized_source = serialized_source

    def process_after_loading(self, layer: nn.Module) -> None:
        pending = getattr(layer, "_nvfp4_pending_weight", None)
        if pending is not None:
            self.quantize_loaded_weight(layer, pending)
            layer._nvfp4_pending_weight = None
        elif layer._nvfp4_serialized_source:
            input_scale = layer.input_global_scale.detach().float().reshape(-1)
            if not torch.isfinite(input_scale).all() or (input_scale <= 0).any():
                raise ValueError(
                    "serialized NVFP4 input scale must be finite and strictly positive"
                )
            delattr(layer, "input_global_scale")
            self.convert_serialized_weight(layer)
        weight_scale = layer.weight_scale.detach().float()
        global_scale = layer.weight_global_scale.detach().float()
        if not torch.isfinite(weight_scale).all() or (weight_scale < 0).any():
            raise ValueError("NVFP4 weight_scale must be finite and nonnegative")
        if not torch.isfinite(global_scale).all() or (global_scale <= 0).any():
            raise ValueError(
                "NVFP4 weight_global_scale must be finite and strictly positive"
            )

    def interleave_scale(self, scale: torch.Tensor) -> torch.Tensor:
        if not scale.is_cuda:
            raise RuntimeError(
                "converting compressed-tensors NVFP4 scale factors to FlashInfer "
                "128x4 layout requires a CUDA tensor on sm_100+"
            )
        try:
            from flashinfer.quantization import block_scale_interleave
        except Exception as exc:
            raise RuntimeError(
                "converting compressed-tensors NVFP4 requires FlashInfer with "
                "block_scale_interleave support"
            ) from exc
        return block_scale_interleave(scale.contiguous().view(torch.uint8))

    def convert_serialized_weight(self, layer: nn.Module) -> None:
        scale = layer.weight_scale.detach()
        global_scale = layer.weight_global_scale.detach().float().reshape(-1)
        if not torch.isfinite(global_scale).all() or (global_scale <= 0).any():
            raise ValueError(
                "serialized NVFP4 weight global scale must be finite and "
                "strictly positive"
            )
        if not torch.isfinite(scale.float()).all() or (scale.float() < 0).any():
            raise ValueError(
                "serialized NVFP4 weight_scale must be finite and nonnegative"
            )

        descales = (
            global_scale
            if layer._nvfp4_source_method == "modelopt"
            else global_scale.reciprocal()
        )
        common_descale = descales[0]
        if not torch.equal(descales, common_descale.expand_as(descales)):
            if len(layer.logical_widths) != descales.numel():
                raise RuntimeError(
                    "serialized NVFP4 fused global-scale count does not "
                    "match the number of logical weight legs"
                )
            common_descale = descales.max()
            rows = []
            offset = 0
            for width, descale in zip(layer.logical_widths, descales, strict=True):
                leg = scale.narrow(0, offset, width).float() * (
                    descale / common_descale
                )
                rows.append(leg.to(torch.float8_e4m3fn))
                offset += width
            scale = torch.cat(rows, dim=0)

        interleaved = self.interleave_scale(scale)
        expected_shape = _scale_shape(
            layer._nvfp4_out_per_rank,
            layer._nvfp4_in_per_rank,
            "128x4",
        )
        if interleaved.numel() != expected_shape[0] * expected_shape[1]:
            raise RuntimeError(
                "FlashInfer block_scale_interleave returned an unexpected number "
                f"of scale factors: got {interleaved.numel()}, expected "
                f"{expected_shape[0] * expected_shape[1]}"
            )
        if interleaved.dtype is torch.uint8:
            interleaved = interleaved.view(torch.float8_e4m3fn)
        elif interleaved.dtype is not torch.float8_e4m3fn:
            raise TypeError(
                "FlashInfer block_scale_interleave must return uint8 or "
                f"float8_e4m3fn, got {interleaved.dtype}"
            )
        layer.weight_scale = nn.Parameter(
            interleaved.reshape(expected_shape),
            requires_grad=False,
        )
        layer.weight_global_scale = nn.Parameter(
            common_descale.reshape(1).to(
                device=layer.weight.device,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

    def load_weight(
        self,
        layer: nn.Module,
        loaded: torch.Tensor,
        shard_id: "int | str | None",
        default_loader: Callable[[nn.Parameter, torch.Tensor, object], None],
    ) -> None:
        logical_shape = (layer.weight.shape[0], layer.weight.shape[1] * 2)
        if loaded.dtype in (torch.float16, torch.bfloat16, torch.float32):
            if self.raw_config is not None and not self.online:
                raise TypeError(
                    f"serialized NVFP4 expects packed uint8 weight, got {loaded.dtype}"
                )
            pending = getattr(layer, "_nvfp4_pending_weight", None)
            if pending is None:
                pending = torch.empty(
                    logical_shape,
                    dtype=loaded.dtype,
                    device=loaded.device,
                )
                layer._nvfp4_pending_weight = pending
            elif pending.dtype != loaded.dtype:
                raise TypeError(
                    "all fused high-precision NVFP4 source weights must share a dtype"
                )
            default_loader(pending, loaded, shard_id)
            return

        if loaded.dtype is not torch.uint8:
            raise TypeError(
                f"serialized NVFP4 weight must use packed uint8, got {loaded.dtype}"
            )
        default_loader(layer.weight, loaded, shard_id)

    def quantize_loaded_weight(self, layer: nn.Module, weight: torch.Tensor) -> None:
        src = weight.detach().to(device=layer.weight.device).contiguous()
        if src.dtype not in (torch.bfloat16, torch.float32, torch.float16):
            src = src.to(torch.bfloat16)
        if not torch.isfinite(src).all():
            raise ValueError("online NVFP4 source weight must be finite")

        if self.scale_layout == "linear":
            packed, scale, global_scale = _quantize_nvfp4_linear(src, self.block_size)
        elif self.scale_layout == "128x4":
            if not src.is_cuda:
                raise RuntimeError(
                    "Nvfp4Spec(scale_layout='128x4') needs CUDA FlashInfer to "
                    "quantize high-precision weights on load. Use "
                    "scale_layout='linear' for CPU reference quantization or load "
                    "a pre-quantized 128x4 checkpoint."
                )
            from flashinfer.quantization import SfLayout, nvfp4_quantize

            flashinfer_global_scale = (flashinfer_nvfp4_e4m3_max() * _NVFP4_MAX) / (
                src.float().abs().amax().clamp_min(1e-12)
            )
            flashinfer_global_scale = flashinfer_global_scale.reshape(1)
            if src.dtype == torch.float32:
                src = src.to(torch.bfloat16)
            packed, scale = nvfp4_quantize(
                src,
                flashinfer_global_scale,
                sfLayout=SfLayout.layout_128x4,
                do_shuffle=False,
                enable_pdl=False,
            )
            global_scale = 1.0 / flashinfer_global_scale
        else:
            raise RuntimeError(
                f"unhandled Nvfp4Spec scale_layout={self.scale_layout!r}"
            )

        layer.weight.data.copy_(packed.to(device=layer.weight.device).view(torch.uint8))
        layer.weight_scale.data.copy_(
            scale.to(device=layer.weight_scale.device).view(layer.weight_scale.dtype)
        )
        layer.weight_global_scale.data.copy_(
            global_scale.to(device=layer.weight_global_scale.device)
        )


__all__ = ["Nvfp4Spec", "flashinfer_nvfp4_e4m3_max"]
