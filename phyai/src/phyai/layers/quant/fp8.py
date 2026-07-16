"""Fp8Spec — ``torch.float8_e4m3fn`` weight with configurable scale granularity.

Linear-only today: ``allocate`` reads ``weight_shape`` as ``(out_per_rank,
in_per_rank)`` and the per-tensor / per-channel / block scale shapes are
all derived under that 2-D assumption. Implements both
:class:`WeightSpec` (storage) and
:class:`phyai.layers.quant.linear.LinearActivationQuant` (activation
quant hook).

``granularity`` selects between:

* ``PER_TENSOR``  — one shared weight scalar plus one dynamically computed
  activation scalar.
* ``PER_CHANNEL`` — legacy/reference per-output-row weight scale plus a
  dynamically computed per-token activation scale.
* ``BLOCK``       — ``(out_per_rank // block_n, in_per_rank // block_k)``
  weight scale plus ``(M, K // block_k)`` dynamic activation scales.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from phyai_kernel import (
    fp8_requantize_with_scale_ratio,
    fp8_quantize_per_group,
    fp8_quantize_per_tensor,
    fp8_quantize_per_tensor_with_scale,
    fp8_quantize_per_token,
    fp8_quantize_weight_per_block,
)

from phyai.layers.quant.base import AllocationRequest
from phyai.layers.quant.granularity import Granularity
from phyai.layers.quant.linear import ActivationView
from phyai.layers.quant.scheme import QDType, TensorQuant


def _convert_to_channelwise(
    scale_per_logical: torch.Tensor,
    logical_widths: list[int],
) -> torch.Tensor:
    """Expand legacy logical-matrix scales without changing FP8 load semantics."""
    return torch.cat(
        [scale_per_logical[i].expand(width) for i, width in enumerate(logical_widths)]
    )


def _raw_quant_method(raw_config: dict | None) -> str:
    return str((raw_config or {}).get("quant_method", "")).lower()


def _is_modelopt_block(raw_config: dict | None) -> bool:
    return (
        _raw_quant_method(raw_config) == "modelopt"
        and str((raw_config or {}).get("quant_algo", "")).upper() == "FP8_PB_WO"
    )


def _weight_scale_source_name(
    raw_config: dict | None,
    granularity: Granularity,
) -> str:
    if _raw_quant_method(raw_config) == "fp8" and granularity is Granularity.BLOCK:
        return "weight_scale_inv"
    return "weight_scale"


def _replace_parameter(
    layer: nn.Module,
    name: str,
    value: torch.Tensor,
    *,
    source_name: str | None = None,
    quant_attrs: dict | None = None,
) -> None:
    param = nn.Parameter(value.detach().contiguous(), requires_grad=False)
    if source_name is not None:
        param._quant_source_name = source_name
    if quant_attrs is not None:
        param._quant_attrs = quant_attrs
    setattr(layer, name, param)


@dataclass
class Fp8Spec:
    granularity: Granularity = Granularity.PER_CHANNEL
    block_shape: tuple[int, int] | None = None
    activation: TensorQuant | None = None
    input_quant: TensorQuant | None = None
    raw_config: dict | None = None
    online: bool = False
    weight_dtype: torch.dtype = torch.float8_e4m3fn
    needs_act_quant: bool = True

    def __post_init__(self) -> None:
        if self.granularity == Granularity.BLOCK and self.block_shape is None:
            raise ValueError("Fp8Spec(granularity=BLOCK) requires block_shape")
        if self.activation is not None and self.input_quant is not None:
            if self.activation != self.input_quant:
                raise ValueError("Fp8Spec activation and input_quant disagree")
        if self.activation is None:
            self.activation = self.input_quant
        if self.activation is None:
            group_size = self.block_shape[1] if self.block_shape is not None else 0
            activation_granularity = (
                Granularity.PER_TENSOR
                if self.granularity is Granularity.PER_TENSOR
                else Granularity.PER_CHANNEL
            )
            self.activation = TensorQuant(
                QDType.FP8_E4M3,
                activation_granularity,
                dynamic=True,
                group_size=group_size,
            )
        self.input_quant = self.activation
        if self.activation.dtype is not QDType.FP8_E4M3:
            raise ValueError("Fp8Spec requires FP8 E4M3 input activations")
        if self.granularity is Granularity.PER_TENSOR:
            if (
                self.activation.granularity is not Granularity.PER_TENSOR
                or self.activation.group_size
            ):
                raise ValueError(
                    "Fp8Spec tensorwise weights require tensorwise activations"
                )
        elif self.granularity is Granularity.BLOCK:
            assert self.block_shape is not None
            if (
                self.activation.granularity is not Granularity.PER_CHANNEL
                or self.activation.group_size != self.block_shape[1]
            ):
                raise ValueError(
                    "Fp8Spec block weights require K-grouped activations matching block_k"
                )
            if not self.activation.dynamic:
                raise ValueError("Fp8Spec block weights require dynamic activations")
        elif not self.activation.dynamic:
            raise ValueError(
                "Fp8Spec static activations are only supported for tensorwise FP8"
            )

    @property
    def spec_id(self) -> str:
        if self.granularity == Granularity.BLOCK:
            assert self.block_shape is not None
            bn, bk = self.block_shape
            return f"fp8_block_{bn}_{bk}"
        return f"fp8_{self.granularity.value}"

    def allocate(self, layer: nn.Module, request: AllocationRequest) -> None:
        if len(request.weight_shape) != 2:
            raise ValueError(
                f"Fp8Spec.allocate expects a 2-D weight_shape (N, K), "
                f"got {request.weight_shape!r}"
            )
        out_per_rank, in_per_rank = request.weight_shape
        device = request.device

        layer.weight = nn.Parameter(
            torch.empty(
                out_per_rank, in_per_rank, dtype=self.weight_dtype, device=device
            ),
            requires_grad=False,
        )
        layer.weight._quant_source_name = "weight"
        layer.weight._accepted_source_dtypes = (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float8_e4m3fn,
        )

        scale_source_name = _weight_scale_source_name(
            self.raw_config,
            self.granularity,
        )

        if self.granularity == Granularity.PER_TENSOR:
            scale_count = len(request.logical_widths)
            layer.weight_scale = nn.Parameter(
                torch.full(
                    (scale_count,),
                    float("nan"),
                    dtype=torch.float32,
                    device=device,
                ),
                requires_grad=False,
            )
            layer.weight_scale._quant_source_name = scale_source_name
            layer.weight_scale._quant_attrs = {
                "scale_type": "tensor",
                "stacked_scalar": scale_count > 1,
            }
        elif self.granularity == Granularity.PER_CHANNEL:
            layer.weight_scale = nn.Parameter(
                torch.full(
                    (out_per_rank,),
                    float("nan"),
                    dtype=torch.float32,
                    device=device,
                ),
                requires_grad=False,
            )
            layer.weight_scale._quant_source_name = scale_source_name
            layer.weight_scale._quant_attrs = {
                "output_dim": 0,
                "scale_type": "channel",
            }
        elif self.granularity == Granularity.BLOCK:
            assert self.block_shape is not None
            bn, bk = self.block_shape
            if out_per_rank % bn != 0:
                raise ValueError(
                    f"Fp8Spec(BLOCK): out_per_rank={out_per_rank} "
                    f"not divisible by block_n={bn}"
                )
            unaligned_widths = [
                width for width in request.logical_widths if width % bn != 0
            ]
            if unaligned_widths:
                raise ValueError(
                    "Fp8Spec(BLOCK): every fused logical width must be divisible "
                    f"by block_n={bn}, got {request.logical_widths!r}"
                )
            if in_per_rank % bk != 0:
                raise ValueError(
                    f"Fp8Spec(BLOCK): in_per_rank={in_per_rank} "
                    f"not divisible by block_k={bk}"
                )
            if _is_modelopt_block(self.raw_config):
                scale_shape = (out_per_rank // bn, 1, in_per_rank // bk, 1)
                quant_attrs = {
                    "output_dim": 0,
                    "input_dim": 2,
                    "scale_type": "block",
                }
                layer._fp8_modelopt_block_scale = True
            else:
                scale_shape = (out_per_rank // bn, in_per_rank // bk)
                quant_attrs = {
                    "output_dim": 0,
                    "input_dim": 1,
                    "scale_type": "block",
                }
                layer._fp8_modelopt_block_scale = False
            layer.weight_scale = nn.Parameter(
                torch.full(
                    scale_shape,
                    float("nan"),
                    dtype=torch.float32,
                    device=device,
                ),
                requires_grad=False,
            )
            layer.weight_scale._quant_source_name = scale_source_name
            layer.weight_scale._quant_attrs = quant_attrs

        if not self.activation.dynamic:
            input_scale_count = len(request.logical_widths)
            layer.input_scale = nn.Parameter(
                torch.full(
                    (input_scale_count,),
                    float("nan"),
                    dtype=torch.float32,
                    device=device,
                ),
                requires_grad=False,
            )
            layer.input_scale._quant_source_name = "input_scale"
            layer.input_scale._quant_attrs = {
                "scale_type": "tensor",
                "stacked_scalar": input_scale_count > 1,
            }

        layer.logical_widths = list(request.logical_widths)
        layer._fp8_pending_weight = None
        layer._fp8_logical_weight_shape = (out_per_rank, in_per_rank)

    def process_after_loading(self, layer: nn.Module) -> None:
        pending = getattr(layer, "_fp8_pending_weight", None)
        if pending is not None:
            self.quantize_loaded_weight(layer, pending)
            layer._fp8_pending_weight = None

        if getattr(layer, "_fp8_modelopt_block_scale", False):
            scale = layer.weight_scale.detach()
            if scale.ndim != 4 or scale.shape[1] != 1 or scale.shape[3] != 1:
                raise ValueError(
                    "ModelOpt FP8_PB_WO weight_scale must have shape "
                    f"[out_blocks,1,in_blocks,1], got {tuple(scale.shape)}"
                )
            _replace_parameter(
                layer,
                "weight_scale",
                scale.squeeze(1).squeeze(-1),
                source_name="weight_scale",
                quant_attrs={
                    "output_dim": 0,
                    "input_dim": 1,
                    "scale_type": "block",
                },
            )

        scale = layer.weight_scale.detach()
        if not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("FP8 weight_scale must be finite and strictly positive")
        if not torch.isfinite(layer.weight.detach().float()).all():
            raise ValueError("FP8 weight must not contain NaN or infinity")
        if self.granularity == Granularity.PER_TENSOR:
            flat = scale.reshape(-1)
            if flat.numel() not in (1, len(layer.logical_widths)):
                raise ValueError(
                    "FP8 tensorwise scale count must be one or match logical "
                    f"weight legs; got {flat.numel()} for {layer.logical_widths!r}"
                )
            common = flat.max().reshape(1)
            if flat.numel() > 1 and not torch.equal(flat, common.expand_as(flat)):
                if sum(layer.logical_widths) != layer.weight.shape[0]:
                    raise RuntimeError(
                        "FP8 fused logical widths do not cover the local weight rows"
                    )
                offset = 0
                for width, old_scale in zip(
                    layer.logical_widths,
                    flat,
                    strict=True,
                ):
                    if width == 0:
                        continue
                    leg = layer.weight.data.narrow(0, offset, width)
                    ratio = (old_scale / common).reshape(1).float().contiguous()
                    if leg.is_cuda:
                        requant = fp8_requantize_with_scale_ratio(leg, ratio)
                    else:
                        requant = torch.clamp(
                            leg.float() * ratio,
                            -448.0,
                            448.0,
                        ).to(torch.float8_e4m3fn)
                    leg.copy_(requant)
                    offset += width
            _replace_parameter(
                layer,
                "weight_scale",
                common,
                source_name=_weight_scale_source_name(
                    self.raw_config,
                    self.granularity,
                ),
                quant_attrs={"scale_type": "tensor"},
            )

        if not self.activation.dynamic:
            input_scale = layer.input_scale.detach().float().reshape(-1)
            if not torch.isfinite(input_scale).all() or (input_scale <= 0).any():
                raise ValueError(
                    "static FP8 input_scale must be finite and strictly positive"
                )
            _replace_parameter(
                layer,
                "input_scale",
                input_scale.max().reshape(1),
                source_name="input_scale",
                quant_attrs={"scale_type": "tensor"},
            )

    def load_weight(
        self,
        layer: nn.Module,
        loaded: torch.Tensor,
        shard_id: "int | str | None",
        default_loader: Callable[[nn.Parameter, torch.Tensor, object], None],
    ) -> None:
        if loaded.dtype in (torch.float16, torch.bfloat16, torch.float32):
            online = self.online or self.raw_config is None
            if not online:
                raise TypeError(
                    "serialized FP8 weight source has high-precision dtype "
                    f"{loaded.dtype}; declare an online QuantScheme instead"
                )
            pending = getattr(layer, "_fp8_pending_weight", None)
            if pending is None:
                pending = torch.empty(
                    layer._fp8_logical_weight_shape,
                    dtype=loaded.dtype,
                    device=loaded.device,
                )
                layer._fp8_pending_weight = pending
            elif pending.dtype != loaded.dtype:
                raise TypeError(
                    "all fused high-precision FP8 source weights must share a dtype"
                )
            default_loader(pending, loaded, shard_id)
            return
        if loaded.dtype is not torch.float8_e4m3fn:
            raise TypeError(
                "FP8 weight source must be BF16/FP16/FP32 for online quantization "
                f"or float8_e4m3fn when serialized, got {loaded.dtype}"
            )
        default_loader(layer.weight, loaded, shard_id)

    def quantize_loaded_weight(
        self,
        layer: nn.Module,
        weight: torch.Tensor,
    ) -> None:
        src = weight.detach().to(device=layer.weight.device).contiguous()
        if src.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(f"cannot online-quantize FP8 weight from {src.dtype}")
        if not torch.isfinite(src).all():
            raise ValueError("online FP8 source weight must be finite")

        if src.is_cuda:
            if self.granularity is Granularity.PER_TENSOR:
                quant, scale = fp8_quantize_per_tensor(src)
            elif self.granularity is Granularity.PER_CHANNEL:
                quant, scale = fp8_quantize_per_token(src)
                scale = scale.reshape(-1)
            elif self.granularity is Granularity.BLOCK:
                assert self.block_shape is not None
                quant, scale = fp8_quantize_weight_per_block(
                    src,
                    self.block_shape,
                )
            else:
                raise RuntimeError(
                    f"unhandled online FP8 granularity {self.granularity!r}"
                )
        else:
            src_f = src.float()
            if self.granularity is Granularity.PER_TENSOR:
                scale = src_f.abs().amax().clamp_min(1e-12).reshape(1) / 448.0
                quant = torch.clamp(src_f / scale, -448.0, 448.0).to(
                    torch.float8_e4m3fn
                )
            elif self.granularity is Granularity.PER_CHANNEL:
                scale = src_f.abs().amax(dim=1).clamp_min(1e-12) / 448.0
                quant = torch.clamp(
                    src_f / scale[:, None],
                    -448.0,
                    448.0,
                ).to(torch.float8_e4m3fn)
            elif self.granularity is Granularity.BLOCK:
                assert self.block_shape is not None
                bn, bk = self.block_shape
                N, K = src_f.shape
                view = src_f.reshape(N // bn, bn, K // bk, bk)
                scale = view.abs().amax(dim=(1, 3)).clamp_min(1e-12) / 448.0
                quant = (
                    torch.clamp(
                        view / scale[:, None, :, None],
                        -448.0,
                        448.0,
                    )
                    .reshape(N, K)
                    .to(torch.float8_e4m3fn)
                )
            else:
                raise RuntimeError(
                    f"unhandled online FP8 granularity {self.granularity!r}"
                )

        layer.weight.data.copy_(quant)
        target_scale = layer.weight_scale
        if getattr(layer, "_fp8_modelopt_block_scale", False):
            scale = scale[:, None, :, None]
        if (
            self.granularity is Granularity.PER_TENSOR
            and scale.numel() == 1
            and target_scale.numel() > 1
        ):
            scale = scale.expand_as(target_scale)
        if target_scale.shape != scale.shape:
            raise RuntimeError(
                "online FP8 scale shape mismatch: "
                f"allocated {tuple(target_scale.shape)}, produced {tuple(scale.shape)}"
            )
        target_scale.data.copy_(scale)

    def quantize_activation(
        self,
        x: torch.Tensor,
        layer: nn.Module,
    ) -> ActivationView:
        if self.granularity == Granularity.PER_TENSOR:
            if self.activation.dynamic:
                x_q, x_scale = fp8_quantize_per_tensor(x)
            else:
                x_q, x_scale = fp8_quantize_per_tensor_with_scale(
                    x,
                    layer.input_scale,
                )
            return ActivationView(x_q, x_scale, Granularity.PER_TENSOR)

        if self.granularity == Granularity.PER_CHANNEL:
            x_q, x_scale = fp8_quantize_per_token(x)
            return ActivationView(x_q, x_scale, self.granularity)

        if self.granularity == Granularity.BLOCK:
            assert self.block_shape is not None
            block_k = self.block_shape[1]
            K = x.shape[-1]
            if K % block_k != 0:
                raise ValueError(
                    f"FP8 block activation K={K} not divisible by block_k={block_k}"
                )
            x_q, x_scale = fp8_quantize_per_group(x, block_k)
            return ActivationView(x_q, x_scale, Granularity.BLOCK)

        raise RuntimeError(f"unhandled Fp8Spec granularity {self.granularity!r}")


__all__ = ["Fp8Spec"]
