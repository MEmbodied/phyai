"""Round-to-nearest data-free weight quantization."""

from __future__ import annotations

import torch.nn as nn

from phyai_model_optimizer.modifiers.base import Modifier, QuantResult
from phyai_model_optimizer.observers.base import Observer
from phyai_model_optimizer.quant_math import (
    FP8Scheme,
    QuantDType,
    WeightQuant,
    compute_scale_zp,
    fake_quantize,
    quantize_fp4,
    quantize_to_codes,
)


def _resolve_activation_dtype(
    weight_dtype: QuantDType,
    activation_dtype: QuantDType | str | None,
) -> QuantDType | None:
    if activation_dtype == "auto":
        if weight_dtype in (QuantDType.NVFP4, QuantDType.FP8_E4M3):
            return weight_dtype
        return None
    if activation_dtype is None or activation_dtype == "none":
        resolved = None
    else:
        resolved = QuantDType(activation_dtype)

    if resolved is not None and not resolved.supports_activation:
        raise ValueError(
            f"{resolved.value} is not supported for activation quantization"
        )
    if weight_dtype is QuantDType.FP8_E4M3 and resolved is not QuantDType.FP8_E4M3:
        raise ValueError(
            "fp8_e4m3 RTN requires dynamic activation_dtype='fp8_e4m3' or 'auto'"
        )
    if weight_dtype is QuantDType.NVFP4 and resolved is not QuantDType.NVFP4:
        raise ValueError("nvfp4 RTN requires activation_dtype='nvfp4' or 'auto'")
    if weight_dtype is QuantDType.MXFP4 and resolved not in (
        None,
        QuantDType.FP8_E4M3,
        QuantDType.MXFP4,
    ):
        raise ValueError(
            "mxfp4 RTN supports only A16, activation_dtype='fp8_e4m3', "
            "or activation_dtype='mxfp4'"
        )
    if resolved is QuantDType.MXFP4 and weight_dtype is not QuantDType.MXFP4:
        raise ValueError("mxfp4 activation requires weight_dtype='mxfp4'")
    if resolved is QuantDType.NVFP4 and weight_dtype is not QuantDType.NVFP4:
        raise ValueError("nvfp4 activation requires weight_dtype='nvfp4'")
    return resolved


class RTNModifier(Modifier):
    def __init__(
        self,
        weight_dtype: QuantDType | str = QuantDType.INT4,
        *,
        activation_dtype: QuantDType | str | None = "auto",
        symmetric: bool = True,
        group_size: int | None = None,
        fp8_scheme: FP8Scheme | str | None = None,
        targets: list[str] | None = None,
        ignore: list[str] | None = None,
    ) -> None:
        dtype = QuantDType(weight_dtype)
        if dtype.is_fp8:
            if group_size is not None:
                raise ValueError(
                    "fp8 weight quant uses fp8_scheme; do not pass group_size"
                )
            if not symmetric:
                raise ValueError("fp8 weight quant must be symmetric")
            resolved_group_size = 0
            resolved_fp8_scheme = (
                FP8Scheme.BLOCK_128 if fp8_scheme is None else fp8_scheme
            )
        elif dtype is QuantDType.MXFP4:
            resolved_group_size = 32 if group_size is None else group_size
            resolved_fp8_scheme = None
        elif dtype is QuantDType.NVFP4:
            resolved_group_size = 16 if group_size is None else group_size
            resolved_fp8_scheme = None
        else:
            resolved_group_size = 0 if group_size is None else group_size
            resolved_fp8_scheme = None
            if fp8_scheme is not None:
                raise ValueError("fp8_scheme is only valid for fp8 weight quant")
        resolved_activation = _resolve_activation_dtype(dtype, activation_dtype)
        super().__init__(
            targets=targets,
            ignore=ignore,
            activation_dtype=resolved_activation,
        )
        self._q = WeightQuant(
            dtype=dtype,
            symmetric=symmetric,
            group_size=resolved_group_size,
            fp8_scheme=resolved_fp8_scheme,
        )

    def weight_quant(self) -> WeightQuant:
        return self._q

    @property
    def activation_dtype(self) -> QuantDType | None:
        return self._activation_dtype

    @property
    def requires_calibration(self) -> bool:
        return False

    def quantize_layer(
        self, module: nn.Module, observer: Observer | None
    ) -> QuantResult:
        weight = module.weight.data
        if self._q.is_fp4:
            fake, packed, scale, global_scale = quantize_fp4(weight, self._q)
            result = QuantResult(
                q=self._q,
                scale=scale,
                zero_point=None,
                fake_weight=fake,
                packed_weight=packed,
                global_scale=global_scale,
            )
        else:
            scale, zero_point = compute_scale_zp(weight, self._q)
            fake = fake_quantize(weight, self._q, scale, zero_point)
            result = QuantResult(
                q=self._q,
                scale=scale,
                zero_point=zero_point,
                fake_weight=fake,
            )
            if self._q.is_integer:
                result.int_codes = quantize_to_codes(weight, self._q, scale, zero_point)
        module.weight.data.copy_(result.fake_weight)
        module._ptq_result = result
        return result


__all__ = ["RTNModifier"]
