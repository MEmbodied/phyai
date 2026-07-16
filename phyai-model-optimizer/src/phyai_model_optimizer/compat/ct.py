"""Thin wrappers over the compressed-tensors library.

compressed-tensors is the source of truth for the config schema and int32 packing.
This module converts the toolkit's backend-agnostic :class:`WeightQuant` into CT
dataclasses and centralizes the CT symbols we depend on, keeping the rest of the
package decoupled.
"""

from __future__ import annotations

from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
)

from phyai_model_optimizer.quant_math import WeightQuant, to_ct_args

_INT_FORMAT = "pack-quantized"
_FLOAT_FORMAT = "float-quantized"
_MXFP4_FORMAT = "mxfp4-pack-quantized"
_NVFP4_FORMAT = "nvfp4-pack-quantized"


def weight_args(q: WeightQuant) -> QuantizationArgs:
    # note(chenghua): single source of truth for the CT args mapping lives in quant_math.
    return to_ct_args(q)


def compression_format(q: WeightQuant) -> str:
    if q.is_mxfp4:
        return _MXFP4_FORMAT
    if q.is_nvfp4:
        return _NVFP4_FORMAT
    return _FLOAT_FORMAT if q.is_float else _INT_FORMAT


def build_scheme(
    targets: list[str],
    q: WeightQuant,
    input_acts: QuantizationArgs | None = None,
) -> QuantizationScheme:
    return QuantizationScheme(
        targets=list(targets),
        weights=weight_args(q),
        input_activations=input_acts,
        format=compression_format(q),
    )


def build_config(
    schemes: dict[str, QuantizationScheme],
    ignore: list[str],
    fmt: str,
) -> QuantizationConfig:
    return QuantizationConfig(
        config_groups=dict(schemes),
        quant_method="compressed-tensors",
        format=fmt,
        quantization_status=QuantizationStatus.COMPRESSED,
        ignore=list(ignore),
    )


__all__ = [
    "weight_args",
    "compression_format",
    "build_scheme",
    "build_config",
    "QuantizationArgs",
    "QuantizationConfig",
    "QuantizationScheme",
]
