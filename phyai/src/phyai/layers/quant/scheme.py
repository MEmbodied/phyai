"""Semantic quantization IR — what a tensor is quantized to, nothing else.

A :class:`QuantScheme` is pure semantics: it carries no scale layout,
kernel, or device decision. Those are lowered by
:func:`phyai.layers.quant.materialize.materialize` into a physical
:class:`phyai.layers.quant.base.WeightSpec`. Weight and activation are
described by the same :class:`TensorQuant`, so ``input=None`` cleanly
means weight-only (e.g. W4A16).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from phyai.layers.quant.granularity import Granularity


class QDType(Enum):
    """Quantized element type. ``BF16`` is the sentinel for 'not quantized'.

    The sub-8-bit integer and fp6 members mirror humming's weight dtype set
    (``int2``/``int3``/``int4``/``int6``/``int8``, ``float6e2m3``/``float6e3m2``,
    ``float4e2m1``); these are weight-only (humming activations are limited to
    int4/int8/fp8/bf16/fp16/fp4).
    """

    BF16 = "bf16"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    INT8 = "int8"
    INT6 = "int6"
    INT4 = "int4"
    INT3 = "int3"
    INT2 = "int2"
    NVFP4 = "nvfp4"
    MXFP4 = "mxfp4"
    FP6_E2M3 = "fp6_e2m3"
    FP6_E3M2 = "fp6_e3m2"


@dataclass(frozen=True)
class TensorQuant:
    """How one tensor (weight OR activation) is quantized.

    ``dynamic`` is meaningful only for activations (True = scale computed
    at runtime); it is always False for weights. ``micro_scaled`` marks
    block-microscaled formats (NVFP4/MXFP4: an in-block low-precision
    scale plus an outer global scale). ``block_shape`` is set for
    block-granularity weights. ``group_size`` is the K-direction group
    size for group-quantized weights or activations (0 = per-channel /
    per-tensor), as used by AWQ/GPTQ and llm-compressor ``group`` strategy.
    """

    dtype: QDType
    granularity: Granularity
    symmetric: bool = True
    dynamic: bool = False
    micro_scaled: bool = False
    block_shape: tuple[int, int] | None = None
    group_size: int = 0


@dataclass(frozen=True)
class QuantScheme:
    """The complete quantization decision for one layer."""

    weight: TensorQuant
    input: TensorQuant | None = None
    online: bool = False
    raw_config: dict | None = None
    input_raw_config: dict | None = None
    pack_format: str | None = None

    @property
    def weight_only(self) -> bool:
        return self.input is None


__all__ = ["QDType", "TensorQuant", "QuantScheme"]
