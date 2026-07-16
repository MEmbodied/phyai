"""SmoothQuant"""

from __future__ import annotations

import torch.nn as nn

from phyai_model_optimizer.modifiers.base import Modifier, QuantResult
from phyai_model_optimizer.observers.base import Observer
from phyai_model_optimizer.observers.minmax import MinMaxObserver
from phyai_model_optimizer.quant_math import QuantDType, WeightQuant


class SmoothQuantModifier(Modifier):
    def __init__(
        self,
        weight_dtype: QuantDType | str = QuantDType.INT8,
        *,
        symmetric: bool = True,
        smoothing_strength: float = 0.5,
        targets: list[str] | None = None,
        ignore: list[str] | None = None,
    ) -> None:
        # TODO(chenghua): Implement SmoothQuant smoothing and calibration before enabling it.
        raise NotImplementedError(
            "SmoothQuantModifier is not available because SmoothQuant is not implemented"
        )

        dtype = QuantDType(weight_dtype)
        if not dtype.is_integer:
            raise ValueError(
                f"SmoothQuant only supports integer weight_dtype, got {dtype.value}"
            )
        super().__init__(targets=targets, ignore=ignore)
        self._q = WeightQuant(
            dtype=dtype,
            symmetric=symmetric,
            group_size=0,
        )
        self.smoothing_strength = smoothing_strength

    def weight_quant(self) -> WeightQuant:
        return self._q

    @property
    def requires_calibration(self) -> bool:
        return True

    def make_observer(self, module: nn.Module) -> Observer:
        return MinMaxObserver(per_channel=True)

    def quantize_layer(
        self, module: nn.Module, observer: Observer | None
    ) -> QuantResult:
        raise NotImplementedError(
            "SmoothQuantModifier is a phase-2 scaffold; use RTNModifier/GPTQModifier for v1"
        )


__all__ = ["SmoothQuantModifier"]
