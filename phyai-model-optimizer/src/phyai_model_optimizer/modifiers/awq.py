"""AWQ"""

from __future__ import annotations

import torch.nn as nn

from phyai_model_optimizer.modifiers.base import Modifier, QuantResult
from phyai_model_optimizer.observers.base import Observer
from phyai_model_optimizer.observers.minmax import MinMaxObserver
from phyai_model_optimizer.quant_math import QuantDType, WeightQuant


class AWQModifier(Modifier):
    def __init__(
        self,
        weight_dtype: QuantDType | str = QuantDType.INT4,
        *,
        symmetric: bool = False,
        group_size: int = 128,
        targets: list[str] | None = None,
        ignore: list[str] | None = None,
    ) -> None:
        # TODO(chenghua): Implement AWQ search and add calibration coverage before enabling it.
        raise NotImplementedError(
            "AWQModifier is not available because AWQ is not implemented"
        )

        dtype = QuantDType(weight_dtype)
        if not dtype.is_integer:
            raise ValueError(
                f"AWQ only supports integer weight_dtype, got {dtype.value}"
            )
        super().__init__(targets=targets, ignore=ignore)
        self._q = WeightQuant(
            dtype=dtype,
            symmetric=symmetric,
            group_size=group_size,
        )

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
            "AWQModifier is a phase-2 scaffold; use RTNModifier/GPTQModifier for v1"
        )


__all__ = ["AWQModifier"]
