"""Modifier base."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn

from phyai_model_optimizer.observers.base import Observer
from phyai_model_optimizer.quant_math import QuantDType, WeightQuant, activation_args


@dataclass
class QuantResult:
    """Per-module quantization artifacts, stashed on ``module._ptq_result`` for the
    serializer. Integer ``int_codes`` and FP4 ``packed_weight`` are authoritative;
    ``scale`` is already storage-ready for FP4, and ``global_scale`` follows the
    compressed-tensors divisor convention. ``fake_weight`` drives sequential replay
    and evaluation."""

    q: WeightQuant
    scale: torch.Tensor
    zero_point: torch.Tensor | None
    fake_weight: torch.Tensor
    int_codes: torch.Tensor | None = None
    packed_weight: torch.Tensor | None = None
    global_scale: torch.Tensor | None = None


def _compile_target(pattern: str) -> tuple[str, object]:
    """Targets accept ``re:<regex>`` (fqn regex), a bare CamelCase class name, or
    a plain fqn substring/suffix."""
    if pattern.startswith("re:"):
        return ("regex", re.compile(pattern[3:]))
    if "." not in pattern and pattern[:1].isupper():
        return ("module_cls", pattern)
    return ("name", pattern)


def _is_linear_like(module: nn.Module) -> bool:
    """A quantizable linear: exposes a 2-D ``weight`` Parameter. Covers
    ``nn.Linear``, phyai ``LinearBase`` subclasses, and the model-free holder."""
    w = getattr(module, "weight", None)
    return isinstance(w, nn.Parameter) and w.ndim == 2


def _match_one(kind: str, needle: object, name: str, module: nn.Module) -> bool:
    if kind == "regex":
        if not isinstance(needle, re.Pattern):
            raise TypeError(f"regex target must compile to re.Pattern, got {needle!r}")
        return needle.search(name) is not None
    if kind == "module_cls":
        n = str(needle)
        # note(chenghua): "Linear"/"LinearBase" mean any linear-like module; other names
        # match a class exactly against the MRO (no loose substring — use ``re:`` for that).
        if n in ("Linear", "LinearBase"):
            return _is_linear_like(module)
        return n in {c.__name__ for c in type(module).__mro__}
    # note(chenghua): full-fqn or dotted-suffix equality only (no loose substring —
    # use an ``re:`` target for that).
    n = str(needle)
    return name == n or name.endswith("." + n)


class Modifier(ABC):
    def __init__(
        self,
        targets: list[str] | None = None,
        ignore: list[str] | None = None,
        *,
        activation_dtype: QuantDType | str | None = None,
    ) -> None:
        self._targets = list(targets or ["Linear"])
        self._ignore = list(ignore or [])
        self._compiled_t = [_compile_target(t) for t in self._targets]
        self._compiled_i = [_compile_target(t) for t in self._ignore]
        self._activation_dtype = (
            None if activation_dtype is None else QuantDType(activation_dtype)
        )

    @abstractmethod
    def weight_quant(self) -> WeightQuant: ...

    def input_args(self):
        """Return dynamic compressed-tensors activation args, or weight-only None."""
        if self._activation_dtype is None:
            return None
        q = self.weight_quant()
        return activation_args(
            self._activation_dtype,
            fp8_scheme=q.fp8_scheme if q.is_fp8 else None,
        )

    def targets(self) -> list[str]:
        return list(self._targets)

    def ignore(self) -> list[str]:
        return list(self._ignore)

    @property
    @abstractmethod
    def requires_calibration(self) -> bool: ...

    def matches(self, name: str, module: nn.Module) -> bool:
        if any(_match_one(k, v, name, module) for k, v in self._compiled_i):
            return False
        return any(_match_one(k, v, name, module) for k, v in self._compiled_t)

    def make_observer(self, module: nn.Module) -> Observer | None:
        """Return a fresh observer for this target module, or None if the
        modifier needs no per-layer activation statistics."""
        return None

    @abstractmethod
    def quantize_layer(
        self, module: nn.Module, observer: Observer | None
    ) -> QuantResult:
        """Compute qparams, write the fake-quantized weight into
        ``module.weight.data``, stash + return a :class:`QuantResult`."""
        ...


__all__ = ["Modifier", "QuantResult"]
