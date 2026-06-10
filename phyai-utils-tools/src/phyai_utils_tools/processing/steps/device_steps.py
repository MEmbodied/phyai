"""Device step — move transition tensors to a target device / dtype.

Moves the canonical model-input/action tensor fields onto ``device`` and, for
float tensors, optionally to ``float_dtype``. Registry name + config schema
(``{device, float_dtype}``) match lerobot's ``device_processor`` so it
round-trips with ``policy_*processor.json`` (where ``float_dtype`` is a string
like ``"float32"`` or ``null``). Non-tensor entries (task strings) are untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from phyai_utils_tools.processing.pipeline import (
    ProcessorStep,
    ProcessorStepRegistry,
)
from phyai_utils_tools.processing.transition import (
    ACTION,
    INPUT_IDS,
    LANG_LENS,
    PIXEL_VALUES,
    STATE,
    Transition,
)

# Tensor fields the step moves. Kept internal (not serialized) — lerobot's
# device_processor config is only {device, float_dtype}.
_DEFAULT_FIELDS = (PIXEL_VALUES, INPUT_IDS, LANG_LENS, STATE, ACTION)

_DTYPE_BY_NAME: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float": torch.float32,
    "float64": torch.float64,
    "double": torch.float64,
}


def _resolve_dtype(float_dtype: str | torch.dtype | None) -> torch.dtype | None:
    """Accept a torch.dtype or a lerobot dtype string; return a torch.dtype."""
    if float_dtype is None or isinstance(float_dtype, torch.dtype):
        return float_dtype
    key = str(float_dtype).replace("torch.", "")
    if key not in _DTYPE_BY_NAME:
        raise ValueError(
            f"device_processor: unknown float_dtype {float_dtype!r}; "
            f"expected one of {sorted(_DTYPE_BY_NAME)} or None."
        )
    return _DTYPE_BY_NAME[key]


@ProcessorStepRegistry.register("device_processor")
@dataclass
class DeviceStep(ProcessorStep):
    """Move tensor fields to ``device`` (and float tensors to ``float_dtype``)."""

    device: torch.device | str = "cpu"
    float_dtype: str | torch.dtype | None = None

    def __post_init__(self) -> None:
        self._float_dtype = _resolve_dtype(self.float_dtype)

    def __call__(self, transition: Transition) -> Transition:
        out = transition.copy()
        for name in _DEFAULT_FIELDS:
            t = out.get(name)
            if not isinstance(t, torch.Tensor):
                continue
            t = t.to(device=self.device)
            if self._float_dtype is not None and t.is_floating_point():
                t = t.to(dtype=self._float_dtype)
            out[name] = t
        return out

    def get_config(self) -> dict[str, Any]:
        # lerobot schema: device string + float_dtype string (or null).
        dt = self._float_dtype
        return {
            "device": str(self.device).replace("torch.device", "").strip("()'\""),
            "float_dtype": (str(dt).replace("torch.", "") if dt is not None else None),
        }


__all__ = ["DeviceStep"]
