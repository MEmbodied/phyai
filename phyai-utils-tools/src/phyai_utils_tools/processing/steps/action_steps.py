"""Action postprocessing step — trim the action chunk to the real action dim.

pi0.5 (and openpi VLAs generally) pad the action vector to a fixed
``max_action_dim`` for the model, then slice back to the dataset's true action
dimensionality on the way out. :class:`SliceActionStep` is that slice — the
minimal postprocess every model needs after (optional) unnormalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from phyai_utils_tools.processing.pipeline import (
    ProcessorStep,
    ProcessorStepRegistry,
)
from phyai_utils_tools.processing.transition import ACTION, Transition


@ProcessorStepRegistry.register("slice_action_step")
@dataclass
class SliceActionStep(ProcessorStep):
    """Trim ``ACTION`` to ``[..., :action_dim]``.

    ``action_dim`` is the dataset's real action width (``<= max_action_dim``).
    ``None`` leaves the action untouched (pass-through), useful when the caller
    already wants the full padded chunk.
    """

    action_dim: int | None = None

    def __call__(self, transition: Transition) -> Transition:
        if self.action_dim is None:
            return transition
        action = transition.get(ACTION)
        if action is None:
            raise ValueError("SliceActionStep requires an ACTION entry.")
        out = transition.copy()
        out[ACTION] = action[..., : self.action_dim]
        return out

    def get_config(self) -> dict[str, Any]:
        return {"action_dim": self.action_dim}


__all__ = ["SliceActionStep"]
