"""``to_batch_processor`` — add a leading batch dim to unbatched inputs.

lerobot's pi05 preprocessor includes this to batchify single-sample inputs.
phyai's canonical request tensors are already batched ``(B, ...)``, so for the
phyai inference path this step is normally a no-op; it is implemented (and
registered under lerobot's name) so the lerobot json loads, and it still
correctly batchifies a genuinely unbatched ``STATE`` / ``ACTION`` tensor or a
bare task string if one is passed.
"""

from __future__ import annotations

import torch

from phyai_utils_tools.processing.pipeline import (
    ProcessorStep,
    ProcessorStepRegistry,
)
from phyai_utils_tools.processing.transition import (
    ACTION,
    STATE,
    TASK,
    Transition,
)


@ProcessorStepRegistry.register("to_batch_processor")
class AddBatchDimensionStep(ProcessorStep):
    """Add a batch dim where an input looks unbatched.

    Heuristics (only when the field is present):
    * ``STATE`` 1-D ``(D,)`` -> ``(1, D)``.
    * ``ACTION`` 1-D -> unsqueezed.
    * ``TASK`` a bare ``str`` -> ``[str]``.
    Already-batched inputs are left untouched. Empty ``config`` (matches
    lerobot's ``to_batch_processor``).
    """

    def __call__(self, transition: Transition) -> Transition:
        out = transition.copy()
        state = out.get(STATE)
        if isinstance(state, torch.Tensor) and state.dim() == 1:
            out[STATE] = state.unsqueeze(0)
        action = out.get(ACTION)
        if isinstance(action, torch.Tensor) and action.dim() == 1:
            out[ACTION] = action.unsqueeze(0)
        task = out.get(TASK)
        if isinstance(task, str):
            out[TASK] = [task]
        return out


__all__ = ["AddBatchDimensionStep"]
