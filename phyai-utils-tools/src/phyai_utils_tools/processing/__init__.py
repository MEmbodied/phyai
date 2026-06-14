"""Composable pre/post-process framework for phyai models.

Re-exports the framework surface:

* :class:`ProcessorStep`, :class:`ProcessorStepRegistry`,
  :class:`ProcessorPipeline` — the step ABC, the name registry, and the chainer.
* :class:`ProcessorStepError` — raised on an unresolvable step in a config json.
* :class:`BaseModelProcessor` — the per-model ABC each model subclasses.
* the canonical transition key constants + :class:`Transition` type alias.
"""

from __future__ import annotations

from phyai_utils_tools.processing.base_processor import BaseModelProcessor
from phyai_utils_tools.processing.pipeline import (
    ProcessorPipeline,
    ProcessorStep,
    ProcessorStepError,
    ProcessorStepRegistry,
)
from phyai_utils_tools.processing.transition import (
    ACTION,
    IMAGES,
    INPUT_IDS,
    LANG_LENS,
    PIXEL_VALUES,
    PROMPT,
    STATE,
    TASK,
    Transition,
    identity_adapter,
)

__all__ = [
    "ACTION",
    "BaseModelProcessor",
    "IMAGES",
    "INPUT_IDS",
    "LANG_LENS",
    "PIXEL_VALUES",
    "PROMPT",
    "ProcessorPipeline",
    "ProcessorStep",
    "ProcessorStepError",
    "ProcessorStepRegistry",
    "STATE",
    "TASK",
    "Transition",
    "identity_adapter",
]
