"""Per-model processor base class.

A :class:`BaseModelProcessor` bundles a model's preprocessing and
postprocessing into two :class:`~phyai_utils_tools.processing.pipeline.ProcessorPipeline`
instances and exposes them through :meth:`preprocess` / :meth:`postprocess`.
Each concrete model (pi0.5, and future models) subclasses this and implements
:meth:`build_preprocessor` / :meth:`build_postprocessor`, composing the shared
registered steps into its own order. The two pipelines are built once in
``__init__`` and cached, so per-call cost is just running the steps.

This is intentionally thin: all real logic lives in the composable steps
(:mod:`phyai_utils_tools.processing.steps`); the subclass only declares which
steps run, in what order, with what config.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from phyai_utils_tools.processing.pipeline import ProcessorPipeline


class BaseModelProcessor(ABC):
    """ABC bundling a model's pre/post-process pipelines.

    Subclasses implement :meth:`build_preprocessor` and
    :meth:`build_postprocessor`; the base wires them to :meth:`preprocess` /
    :meth:`postprocess`. The pipelines are constructed eagerly in
    :meth:`__init__` (so any tokenizer load / stats tensor-ization happens once
    up front, not per call).
    """

    def __init__(self) -> None:
        self._preprocessor: ProcessorPipeline = self.build_preprocessor()
        self._postprocessor: ProcessorPipeline = self.build_postprocessor()

    @abstractmethod
    def build_preprocessor(self) -> ProcessorPipeline:
        """Return the model's preprocessing pipeline (raw inputs -> model inputs)."""
        raise NotImplementedError

    @abstractmethod
    def build_postprocessor(self) -> ProcessorPipeline:
        """Return the model's postprocessing pipeline (model output -> result)."""
        raise NotImplementedError

    @property
    def preprocessor(self) -> ProcessorPipeline:
        """The (cached) preprocessing pipeline."""
        return self._preprocessor

    @property
    def postprocessor(self) -> ProcessorPipeline:
        """The (cached) postprocessing pipeline."""
        return self._postprocessor

    def preprocess(self, raw: Any) -> Any:
        """Run the raw caller payload through the preprocessing pipeline."""
        return self._preprocessor(raw)

    def postprocess(self, model_output: Any) -> Any:
        """Run the model output through the postprocessing pipeline."""
        return self._postprocessor(model_output)


__all__ = ["BaseModelProcessor"]
