"""Pipelines Decide how data flows during quantization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Protocol, runtime_checkable

import torch.nn as nn

from phyai_model_optimizer.modifiers.base import Modifier

Target = tuple[str, nn.Module, Modifier]


class Pipeline(ABC):
    @abstractmethod
    def run(
        self,
        model: nn.Module,
        targets: list[Target],
        *,
        dataloader: Iterable | None = None,
        driver: "SequentialDriver | None" = None,
    ) -> None: ...


@runtime_checkable
class SequentialDriver(Protocol):
    """Adapt a model's forward structure to block-by-block calibration.

    The sequential calibration pipeline owns the quantization algorithm: it attaches
    observers, gathers statistics, quantizes the current block, and replays that block
    so the next block sees quantized rather than baseline activations. The driver owns
    only the model-specific execution details needed to make that process possible:

    * where the ordered block boundaries are;
    * how to run the model prefix and cache inputs for the first block;
    * how to invoke one block from a cached ``(args, kwargs)`` pair; and
    * how to replace the hidden state in that pair while preserving context such as
      attention masks, position IDs, RoPE data, conditioning, or KV state.

    A driver does not select quantization targets, calculate scales, create observers,
    or serialize weights. Keeping those responsibilities in the pipeline and modifiers
    lets the same calibration implementation serve models with different forward
    signatures.

    For a simple model whose dataloader already yields the first block's hidden state
    and whose blocks all implement ``block(x) -> x``, a driver can be as small as::

        class TwoBlockDriver:
            def __init__(self, model):
                self.model = model

            def blocks(self):
                return [
                    ("blocks.0", self.model.blocks[0]),
                    ("blocks.1", self.model.blocks[1]),
                ]

            def seed(self, dataloader):
                return [((x,), {}) for x in dataloader]

            def replay(self, block, args, kwargs):
                return block(*args, **kwargs)

            def advance(self, args, kwargs, output):
                return ((output,), {})

    The pipeline consumes this driver as follows: collect block 0 statistics from the
    seeded inputs, quantize block 0, replay its quantized forward, pass those outputs to
    block 1 through :meth:`advance`, and repeat. More complex drivers may run embeddings
    or vision encoders in :meth:`seed` and carry model-specific context through every
    block without changing the generic calibration pipeline.
    """

    def blocks(self) -> list[tuple[str, nn.Module]]:
        """Return the ordered ``(module_name, module)`` calibration boundaries.

        Names must use the same namespace as target module names so the pipeline can
        assign each target Linear to its containing block.
        """
        ...

    def seed(self, dataloader: Iterable) -> list[tuple[tuple, dict]]:
        """Return cached calls for the first block, one per calibration unit.

        Each item is an ``(args, kwargs)`` pair that can be passed directly to
        :meth:`replay`. Implementations may run the model prefix before producing the
        cache. For a diffusion tower, one sample may expand into multiple timestep
        units so calibration covers the denoising activation distribution.
        """
        ...

    def replay(self, block: nn.Module, args: tuple, kwargs: dict):
        """Execute one block from a cached call and return its propagated output.

        If the block returns auxiliary values, the driver must extract or preserve the
        value that :meth:`advance` needs for the next block.
        """
        ...

    def advance(self, args: tuple, kwargs: dict, output) -> tuple[tuple, dict]:
        """Build the next block's cached call from the current call and output.

        Usually this replaces only the hidden-state argument while retaining immutable
        context in the original ``args`` and ``kwargs``.
        """
        ...


def select_pipeline(modifiers: list[Modifier]) -> Pipeline:
    """data-free unless any modifier needs calibration data."""
    from phyai_model_optimizer.pipelines.datafree import DatafreePipeline
    from phyai_model_optimizer.pipelines.sequential import SequentialCalibrationPipeline

    if any(m.requires_calibration for m in modifiers):
        return SequentialCalibrationPipeline()
    return DatafreePipeline()


__all__ = ["Pipeline", "SequentialDriver", "Target", "select_pipeline"]
