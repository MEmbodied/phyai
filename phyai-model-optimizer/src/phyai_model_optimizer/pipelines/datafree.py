"""DatafreePipeline. RTN and friends: quantize weights, no forward, no data."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

from phyai_model_optimizer.pipelines.base import Pipeline, SequentialDriver, Target


class DatafreePipeline(Pipeline):
    def run(
        self,
        model: nn.Module,
        targets: list[Target],
        *,
        dataloader: Iterable | None = None,
        driver: SequentialDriver | None = None,
    ) -> None:
        with torch.no_grad():
            for name, module, modifier in targets:
                if modifier.requires_calibration:
                    raise RuntimeError(
                        f"DatafreePipeline got a calibration-requiring modifier for {name!r}"
                    )
                modifier.quantize_layer(module, None)


__all__ = ["DatafreePipeline"]
