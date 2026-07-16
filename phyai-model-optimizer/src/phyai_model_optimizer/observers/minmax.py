"""MinMaxObserver. Running min/max of observed activations."""

from __future__ import annotations

import torch

from phyai_model_optimizer.observers.base import Observer


class MinMaxObserver(Observer):
    def __init__(self, per_channel: bool = True) -> None:
        super().__init__()
        self.per_channel = per_channel
        self.min: torch.Tensor | None = None
        self.max: torch.Tensor | None = None

    def observe(self, x: torch.Tensor) -> None:
        x = x.detach().reshape(-1, x.shape[-1]).float()
        if x.numel() == 0:
            return
        if self.per_channel:
            cur_min = x.amin(dim=0)
            cur_max = x.amax(dim=0)
        else:
            cur_min = x.amin()
            cur_max = x.amax()
        if self.min is None:
            self.min, self.max = cur_min, cur_max
        else:
            self.min = torch.minimum(self.min, cur_min)
            self.max = torch.maximum(self.max, cur_max)
        self.nsamples += x.shape[0]

    def minmax(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.min is None or self.max is None:
            raise RuntimeError("MinMaxObserver saw no data")
        return self.min, self.max

    def reset(self) -> None:
        super().reset()
        self.min = None
        self.max = None


__all__ = ["MinMaxObserver"]
