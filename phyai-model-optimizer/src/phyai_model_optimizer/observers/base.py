"""Observers

Owned by a Modifier, attached transiently via ``register_forward_pre_hook`` and
dropped after the block is processed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Observer(ABC):
    """Base accumulator. ``observe`` is called once per forward with the tensor
    of interest (typically the input activation to a Linear, shape ``(..., K)``)."""

    def __init__(self) -> None:
        self.nsamples: int = 0

    @abstractmethod
    def observe(self, x: torch.Tensor) -> None: ...

    def reset(self) -> None:
        self.nsamples = 0


__all__ = ["Observer"]
