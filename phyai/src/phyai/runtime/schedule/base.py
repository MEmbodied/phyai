"""Abstract base for inference schedulers.

A :class:`Scheduler` owns one or more :class:`~phyai.runtime.model_runner.ModelRunner`
instances and orchestrates their forward calls per inference round. The
contract is intentionally minimal — :meth:`setup` brings everything to
the warmed / graph-captured state, :meth:`step` runs one inference, and
:meth:`close` releases pinned resources.

Subclasses are model-specific: each model package contributes its own
scheduler that composes its runners (vision, LLM, expert, etc.) into
the right per-step sequence.

This file deliberately stays minimal — no internal event loop, no
worker pool. The :class:`Scheduler` here is a pure object:
single-threaded, single-batch, no internal queue. Higher-level
multi-batch / continuous-batching schedulers can be layered on top
later in this package.
"""

from __future__ import annotations

import abc
from typing import Any


class Scheduler(abc.ABC):
    """Abstract per-model inference orchestrator."""

    @abc.abstractmethod
    def setup(self, *args: Any, **kwargs: Any) -> None:
        """Warm up every runner, capture graphs, allocate caches."""

    @abc.abstractmethod
    def step(self, request: Any) -> Any:
        """Run one inference round end-to-end."""

    def close(self) -> None:
        """Release runners' resources. Default: no-op."""
        return None


__all__ = ["Scheduler"]
