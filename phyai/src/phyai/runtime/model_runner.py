"""Abstract base for model runners.

A :class:`ModelRunner` is the smallest unit of compute the scheduler
hands a forward batch to. Each runner owns one or more captured CUDA
graphs (registered in a :class:`~phyai.runtime.cuda_graph_manager.CudaGraphRegistry`),
the model module(s) those graphs wrap, plus any auxiliary state
(flashinfer wrappers, RoPE caches, …) that should outlive a single call.

Lifecycle (driven by the scheduler):

1. ``__init__`` — accepts the model module(s) it will wrap and any
   shape parameters it needs to capture for. No CUDA work yet.
2. ``setup`` — runs the warmup + graph capture. Heavy CUDA call;
   typically invoked once at program start.
3. ``warmup`` — optional secondary warmup hook; the default
   implementation is a no-op. Subclasses override when they need to
   spin up extra resources after :meth:`setup` (e.g. the flashinfer
   workspace).
4. ``plan_inference`` — eager metadata staging hook. Called by the
   scheduler outside any captured region with the per-stack metadata
   describing the next forward step (``ARAttnMetadata`` /
   ``DiffusionAttnMetadata`` / ``AttnMetadata`` depending on which
   attention stack the runner drives). Runners forward this to their
   backend's ``replay_metadata`` (graph mode) or
   ``init_forward_metadata`` (eager mode). Default: no-op for runners
   without attention (e.g. the vision tower).
5. ``forward`` — the hot path. Receives a forward-batch payload;
   returns the runner's output (depends on the runner type — vision
   returns image embeddings, LLM returns nothing, expert returns
   ``v_t``).
6. ``close`` — release the captured graphs and any GPU memory the
   runner pinned. Optional; the default implementation is a no-op.

The base class deliberately does not enforce a payload type — different
runners consume different forward-batch flavors and the type lives in
the subclass signature, not on the base. Same applies to ``meta`` —
each concrete runner annotates its ``plan_inference`` with the
matching metadata type.
"""

from __future__ import annotations

import abc
from typing import Any


class ModelRunner(abc.ABC):
    """Abstract single-purpose model runner.

    Subclasses provide the model + warmup / capture logic; the
    :class:`~phyai.runtime.schedule.base.Scheduler` orchestrates the
    sequence of runner calls per inference.
    """

    @abc.abstractmethod
    def setup(self, *args: Any, **kwargs: Any) -> None:
        """One-time setup: warmup + capture every CUDA graph this runner uses."""

    def warmup(self) -> None:
        """Optional post-setup warmup. Default: no-op."""
        return None

    def plan_inference(self, meta: Any) -> None:
        """Stage attention metadata for the next :meth:`forward` call.

        Called by the scheduler OUTSIDE any captured region. Runners
        with a paged-attention backend forward this to either
        ``replay_metadata`` (graph mode) or ``init_forward_metadata``
        (eager mode) on the runner's backend instance. The concrete
        runner's signature narrows ``meta`` to its per-stack metadata
        type (``ARAttnMetadata`` / ``DiffusionAttnMetadata`` / etc.).

        Default: no-op. Runners that don't drive attention (e.g.
        vision tower) inherit the no-op directly.
        """
        del meta
        return None

    @abc.abstractmethod
    def forward(self, batch: Any) -> Any:
        """Run one forward pass against ``batch`` and return the runner's output."""

    def close(self) -> None:
        """Release captured graphs / scratch buffers. Default: no-op."""
        return None


__all__ = ["ModelRunner"]
