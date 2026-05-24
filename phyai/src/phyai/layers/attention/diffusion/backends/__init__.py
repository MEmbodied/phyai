"""Concrete diffusion / action-expert paged attention backend implementations.

Two backend names — ``"flashinfer"`` (paged-KV) and ``"eager"``
(contiguous-slab reference). Self-register via the
:func:`~phyai.layers.attention.diffusion.registry.register_backend`
decorator at module import.
"""

from __future__ import annotations

from phyai.layers.attention.diffusion.backends.eager import (
    EagerDiffusionBackend,
    EagerDiffusionPlan,
)
from phyai.layers.attention.diffusion.backends.flashinfer import (
    FlashInferDiffusionBackend,
    FlashInferDiffusionPlan,
)


__all__ = [
    "EagerDiffusionBackend",
    "EagerDiffusionPlan",
    "FlashInferDiffusionBackend",
    "FlashInferDiffusionPlan",
]
