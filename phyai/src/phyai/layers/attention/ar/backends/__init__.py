"""Concrete AR (LM-side) paged attention backend implementations.

Each ``<vendor>.py`` module owns one backend class (and the plan
handle types it consumes) and self-registers via the
:func:`~phyai.layers.attention.ar.registry.register_backend`
decorator at module import.

Two backend names — ``"flashinfer"`` (paged-KV) and ``"eager"``
(contiguous-slab reference).
"""

from __future__ import annotations

from phyai.layers.attention.ar.backends.eager import (
    EagerARBackend,
    EagerARPlan,
)
from phyai.layers.attention.ar.backends.flashinfer import (
    FlashInferARBackend,
    FlashInferARPlan,
)


__all__ = [
    "EagerARBackend",
    "EagerARPlan",
    "FlashInferARBackend",
    "FlashInferARPlan",
]
