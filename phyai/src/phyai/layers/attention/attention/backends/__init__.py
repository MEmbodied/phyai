"""Concrete no-cache attention backend implementations.

Each ``<vendor>.py`` module owns one backend class (and the plan
handle types it consumes) and self-registers via the
:func:`~phyai.layers.attention.attention.registry.register_backend`
decorator at module import. Importing this package executes those
side effects and surfaces the public classes through the names below.

Three backend names — ``"sdpa"`` / ``"flashinfer"`` / ``"eager"``.
"""

from __future__ import annotations

from phyai.layers.attention.attention.backends.eager import (
    EagerAttentionBackend,
    EagerAttentionPlan,
)
from phyai.layers.attention.attention.backends.flashinfer import (
    FlashInferAttentionBackend,
    FlashInferAttentionPlan,
)
from phyai.layers.attention.attention.backends.sdpa import (
    SdpaAttentionBackend,
    SdpaAttentionPlan,
)


__all__ = [
    "EagerAttentionBackend",
    "EagerAttentionPlan",
    "FlashInferAttentionBackend",
    "FlashInferAttentionPlan",
    "SdpaAttentionBackend",
    "SdpaAttentionPlan",
]
