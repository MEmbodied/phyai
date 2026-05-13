"""Scale granularity enum, shared by every quant scheme.

Lives in ``quant`` so any op (linear, embedding, MoE, KV cache) can pick
it up without leaning on ``layers.linear``.
"""

from __future__ import annotations

from enum import Enum


class Granularity(Enum):
    """How a scale tensor is laid out relative to the weight."""

    PER_TENSOR = "per_tensor"
    PER_CHANNEL = "per_channel"
    BLOCK = "block"


__all__ = ["Granularity"]
