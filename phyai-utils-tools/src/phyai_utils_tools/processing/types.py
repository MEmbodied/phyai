"""Feature-type and normalization-mode enums — byte-exact with lerobot.

The processor steps serialize to / deserialize from lerobot-format
``policy_*processor.json``. The ``norm_map`` in those files keys a feature
*bucket* (:class:`FeatureType`) to a :class:`NormalizationMode`, and the
``features`` block records each feature's ``{type, shape}``. The enum *values*
here must match lerobot's exactly (``lerobot/configs/types.py``) so a config
round-trips through ``json`` without translation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FeatureType(str, Enum):
    """Feature buckets used by the normalizer's ``norm_map`` (lerobot-exact)."""

    STATE = "STATE"
    VISUAL = "VISUAL"
    ENV = "ENV"
    ACTION = "ACTION"
    REWARD = "REWARD"
    LANGUAGE = "LANGUAGE"


class NormalizationMode(str, Enum):
    """Normalization modes the normalizer steps support (lerobot-exact)."""

    MIN_MAX = "MIN_MAX"  # to/from [-1, 1] via min/max
    MEAN_STD = "MEAN_STD"  # (x - mean) / std
    IDENTITY = "IDENTITY"  # passthrough
    QUANTILES = "QUANTILES"  # to/from [-1, 1] via q01/q99
    QUANTILE10 = "QUANTILE10"  # to/from [-1, 1] via q10/q90


@dataclass(frozen=True)
class PolicyFeature:
    """A feature's type bucket + shape, mirroring lerobot's ``PolicyFeature``.

    Serialized inside the normalizer's ``features`` block as
    ``{"type": <FeatureType value>, "shape": [...]}``.
    """

    type: FeatureType
    shape: tuple[int, ...]


__all__ = ["FeatureType", "NormalizationMode", "PolicyFeature"]
