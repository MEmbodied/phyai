"""phyai unified multi-modality / multi-pattern radix-cache layer.

This package provides Python-side composition on top of
``phyai_ext.radix_cache.PrefixCache`` / ``HybridPrefixCache``.

* :class:`MultimodalCache`   — modality routing (text / image / video / audio).
* :class:`PairedCache`       — Full + SWA dual cache.
* :class:`HybridCache`       — Mamba + Attention shared-tree.
* :class:`MultiPatternCache` — generic per-pattern routing.
* :class:`NestedCache`       — composes any of the above (e.g. a
  :class:`MultimodalCache` whose entries are themselves :class:`HybridCache`).

All facades remain pure Python; the heavy lifting (radix walk, eviction,
RAII) stays in C++.
"""

from __future__ import annotations

from .config import CacheConfig, Modality
from .multimodal import MultimodalCache, MultiPatternCache, NestedCache
from .paired import PairedCache, PairedCacheConfig, PairedMatchResult
from .hybrid import HybridCache, HybridCacheConfig
from . import encoding

__all__ = [
    "Modality",
    "CacheConfig",
    "MultimodalCache",
    "MultiPatternCache",
    "NestedCache",
    "PairedCache",
    "PairedCacheConfig",
    "PairedMatchResult",
    "HybridCache",
    "HybridCacheConfig",
    "encoding",
]
