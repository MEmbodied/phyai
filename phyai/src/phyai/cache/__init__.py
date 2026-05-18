"""phyai.cache — KV-cache primitives.

The package layout::

    phyai/cache/
        kv_cache_pool.py   # KVCachePool: per-layer K/V tensor pool
        static_cache.py    # StaticCache: one-shot contiguous allocator
        radix_cache/       # RadixCache facades (multimodal / paired / hybrid)

:class:`KVCachePool` is the storage primitive: a passive container
holding ``(num_slots, page_size, num_kv_heads, head_dim)`` K and V
tensors per transformer layer. Allocation policy lives in higher-level
wrappers — :class:`StaticCache` for one-shot inference (allocate, fill,
attend, release) and the :mod:`phyai.cache.radix_cache` facades for
prefix-shared LLM serving.

The buffer layout matches flashinfer's paged-KV-cache convention so
``KVCachePool.kv_buffer(layer_id)`` can be passed directly to
``BatchPrefillWithPagedKVCacheWrapper.run``.
"""

from __future__ import annotations

from phyai.cache.kv_cache_pool import KVCachePool
from phyai.cache.static_cache import StaticCache, StaticCacheError

# Radix-cache facades stay accessible at the cache top level for callers
# that compose paged storage with prefix sharing.
from phyai.cache.radix_cache import (
    CacheConfig,
    HybridCache,
    HybridCacheConfig,
    Modality,
    MultiPatternCache,
    MultimodalCache,
    NestedCache,
    PairedCache,
    PairedCacheConfig,
    PairedMatchResult,
    encoding,
)


__all__ = [
    # Storage / one-shot path.
    "KVCachePool",
    "StaticCache",
    "StaticCacheError",
    # Radix-cache facades (LLM serving with prefix sharing).
    "CacheConfig",
    "Modality",
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
