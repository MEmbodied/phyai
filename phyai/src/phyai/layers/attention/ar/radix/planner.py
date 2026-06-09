"""Radix-cache → :class:`ARAttnMetadata` bridge for the AR attention stack.

:class:`RadixAttentionPlanner` is the per-request lifecycle glue between
:class:`phyai_ext.radix_cache.PrefixCache` and the paged AR attention
backends. It implements the prefill-with-prefix contract: match the
longest cached prefix, reuse those slots, allocate only the uncached
suffix, build an :class:`ARAttnMetadata` whose query is the suffix tokens
(KV = prefix + suffix), and — after the forward writes the suffix K/V —
``insert`` the suffix back into the tree for future reuse.

Model- and encoding-agnostic: callers hand it :class:`RadixSequence`
objects carrying pre-encoded ``atoms`` (see
:mod:`phyai.cache.radix_cache.encoding`). pi0.5 keeps its ``StaticCache``
path; this is the foundation a radix-enabled AR runner (e.g. cosmos)
builds on.
"""

from __future__ import annotations

import torch

from phyai_ext.radix_cache import PrefixCache, Tier

from phyai.cache.kv_cache_pool import KVCachePool


class RadixAttentionPlanner:
    """Builds radix-prefix-reusing :class:`ARAttnMetadata` for AR attention.

    Parameters
    ----------
    cache:
        A built :class:`phyai_ext.radix_cache.PrefixCache` (e.g. via
        :class:`phyai.cache.radix_cache.CacheConfig`). ``cache.atoms_per_unit``
        must equal ``kv_pool.page_size`` so one radix unit maps to one slot.
    kv_pool:
        The KV slot pool the unit ids index into — used for device + slot
        bounds. The planner never reads/writes K/V (the backend does).
    tier:
        Cache tier to match/allocate on. Device tier only for now.
    """

    def __init__(
        self,
        cache: PrefixCache,
        kv_pool: KVCachePool,
        *,
        tier: Tier = Tier.DEVICE,
    ) -> None:
        if cache.atoms_per_unit != kv_pool.page_size:
            raise ValueError(
                f"cache.atoms_per_unit ({cache.atoms_per_unit}) must equal "
                f"kv_pool.page_size ({kv_pool.page_size}) so one radix unit "
                f"maps to one KV pool slot."
            )
        if not cache.tier_enabled(tier):
            raise ValueError(f"cache tier {tier!r} is not enabled.")
        self.cache = cache
        self.kv_pool = kv_pool
        self.tier = tier
        self._tier_i = int(tier)
        self.page_bytes = int(cache.page_bytes)
        self.atoms_per_unit = int(cache.atoms_per_unit)
        self.device = kv_pool.device
        self.num_slots = int(kv_pool.num_slots)


__all__ = ["RadixAttentionPlanner"]
