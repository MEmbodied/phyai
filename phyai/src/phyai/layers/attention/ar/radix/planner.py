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
from phyai.layers.attention.ar.base import ARAttnMetadata
from phyai.layers.attention.ar.radix.sequence import RadixSequence
from phyai.layers.attention.enums import AttnLayout, AttnMode


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
        if tier != Tier.DEVICE:
            raise ValueError(
                f"RadixAttentionPlanner is device-slot-only; tier must be "
                f"Tier.DEVICE, got {tier!r}. Its unit ids are used directly as "
                f"KVCachePool slot indices, so a non-device tier would index "
                f"the device pool with foreign ids."
            )
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

    # ------------------------------------------------------------------ #
    # Plan                                                               #
    # ------------------------------------------------------------------ #

    def plan(
        self,
        sequences: list[RadixSequence],
        *,
        mode: AttnMode = AttnMode.PREFILL,
    ) -> ARAttnMetadata:
        """Match + allocate every sequence and assemble one ARAttnMetadata.

        Mutates each :class:`RadixSequence` with its prefix/suffix slot
        split and the radix handles (lock + suffix units) that ``commit`` /
        ``release`` consume. Query rows are the per-sequence suffixes.
        """
        if not sequences:
            raise ValueError("plan() requires at least one RadixSequence.")

        suffix_lens: list[int] = []
        total_lens: list[int] = []
        indices_parts: list[torch.Tensor] = []
        write_parts: list[torch.Tensor] = []
        pos_parts: list[torch.Tensor] = []

        for seq in sequences:
            if seq.released:
                raise ValueError("cannot plan a released RadixSequence.")
            num_units = self._num_units(seq.atoms)
            prefix_units, prefix_slots, node_ref = self._match_prefix(
                seq.atoms, num_units
            )
            seq.node_ref = node_ref

            suffix_units = num_units - prefix_units
            if suffix_units > 0:
                self.cache.ensure_capacity(self.tier, suffix_units)
                owned = self.cache.allocate(self.tier, suffix_units)
                suffix_slots = torch.as_tensor(
                    list(owned.ids()), dtype=torch.int64, device=self.device
                )
                seq.suffix_units = owned
            else:
                suffix_slots = torch.empty(0, dtype=torch.int64, device=self.device)
                seq.suffix_units = None

            if int(suffix_slots.numel()) and int(suffix_slots.max()) >= self.num_slots:
                raise ValueError(
                    f"allocated slot id {int(suffix_slots.max())} >= "
                    f"kv_pool.num_slots={self.num_slots}; pool smaller than the "
                    f"cache's device tier."
                )

            seq.prefix_len = prefix_units
            seq.prefix_slots = prefix_slots
            seq.suffix_slots = suffix_slots
            seq.committed = False

            s = int(suffix_slots.numel())
            suffix_lens.append(s)
            total_lens.append(prefix_units + s)
            indices_parts.append(torch.cat([prefix_slots, suffix_slots]))
            write_parts.append(suffix_slots)
            pos_parts.append(
                torch.arange(
                    prefix_units,
                    prefix_units + s,
                    dtype=torch.int32,
                    device=self.device,
                )
            )

        return self._assemble(
            len(sequences),
            suffix_lens,
            total_lens,
            indices_parts,
            write_parts,
            pos_parts,
            mode,
        )

    # ------------------------------------------------------------------ #
    # Commit / release                                                   #
    # ------------------------------------------------------------------ #

    def commit(self, sequences: list[RadixSequence]) -> None:
        """Seed fully-new sequences into the radix tree for future reuse.

        Call after the forward has scattered the suffix K/V into the pool. Only
        sequences with **no cached prefix** (``overlap == 0``) are inserted: the
        whole sequence is new, so the suffix units ``plan`` allocated are exactly
        the units the C++ ``insert`` needs.

        A sequence that **reused** a cached prefix is left uncommitted (a no-op).
        Growing an existing prefix would mean handing ``insert`` a full-length
        unit list (it frees the matched overlap), i.e. transiently reserving
        ``overlap`` extra slots while the prefix is still locked — which can
        evict useful entries or fail under capacity pressure for no net gain. The
        reuse already delivered the compute savings; ``release`` frees the
        uncommitted suffix. Extending a cached prefix awaits a C++
        ``insert_suffix_from_node``.
        """
        for seq in sequences:
            if seq.committed or seq.suffix_units is None:
                continue
            num_units = self._num_units(seq.atoms)
            overlap = min(
                int(self.cache.match(seq.atoms).matched_atoms[self._tier_i])
                // self.atoms_per_unit,
                num_units,
            )
            if overlap > 0:
                continue  # reused/already-cached prefix — see docstring
            self.cache.insert(self.tier, seq.atoms, seq.suffix_units)
            seq.suffix_units = None
            seq.committed = True

    def release(self, sequences: list[RadixSequence]) -> None:
        """Drop prefix locks and free any uncommitted suffix units.

        Idempotent. After release the matched prefix becomes evictable and
        any planned-but-uncommitted suffix returns to the allocator.
        """
        for seq in sequences:
            seq.node_ref = None  # RAII unlock
            if not seq.committed:
                seq.suffix_units = None  # RAII free
            seq.released = True

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _num_units(self, atoms: bytes) -> int:
        n = len(atoms)
        if n % self.page_bytes != 0:
            raise ValueError(
                f"atoms length {n} is not a multiple of page_bytes="
                f"{self.page_bytes}; page-align before planning."
            )
        return n // self.page_bytes

    def _match_prefix(self, atoms: bytes, num_units: int):
        empty = torch.empty(0, dtype=torch.int64, device=self.device)
        if num_units == 0:
            return 0, empty, None
        mr = self.cache.match(atoms)
        prefix_units = int(mr.matched_atoms[self._tier_i]) // self.atoms_per_unit
        prefix_units = min(prefix_units, num_units)
        if prefix_units == 0:
            return 0, empty, None
        node = int(mr.last_node[self._tier_i])
        node_ref = self.cache.lock(self.tier, node)
        slots = torch.from_dlpack(self.cache.collect_units(node, self.tier)).to(
            device=self.device, dtype=torch.int64
        )
        if int(slots.numel()) != prefix_units:
            raise RuntimeError(
                f"collect_units returned {int(slots.numel())} slots but match "
                f"reported {prefix_units} prefix units."
            )
        return prefix_units, slots, node_ref

    def _assemble(
        self, B, suffix_lens, total_lens, indices_parts, write_parts, pos_parts, mode
    ):
        suffix_t = torch.tensor(suffix_lens, dtype=torch.int32, device=self.device)
        total_t = torch.tensor(total_lens, dtype=torch.int32, device=self.device)
        cu_seqlens_q = torch.zeros(B + 1, dtype=torch.int32, device=self.device)
        cu_seqlens_q[1:] = torch.cumsum(suffix_t, 0)
        paged_kv_indptr = torch.zeros(B + 1, dtype=torch.int32, device=self.device)
        paged_kv_indptr[1:] = torch.cumsum(total_t, 0)
        return ARAttnMetadata(
            mode=mode,
            layout=AttnLayout.RAGGED_3D,
            batch_size=B,
            num_query_tokens=int(sum(suffix_lens)),
            cu_seqlens_q=cu_seqlens_q,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=torch.cat(indices_parts).to(torch.int32),
            paged_kv_last_page_len=(total_t > 0).to(torch.int32),
            write_indices=torch.cat(write_parts).to(torch.int64),
            position_ids=torch.cat(pos_parts).to(torch.int32),
        )


__all__ = ["RadixAttentionPlanner"]
