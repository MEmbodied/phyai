"""Hybrid Mamba + Attention cache facade.

Wraps a single ``HybridPrefixCache`` and adds Python-side layer routing.
Mamba layers consume the mamba slot at the matched node; attention layers
consume the KV unit ids of the same tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from phyai_ext.radix_cache import (
    HybridMatchResult,
    HybridPrefixCache,
    MambaSlot,
    PrefixCache,
)

from .config import CacheConfig


@dataclass(frozen=True)
class HybridCacheConfig:
    kv: CacheConfig
    num_mamba_slots: int
    layer_kinds: tuple[Literal["attn", "mamba"], ...]


class HybridCache:
    """Mamba + Attention shared-tree cache with per-layer routing."""

    def __init__(self, cfg: HybridCacheConfig) -> None:
        kv = cfg.kv.build()
        self.hybrid = HybridPrefixCache(kv, cfg.num_mamba_slots)
        self.layer_kinds = tuple(cfg.layer_kinds)

    @property
    def kv(self) -> PrefixCache:
        return self.hybrid.kv

    @property
    def num_layers(self) -> int:
        return len(self.layer_kinds)

    def is_mamba_layer(self, layer_id: int) -> bool:
        return self.layer_kinds[layer_id] == "mamba"

    def match(self, atoms: bytes) -> HybridMatchResult:
        return self.hybrid.match(atoms)

    def allocate_mamba_slot(self) -> Optional[MambaSlot]:
        return self.hybrid.allocate_mamba_slot()

    def attach_mamba(self, node_handle: int, slot: MambaSlot) -> None:
        self.hybrid.attach_mamba(node_handle, slot)

    def detach_mamba(self, node_handle: int) -> Optional[MambaSlot]:
        return self.hybrid.detach_mamba(node_handle)

    def ensure_mamba_capacity_by_evict(self, n: int) -> bool:
        return self.hybrid.ensure_mamba_capacity_by_evict(n)

    @property
    def available_slots(self) -> int:
        return self.hybrid.available_slots

    @property
    def total_slots(self) -> int:
        return self.hybrid.total_slots

    @property
    def active_slots(self) -> int:
        return self.hybrid.active_slots
