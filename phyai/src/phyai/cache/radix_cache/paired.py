"""SWA + Full paired cache.

Two independent ``PrefixCache`` instances share the same atom encoding; SWA
window-out evictions are driven by ``advance(global_step)``, which uses the
native ``step_le`` predicate to drop nodes that have aged out of the window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from phyai_ext.radix_cache import (
    MatchResult,
    OwnedUnits,
    PrefixCache,
    Tier,
)

from .config import CacheConfig


@dataclass(frozen=True)
class PairedCacheConfig:
    full: CacheConfig
    swa: CacheConfig
    window_size: int  # in atoms (typically tokens)


class PairedMatchResult(NamedTuple):
    full: MatchResult
    swa: MatchResult


class PairedCache:
    """Full + SWA dual pool with synchronised inserts and per-step SWA evict."""

    def __init__(self, cfg: PairedCacheConfig) -> None:
        self.cfg = cfg
        self.full: PrefixCache = cfg.full.build()
        self.swa: PrefixCache = cfg.swa.build()
        self.window_size = int(cfg.window_size)

    @property
    def cur_step(self) -> int:
        # Both caches advance in lock-step; either counter answers.
        return self.full.current_step

    def allocate_pair(self, n: int) -> tuple[OwnedUnits, OwnedUnits]:
        full_units = self.full.allocate(Tier.DEVICE, n)
        try:
            swa_units = self.swa.allocate(Tier.DEVICE, n)
        except Exception:
            del full_units  # RAII frees it on scope exit
            raise
        return full_units, swa_units

    def insert_pair(
        self,
        atoms: bytes,
        full_units: OwnedUnits,
        swa_units: OwnedUnits,
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        full_res = self.full.insert(Tier.DEVICE, atoms, full_units)
        swa_res = self.swa.insert(Tier.DEVICE, atoms, swa_units)
        return full_res, swa_res

    def match(self, atoms: bytes) -> PairedMatchResult:
        return PairedMatchResult(full=self.full.match(atoms), swa=self.swa.match(atoms))

    def lock_pair(self, full_node: int, swa_node: int):
        return (
            self.full.lock(Tier.DEVICE, full_node),
            self.swa.lock(Tier.DEVICE, swa_node),
        )

    def advance(self, n_new_atoms: int) -> int:
        """Advance the SWA window by ``n_new_atoms`` and evict any node whose
        last access step is at or before ``cur_step - window_size``.

        Returns the unit count freed in the SWA cache.
        """
        self.full.advance_step(int(n_new_atoms))
        self.swa.advance_step(int(n_new_atoms))
        cutoff = max(0, self.swa.current_step - self.window_size)
        return self.swa.evict_by_named_predicate(Tier.DEVICE, "step_le", int(cutoff))

    def stats(self) -> dict[str, int]:
        return {
            "full_avail": self.full.available(Tier.DEVICE),
            "full_active": self.full.active(Tier.DEVICE),
            "swa_avail": self.swa.available(Tier.DEVICE),
            "swa_active": self.swa.active(Tier.DEVICE),
            "cur_step": int(self.cur_step),
            "window_size": self.window_size,
        }
