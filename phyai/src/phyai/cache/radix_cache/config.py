"""Cache configuration dataclasses."""

from __future__ import annotations

import enum
from dataclasses import dataclass

from phyai_ext.radix_cache import PrefixCache


class Modality(str, enum.Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


@dataclass(frozen=True)
class CacheConfig:
    """Builder for a :class:`phyai_ext.radix_cache.PrefixCache` instance.

    The defaults open only the device tier with the LRU policy. Set host /
    disk / remote totals to enable additional tiers; set the corresponding
    ``*_max_pending_units`` to bound async I/O reservations.
    """

    atom_bytes: int
    atoms_per_unit: int
    device_total_units: int
    host_total_units: int = 0
    disk_total_units: int = 0
    remote_total_units: int = 0
    eviction_policy: str = "lru"
    slru_threshold: int = 2
    max_events_buffered: int = 16384
    device_max_pending_units: int = 0
    host_max_pending_units: int = 0
    disk_max_pending_units: int = 0
    remote_max_pending_units: int = 0

    def build(self) -> PrefixCache:
        return PrefixCache(
            atom_bytes=self.atom_bytes,
            atoms_per_unit=self.atoms_per_unit,
            device_total_units=self.device_total_units,
            host_total_units=self.host_total_units,
            disk_total_units=self.disk_total_units,
            remote_total_units=self.remote_total_units,
            eviction_policy=self.eviction_policy,
            slru_threshold=self.slru_threshold,
            max_events_buffered=self.max_events_buffered,
            device_max_pending_units=self.device_max_pending_units,
            host_max_pending_units=self.host_max_pending_units,
            disk_max_pending_units=self.disk_max_pending_units,
            remote_max_pending_units=self.remote_max_pending_units,
        )
