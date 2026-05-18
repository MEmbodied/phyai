"""Per-modality and per-attention-pattern routing facades."""

from __future__ import annotations

from typing import Any, Mapping

from phyai_ext.radix_cache import PrefixCache, Tier

from .config import CacheConfig, Modality


class MultimodalCache:
    """Modality-keyed dictionary of :class:`PrefixCache` instances."""

    def __init__(self, configs: Mapping[Modality, CacheConfig]) -> None:
        self.caches: dict[Modality, PrefixCache] = {
            m: cfg.build() for m, cfg in configs.items()
        }
        self.configs: dict[Modality, CacheConfig] = dict(configs)

    def __getitem__(self, m: Modality) -> PrefixCache:
        return self.caches[m]

    def __contains__(self, m: Modality) -> bool:
        return m in self.caches

    def __iter__(self):  # pragma: no cover
        return iter(self.caches)

    def keys(self):  # pragma: no cover
        return self.caches.keys()

    def items(self):
        return self.caches.items()

    def values(self):
        return self.caches.values()

    def stats(self) -> dict[Modality, dict[str, int]]:
        out: dict[Modality, dict[str, int]] = {}
        for m, c in self.caches.items():
            row: dict[str, int] = {}
            for tier in Tier:
                if c.tier_enabled(tier):
                    row[f"{tier.name.lower()}_avail"] = c.available(tier)
                    row[f"{tier.name.lower()}_active"] = c.active(tier)
            out[m] = row
        return out


class MultiPatternCache:
    """Same shape as :class:`MultimodalCache` but keyed by a free-form name
    such as ``"full"`` / ``"swa"`` / ``"sparse"``.
    """

    def __init__(self, configs: Mapping[str, CacheConfig]) -> None:
        self.caches: dict[str, PrefixCache] = {k: c.build() for k, c in configs.items()}

    def cache_of(self, pattern: str) -> PrefixCache:
        return self.caches[pattern]

    def __getitem__(self, pattern: str) -> PrefixCache:
        return self.caches[pattern]

    def __contains__(self, pattern: str) -> bool:
        return pattern in self.caches

    def keys(self):  # pragma: no cover
        return self.caches.keys()

    def items(self):  # pragma: no cover
        return self.caches.items()


class NestedCache:
    """Generic key→cache router that accepts any cache-like value (PrefixCache,
    HybridCache, PairedCache, MultiPatternCache).

    Used to compose layered routing such as
    ``NestedCache({Modality.TEXT: HybridCache(...), Modality.IMAGE: PrefixCache(...)})``.
    """

    def __init__(self, children: Mapping[Any, Any]) -> None:
        self.children: dict[Any, Any] = dict(children)

    def __getitem__(self, key: Any) -> Any:
        return self.children[key]

    def __contains__(self, key: Any) -> bool:
        return key in self.children

    def keys(self):  # pragma: no cover
        return self.children.keys()

    def items(self):  # pragma: no cover
        return self.children.items()

    def values(self):  # pragma: no cover
        return self.children.values()
