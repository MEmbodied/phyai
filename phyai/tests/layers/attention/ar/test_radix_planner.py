"""CPU planner-logic tests for the radix → AR attention bridge."""

from __future__ import annotations

import struct

import pytest
import torch

from phyai_ext.radix_cache import CacheCapacityError, PrefixCache, Tier

from phyai.cache import KVCachePool
from phyai.layers.attention.ar.radix import RadixAttentionPlanner, RadixSequence


def _atoms(tokens):
    return struct.pack(f"{len(tokens)}i", *tokens)


def _build(num_slots: int = 64, total_units: int = 64):
    cache = PrefixCache(atom_bytes=4, atoms_per_unit=1, device_total_units=total_units)
    pool = KVCachePool(
        num_layers=1,
        num_slots=num_slots,
        num_kv_heads=2,
        head_dim=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    return cache, pool, RadixAttentionPlanner(cache, pool)


def test_sequence_defaults():
    s = RadixSequence(atoms=_atoms([7]))
    assert s.prefix_len == 0
    assert s.suffix_len == 0
    assert s.total_len == 0
    assert not s.committed and not s.released


def test_construction_rejects_unit_size_mismatch():
    cache = PrefixCache(atom_bytes=4, atoms_per_unit=2, device_total_units=64)
    pool = KVCachePool(
        num_layers=1,
        num_slots=64,
        num_kv_heads=2,
        head_dim=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )  # page_size=1 != atoms_per_unit=2
    with pytest.raises(ValueError, match="atoms_per_unit"):
        RadixAttentionPlanner(cache, pool)


def test_construction_rejects_disabled_tier():
    cache, pool, _ = _build()
    with pytest.raises(ValueError, match="not enabled"):
        RadixAttentionPlanner(cache, pool, tier=Tier.HOST)
