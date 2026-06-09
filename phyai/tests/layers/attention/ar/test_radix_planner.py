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


def test_construction_rejects_non_device_tier():
    # Even with the host tier enabled, the device-slot-only planner must reject
    # it: host unit ids are not device-pool slots.
    cache = PrefixCache(
        atom_bytes=4, atoms_per_unit=1, device_total_units=64, host_total_units=64
    )
    pool = KVCachePool(
        num_layers=1,
        num_slots=64,
        num_kv_heads=2,
        head_dim=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="Tier.DEVICE"):
        RadixAttentionPlanner(cache, pool, tier=Tier.HOST)


def test_plan_fresh_sequence_allocates_full_suffix():
    cache, pool, planner = _build()
    seq = RadixSequence(_atoms([10, 11, 12]))
    meta = planner.plan([seq])
    assert seq.prefix_len == 0
    assert seq.suffix_slots.tolist() == [1, 2, 3]
    assert seq.node_ref is None
    assert meta.batch_size == 1
    assert meta.num_query_tokens == 3
    assert meta.cu_seqlens_q.tolist() == [0, 3]
    assert meta.paged_kv_indptr.tolist() == [0, 3]
    assert meta.paged_kv_indices.tolist() == [1, 2, 3]
    assert meta.paged_kv_indices.dtype == torch.int32
    assert meta.write_indices.tolist() == [1, 2, 3]
    assert meta.write_indices.dtype == torch.int64
    assert meta.paged_kv_last_page_len.tolist() == [1]
    assert meta.position_ids.tolist() == [0, 1, 2]


def test_plan_rejects_unaligned_atoms():
    cache, pool, planner = _build()
    with pytest.raises(ValueError, match="page_bytes"):
        planner.plan([RadixSequence(b"abcde")])  # 5 bytes, page_bytes=4


def test_plan_reuse_shares_prefix_slots_non_contiguous():
    cache, pool, planner = _build()
    a = RadixSequence(_atoms([10, 11, 12]))
    planner.plan([a])
    planner.commit([a])  # tree: [10,11,12] -> [1,2,3]
    d = RadixSequence(_atoms([55, 66]))  # decoy consumes [4,5]
    planner.plan([d])
    planner.commit([d])
    c = RadixSequence(_atoms([10, 11, 12, 77, 88]))
    meta = planner.plan([c])
    assert c.prefix_len == 3
    assert c.prefix_slots.tolist() == [1, 2, 3]
    assert c.suffix_slots.tolist() == [6, 7]
    assert meta.paged_kv_indices.tolist() == [1, 2, 3, 6, 7]  # gap 3->6
    assert meta.cu_seqlens_q.tolist() == [0, 2]
    assert meta.paged_kv_indptr.tolist() == [0, 5]
    assert meta.write_indices.tolist() == [6, 7]
    assert meta.position_ids.tolist() == [3, 4]
    assert c.node_ref is not None


def test_plan_fully_cached_sequence_has_no_query_rows():
    cache, pool, planner = _build()
    a = RadixSequence(_atoms([10, 11, 12]))
    planner.plan([a])
    planner.commit([a])
    b = RadixSequence(_atoms([10, 11, 12]))  # identical -> fully cached
    meta = planner.plan([b])
    assert b.prefix_len == 3
    assert b.suffix_len == 0
    assert meta.num_query_tokens == 0
    assert meta.cu_seqlens_q.tolist() == [0, 0]
    assert meta.paged_kv_indices.tolist() == [1, 2, 3]
    assert meta.write_indices.numel() == 0


def test_commit_reuse_grows_tree_at_written_slots():
    cache, pool, planner = _build()
    a = RadixSequence(_atoms([10, 11, 12]))
    planner.plan([a])
    planner.commit([a])
    c = RadixSequence(_atoms([10, 11, 12, 77, 88]))
    planner.plan([c])  # prefix [1,2,3], suffix [4,5]
    planner.commit([c])  # dummy-append insert
    assert c.committed and c.suffix_units is None
    # full sequence now cached; suffix slots are the ones we wrote
    mr = cache.match(_atoms([10, 11, 12, 77, 88]))
    assert int(mr.matched_atoms[int(Tier.DEVICE)]) == 5
    full = torch.from_dlpack(
        cache.collect_units(int(mr.last_node[int(Tier.DEVICE)]), Tier.DEVICE)
    ).tolist()
    assert full == [1, 2, 3, 4, 5]


def test_release_frees_lock_and_uncommitted_units():
    cache, pool, planner = _build()
    a = RadixSequence(_atoms([10, 11, 12]))
    planner.plan([a])
    planner.commit([a])
    avail_before = cache.available(Tier.DEVICE)
    c = RadixSequence(_atoms([10, 11, 12, 77, 88]))
    planner.plan([c])  # locks prefix, allocates 2 suffix
    assert c.node_ref is not None
    assert cache.available(Tier.DEVICE) == avail_before - 2
    planner.release([c])  # not committed -> free suffix + drop lock
    assert c.node_ref is None
    assert c.suffix_units is None
    assert c.released
    assert cache.available(Tier.DEVICE) == avail_before


def test_plan_raises_capacity_error_when_locked_and_full():
    cache, pool, planner = _build(num_slots=8, total_units=4)  # ids 1..3 usable
    a = RadixSequence(_atoms([10, 11, 12]))
    planner.plan([a])
    planner.commit([a])  # avail 0, tree=[10,11,12]->[1,2,3]
    c = RadixSequence(_atoms([10, 11, 12, 20]))  # reuse + 1 new, but no free slot
    with pytest.raises(CacheCapacityError):
        planner.plan([c])


def test_bridge_reexported_from_ar_package():
    import phyai.layers.attention.ar as ar

    assert ar.RadixAttentionPlanner is RadixAttentionPlanner
    assert ar.RadixSequence is RadixSequence


def test_ar_import_does_not_pull_radix_extension():
    """phyai-ext is an optional extra; importing the base AR package must not
    import phyai_ext, so non-[ext] installs can still use the attention
    layers/backends. The radix bridge is re-exported lazily."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import phyai.layers.attention.ar\n"
        "assert 'phyai_ext' not in sys.modules, 'phyai_ext imported eagerly'\n"
        "assert 'phyai.layers.attention.ar.radix' not in sys.modules, 'radix eager'\n"
        "from phyai.layers.attention.ar import RadixAttentionPlanner\n"
        "assert 'phyai_ext' in sys.modules, 'lazy access should load phyai_ext'\n"
        "print('LAZY_OK')\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "LAZY_OK" in res.stdout
