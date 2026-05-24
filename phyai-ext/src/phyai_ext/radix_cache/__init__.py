"""phyai radix prefix cache (FFI binding layer).

This module exposes the C++ ``prefix_cache`` / ``hybrid_prefix_cache`` /
``storage_backend`` objects through tvm-ffi-registered globals. The classes
here are thin Python wrappers; they call the registered
``phyai_ext.radix_cache.*`` functions.

Higher-level facades (``MultimodalCache``, ``PairedCache``, ``HybridCache``)
live in ``phyai/radix_cache/`` since they are pure-Python composition.
"""

from __future__ import annotations

import enum
from typing import Any, Callable, Optional, Sequence

import tvm_ffi

from phyai_ext._ffi_api import LIB  # noqa: F401  -- triggers dlopen + registration

__all__ = [
    "Tier",
    "CacheEventKind",
    "CacheEvent",
    "OwnedUnits",
    "NodeRef",
    "CompositeNodeRef",
    "MambaSlot",
    "MatchResult",
    "HybridMatchResult",
    "PrefixCache",
    "HybridPrefixCache",
    "StorageBackend",
    "in_memory_storage_backend",
    "file_storage_backend",
    "xxh3_64",
    "MAX_TIERS",
    # Exception types raised by the FFI layer when the C++ side throws one of
    # the named cache_* errors.
    "CacheCapacityError",
    "CacheUsageError",
    "CacheInvariantError",
]

MAX_TIERS = 4


class Tier(enum.IntEnum):
    DEVICE = 0
    HOST = 1
    DISK = 2
    REMOTE = 3


class CacheEventKind(enum.IntEnum):
    INSERT = 0
    EVICT = 1
    PROMOTE_START = 2
    PROMOTE_DONE = 3
    PROMOTE_FAIL = 4
    DEMOTE_START = 5
    DEMOTE_DONE = 6
    DEMOTE_FAIL = 7
    SPLIT = 8


# ---------------------------------------------------------------------------
# Exception type hierarchy (raised by translating C++ messages — the FFI layer
# delivers these as RuntimeError or ValueError; we wrap user-facing API
# methods to re-raise as the typed subclass).
# ---------------------------------------------------------------------------


class CacheError(Exception):
    """Base class for all radix-cache errors."""


class CacheCapacityError(CacheError, RuntimeError):
    """Raised when an allocation or pending-budget request cannot be served."""


class CacheUsageError(CacheError, ValueError):
    """Raised on misuse — invalid arguments, alignment, disabled tier."""


class CacheInvariantError(CacheError, RuntimeError):
    """Raised when an internal invariant is violated (e.g. tier mismatch on insert)."""


_USAGE_MARKERS = (
    "tier not enabled",
    "must be > 0",
    "atoms not page-aligned",
    "atom_bytes",
    "must be multiple of atoms_per_unit",
    "unknown predicate",
    "unknown eviction policy",
    "tier_total_units",
    "tier_max_pending_units",
    "negative",
    "complete_op: unknown handle",
    "invalid tier index",
    "kv cache is null",
    "src/dst tier",
)
_INVARIANT_MARKERS = (
    "owned_units tier mismatch",
    "underflow",
    "node has no src tier resource",
    "tier mismatch",
    "split_self",
)
_CAPACITY_MARKERS = (
    "not enough units available",
    "pending units cap reached",
)


def _translate(exc: BaseException) -> BaseException:
    msg = str(exc)
    for marker in _CAPACITY_MARKERS:
        if marker in msg:
            return CacheCapacityError(msg)
    for marker in _INVARIANT_MARKERS:
        if marker in msg:
            return CacheInvariantError(msg)
    for marker in _USAGE_MARKERS:
        if marker in msg:
            return CacheUsageError(msg)
    return exc


def _wrap(fn):
    def inner(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (RuntimeError, ValueError) as e:
            translated = _translate(e)
            if translated is e:
                raise
            raise translated from e

    return inner


# ---------------------------------------------------------------------------
# Object wrappers (registered with tvm-ffi)
# ---------------------------------------------------------------------------


@tvm_ffi.register_object("phyai_ext.radix_cache.owned_units")
class OwnedUnits(tvm_ffi.Object):
    """RAII handle for a contiguous batch of allocator unit ids."""

    def tier(self) -> Tier:
        return Tier(int(_F["owned_units.tier"](self)))

    def size(self) -> int:
        return int(_F["owned_units.size"](self))

    def __len__(self) -> int:  # pragma: no cover
        return self.size()

    def ids(self) -> list[int]:
        return [int(x) for x in _F["owned_units.ids"](self)]

    def take_first(self, n: int) -> "OwnedUnits":
        return _F["owned_units.take_first"](self, int(n))

    def take_last(self, n: int) -> "OwnedUnits":
        return _F["owned_units.take_last"](self, int(n))

    def append(self, other: "OwnedUnits") -> None:
        _F["owned_units.append"](self, other)


@tvm_ffi.register_object("phyai_ext.radix_cache.node_ref")
class NodeRef(tvm_ffi.Object):
    """RAII pin against eviction. Drop the reference to release."""

    def node_handle(self) -> int:
        return int(_F["node_ref.node"](self))

    def tier(self) -> Tier:
        return Tier(int(_F["node_ref.tier"](self)))

    def valid(self) -> bool:
        return bool(_F["node_ref.valid"](self))


@tvm_ffi.register_object("phyai_ext.radix_cache.composite_node_ref")
class CompositeNodeRef(tvm_ffi.Object):
    """RAII pin across multiple tiers simultaneously."""

    def size(self) -> int:
        return int(_F["composite_node_ref.size"](self))

    def __len__(self) -> int:  # pragma: no cover
        return self.size()

    def valid(self) -> bool:
        return bool(_F["composite_node_ref.valid"](self))

    def node_handle(self) -> int:
        return int(_F["composite_node_ref.node"](self))


@tvm_ffi.register_object("phyai_ext.radix_cache.mamba_slot")
class MambaSlot(tvm_ffi.Object):
    """RAII handle for a single mamba state slot."""

    def index(self) -> int:
        return int(_F["mamba_slot.index"](self))

    def valid(self) -> bool:
        return bool(_F["mamba_slot.valid"](self))


@tvm_ffi.register_object("phyai_ext.radix_cache.cache_event")
class CacheEvent(tvm_ffi.Object):
    @property
    def kind_enum(self) -> CacheEventKind:
        return CacheEventKind(int(self.kind))

    @property
    def tier_from_enum(self) -> Tier:
        return Tier(int(self.tier_from))

    @property
    def tier_to_enum(self) -> Tier:
        return Tier(int(self.tier_to))


@tvm_ffi.register_object("phyai_ext.radix_cache.prefix_cache")
class _PrefixCacheObj(tvm_ffi.Object):
    pass


@tvm_ffi.register_object("phyai_ext.radix_cache.hybrid_prefix_cache")
class _HybridPrefixCacheObj(tvm_ffi.Object):
    pass


@tvm_ffi.register_object("phyai_ext.radix_cache.storage_backend")
class _StorageBackendObj(tvm_ffi.Object):
    pass


# ---------------------------------------------------------------------------
# Lazy lookup map for the C++ globals.
# ---------------------------------------------------------------------------


class _Funcs:
    _cache: dict[str, tvm_ffi.Function] = {}

    def __getitem__(self, suffix: str) -> tvm_ffi.Function:
        if suffix not in self._cache:
            self._cache[suffix] = tvm_ffi.get_global_func(
                f"phyai_ext.radix_cache.{suffix}"
            )
        return self._cache[suffix]


_F = _Funcs()


# ---------------------------------------------------------------------------
# MatchResult / HybridMatchResult
# ---------------------------------------------------------------------------


class MatchResult:
    """Result of ``PrefixCache.match`` (per-tier views).

    The actual unit-id list is intentionally NOT returned here — fetch it on
    demand with :meth:`PrefixCache.collect_units` (zero-copy DLPack tensor)
    when the caller actually needs to feed it to a kernel.
    """

    __slots__ = ("last_node", "matched_atoms")

    def __init__(self, last_node: list[int], matched_atoms: list[int]) -> None:
        self.last_node = last_node
        self.matched_atoms = matched_atoms

    @classmethod
    def _from_ffi(cls, raw: Sequence[Any]) -> "MatchResult":
        last_nodes_arr, matched_atoms_arr = raw
        last_node = [int(x) for x in last_nodes_arr]
        matched_atoms = [int(x) for x in matched_atoms_arr]
        return cls(last_node, matched_atoms)


class HybridMatchResult:
    __slots__ = ("kv", "last_mamba_node", "mamba_branching_atoms")

    def __init__(
        self, kv: MatchResult, last_mamba_node: int, mamba_branching_atoms: int
    ) -> None:
        self.kv = kv
        self.last_mamba_node = last_mamba_node
        self.mamba_branching_atoms = mamba_branching_atoms


# ---------------------------------------------------------------------------
# PrefixCache
# ---------------------------------------------------------------------------


class PrefixCache:
    """Modality-agnostic byte-erased radix prefix cache.

    All atom data passes through as ``bytes``-like objects. The cache holds
    only unit ids — actual KV/embedding tensors are owned by the caller.
    """

    def __init__(
        self,
        atom_bytes: int,
        atoms_per_unit: int,
        device_total_units: int,
        host_total_units: int = 0,
        disk_total_units: int = 0,
        remote_total_units: int = 0,
        eviction_policy: str = "lru",
        slru_threshold: int = 2,
        max_events_buffered: int = 16384,
        device_max_pending_units: int = 0,
        host_max_pending_units: int = 0,
        disk_max_pending_units: int = 0,
        remote_max_pending_units: int = 0,
    ) -> None:
        self._atom_bytes = int(atom_bytes)
        self._atoms_per_unit = int(atoms_per_unit)
        self._page_bytes = self._atom_bytes * self._atoms_per_unit
        tier_units = [
            int(device_total_units),
            int(host_total_units),
            int(disk_total_units),
            int(remote_total_units),
        ]
        max_pending = [
            int(device_max_pending_units),
            int(host_max_pending_units),
            int(disk_max_pending_units),
            int(remote_max_pending_units),
        ]
        self._impl = _wrap(_F["prefix_cache.create"])(
            self._atom_bytes,
            self._atoms_per_unit,
            tier_units,
            max_pending,
            str(eviction_policy),
            int(slru_threshold),
            int(max_events_buffered),
        )

    # ── Configuration ────────────────────────────────────────────────────
    @property
    def atom_bytes(self) -> int:
        return self._atom_bytes

    @property
    def atoms_per_unit(self) -> int:
        return self._atoms_per_unit

    @property
    def page_bytes(self) -> int:
        return self._page_bytes

    @property
    def policy_name(self) -> str:
        return str(_F["prefix_cache.policy_name"](self._impl))

    @property
    def slru_threshold(self) -> int:
        return int(_F["prefix_cache.slru_threshold"](self._impl))

    def tier_enabled(self, tier: Tier) -> bool:
        return bool(_F["prefix_cache.tier_enabled"](self._impl, int(tier)))

    @property
    def impl(self) -> _PrefixCacheObj:
        return self._impl

    # ── Query / write ────────────────────────────────────────────────────
    def match(self, atoms: bytes | bytearray | memoryview) -> MatchResult:
        raw = _wrap(_F["prefix_cache.match"])(self._impl, _to_bytes(atoms))
        return MatchResult._from_ffi(raw)

    def collect_units(self, last_node: int, tier: Tier) -> Any:
        """Return a zero-copy CPU int32 ``tvm_ffi.Tensor`` of every unit id
        on the path root -> ``last_node`` for the given tier.

        The returned object supports the ``__dlpack__`` protocol; consume it
        via ``torch.from_dlpack(...)`` / ``numpy.from_dlpack(...)``. The
        tensor owns its buffer and is safely reference-counted across the
        FFI boundary — there is no shared lifetime with the cache, so it is
        valid even if a subsequent ``ensure_capacity`` evicts the source
        node.
        """
        return _F["prefix_cache.collect_units"](self._impl, int(last_node), int(tier))

    def insert(
        self,
        tier: Tier,
        atoms: bytes | bytearray | memoryview,
        units: OwnedUnits,
    ) -> tuple[int, int, int]:
        out = _wrap(_F["prefix_cache.insert"])(
            self._impl, int(tier), _to_bytes(atoms), units
        )
        return int(out[0]), int(out[1]), int(out[2])

    def allocate(self, tier: Tier, n: int) -> OwnedUnits:
        return _wrap(_F["prefix_cache.allocate"])(self._impl, int(tier), int(n))

    def lock(self, tier: Tier, last_node: int) -> NodeRef:
        return _F["prefix_cache.lock"](self._impl, int(tier), int(last_node))

    def lock_multi(self, last_node: int, tiers: Sequence[Tier]) -> CompositeNodeRef:
        return _F["prefix_cache.lock_multi"](
            self._impl, int(last_node), [int(t) for t in tiers]
        )

    def ensure_capacity(
        self, tier: Tier, n: int, promote_to: Optional[Tier] = None
    ) -> None:
        promote_int = -1 if promote_to is None else int(promote_to)
        _wrap(_F["prefix_cache.ensure_capacity"])(
            self._impl, int(tier), int(n), int(promote_int)
        )

    def evict_by_predicate(
        self, tier: Tier, callback: Callable[[int, int, int, int, int], bool]
    ) -> int:
        """Evict any node where ``callback(node_handle, depth, hits, step, prio)`` returns True.

        Use :meth:`evict_by_named_predicate` for common predicates — it avoids
        the Python callback overhead for large candidate sets.
        """
        return int(
            _F["prefix_cache.evict_by_predicate"](self._impl, int(tier), callback)
        )

    def evict_by_named_predicate(self, tier: Tier, predicate: str, arg: int) -> int:
        """Evict using a built-in C++ predicate for speed.

        Recognised predicates:

        * ``"step_le"`` — evict where ``last_access_step <= arg``
        * ``"step_lt"`` — evict where ``last_access_step <  arg``
        * ``"hits_lt"`` — evict where ``hit_count <  arg``
        * ``"priority_le"`` — evict where ``user_priority <= arg``
        * ``"age_ns_ge"`` — evict where access age (now-last_access) ``>= arg``
        """
        return int(
            _wrap(_F["prefix_cache.evict_by_named_predicate"])(
                self._impl, int(tier), str(predicate), int(arg)
            )
        )

    def available(self, tier: Tier) -> int:
        return int(_F["prefix_cache.available"](self._impl, int(tier)))

    def total(self, tier: Tier) -> int:
        return int(_F["prefix_cache.total"](self._impl, int(tier)))

    def active(self, tier: Tier) -> int:
        return int(_F["prefix_cache.active"](self._impl, int(tier)))

    def pending_units(self, tier: Tier) -> int:
        return int(_F["prefix_cache.pending_units"](self._impl, int(tier)))

    def max_pending_units(self, tier: Tier) -> int:
        return int(_F["prefix_cache.max_pending_units"](self._impl, int(tier)))

    # ── Async ops ────────────────────────────────────────────────────────
    def start_demote(self, node: int, src: Tier, dst: Tier) -> int:
        return int(
            _wrap(_F["prefix_cache.start_demote"])(
                self._impl, int(node), int(src), int(dst)
            )
        )

    def start_promote(self, node: int, src: Tier, dst: Tier) -> int:
        return int(
            _wrap(_F["prefix_cache.start_promote"])(
                self._impl, int(node), int(src), int(dst)
            )
        )

    def complete_op(self, handle: int, success: bool) -> None:
        _wrap(_F["prefix_cache.complete_op"])(self._impl, int(handle), bool(success))

    def wait_op(self, handle: int, timeout_ms: int = 0) -> bool:
        """Block until ``handle`` completes or ``timeout_ms`` milliseconds elapse.

        Returns ``True`` if the op completed (regardless of success), ``False``
        on timeout.
        """
        return bool(
            _F["prefix_cache.wait_op"](self._impl, int(handle), int(timeout_ms))
        )

    def fail_all_inflight(self) -> int:
        """Mark every in-flight op as failed. Returns the count.

        Use after a process restart that finds leftover ops from the previous
        run — the data they were carrying is no longer valid.
        """
        return int(_F["prefix_cache.fail_all_inflight"](self._impl))

    def inflight_ops(self) -> list[int]:
        return [int(h) for h in _F["prefix_cache.inflight_ops"](self._impl)]

    # ── Eviction observers ───────────────────────────────────────────────
    def add_evict_observer(self, callback: Callable[[int, int], None]) -> int:
        """Register ``callback(node_handle, tier_int)``; returns observer id."""
        return int(_F["prefix_cache.add_evict_observer"](self._impl, callback))

    def remove_evict_observer(self, observer_id: int) -> bool:
        return bool(
            _F["prefix_cache.remove_evict_observer"](self._impl, int(observer_id))
        )

    # ── Events ───────────────────────────────────────────────────────────
    def take_events(self) -> list[CacheEvent]:
        return list(_F["prefix_cache.take_events"](self._impl))

    @property
    def dropped_events(self) -> int:
        return int(_F["prefix_cache.dropped_events"](self._impl))

    # ── Step counter / hashes ────────────────────────────────────────────
    @property
    def current_step(self) -> int:
        return int(_F["prefix_cache.current_step"](self._impl))

    def advance_step(self, n: int) -> None:
        _F["prefix_cache.advance_step"](self._impl, int(n))

    def touch_step(self, node_handle: int) -> None:
        _F["prefix_cache.touch_step"](self._impl, int(node_handle))

    def node_path_hash(self, node_handle: int) -> int:
        """Return the 64-bit content hash of the path root -> node_handle.

        Useful as a content-addressed key for storage backends.
        """
        return int(_F["prefix_cache.node_path_hash"](self._impl, int(node_handle))) & (
            (1 << 64) - 1
        )


# ---------------------------------------------------------------------------
# HybridPrefixCache
# ---------------------------------------------------------------------------


class HybridPrefixCache:
    """Mamba + Attention shared-tree cache."""

    def __init__(self, kv: PrefixCache, num_mamba_slots: int) -> None:
        self._kv = kv
        self._impl = _F["hybrid_prefix_cache.create"](kv.impl, int(num_mamba_slots))

    @property
    def kv(self) -> PrefixCache:
        return self._kv

    def match(self, atoms: bytes | bytearray | memoryview) -> HybridMatchResult:
        raw = _F["hybrid_prefix_cache.match"](self._impl, _to_bytes(atoms))
        kv_pkg, last_mamba, branching = raw
        return HybridMatchResult(
            MatchResult._from_ffi(kv_pkg), int(last_mamba), int(branching)
        )

    def allocate_mamba_slot(self) -> Optional[MambaSlot]:
        return _F["hybrid_prefix_cache.allocate_mamba_slot"](self._impl)

    def attach_mamba(self, node_handle: int, slot: MambaSlot) -> None:
        _F["hybrid_prefix_cache.attach_mamba"](self._impl, int(node_handle), slot)

    def detach_mamba(self, node_handle: int) -> Optional[MambaSlot]:
        return _F["hybrid_prefix_cache.detach_mamba"](self._impl, int(node_handle))

    def find_last_mamba_node(self, from_handle: int) -> int:
        return int(
            _F["hybrid_prefix_cache.find_last_mamba_node"](self._impl, int(from_handle))
        )

    def ensure_mamba_capacity_by_evict(self, n: int) -> bool:
        return bool(
            _F["hybrid_prefix_cache.ensure_mamba_capacity_by_evict"](self._impl, int(n))
        )

    @property
    def available_slots(self) -> int:
        return int(_F["hybrid_prefix_cache.available_slots"](self._impl))

    @property
    def total_slots(self) -> int:
        return int(_F["hybrid_prefix_cache.total_slots"](self._impl))

    @property
    def active_slots(self) -> int:
        return int(_F["hybrid_prefix_cache.active_slots"](self._impl))


# ---------------------------------------------------------------------------
# Storage backends
# ---------------------------------------------------------------------------


class StorageBackend:
    """Python wrapper for a content-addressed storage backend.

    Use :func:`in_memory_storage_backend` or :func:`file_storage_backend` to
    construct a backend; this class is the typed Python view over the C++
    object so the user can pass it around and inspect it.
    """

    def __init__(self, impl: _StorageBackendObj) -> None:
        self._impl = impl

    @property
    def impl(self) -> _StorageBackendObj:
        return self._impl

    @property
    def name(self) -> str:
        return str(_F["storage_backend.name"](self._impl))

    @property
    def unit_bytes(self) -> int:
        return int(_F["storage_backend.unit_bytes"](self._impl))

    def drain(self) -> None:
        _F["storage_backend.drain"](self._impl)

    # In-memory introspection hooks (no-op for non-in-memory backends).
    def contains(self, key: int) -> bool:
        return bool(_F["storage_backend.in_memory_contains"](self._impl, int(key)))

    def entries(self) -> int:
        return int(_F["storage_backend.in_memory_entries"](self._impl))

    # Synchronous round-trip helpers (drive the backend without an event loop).
    def write_sync(self, op_handle: int, key: int, ids: Sequence[int]) -> bool:
        return bool(
            _F["storage_backend.write_sync"](
                self._impl, int(op_handle), int(key), [int(x) for x in ids]
            )
        )

    def read_sync(self, op_handle: int, key: int, ids: Sequence[int]) -> bool:
        return bool(
            _F["storage_backend.read_sync"](
                self._impl, int(op_handle), int(key), [int(x) for x in ids]
            )
        )


def in_memory_storage_backend(unit_bytes: int) -> StorageBackend:
    return StorageBackend(_F["storage_backend.in_memory"](int(unit_bytes)))


def file_storage_backend(path: str, unit_bytes: int) -> StorageBackend:
    return StorageBackend(_F["storage_backend.file"](str(path), int(unit_bytes)))


# ---------------------------------------------------------------------------
# Helpers / utilities
# ---------------------------------------------------------------------------


def _to_bytes(x: bytes | bytearray | memoryview) -> bytes:
    if isinstance(x, bytes):
        return x
    if isinstance(x, (bytearray, memoryview)):
        return bytes(x)
    return bytes(x)


def xxh3_64(data: bytes | bytearray | memoryview) -> int:
    """Compute the 64-bit xxh3 hash of ``data``."""
    return int(_F["xxh3_64"](_to_bytes(data))) & ((1 << 64) - 1)


# Tree-node telemetry helpers (handles do not extend lifetime; use only
# between match() and lock()/insert()).


def tree_node_depth_in_atoms(node_handle: int) -> int:
    return int(_F["tree_node.depth_in_atoms"](int(node_handle)))


def tree_node_atom_count(node_handle: int) -> int:
    return int(_F["tree_node.atom_count"](int(node_handle)))


def tree_node_hit_count(node_handle: int) -> int:
    return int(_F["tree_node.hit_count"](int(node_handle)))


def tree_node_last_access_step(node_handle: int) -> int:
    return int(_F["tree_node.last_access_step"](int(node_handle)))


def tree_node_user_priority(node_handle: int) -> int:
    return int(_F["tree_node.user_priority"](int(node_handle)))


def tree_node_set_user_priority(node_handle: int, priority: int) -> None:
    _F["tree_node.set_user_priority"](int(node_handle), int(priority))


def tree_node_has_mamba(node_handle: int) -> bool:
    return bool(_F["tree_node.has_mamba"](int(node_handle)))


def tree_node_mamba_index(node_handle: int) -> int:
    return int(_F["tree_node.mamba_index"](int(node_handle)))
