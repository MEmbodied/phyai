"""One-shot contiguous KV-slot allocator on top of a :class:`KVCachePool`.

:class:`StaticCache` is the simplest possible allocator: it owns a
contiguous range ``[base_offset, base_offset + capacity)`` of slots in a
shared :class:`~phyai.cache.kv_cache_pool.KVCachePool`, hands them out
in order via :meth:`allocate`, and resets the cursor on
:meth:`reset`. There is no prefix sharing across rounds, no eviction,
and no LRU bookkeeping — one-shot inference releases its entire cache
slab between requests, so the radix-cache machinery is overkill.

The allocator deliberately returns slot indices as ``int64`` tensors:
that is the dtype PyTorch's ``index_put_`` / ``index_select`` accept
without an extra cast on the hot path. flashinfer's
``paged_kv_indices_buf`` is ``int32``; the runner casts once when
copying into the wrapper's pre-allocated buffer.

Two-or-more :class:`StaticCache` instances may share the same
:class:`KVCachePool` as long as their slot ranges do not overlap — for
example, one for a prefix slab and one for a suffix slab.
"""

from __future__ import annotations

import torch

from phyai.cache.kv_cache_pool import KVCachePool


class StaticCacheError(RuntimeError):
    """Raised when a :class:`StaticCache` operation cannot be served."""


class StaticCache:
    """Contiguous one-shot allocator over a slice of a :class:`KVCachePool`.

    Parameters
    ----------
    pool:
        The backing pool. Slots in ``[base_offset, base_offset + capacity)``
        belong to this :class:`StaticCache`; the caller must ensure no
        other allocator touches the same range.
    base_offset:
        First slot index this allocator owns (inclusive).
    capacity:
        Number of slots in the owned range.

    The cursor starts at ``0`` (relative to ``base_offset``); each
    :meth:`allocate` advances it. :meth:`reset` rewinds the cursor to
    ``0`` so the same physical slots can be reused for the next inference
    round.
    """

    def __init__(
        self,
        pool: KVCachePool,
        *,
        base_offset: int,
        capacity: int,
    ) -> None:
        if base_offset < 0:
            raise ValueError(f"base_offset must be >= 0, got {base_offset}.")
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}.")
        if base_offset + capacity > pool.num_slots:
            raise ValueError(
                f"base_offset({base_offset}) + capacity({capacity}) = "
                f"{base_offset + capacity} exceeds pool.num_slots="
                f"{pool.num_slots}."
            )
        self.pool = pool
        self.base_offset = int(base_offset)
        self.capacity = int(capacity)
        self.cursor = 0

    # ------------------------------------------------------------------ #
    # Properties                                                         #
    # ------------------------------------------------------------------ #

    @property
    def used(self) -> int:
        """Number of slots handed out since the last :meth:`reset`."""
        return self.cursor

    @property
    def remaining(self) -> int:
        return self.capacity - self.cursor

    # ------------------------------------------------------------------ #
    # Allocation                                                         #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Rewind the cursor to 0 (does NOT touch underlying tensors).

        The pool's K/V buffers keep whatever values the previous round
        wrote; that is fine because attention only reads slots whose
        indices appear in the new ``paged_kv_indices`` buffer.
        """
        self.cursor = 0

    def allocate(self, n: int) -> torch.Tensor:
        """Hand out the next ``n`` slot indices as an ``int64`` tensor.

        The returned tensor has shape ``(n,)`` and lives on
        ``pool.device``. Calling sites typically slice / scatter it
        further before handing it to ``write_kv`` /
        ``paged_kv_indices``. After the call ``self.used`` advances by
        ``n``.

        Raises :class:`StaticCacheError` when the request would exceed
        :attr:`capacity` — the caller almost always has a static upper
        bound, so this is a misconfiguration, not a runtime contention.
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}.")
        if self.cursor + n > self.capacity:
            raise StaticCacheError(
                f"cannot allocate {n} slot(s): used={self.cursor}, "
                f"capacity={self.capacity}, remaining="
                f"{self.remaining}."
            )
        start = self.base_offset + self.cursor
        self.cursor += n
        # int64 is required by ``torch.Tensor.index_put_`` /
        # ``index_select``; flashinfer's int32 buffer expects an
        # explicit cast at the runner boundary.
        return torch.arange(
            start, start + n, dtype=torch.int64, device=self.pool.device
        )

    def slot_range(self) -> tuple[int, int]:
        """Return ``(base_offset, base_offset + capacity)`` of this allocator.

        Convenience for callers that need to know the absolute slot
        bounds in the pool (e.g. building static ``paged_kv_indices``
        outside the cache for warmup).
        """
        return self.base_offset, self.base_offset + self.capacity

    def used_slot_range(self) -> tuple[int, int]:
        """Return ``(base_offset, base_offset + used)`` — currently active range."""
        return self.base_offset, self.base_offset + self.cursor

    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"StaticCache(base_offset={self.base_offset}, "
            f"capacity={self.capacity}, used={self.cursor})"
        )


__all__ = ["StaticCache", "StaticCacheError"]
