"""phyai.layers.attention — attention ops.

Two flavors are exposed:

* :class:`NoStateAttention` — prefill-only attention with sdpa / eager /
  flashinfer (ragged) backends. No KV cache; Q/K/V come in projected and
  attention goes back out.
* :class:`StaticCachedAttention` — varlen attention that scatters K/V
  into a :class:`~phyai.cache.kv_cache_pool.KVCachePool` and reads it
  back via flashinfer's paged-KV-cache wrapper or an sdpa fallback.
  "Static" pairs lexically with :class:`~phyai.cache.static_cache.StaticCache`,
  the one-shot contiguous slot allocator used by single-request
  inference; both follow the no-eviction policy as opposed to the
  radix-cache facades. The attention module is designed to sit inside
  a captured CUDA graph: the runner plans the wrapper out of the graph
  and the captured ``run()`` reads pre-allocated metadata buffers.

The flashinfer scratch buffer is process-global and per-device; see
:func:`get_global_fi_workspace` for the entry point and the
``PHYAI_FLASHINFER_WORKSPACE_BYTES`` env var for sizing.
"""

from __future__ import annotations

from phyai.layers.attention.no_state_attention import NoStateAttention
from phyai.layers.attention.static_cached_attention import (
    StaticCachedAttention,
    StaticCachedAttnCtx,
)
from phyai.layers.attention.utils import (
    get_global_fi_workspace,
    register_global_fi_workspace,
    resolve_workspace_bytes,
)

__all__ = [
    "NoStateAttention",
    "StaticCachedAttention",
    "StaticCachedAttnCtx",
    "get_global_fi_workspace",
    "register_global_fi_workspace",
    "resolve_workspace_bytes",
]
