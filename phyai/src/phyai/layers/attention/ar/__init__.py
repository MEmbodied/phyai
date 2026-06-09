"""`phyai.layers.attention.ar` — autoregressive LM-side paged attention.

Layer + backends + types for the LM side of a model. K/V are scattered
into a :class:`~phyai.cache.kv_cache_pool.KVCachePool` then read back
via flashinfer's paged kernel (or an eager contiguous-slab fallback).

Backends: ``"flashinfer"`` (default, paged-KV with cuda-graph capture
support) and ``"eager"`` (CPU/CI reference).

Sibling stacks: :mod:`phyai.layers.attention.attention` (no cache,
ViT use case) and :mod:`phyai.layers.attention.diffusion`
(action-expert / diffusion role; same paged kernel today, separate
type tree).
"""

from __future__ import annotations

from phyai.layers.attention.ar.backends import (
    EagerARBackend,
    EagerARPlan,
    FlashInferARBackend,
    FlashInferARPlan,
)
from phyai.layers.attention.ar.base import (
    ARAttentionBackend,
    ARAttentionLayerProto,
    ARAttnCtx,
    ARAttnMetadata,
    ARAttnPlanHandle,
)
from phyai.layers.attention.ar.layer import ARAttention
from phyai.layers.attention.ar.registry import (
    BackendFactory,
    get_backend_factory,
    list_backends,
    register_backend,
)


__all__ = [
    "ARAttention",
    "ARAttentionBackend",
    "ARAttentionLayerProto",
    "ARAttnCtx",
    "ARAttnMetadata",
    "ARAttnPlanHandle",
    "BackendFactory",
    "EagerARBackend",
    "EagerARPlan",
    "FlashInferARBackend",
    "FlashInferARPlan",
    "RadixAttentionPlanner",
    "RadixSequence",
    "get_backend_factory",
    "list_backends",
    "register_backend",
]


def __getattr__(name: str):
    # Lazy re-export: the radix bridge pulls in the optional ``phyai-ext``
    # extra, so importing the base AR package (layers/backends) must not import
    # it eagerly. Resolved on first attribute access.
    # See tests/.../test_radix_planner.py::test_ar_import_does_not_pull_radix_extension.
    if name in ("RadixAttentionPlanner", "RadixSequence"):
        from phyai.layers.attention.ar import radix

        value = getattr(radix, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
