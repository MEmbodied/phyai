"""`phyai.layers.attention.diffusion` — diffusion / action-expert paged attention.

Layer + backends + types for the diffusion / action-expert side of a
model. K/V are scattered into a
:class:`~phyai.cache.kv_cache_pool.KVCachePool` then read back via
flashinfer's paged kernel (or an eager contiguous-slab fallback).

Backends: ``"flashinfer"`` (default) and ``"eager"`` (CPU/CI).

Sibling stacks: :mod:`phyai.layers.attention.attention` (no cache,
ViT use case) and :mod:`phyai.layers.attention.ar` (LM-side; same
paged kernel today, separate type tree).
"""

from __future__ import annotations

from phyai.layers.attention.diffusion.backends import (
    EagerDiffusionBackend,
    EagerDiffusionPlan,
    FlashInferDiffusionBackend,
    FlashInferDiffusionPlan,
)
from phyai.layers.attention.diffusion.base import (
    DiffusionAttentionBackend,
    DiffusionAttentionLayerProto,
    DiffusionAttnCtx,
    DiffusionAttnMetadata,
    DiffusionAttnPlanHandle,
)
from phyai.layers.attention.diffusion.layer import DiffusionAttention
from phyai.layers.attention.diffusion.registry import (
    BackendFactory,
    get_backend_factory,
    list_backends,
    register_backend,
)


__all__ = [
    "BackendFactory",
    "DiffusionAttention",
    "DiffusionAttentionBackend",
    "DiffusionAttentionLayerProto",
    "DiffusionAttnCtx",
    "DiffusionAttnMetadata",
    "DiffusionAttnPlanHandle",
    "EagerDiffusionBackend",
    "EagerDiffusionPlan",
    "FlashInferDiffusionBackend",
    "FlashInferDiffusionPlan",
    "get_backend_factory",
    "list_backends",
    "register_backend",
]
