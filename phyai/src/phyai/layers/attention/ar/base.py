"""ABC + per-call types for `phyai.layers.attention.ar` (LM-side paged attention).

This subpackage is **paged-KV attention** for the autoregressive
language-model side of a model: K/V are scattered into a
:class:`~phyai.cache.kv_cache_pool.KVCachePool` then read back via
flashinfer's paged kernel (or an eager contiguous-slab fallback).

Per-call lifecycle
------------------
The runner owns one backend instance per stack and threads an
:class:`ARAttnCtx` through every layer. Layers do not store backends;
they route via ``ctx.backend.forward(layer, q, k, v, ctx)``. The
four-hook contract drives metadata staging:

1. :meth:`ARAttentionBackend.init_cuda_graph_state` — once at runner
   setup; backend allocates static buffers + builds wrapper.
2. :meth:`ARAttentionBackend.init_capture_metadata` — once at capture
   warmup; produces a plan handle whose tensor refs stay stable
   across replays.
3. :meth:`ARAttentionBackend.replay_metadata` — every step before
   ``graph.replay()``; backend writes new metadata into its static
   buffers in place.
4. :meth:`ARAttentionBackend.init_forward_metadata` — every step in
   the eager (non-graph) path.

Sibling stack: :mod:`phyai.layers.attention.diffusion` is structurally
identical (paged kernel, write-pool-then-read) but typed independently
for the action-expert / diffusion role.

AR vs Diffusion
---------------
The class name marks the layer's **role** in the model (LM side vs
action expert side), not a causality contract. Both stacks accept a
``causal`` flag at construction time. In pi0.5, both PaliGemma
(``ARAttention``) and the action expert (``DiffusionAttention``) use
``causal=False`` because the block-prefix mask is implemented at the
runner level rather than as a per-token causal mask.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

import torch

from phyai.layers.attention.enums import AttnLayout, AttnMode


if TYPE_CHECKING:
    from phyai.cache import KVCachePool


@dataclass(frozen=True)
class ARAttnMetadata:
    """Host-side description of the next AR attention step.

    Built by the scheduler from per-batch tensors, handed to an
    :class:`ARAttentionBackend` via :meth:`init_forward_metadata`
    (eager) or :meth:`replay_metadata` (graph replay).
    """

    mode: AttnMode
    layout: AttnLayout
    batch_size: int
    num_query_tokens: int
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_kv: torch.Tensor | None = None
    seq_lens_kv: torch.Tensor | None = None
    paged_kv_indptr: torch.Tensor | None = None
    paged_kv_indices: torch.Tensor | None = None
    paged_kv_last_page_len: torch.Tensor | None = None
    write_indices: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.batch_size < 0 or self.num_query_tokens < 0:
            raise ValueError(
                f"ARAttnMetadata: batch_size={self.batch_size}, "
                f"num_query_tokens={self.num_query_tokens} must be non-negative."
            )
        if self.mode == AttnMode.IDLE:
            return
        if self.layout == AttnLayout.RAGGED_3D and self.cu_seqlens_q is None:
            raise ValueError("ARAttnMetadata: layout=RAGGED_3D requires cu_seqlens_q.")


class ARAttnPlanHandle:
    """Backend-private per-step state for AR attention.

    Stability invariant
    -------------------
    Backends that support CUDA graph capture MUST keep the handle's
    tensor / wrapper references stable across replays: the graph
    captures Python identity, so substituting a fresh handle on replay
    invalidates capture.
    """


@dataclass(frozen=True)
class ARAttnCtx:
    """Per-call context for AR attention layers.

    The runner builds one ctx per inference step and threads it
    through every layer's forward. ``kv_pool`` and ``write_indices``
    are mandatory — AR is paged-KV by definition.
    """

    backend: "ARAttentionBackend"
    plan: ARAttnPlanHandle
    mode: AttnMode
    layout: AttnLayout
    kv_pool: "KVCachePool"
    write_indices: torch.Tensor
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_kv: torch.Tensor | None = None


@runtime_checkable
class ARAttentionLayerProto(Protocol):
    """Static config a backend reads off the AR layer instance."""

    num_heads: int
    num_kv_heads: int
    head_dim: int
    scale: float
    causal: bool
    layer_id: int


class ARAttentionBackend(ABC):
    """ABC for AR-side paged attention backends.

    Subclasses register themselves through
    :func:`~phyai.layers.attention.ar.registry.register_backend`,
    which sets :attr:`name` in place and stores a factory in this
    subpackage's registry.
    """

    name: ClassVar[str]

    def supports_capture(self) -> bool:
        return False

    def init_cuda_graph_state(
        self,
        *,
        max_batch_size: int,
        max_num_tokens: int,
        max_paged_kv_indices: int,
        device: torch.device,
        params_dtype: torch.dtype,
        layer_proto: ARAttentionLayerProto,
    ) -> None:
        """Allocate every static buffer the backend touches inside a captured graph.

        Called once at runner setup. After this returns the backend
        MUST hold every device-resident tensor at a stable address —
        :meth:`replay_metadata` may then update their contents but
        not their identity.

        Default no-op for backends without static state (eager).
        """
        return None

    def init_capture_metadata(self, seed_meta: ARAttnMetadata) -> ARAttnPlanHandle:
        """Plan with a representative shape so capture has valid kernel state.

        Default delegates to :meth:`init_forward_metadata`.
        """
        return self.init_forward_metadata(seed_meta)

    def replay_metadata(
        self,
        plan: ARAttnPlanHandle,
        replay_meta: ARAttnMetadata,
    ) -> None:
        """Update the backend's static buffers in place. Default no-op."""
        return None

    @abstractmethod
    def init_forward_metadata(self, meta: ARAttnMetadata) -> ARAttnPlanHandle:
        """Eagerly plan one step. Returns a handle written to ``ctx.plan``."""

    @abstractmethod
    def forward(
        self,
        layer: ARAttentionLayerProto,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: ARAttnCtx,
    ) -> torch.Tensor:
        """Run AR attention.

        Backends are responsible for scattering ``k`` / ``v`` into
        ``ctx.kv_pool`` (the layer no longer does it). For
        ``ctx.mode == IDLE`` the backend MUST return zeros without
        any kernel launch.
        """


__all__ = [
    "ARAttentionBackend",
    "ARAttentionLayerProto",
    "ARAttnCtx",
    "ARAttnMetadata",
    "ARAttnPlanHandle",
]
