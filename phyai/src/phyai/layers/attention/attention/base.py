"""ABC + per-call types for `phyai.layers.attention.attention` (the no-cache stack).

This subpackage is **stateless attention**: prefill-only, no KV cache,
no pool reads, no per-layer state across calls. Used today by the
SigLIP vision tower and by ad-hoc unit tests.

Per-call lifecycle
------------------
The runner (or the layer's convenience-ctx builder) hands the layer an
:class:`AttnCtx`. Layers do not store backends; they route via
``ctx.backend.forward(layer, q, k, v, ctx)``. The runner-driven path
uses :meth:`AttentionBackend.init_forward_metadata` to build a plan
once per step; the convenience path (vision tower / unit tests) lazily
builds a degenerate ctx on the first ctx-less ``forward`` call.

There is no `init_cuda_graph_state` / `replay_metadata` distinction
because no-cache backends carry no per-step static buffers — the
default :meth:`init_capture_metadata` simply delegates to
:meth:`init_forward_metadata` and :meth:`replay_metadata` is a no-op.

Sibling stacks: ``phyai.layers.attention.ar`` (LM-side paged attention)
and ``phyai.layers.attention.diffusion`` (action-expert paged
attention). They are typed independently — :class:`AttnCtx` here
deliberately does NOT carry a ``kv_pool``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable

import torch

from phyai.layers.attention.enums import AttnLayout, AttnMode


@dataclass(frozen=True)
class AttnMetadata:
    """Host-side description of the next attention step (no-cache).

    Built by the runner from per-batch tensors, handed to a
    :class:`AttentionBackend` via :meth:`init_forward_metadata`.
    Backends pick the fields they care about; unused fields stay
    ``None``.

    Fields
    ------
    mode:
        Forward stage. ``IDLE`` skips kernel launches.
    layout:
        Q/K/V layout the backend should expect on the matching
        :meth:`AttentionBackend.forward` call.
    batch_size:
        Logical samples in this step (pre-pad).
    num_query_tokens:
        ``N`` for ragged, ``B * S_q`` for padded.
    cu_seqlens_q, cu_seqlens_kv:
        ``(B+1,)`` int32 cumulative offsets. Required when
        ``layout == RAGGED_3D``; backends that own a wrapper plan
        with these.
    seq_lens_kv:
        ``(B,)`` int32 — per-sample full KV length. Mostly informational.
    position_ids:
        ``(N,)`` int32 absolute positions. Some runners use these for
        rope or for staging into static buffers outside captured regions.
    extras:
        Backend-specific overrides. The flashinfer backend reads
        ``extras['layer_proto']`` / ``extras['q_dtype']`` /
        ``extras['kv_dtype']`` for the B>1 plan path.
    """

    mode: AttnMode
    layout: AttnLayout
    batch_size: int
    num_query_tokens: int
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_kv: torch.Tensor | None = None
    seq_lens_kv: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.batch_size < 0 or self.num_query_tokens < 0:
            raise ValueError(
                f"AttnMetadata: batch_size={self.batch_size}, "
                f"num_query_tokens={self.num_query_tokens} must be non-negative."
            )
        if self.mode == AttnMode.IDLE:
            return
        if self.layout == AttnLayout.RAGGED_3D and self.cu_seqlens_q is None:
            raise ValueError("AttnMetadata: layout=RAGGED_3D requires cu_seqlens_q.")


class AttnPlanHandle:
    """Backend-private per-step state.

    Concrete backends subclass this with whatever they need to carry
    forward from a planning call to the matching
    :meth:`AttentionBackend.forward`. The layer threads the handle
    through :class:`AttnCtx` opaquely — only the matching backend
    cracks it open.
    """


@dataclass(frozen=True)
class AttnCtx:
    """Per-call context handed to the layer's ``forward``.

    The runner builds one ctx per inference step (when integrating
    with a runner-driven attention stack); the convenience path used
    by the vision tower / unit tests lazily builds a degenerate ctx
    on the first ctx-less forward call. Layers route via
    ``ctx.backend.forward(self, q, k, v, ctx)``.
    """

    backend: "AttentionBackend"
    plan: AttnPlanHandle
    mode: AttnMode
    layout: AttnLayout
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_kv: torch.Tensor | None = None


@runtime_checkable
class AttentionLayerProto(Protocol):
    """Static config a backend reads off the layer instance.

    :class:`~phyai.layers.attention.attention.layer.Attention` satisfies
    this Protocol. Backends type their ``forward(layer, ...)`` against
    it so they can read config without coupling to the concrete layer.
    """

    num_heads: int
    num_kv_heads: int
    head_dim: int
    scale: float
    causal: bool
    sliding_window: int | None
    logits_soft_cap: float | None


class AttentionBackend(ABC):
    """ABC for every no-cache attention backend.

    Subclasses register themselves through
    :func:`~phyai.layers.attention.attention.registry.register_backend`,
    which sets :attr:`name` in place and stores a factory in this
    subpackage's registry.

    Lifecycle
    ---------
    No-cache backends typically do not need ``init_cuda_graph_state``
    nor ``replay_metadata`` (they carry no per-step static buffers).
    The default implementations are no-ops; backends override only if
    they really need static state. ``init_capture_metadata`` defaults
    to delegating to :meth:`init_forward_metadata`.
    """

    name: ClassVar[str]

    def supports_capture(self) -> bool:
        """Whether the per-call hot path is safe inside a captured graph."""
        return False

    def init_cuda_graph_state(
        self,
        *,
        max_batch_size: int,
        max_num_tokens: int,
        device: torch.device,
        params_dtype: torch.dtype,
        layer_proto: AttentionLayerProto,
    ) -> None:
        """Allocate any static buffer the backend touches inside a captured graph.

        Default no-op — no-cache backends rarely need static state.
        """
        return None

    def init_capture_metadata(self, seed_meta: AttnMetadata) -> AttnPlanHandle:
        """Plan with a representative shape so capture has valid kernel state."""
        return self.init_forward_metadata(seed_meta)

    def replay_metadata(
        self,
        plan: AttnPlanHandle,
        replay_meta: AttnMetadata,
    ) -> None:
        """Update the backend's static buffers in place. Default no-op."""
        return None

    @abstractmethod
    def init_forward_metadata(self, meta: AttnMetadata) -> AttnPlanHandle:
        """Eagerly plan one step. Returns a handle written to ``ctx.plan``."""

    @abstractmethod
    def forward(
        self,
        layer: AttentionLayerProto,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: AttnCtx,
    ) -> torch.Tensor:
        """Run attention.

        Backend dispatches internally on ``ctx.mode`` and ``ctx.layout``.
        For ``mode == IDLE`` the backend MUST return zeros without any
        kernel launch.
        """


__all__ = [
    "AttentionBackend",
    "AttentionLayerProto",
    "AttnCtx",
    "AttnMetadata",
    "AttnPlanHandle",
]
