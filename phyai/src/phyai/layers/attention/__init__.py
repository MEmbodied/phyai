"""phyai.layers.attention — three-flavor attention API.

Three structurally independent attention stacks live as subpackages:

* :mod:`phyai.layers.attention.attention` — :class:`Attention`, no
  cache. ViT / encoder use case. Backends:
  ``"flashinfer"`` / ``"sdpa"`` / ``"eager"``.
* :mod:`phyai.layers.attention.ar` — :class:`ARAttention`, paged-KV
  for the autoregressive language-model side. Backends:
  ``"flashinfer"`` / ``"eager"``.
* :mod:`phyai.layers.attention.diffusion` — :class:`DiffusionAttention`,
  paged-KV for the diffusion / action-expert side. Backends:
  ``"flashinfer"`` / ``"eager"``.

Each subpackage has its own ABC, ``Ctx``, ``Metadata``, ``PlanHandle``,
and registry. Backends are resolved per-stack. Adding a new backend
means picking the right subpackage's ``register_backend`` decorator,
subclassing the matching ABC, and importing the new module from the
subpackage's ``backends/__init__.py``.

Shared primitives across the three stacks:

* :class:`AttnMode` and :class:`AttnLayout` enums (in :mod:`enums`).
* :func:`eager_attn` / :func:`repeat_kv` / :func:`build_padded_mask`
  (in :mod:`common`, used internally by the backend modules).
* The flashinfer scratch helpers (in :mod:`utils`) — one process-global
  uint8 workspace per device shared across all three stacks.

AR vs Diffusion
---------------
The class names mark the layer's **role** (LM side vs action expert
side), not a causality contract. Both stacks accept a ``causal``
flag. AR's default is ``causal=True`` and Diffusion's default is
``causal=False``, but pi0.5 overrides both to ``causal=False``
because the block-prefix mask is implemented at the runner level.

The two paged backends (``FlashInferARBackend`` /
``FlashInferDiffusionBackend`` and the two eager equivalents) are
**byte-identical implementations today** — sibling code paths kept
in sync manually so the two stacks can evolve independently later.
"""

from __future__ import annotations

from phyai.layers.attention.ar import (
    ARAttention,
    ARAttentionBackend,
    ARAttentionLayerProto,
    ARAttnCtx,
    ARAttnMetadata,
    ARAttnPlanHandle,
    EagerARBackend,
    EagerARPlan,
    FlashInferARBackend,
    FlashInferARPlan,
)
from phyai.layers.attention.ar import (
    get_backend_factory as get_ar_backend_factory,
)
from phyai.layers.attention.ar import (
    list_backends as list_ar_backends,
)
from phyai.layers.attention.ar import (
    register_backend as register_ar_backend,
)
from phyai.layers.attention.attention import (
    Attention,
    AttentionBackend,
    AttentionLayerProto,
    AttnCtx,
    AttnMetadata,
    AttnPlanHandle,
    EagerAttentionBackend,
    EagerAttentionPlan,
    FlashInferAttentionBackend,
    FlashInferAttentionPlan,
    SdpaAttentionBackend,
    SdpaAttentionPlan,
)
from phyai.layers.attention.attention import (
    get_backend_factory as get_attention_backend_factory,
)
from phyai.layers.attention.attention import (
    list_backends as list_attention_backends,
)
from phyai.layers.attention.attention import (
    register_backend as register_attention_backend,
)
from phyai.layers.attention.diffusion import (
    DiffusionAttention,
    DiffusionAttentionBackend,
    DiffusionAttentionLayerProto,
    DiffusionAttnCtx,
    DiffusionAttnMetadata,
    DiffusionAttnPlanHandle,
    EagerDiffusionBackend,
    EagerDiffusionPlan,
    FlashInferDiffusionBackend,
    FlashInferDiffusionPlan,
)
from phyai.layers.attention.diffusion import (
    get_backend_factory as get_diffusion_backend_factory,
)
from phyai.layers.attention.diffusion import (
    list_backends as list_diffusion_backends,
)
from phyai.layers.attention.diffusion import (
    register_backend as register_diffusion_backend,
)
from phyai.layers.attention.enums import AttnLayout, AttnMode
from phyai.layers.attention.utils import (
    get_global_fi_workspace,
    register_global_fi_workspace,
    resolve_workspace_bytes,
)


__all__ = [
    # === Layers ===
    "Attention",
    "ARAttention",
    "DiffusionAttention",
    # === Shared enums ===
    "AttnLayout",
    "AttnMode",
    # === attention/ stack ===
    "AttentionBackend",
    "AttentionLayerProto",
    "AttnCtx",
    "AttnMetadata",
    "AttnPlanHandle",
    "EagerAttentionBackend",
    "EagerAttentionPlan",
    "FlashInferAttentionBackend",
    "FlashInferAttentionPlan",
    "SdpaAttentionBackend",
    "SdpaAttentionPlan",
    "get_attention_backend_factory",
    "list_attention_backends",
    "register_attention_backend",
    # === ar/ stack ===
    "ARAttentionBackend",
    "ARAttentionLayerProto",
    "ARAttnCtx",
    "ARAttnMetadata",
    "ARAttnPlanHandle",
    "EagerARBackend",
    "EagerARPlan",
    "FlashInferARBackend",
    "FlashInferARPlan",
    "get_ar_backend_factory",
    "list_ar_backends",
    "register_ar_backend",
    # === diffusion/ stack ===
    "DiffusionAttentionBackend",
    "DiffusionAttentionLayerProto",
    "DiffusionAttnCtx",
    "DiffusionAttnMetadata",
    "DiffusionAttnPlanHandle",
    "EagerDiffusionBackend",
    "EagerDiffusionPlan",
    "FlashInferDiffusionBackend",
    "FlashInferDiffusionPlan",
    "get_diffusion_backend_factory",
    "list_diffusion_backends",
    "register_diffusion_backend",
    # === Workspace ===
    "get_global_fi_workspace",
    "register_global_fi_workspace",
    "resolve_workspace_bytes",
]
