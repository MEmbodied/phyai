"""phyai.layers.attention — attention ops.

So far this exposes :class:`NoStateAttention`, a prefill-only attention
op with selectable sdpa / eager / flashinfer backends. There is no KV
cache and no radix tree here; Q/K/V come in projected and attention
goes back out.

The flashinfer scratch buffer is process-global and per-device; see
:func:`get_global_fi_workspace` for the entry point and the
``PHYAI_FLASHINFER_WORKSPACE_BYTES`` env var for sizing.
"""

from __future__ import annotations

from phyai.layers.attention.no_state_attention import NoStateAttention
from phyai.layers.attention.utils import (
    get_global_fi_workspace,
    register_global_fi_workspace,
    resolve_workspace_bytes,
)

__all__ = [
    "NoStateAttention",
    "get_global_fi_workspace",
    "register_global_fi_workspace",
    "resolve_workspace_bytes",
]
