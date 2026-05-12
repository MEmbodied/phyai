"""Back-compat shim for ``phyai.layers.linear.loaders``.

The loaders moved up to :mod:`phyai.layers.loaders` once they grew beyond the
linear-only set (RMSNorm and other replicated layers needed the same
protocol). This module re-exports the names so any existing
``from phyai.layers.linear.loaders import ...`` keeps working.
"""

from __future__ import annotations

from phyai.layers.loaders import (
    ColumnShardLoader,
    QKVShardLoader,
    ReplicatedLoader,
    RowShardLoader,
)

__all__ = [
    "ColumnShardLoader",
    "QKVShardLoader",
    "ReplicatedLoader",
    "RowShardLoader",
]
