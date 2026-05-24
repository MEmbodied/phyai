"""Built-in :mod:`phyai.vgpu` backends.

Importing this package has the side effect of registering every available
backend class with the global registry in :mod:`phyai.vgpu.backend`.

The torch backend is unconditionally registered; the flashinfer backend is
registered only when the ``flashinfer`` package can be imported.
"""

from __future__ import annotations

# Side-effect import: registers TorchBackend.
from phyai.vgpu.backends import torch  # noqa: F401

try:
    # Side-effect import: registers FlashInferBackend.
    from phyai.vgpu.backends import flashinfer  # noqa: F401
except ImportError:
    pass


__all__: list[str] = []
