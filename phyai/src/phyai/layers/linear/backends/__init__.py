"""Concrete LinearKernel implementations.

Each ``<vendor>.py`` module owns one kernel class and self-registers
via the :func:`~phyai.layers.linear.registry.register_linear_kernel`
decorator at module import. Importing this package executes those
side effects;  :func:`phyai.layers.linear.init` then materialises a
:class:`~phyai.layers.linear.registry.LinearKernelRegistry` from the
gathered declarations.
"""

from __future__ import annotations

from phyai.layers.linear.backends.flashinfer import FlashInferKernel
from phyai.layers.linear.backends.torch import TorchKernel

__all__ = ["FlashInferKernel", "TorchKernel"]
