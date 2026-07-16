"""LinearKernel Protocol + descriptive dataclasses.

A ``KernelProbe`` is the dispatcher's query packet, and ``LinearKernel``
is a pure-capability Protocol each backend implements. There is no
``score()`` — priority is the Registry's job (see
:mod:`phyai.layers.linear.registry`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from phyai.layers.quant.granularity import Granularity  # noqa: F401
from phyai.layers.quant.linear import ActivationView
from phyai.parallel.state import Mode


# ``Granularity`` was originally defined here. It now lives in
# :mod:`phyai.layers.quant.granularity` so non-linear ops can reuse it; the
# import above re-exports it under the historical path.


@dataclass(frozen=True)
class KernelProbe:
    """The query packet the dispatcher hands to the registry.

    Fields are chosen so that ``(probe)`` is uniquely keyable into a cache
    and carries everything a ``LinearKernel.can_handle`` predicate needs.
    """

    spec_id: str
    M_bucket: int
    N: int
    K: int
    in_dtype: torch.dtype
    out_dtype: torch.dtype
    sm: int
    mode: Mode
    # note(chenghua): Keep exact M beside the coarse regime to separate M=16
    # from M=17 in the capability cache.
    M: int | None = None


@runtime_checkable
class LinearKernel(Protocol):
    """Pure-capability Protocol each matmul backend implements.

    Hot path is :meth:`apply`; :meth:`can_handle` is consulted on cache
    miss only and returns a bool (no scoring).
    """

    name: str

    def can_handle(self, probe: KernelProbe) -> bool: ...

    def supports_capture(self) -> bool: ...

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor: ...


@runtime_checkable
class PrequantizedLinearKernel(Protocol):
    """Optional extension for kernels that consume an :class:`ActivationView`."""

    def apply_prequantized(
        self,
        layer: torch.nn.Module,
        activation: ActivationView,
        bias: torch.Tensor | None,
    ) -> torch.Tensor: ...
