"""Linear-only mixins for the spec abstraction.

A spec that wants to participate in the fp8 / int8 linear path declares
a pre-matmul activation quantisation hook by satisfying
:class:`LinearActivationQuant`. Specs that don't need it (bf16) simply
don't implement it; kernels that need quantised activations check
``isinstance(spec, LinearActivationQuant)`` (or branch on ``spec_id``).

Lives in ``quant.linear`` rather than ``layers.linear`` because it's a
quant *taxonomy* concern — the parallel sibling for MoE will be
``quant.moe`` once that op lands. ``layers.linear`` remains the home of
parallel layer wiring and kernel dispatch.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol, runtime_checkable

import torch
import torch.nn as nn

from phyai.layers.quant.granularity import Granularity


class ActivationView(NamedTuple):
    """The view a kernel receives from :meth:`LinearActivationQuant.quantize_activation`."""

    x: torch.Tensor
    x_scale: torch.Tensor | None
    granularity: Granularity


@runtime_checkable
class LinearActivationQuant(Protocol):
    """Linear-only mixin for specs that need pre-matmul activation quant.

    ``runtime_checkable`` so callers can write
    ``if isinstance(spec, LinearActivationQuant):`` without inspecting
    spec_id strings.
    """

    needs_act_quant: bool

    def quantize_activation(
        self,
        x: torch.Tensor,
        layer: nn.Module,
    ) -> ActivationView: ...


__all__ = ["ActivationView", "LinearActivationQuant"]
