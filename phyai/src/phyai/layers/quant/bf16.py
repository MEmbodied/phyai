"""Bf16Spec — plain bf16 / fp16 weight, no scales, op-agnostic.

Works for any op (linear, vocab embedding, future MoE) because it does
nothing more than allocate a single :class:`nn.Parameter` of the
requested shape. Specifically does NOT carry ``granularity`` or
``needs_act_quant`` — those are quant-format concerns and bf16 is not a
quant format.

``weight_dtype`` is a hint; the actual dtype comes from
``request.params_dtype`` so users can pick fp16 without a new spec class.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from phyai.layers.quant.base import AllocationRequest


@dataclass
class Bf16Spec:
    spec_id: str = "bf16"
    weight_dtype: torch.dtype = torch.bfloat16

    def allocate(self, layer: nn.Module, request: AllocationRequest) -> None:
        layer.weight = nn.Parameter(
            torch.empty(*request.weight_shape, dtype=request.params_dtype),
            requires_grad=False,
        )
        layer.logical_widths = list(request.logical_widths)

    def process_after_loading(self, layer: nn.Module) -> None:
        return None


__all__ = ["Bf16Spec"]
