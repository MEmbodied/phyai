"""Op-agnostic core of the spec abstraction.

A :class:`WeightSpec` knows the storage format of a weight tensor — its
dtype, scale layout, and any per-format setup. It does *not* know which
op consumes the weight (linear, embedding, MoE), which kernel runs, or
how the activation should be pre-processed. Op-specific concerns live on
separate Protocols, e.g. :class:`phyai.layers.quant.linear.LinearActivationQuant`.

The contract between a layer and its spec is :class:`AllocationRequest`.
Layers build the request from their own shape conventions (linear's
``(N, K)``, embedding's ``(V_per, D)``, …) and hand it over; specs only
ever see ``weight_shape`` plus a list of per-logical-matrix widths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol

import torch
import torch.nn as nn


@dataclass(frozen=True)
class AllocationRequest:
    """Op-agnostic "where & how big" packet handed to :meth:`WeightSpec.allocate`.

    ``weight_shape`` is the *local* (per-rank, post-fuse) shape. For a
    fused matmul that's ``(sum(logical_widths), in_per_rank)``; for a
    sharded vocab embedding it's ``(num_embeddings_per_partition, embedding_dim)``.

    ``logical_widths`` is the per-sub-matrix breakdown along the fused
    dim — ``[w]`` for a single unfused matrix, ``[gate, up]`` for a
    merged MLP, etc. ``fused_dim`` says which dim the widths sum over;
    both linear and embedding use ``0``.

    ``extras`` is the controlled escape hatch for op-specific config a
    spec may want to look at without polluting the core fields. Prefer
    a typed Protocol over stuffing things here.
    """

    weight_shape: tuple[int, ...]
    logical_widths: list[int]
    fused_dim: int = 0
    params_dtype: torch.dtype = torch.bfloat16
    extras: Mapping[str, object] = field(default_factory=dict)


class WeightSpec(Protocol):
    """Op-agnostic core every spec satisfies.

    Implementations register parameters on ``layer`` (typically
    ``layer.weight`` plus any scales / zero points) and may attach
    spec-flavoured metadata such as ``layer.logical_widths``. They must
    NOT write op-specific names like ``layer.input_size_per_partition``
    — that's the layer's job.
    """

    spec_id: str
    weight_dtype: torch.dtype

    def allocate(self, layer: nn.Module, request: AllocationRequest) -> None: ...

    def process_after_loading(self, layer: nn.Module) -> None: ...


__all__ = ["AllocationRequest", "WeightSpec"]
