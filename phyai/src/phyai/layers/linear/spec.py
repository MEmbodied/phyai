"""WeightSpec — the "logical form" of a weight tensor.

A spec declares what the parameter storage looks like (dtype, scales,
granularity) and how to quantise an activation to match. It does *not*
know which kernel will consume the weight — that's the dispatcher's job.

Two specs ship here: :class:`Bf16Spec` (identity path) and :class:`Fp8Spec`
with per-tensor / per-channel / block granularity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn

from phyai.layers.linear.backend import Granularity


class ActivationView(NamedTuple):
    """The view a kernel receives from :meth:`WeightSpec.quantize_activation`."""

    x: torch.Tensor
    x_scale: torch.Tensor | None
    granularity: Granularity


def _convert_to_channelwise(
    scale_per_logical: torch.Tensor,
    logical_widths: list[int],
) -> torch.Tensor:
    """Fan a ``[L]`` per-logical-matrix scale out to ``[sum(widths)]`` per-channel."""
    return torch.cat(
        [scale_per_logical[i].expand(w) for i, w in enumerate(logical_widths)]
    )


_FP8_E4M3_AMAX = 448.0


@dataclass
class Bf16Spec:
    """Plain bf16 (or fp16) weight. No quantisation, no scales.

    ``weight_dtype`` is a hint; the actual dtype comes from
    ``params_dtype`` passed to :meth:`allocate`, to let users pick fp16
    without a new spec class.
    """

    spec_id: str = "bf16"
    weight_dtype: torch.dtype = torch.bfloat16
    granularity: Granularity = Granularity.PER_TENSOR
    needs_act_quant: bool = False

    def allocate(
        self,
        layer: nn.Module,
        *,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size_global: int,
        output_size_global: int,
        params_dtype: torch.dtype,
        weight_loader: object | None,
    ) -> None:
        out_per_rank = sum(output_partition_sizes)
        layer.weight = nn.Parameter(
            torch.empty(
                out_per_rank,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.weight.loader = weight_loader  # type: ignore[attr-defined]
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = out_per_rank
        layer.input_size_global = input_size_global
        layer.output_size_global = output_size_global
        layer.params_dtype = params_dtype

    def process_after_loading(self, layer: nn.Module) -> None:
        return None

    def quantize_activation(
        self,
        x: torch.Tensor,
        layer: nn.Module,
    ) -> ActivationView:
        return ActivationView(x, None, Granularity.PER_TENSOR)


@dataclass
class Fp8Spec:
    """``torch.float8_e4m3fn`` weight with configurable scale granularity.

    ``granularity`` selects between:

    * ``PER_TENSOR``  — one scalar per logical matrix (fanned out to
      per-channel by :meth:`process_after_loading`) + one static
      activation scale.
    * ``PER_CHANNEL`` — per-output-row weight scale, per-token
      (rowwise) activation scale computed at runtime.
    * ``BLOCK``       — ``(out_per_rank // block_n, in_per_rank // block_k)``
      weight scale, per-token activation scale at block-K granularity.
    """

    granularity: Granularity = Granularity.PER_CHANNEL
    block_shape: tuple[int, int] | None = None
    weight_dtype: torch.dtype = torch.float8_e4m3fn

    def __post_init__(self) -> None:
        if self.granularity == Granularity.BLOCK and self.block_shape is None:
            raise ValueError("Fp8Spec(granularity=BLOCK) requires block_shape")

    @property
    def spec_id(self) -> str:
        if self.granularity == Granularity.BLOCK:
            assert self.block_shape is not None
            bn, bk = self.block_shape
            return f"fp8_block_{bn}_{bk}"
        return f"fp8_{self.granularity.value}"

    @property
    def needs_act_quant(self) -> bool:
        return True

    def allocate(
        self,
        layer: nn.Module,
        *,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size_global: int,
        output_size_global: int,
        params_dtype: torch.dtype,
        weight_loader: object | None,
    ) -> None:
        out_per_rank = sum(output_partition_sizes)

        layer.weight = nn.Parameter(
            torch.empty(
                out_per_rank,
                input_size_per_partition,
                dtype=self.weight_dtype,
            ),
            requires_grad=False,
        )
        layer.weight.loader = weight_loader  # type: ignore[attr-defined]

        if self.granularity == Granularity.PER_TENSOR:
            layer.weight_scale = nn.Parameter(
                torch.ones(len(output_partition_sizes), dtype=torch.float32),
                requires_grad=False,
            )
            layer.input_scale = nn.Parameter(
                torch.ones(1, dtype=torch.float32),
                requires_grad=False,
            )
        elif self.granularity == Granularity.PER_CHANNEL:
            layer.weight_scale = nn.Parameter(
                torch.ones(out_per_rank, dtype=torch.float32),
                requires_grad=False,
            )
        elif self.granularity == Granularity.BLOCK:
            assert self.block_shape is not None
            bn, bk = self.block_shape
            if out_per_rank % bn != 0:
                raise ValueError(
                    f"Fp8Spec(BLOCK): out_per_rank={out_per_rank} "
                    f"not divisible by block_n={bn}"
                )
            if input_size_per_partition % bk != 0:
                raise ValueError(
                    f"Fp8Spec(BLOCK): in_per_rank={input_size_per_partition} "
                    f"not divisible by block_k={bk}"
                )
            layer.weight_scale = nn.Parameter(
                torch.ones(
                    out_per_rank // bn,
                    input_size_per_partition // bk,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = out_per_rank
        layer.input_size_global = input_size_global
        layer.output_size_global = output_size_global
        layer.params_dtype = params_dtype

    def process_after_loading(self, layer: nn.Module) -> None:
        if self.granularity == Granularity.PER_TENSOR:
            # Fan one scalar per logical matrix out to a per-channel vector
            # so kernels can consume a uniform scale layout.
            layer.weight_scale = nn.Parameter(
                _convert_to_channelwise(layer.weight_scale, layer.logical_widths),
                requires_grad=False,
            )

    def quantize_activation(
        self,
        x: torch.Tensor,
        layer: nn.Module,
    ) -> ActivationView:
        if self.granularity == Granularity.PER_TENSOR:
            x_q = (
                (x.float() / layer.input_scale)
                .clamp(-_FP8_E4M3_AMAX, _FP8_E4M3_AMAX)
                .to(torch.float8_e4m3fn)
            )
            return ActivationView(x_q, layer.input_scale, Granularity.PER_TENSOR)

        if self.granularity in (Granularity.PER_CHANNEL, Granularity.BLOCK):
            # Per-token (rowwise) activation quant.
            x_amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
            x_scale = (x_amax / _FP8_E4M3_AMAX).to(torch.float32)
            x_q = (x / x_scale).to(torch.float8_e4m3fn)
            return ActivationView(x_q, x_scale, self.granularity)

        raise RuntimeError(f"unhandled Fp8Spec granularity {self.granularity!r}")
