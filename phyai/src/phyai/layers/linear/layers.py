"""Linear layers — named-axis TP/SP on top of WeightSpec + KernelDispatcher.

These classes only own the *parallel* aspect (mesh axis, bias placement,
input/output collectives) and delegate everything else to ``self.spec``
and the dispatcher. No fp8 / cutlass / marlin branches live here — the
decision tree is pushed into :class:`KernelDispatcher`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import phyai.parallel as P
from phyai.layers.linear.dispatch import get_linear_dispatcher
from phyai.layers.linear.spec import Bf16Spec
from phyai.layers.loaders import (
    ColumnShardLoader,
    QKVShardLoader,
    ReplicatedLoader,
    RowShardLoader,
)
from phyai.layers.quant import AllocationRequest
from phyai.parallel.state import resolve_mesh


def _M_of(x: torch.Tensor) -> int:
    """Total token count along all batch dims."""
    M = 1
    for s in x.shape[:-1]:
        M *= int(s)
    return M


class LinearBase(nn.Module):
    """Shared state for every Linear variant.

    Subclasses must call :meth:`_alloc_weight` exactly once (usually
    straight from ``__init__``) so ``spec`` gets a chance to register
    parameters on ``self``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        spec: object | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.params_dtype = params_dtype or torch.get_default_dtype()
        self.skip_bias_add = skip_bias_add
        self.spec = spec if spec is not None else Bf16Spec()
        self.prefix = prefix
        self._bias_requested = bias


class ReplicatedLinear(LinearBase):
    """Every rank holds the full weight — no collectives."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        spec: object | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            spec=spec,
            prefix=prefix,
        )
        self.spec.allocate(
            self,
            AllocationRequest(
                weight_shape=(out_features, in_features),
                logical_widths=[out_features],
                fused_dim=0,
                weight_loader=ReplicatedLoader(),
                params_dtype=self.params_dtype,
            ),
        )
        self.input_size_per_partition = in_features
        self.output_size_per_partition = out_features
        self.input_size_global = in_features
        self.output_size_global = out_features
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=self.params_dtype),
                requires_grad=False,
            )
            self.bias.loader = ReplicatedLoader()  # type: ignore[attr-defined]
        else:
            self.register_parameter("bias", None)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        kernel = get_linear_dispatcher().select(
            spec_id=self.spec.spec_id,
            M=_M_of(x),
            N=self.out_features,
            K=self.in_features,
            in_dtype=x.dtype,
            out_dtype=self.params_dtype,
        )
        bias = self.bias if not self.skip_bias_add else None
        y = kernel.apply(self, x, bias)
        return y, (self.bias if self.skip_bias_add else None)


class ColumnParallelLinear(LinearBase):
    """Sharded on the output dim along ``axis`` of ``mesh``.

    With ``sp_axis`` set, the forward first all-gathers ``x`` along that
    axis (dim=0 by convention) — sequence-parallel entry path.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        axis: str = "tp",
        sp_axis: str | None = None,
        gather_output: bool = False,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        spec: object | None = None,
        output_sizes: list[int] | None = None,
        mesh: str = "model",
        prefix: str = "",
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            spec=spec,
            prefix=prefix,
        )
        mesh_obj = resolve_mesh(mesh)
        self.mesh_name = mesh_obj.name
        self.axis = axis
        self.sp_axis = sp_axis
        self.gather_output = gather_output
        self.tp_size = mesh_obj.axis_size(axis)
        self.tp_rank = mesh_obj.axis_local_rank(axis)

        global_sizes = output_sizes if output_sizes is not None else [out_features]
        if sum(global_sizes) != out_features:
            raise ValueError(
                f"output_sizes sum ({sum(global_sizes)}) != out_features ({out_features})"
            )
        per_rank_sizes = []
        for s in global_sizes:
            if s % self.tp_size != 0:
                raise ValueError(
                    f"output partition {s} not divisible by tp_size={self.tp_size}"
                )
            per_rank_sizes.append(s // self.tp_size)
        self.output_partition_sizes = per_rank_sizes
        self.output_sizes_global = list(global_sizes)

        loader = ColumnShardLoader(
            output_partition_sizes=per_rank_sizes,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
        )
        self.spec.allocate(
            self,
            AllocationRequest(
                weight_shape=(sum(per_rank_sizes), in_features),
                logical_widths=per_rank_sizes,
                fused_dim=0,
                weight_loader=loader,
                params_dtype=self.params_dtype,
            ),
        )
        self.input_size_per_partition = in_features
        self.output_size_per_partition = sum(per_rank_sizes)
        self.input_size_global = in_features
        self.output_size_global = out_features

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(sum(per_rank_sizes), dtype=self.params_dtype),
                requires_grad=False,
            )
            # Bias is fused along the output dim just like the weight; the
            # same ColumnShardLoader (narrow(0, ...)) drives both.
            self.bias.loader = loader  # type: ignore[attr-defined]
        else:
            self.register_parameter("bias", None)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.sp_axis is not None:
            x = P.all_gather(x, axis=self.sp_axis, dim=0)

        kernel = get_linear_dispatcher().select(
            spec_id=self.spec.spec_id,
            M=_M_of(x),
            N=self.output_size_per_partition,
            K=self.input_size_per_partition,
            in_dtype=x.dtype,
            out_dtype=self.params_dtype,
        )
        bias = self.bias if not self.skip_bias_add else None
        y = kernel.apply(self, x, bias)

        if self.gather_output and self.tp_size > 1:
            y = P.all_gather(y, axis=self.axis, dim=-1)
        return y, (self.bias if self.skip_bias_add else None)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """Gate/up-style fused ColumnParallelLinear; ``output_sizes=[gate, up]``."""

    def __init__(
        self,
        in_features: int,
        output_sizes: list[int],
        *,
        axis: str = "tp",
        sp_axis: str | None = None,
        gather_output: bool = False,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        spec: object | None = None,
        mesh: str = "model",
        prefix: str = "",
    ) -> None:
        super().__init__(
            in_features,
            sum(output_sizes),
            axis=axis,
            sp_axis=sp_axis,
            gather_output=gather_output,
            bias=bias,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            spec=spec,
            output_sizes=output_sizes,
            mesh=mesh,
            prefix=prefix,
        )


class QKVParallelLinear(ColumnParallelLinear):
    """Q/K/V fused ColumnParallelLinear with GQA support.

    When ``tp_size`` exceeds ``num_kv_heads``, K and V are replicated
    ``tp_size // num_kv_heads`` times so every rank has a full set.
    """

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        *,
        axis: str = "tp",
        sp_axis: str | None = None,
        gather_output: bool = False,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        spec: object | None = None,
        mesh: str = "model",
        prefix: str = "",
    ) -> None:
        mesh_obj = resolve_mesh(mesh)
        tp_size = mesh_obj.axis_size(axis)

        if num_kv_heads is None:
            num_kv_heads = num_heads
        if tp_size >= num_kv_heads:
            if tp_size % num_kv_heads != 0:
                raise ValueError(
                    f"tp_size={tp_size} not a multiple of num_kv_heads={num_kv_heads}"
                )
            num_kv_replicas = tp_size // num_kv_heads
            effective_kv_heads = tp_size
        else:
            if num_kv_heads % tp_size != 0:
                raise ValueError(
                    f"num_kv_heads={num_kv_heads} not divisible by tp_size={tp_size}"
                )
            num_kv_replicas = 1
            effective_kv_heads = num_kv_heads

        q_size = num_heads * head_dim
        kv_size = effective_kv_heads * head_dim
        out_features = q_size + 2 * kv_size

        super().__init__(
            hidden_size,
            out_features,
            axis=axis,
            sp_axis=sp_axis,
            gather_output=gather_output,
            bias=bias,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            spec=spec,
            output_sizes=[q_size, kv_size, kv_size],
            mesh=mesh,
            prefix=prefix,
        )
        # Override the generic loader with a QKV-aware one.
        q_rank = q_size // self.tp_size
        kv_rank = kv_size // self.tp_size
        self.num_kv_replicas = num_kv_replicas
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        loader = QKVShardLoader(
            q_size=q_rank,
            kv_size=kv_rank,
            num_kv_replicas=num_kv_replicas,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
        )
        self.weight.loader = loader  # type: ignore[attr-defined]
        # Bias (when present) shares the same QKV layout along dim 0.
        if self.bias is not None:
            self.bias.loader = loader  # type: ignore[attr-defined]


class RowParallelLinear(LinearBase):
    """Sharded on the input dim along ``axis`` of ``mesh``.

    Exit collective is ``all_reduce`` on ``axis`` unless ``sp_axis`` is
    set, in which case it becomes ``reduce_scatter`` on ``sp_axis``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        axis: str = "tp",
        sp_axis: str | None = None,
        input_is_parallel: bool = True,
        reduce_results: bool = True,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        spec: object | None = None,
        mesh: str = "model",
        prefix: str = "",
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            spec=spec,
            prefix=prefix,
        )
        mesh_obj = resolve_mesh(mesh)
        self.mesh_name = mesh_obj.name
        self.axis = axis
        self.sp_axis = sp_axis
        self.tp_size = mesh_obj.axis_size(axis)
        self.tp_rank = mesh_obj.axis_local_rank(axis)
        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results

        if in_features % self.tp_size != 0:
            raise ValueError(
                f"in_features={in_features} not divisible by tp_size={self.tp_size}"
            )
        in_per_rank = in_features // self.tp_size

        loader = RowShardLoader(tp_rank=self.tp_rank, tp_size=self.tp_size)
        self.spec.allocate(
            self,
            AllocationRequest(
                weight_shape=(out_features, in_per_rank),
                logical_widths=[out_features],
                fused_dim=0,
                weight_loader=loader,
                params_dtype=self.params_dtype,
            ),
        )
        self.input_size_per_partition = in_per_rank
        self.output_size_per_partition = out_features
        self.input_size_global = in_features
        self.output_size_global = out_features
        if bias:
            # RowParallel bias is global (only rank 0 adds it at forward), so
            # every rank loads the full disk tensor unsliced.
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=self.params_dtype),
                requires_grad=False,
            )
            self.bias.loader = ReplicatedLoader()  # type: ignore[attr-defined]
        else:
            self.register_parameter("bias", None)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.input_is_parallel and self.tp_size > 1:
            shard = x.shape[-1] // self.tp_size
            x = x.narrow(-1, self.tp_rank * shard, shard).contiguous()

        kernel = get_linear_dispatcher().select(
            spec_id=self.spec.spec_id,
            M=_M_of(x),
            N=self.output_size_per_partition,
            K=self.input_size_per_partition,
            in_dtype=x.dtype,
            out_dtype=self.params_dtype,
        )
        # Only rank 0 adds the bias to avoid double counting post-reduce.
        add_bias = (
            self.bias
            if (self.bias is not None and self.tp_rank == 0 and not self.skip_bias_add)
            else None
        )
        y = kernel.apply(self, x, add_bias)

        if self.reduce_results and self.tp_size > 1:
            if self.sp_axis is not None:
                y = P.reduce_scatter(y, axis=self.sp_axis, dim=0)
            else:
                y = P.all_reduce(y, axis=self.axis)
        return y, (self.bias if self.skip_bias_add else None)
