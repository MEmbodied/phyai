"""Vocab-parallel input embedding and tied output LM head.

Two classes, intentionally independent:

* :class:`VocabParallelEmbedding` — gather ``W[input_ids]`` along the V axis.
  Per-rank weight shape ``(V_padded // tp_size, D)``. Forward calls the
  fused :func:`phyai::masked_embedding_lookup` op (Triton on CUDA, eager
  fallback elsewhere) and finishes with a single ``all_reduce`` along the
  TP axis. There is no ``masked_fill_`` second pass — the kernel writes
  zeros for out-of-shard positions directly.

* :class:`ParallelLMHead` — column-parallel matmul over the same
  ``(V_padded // tp_size, D)`` weight, producing per-rank logits of shape
  ``(..., V_padded // tp_size)``. Independent class, NOT inheriting
  ``VocabParallelEmbedding``. ``forward`` returns logits (no ``raise``);
  weight tying is a constructor argument (``tied_weight=embed.weight``)
  rather than a post-hoc ``tie_weights()`` mutation.

Padding: ``num_embeddings`` is rounded up to a multiple of
``tp_size * padding_multiple`` (default ``padding_multiple=64``). Padding
rows are zero-filled by :class:`phyai.layers.loaders.VocabShardLoader`,
so embeddings of out-of-range token ids are guaranteed-zero rows and
LM-head logits over padding columns are guaranteed-zero scalars — no
sampler reindex needed.

Differences from SGLang's ``vocab_parallel_embedding.py`` are documented
in ``plans/synchronous-stargazing-moler.md``; the most important are:
no four-section LoRA layout, no ``tie_weights()`` mutation, no
``method_has_implemented_embedding`` quant-method indirection.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

import phyai.parallel as P
from phyai.layers.linear.dispatch import get_linear_dispatcher
from phyai.layers.loaders import VocabShardLoader
from phyai.layers.quant import AllocationRequest
from phyai.layers.quant.bf16 import Bf16Spec
from phyai.layers.vocab_embedding.ops import masked_embedding_lookup
from phyai.parallel.state import resolve_mesh


def pad_vocab_to(num_embeddings: int, tp_size: int, multiple: int = 64) -> int:
    """Round ``num_embeddings`` up to the nearest multiple of ``tp_size * multiple``.

    Combines two requirements at once: per-rank chunks must be equal-size
    (divisible by ``tp_size``) and each chunk should be a multiple of
    ``multiple`` for memory-alignment / packed-quant reasons. ``multiple``
    is exposed so callers with stricter alignment (FP8 wants 128, INT4
    packed wants 256) can override.
    """
    if tp_size <= 0 or multiple <= 0:
        raise ValueError(
            f"tp_size and multiple must be positive, got {tp_size=}, {multiple=}"
        )
    step = tp_size * multiple
    return ((num_embeddings + step - 1) // step) * step


def _M_of(x: torch.Tensor) -> int:
    M = 1
    for s in x.shape[:-1]:
        M *= int(s)
    return M


class VocabParallelEmbedding(nn.Module):
    """V-sharded input embedding with masked-lookup + all-reduce.

    Args:
        num_embeddings: real vocab size ``V``.
        embedding_dim: hidden size ``D``.
        params_dtype: dtype for the weight tensor; defaults to torch default.
        spec: :class:`phyai.layers.quant.WeightSpec` controlling allocation
            (default :class:`Bf16Spec`).
        layout: kept for API symmetry; only ``"vocab_parallel"`` is
            implemented in this version. ``"hidden_parallel"`` (D-shard)
            and ``"replicated"`` are reserved for future work.
        padding_multiple: per-rank chunks are padded to this multiple.
            Default 64 (matches SGLang); FP8/INT4 may want 128/256.
        axis: mesh axis used for the V split. Default ``"tp"``.
        mesh: mesh name (default ``"model"``).
        prefix: state-dict prefix; carried for parity with linear layers.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        params_dtype: torch.dtype | None = None,
        spec: object | None = None,
        layout: Literal["vocab_parallel"] = "vocab_parallel",
        padding_multiple: int = 64,
        axis: str = "tp",
        mesh: str = "model",
        prefix: str = "",
    ) -> None:
        super().__init__()
        if layout != "vocab_parallel":
            raise NotImplementedError(
                f"VocabParallelEmbedding currently only supports "
                f"'vocab_parallel' layout; got {layout!r}. (D-shard and "
                f"replicated layouts are reserved for future work — see "
                f"plans/synchronous-stargazing-moler.md.)"
            )
        if num_embeddings <= 0:
            raise ValueError(f"num_embeddings must be positive, got {num_embeddings}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")

        self.params_dtype = params_dtype or torch.get_default_dtype()
        self.spec = spec if spec is not None else Bf16Spec()
        self.prefix = prefix

        mesh_obj = resolve_mesh(mesh)
        self.mesh_name = mesh_obj.name
        self.axis = axis
        self.tp_size = mesh_obj.axis_size(axis)
        self.tp_rank = mesh_obj.axis_local_rank(axis)

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_embeddings_padded = pad_vocab_to(
            num_embeddings, self.tp_size, multiple=padding_multiple
        )
        self.num_embeddings_per_partition = self.num_embeddings_padded // self.tp_size

        # Real (padding-trimmed) shard bounds. The Triton kernel uses these
        # to mask out positions whose token id falls outside the actual
        # vocabulary; padding rows on the trailing rank never get queried.
        raw_start = self.tp_rank * self.num_embeddings_per_partition
        raw_end = raw_start + self.num_embeddings_per_partition
        self.shard_start = raw_start
        # Clamp to V_real on both sides so the kernel sees a consistent
        # ``shard_end >= shard_start`` even when this rank holds nothing
        # but padding rows (pathological V << V_padded case).
        self.shard_end = max(raw_start, min(raw_end, self.num_embeddings))

        loader = VocabShardLoader(
            num_embeddings=self.num_embeddings,
            num_embeddings_padded=self.num_embeddings_padded,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
        )
        self.spec.allocate(
            self,
            AllocationRequest(
                weight_shape=(self.num_embeddings_per_partition, embedding_dim),
                logical_widths=[self.num_embeddings_per_partition],
                fused_dim=0,
                weight_loader=loader,
                params_dtype=self.params_dtype,
            ),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        local = masked_embedding_lookup(
            input_ids,
            self.weight,
            shard_start=self.shard_start,
            shard_end=self.shard_end,
        )
        if self.tp_size > 1:
            local = P.all_reduce(local, axis=self.axis, mesh=self.mesh_name)
        return local

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, "
            f"per_partition={self.num_embeddings_per_partition}, "
            f"padded={self.num_embeddings_padded}, "
            f"tp_size={self.tp_size}, axis={self.axis!r}"
        )


class ParallelLMHead(nn.Module):
    """V-sharded output projection: ``logits = x @ weight^T``.

    Independent class — does NOT inherit :class:`VocabParallelEmbedding`.
    The matmul side reuses the same dispatcher and ``LinearKernel`` Protocol
    as :class:`phyai.layers.linear.ColumnParallelLinear`, so any future fp8
    / cutlass / marlin Linear kernel automatically applies to the LM head.

    Args:
        embedding_dim: hidden size ``D`` (input dim).
        num_embeddings: real vocab size ``V``.
        bias: not yet supported in this version (raises if True).
        params_dtype, spec, padding_multiple, axis, mesh, prefix: as for
            :class:`VocabParallelEmbedding`.
        tied_weight: if provided, the LM head shares this :class:`nn.Parameter`
            with another layer (typically a :class:`VocabParallelEmbedding`).
            ``spec.allocate`` is skipped; the caller is responsible for
            ensuring the tied weight has the right shape and dtype. Only
            ``Bf16Spec`` is supported in tied mode for now (fp8 etc. would
            need to also tie scale tensors and is deferred).
        gather_output: if True, the per-rank logits are all-gathered along
            the TP axis on the way out so callers see global ``V_padded``
            logits. Default False — the sampler typically gathers itself.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: int,
        *,
        bias: bool = False,
        params_dtype: torch.dtype | None = None,
        spec: object | None = None,
        padding_multiple: int = 64,
        tied_weight: nn.Parameter | None = None,
        gather_output: bool = False,
        axis: str = "tp",
        mesh: str = "model",
        prefix: str = "",
    ) -> None:
        super().__init__()
        if bias:
            raise NotImplementedError(
                "ParallelLMHead bias=True is not supported in this version; "
                "LM heads in modern LLMs (Llama, Qwen, Gemma) are bias-free."
            )
        if num_embeddings <= 0:
            raise ValueError(f"num_embeddings must be positive, got {num_embeddings}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")

        self.params_dtype = params_dtype or torch.get_default_dtype()
        self.spec = spec if spec is not None else Bf16Spec()
        self.prefix = prefix

        mesh_obj = resolve_mesh(mesh)
        self.mesh_name = mesh_obj.name
        self.axis = axis
        self.tp_size = mesh_obj.axis_size(axis)
        self.tp_rank = mesh_obj.axis_local_rank(axis)
        self.gather_output = gather_output

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_embeddings_padded = pad_vocab_to(
            num_embeddings, self.tp_size, multiple=padding_multiple
        )
        self.num_embeddings_per_partition = self.num_embeddings_padded // self.tp_size

        # Linear-kernel-API attributes. The kernel's ``apply`` only reads
        # ``layer.spec`` and ``layer.weight`` (plus scale tensors for
        # quantized specs). The size attributes mirror what
        # ``ColumnParallelLinear`` exposes so any kernel that probes them
        # works unchanged.
        self.input_size_per_partition = embedding_dim
        self.output_size_per_partition = self.num_embeddings_per_partition
        self.input_size_global = embedding_dim
        self.output_size_global = self.num_embeddings_padded

        if tied_weight is not None:
            # Tied path: skip allocation, share the source Parameter. The
            # source is responsible for spec-allocated state (scales etc.);
            # we restrict to bf16 here so tying is unambiguous.
            if self.spec.spec_id != "bf16":
                raise NotImplementedError(
                    f"ParallelLMHead tied_weight is only supported for "
                    f"bf16-style specs in this version; got "
                    f"spec_id={self.spec.spec_id!r}"
                )
            expected_shape = (self.num_embeddings_per_partition, embedding_dim)
            if tuple(tied_weight.shape) != expected_shape:
                raise ValueError(
                    f"tied_weight shape {tuple(tied_weight.shape)} does not "
                    f"match expected {expected_shape}"
                )
            self.weight = tied_weight
            self.logical_widths = [self.num_embeddings_per_partition]
        else:
            loader = VocabShardLoader(
                num_embeddings=self.num_embeddings,
                num_embeddings_padded=self.num_embeddings_padded,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
            )
            self.spec.allocate(
                self,
                AllocationRequest(
                    weight_shape=(self.num_embeddings_per_partition, embedding_dim),
                    logical_widths=[self.num_embeddings_per_partition],
                    fused_dim=0,
                    weight_loader=loader,
                    params_dtype=self.params_dtype,
                ),
            )

        self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel = get_linear_dispatcher().select(
            spec_id=self.spec.spec_id,
            M=_M_of(x),
            N=self.output_size_per_partition,
            K=self.input_size_per_partition,
            in_dtype=x.dtype,
            out_dtype=self.params_dtype,
        )
        y = kernel.apply(self, x, self.bias)
        if self.gather_output and self.tp_size > 1:
            y = P.all_gather(y, axis=self.axis, dim=-1, mesh=self.mesh_name)
        return y

    def extra_repr(self) -> str:
        return (
            f"embedding_dim={self.embedding_dim}, "
            f"num_embeddings={self.num_embeddings}, "
            f"per_partition={self.num_embeddings_per_partition}, "
            f"padded={self.num_embeddings_padded}, "
            f"tp_size={self.tp_size}, axis={self.axis!r}, "
            f"gather_output={self.gather_output}"
        )


__all__ = [
    "VocabParallelEmbedding",
    "ParallelLMHead",
    "pad_vocab_to",
]
