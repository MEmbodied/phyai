"""Shared weight-loader factories — sharding math lives here, once.

Each function returns a :data:`WeightLoader` closure with signature
``(param, loaded, shard_id) -> None``. Layers attach the closure to
their ``nn.Parameter`` so :func:`phyai.weights.load_pretrained` can
dispatch generically. The math runs at load time only — there is no
forward-time cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from phyai.parallel.mesh import Mesh


WeightLoader = Callable[[torch.nn.Parameter, torch.Tensor, "int | str | None"], None]


def replicated() -> WeightLoader:
    """Full-tensor copy. Handles 0-D HF scalars expanding into 1-element params."""

    def load(param: torch.nn.Parameter, loaded: torch.Tensor, _shard_id=None) -> None:
        if loaded.dim() == 0 and param.numel() == 1:
            param.data.fill_(loaded.item())
            return
        param.data.copy_(loaded)

    return load


def sharded(
    *, dim: int, axis: str = "tp", mesh: Mesh, replicate: int = 1
) -> WeightLoader:
    """Single-axis TP / EP shard along ``dim``.

    ``replicate>1`` is the GQA case: ``replicate`` ranks on ``axis``
    each read the same shard slot. The effective world size shrinks by
    ``replicate``; the effective rank is ``rank // replicate``.
    """

    def load(param: torch.nn.Parameter, loaded: torch.Tensor, _shard_id=None) -> None:
        rank = mesh.axis_local_rank(axis) // replicate
        world = mesh.axis_size(axis) // replicate
        size = loaded.shape[dim] // world
        param.data.copy_(loaded.narrow(dim, rank * size, size))

    return load


@dataclass(frozen=True)
class _Leg:
    """One leg of a fused-param load.

    ``offset`` / ``size`` are the **post-shard local** position and size
    in the destination's ``fuse_dim``. ``dim`` is the source's TP-shard
    dim (almost always ``0`` — column-parallel fuse). ``replicate`` is
    the GQA replication factor for K/V legs.
    """

    offset: int
    size: int
    dim: int = 0
    axis: str = "tp"
    replicate: int = 1


def fused(*, fuse_dim: int, legs: dict, mesh: Mesh) -> WeightLoader:
    """Multi-source fused-param loader.

    ``legs`` maps ``shard_id`` -> :class:`_Leg`. The loader looks up the
    leg by ``shard_id`` (received from the param's ``hf_keys`` entry),
    TP-shards the source, and writes into the destination's ``fuse_dim``
    slot at ``[offset, offset+size)``.

    Covers fused QKV with GQA (Q has ``replicate=1``, K/V have
    ``replicate=num_kv_replicas``) and fused gate/up.
    """

    def load(param: torch.nn.Parameter, loaded: torch.Tensor, shard_id) -> None:
        leg = legs[shard_id]
        rank = mesh.axis_local_rank(leg.axis) // leg.replicate
        src = loaded.narrow(leg.dim, rank * leg.size, leg.size)
        param.data.narrow(fuse_dim, leg.offset, leg.size).copy_(src)

    return load


def scale_sharded(
    *,
    extra_attrs: dict,
    axis: str = "tp",
    mesh: Mesh,
    replicate: int = 1,
    row_parallel: bool = False,
) -> WeightLoader:
    """Shard a scale / zero-point param to match its weight's TP split.

    ``extra_attrs`` is the humming param metadata (``output_dim`` / ``input_dim``
    / ``packed_dim`` / ``packed_factor`` / ``scale_type``) attached by
    ``HummingWeightSpec.allocate``. The rule mirrors the weight:

    * Column-family (``row_parallel=False``): shard the N/output dim when the
      param indexes it (``output_dim`` present); otherwise replicate. K is not
      sharded on column-parallel, so a group/block scale only moves along N.
    * Row-family (``row_parallel=True``): shard the K/input dim when the param
      indexes it (``input_dim`` present); a per-channel scale (``output_dim``
      only) is replicated because row-parallel does not shard N.

    A param with neither axis (per-tensor ``global_scale``) is replicated.
    A packed int32 dim is narrowed like any other; the per-rank extent must be a
    multiple of ``packed_factor`` (enforced by the divisibility check).
    """
    out_dim = extra_attrs.get("output_dim")
    in_dim = extra_attrs.get("input_dim")

    def load(param: torch.nn.Parameter, loaded: torch.Tensor, _shard_id=None) -> None:
        dim = in_dim if row_parallel else out_dim
        if dim is None:
            param.data.copy_(loaded)
            return
        rank = mesh.axis_local_rank(axis) // replicate
        world = mesh.axis_size(axis) // replicate
        full = loaded.shape[dim]
        if full % world != 0:
            raise ValueError(
                f"scale_sharded: dim {dim} size {full} not divisible by world {world}"
            )
        size = full // world
        param.data.copy_(loaded.narrow(dim, rank * size, size))

    return load


@dataclass(frozen=True)
class _ScaleLeg:
    """One leg of a fused scale/zero-point load, in *weight* coordinates.

    ``weight_offset`` / ``weight_size`` are the post-shard local N sizes of the
    leg's weight; ``total_weight`` is their sum across legs. The fused scale's
    N-extent is a constant factor of the weight's N (1 for channel/group,
    ``1/group_size_n`` for block, ``num_bits/32`` for packed zero-point), so the
    scale's per-leg slot is derived proportionally from the param's own shape.
    """

    weight_offset: int
    weight_size: int
    total_weight: int
    axis: str = "tp"
    replicate: int = 1


def scale_fused(*, fuse_dim: int, legs: dict, mesh: Mesh) -> WeightLoader:
    """Fused scale/zero-point loader for Merged / QKV linears (column-family).

    Each leg loads from its own disk key, is TP-sharded along ``fuse_dim``
    (N/output) with GQA ``replicate`` for K/V, and written into the fused
    param's proportional N-slot. K (dim 1) is never sharded here — column-family
    linears split N only.
    """

    def load(param: torch.nn.Parameter, loaded: torch.Tensor, shard_id) -> None:
        leg = legs[shard_id]
        d = param.shape[fuse_dim]
        if (d * leg.weight_size) % leg.total_weight != 0:
            raise ValueError(
                f"scale_fused: scale dim {d} not divisible proportionally for leg "
                f"{leg.weight_size}/{leg.total_weight} (check group/pack alignment)"
            )
        dest_size = d * leg.weight_size // leg.total_weight
        dest_off = d * leg.weight_offset // leg.total_weight
        rank = mesh.axis_local_rank(leg.axis) // leg.replicate
        world = mesh.axis_size(leg.axis) // leg.replicate
        src_size = loaded.shape[fuse_dim] // world
        src = loaded.narrow(fuse_dim, rank * src_size, src_size)
        param.data.narrow(fuse_dim, dest_off, dest_size).copy_(src)

    return load


def weight_norm_fold(*, eps: float = 1e-12) -> WeightLoader:
    """Fold a legacy ``weight_norm`` (``weight_g`` / ``weight_v``) pair into a dense weight.

    ``torch.nn.utils.weight_norm`` (now deprecated) reparametrises a weight as a
    magnitude ``g`` and direction ``v``; the forward weight is ``g * v / ‖v‖`` with
    the norm taken over every dim except ``0`` (the ``dim=0`` default — correct for
    both ``Conv`` ``(out, in, *k)`` and ``ConvTranspose`` ``(in, out, *k)`` layouts).
    This loader caches the two source tensors (``shard_id`` ``"g"`` / ``"v"``, either
    arrival order) and, once both are in, writes the dense forward weight. So a layer
    can carry a single dense ``weight`` and never run ``weight_norm`` at inference.

    A fresh closure (with its own per-parameter cache) is returned per call — attach
    one to each parameter, not a shared instance.
    """

    cache: dict[str, torch.Tensor] = {}

    def load(param: torch.nn.Parameter, loaded: torch.Tensor, shard_id) -> None:
        if shard_id not in ("g", "v"):
            raise ValueError(
                f"weight_norm_fold expects shard_id 'g' or 'v', got {shard_id!r}"
            )
        cache[shard_id] = loaded.to(torch.float32)
        if "g" in cache and "v" in cache:
            g = cache.pop("g")
            v = cache.pop("v")
            dims = tuple(range(1, v.dim()))  # all dims except 0 (weight_norm dim=0)
            norm = v.norm(dim=dims, keepdim=True).clamp_min(eps)
            param.data.copy_((g * v / norm).to(param.dtype))

    return load


def vocab(*, axis: str = "tp", mesh: Mesh) -> WeightLoader:
    """Vocab-parallel embedding load with right-edge zero padding.

    The destination's ``shape[0]`` is the per-rank padded size. The HF
    tensor has ``V_real`` rows. Each rank loads the slice of real rows
    that fall in its range and zero-fills the trailing pad on the last
    rank. No full-padded source is materialised.
    """

    def load(param: torch.nn.Parameter, loaded: torch.Tensor, _shard_id=None) -> None:
        per_rank = param.shape[0]
        rank = mesh.axis_local_rank(axis)
        start = rank * per_rank
        v_real = loaded.shape[0]
        if start >= v_real:
            param.data.zero_()
            return
        n_real = min(start + per_rank, v_real) - start
        param.data.narrow(0, 0, n_real).copy_(loaded.narrow(0, start, n_real))
        if n_real < per_rank:
            param.data.narrow(0, n_real, per_rank - n_real).zero_()

    return load


__all__ = [
    "WeightLoader",
    "_Leg",
    "_ScaleLeg",
    "fused",
    "replicated",
    "scale_fused",
    "scale_sharded",
    "sharded",
    "vocab",
    "weight_norm_fold",
]
