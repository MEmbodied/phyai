"""Shared weight-loader factories — sharding math lives here, once.

Each function returns a :data:`WeightLoader` closure with signature
``(param, loaded, shard_id) -> None``. Layers attach the closure to
their ``nn.Parameter`` so :func:`phyai.weights.load_pretrained` can
dispatch generically. The math runs at load time only — there is no
forward-time cost.

Five primitives cover every parallelism shape phyai uses today:

* :func:`replicated` — full copy. Default for norms, replicated linear,
  conv, and replicated quant scales.
* :func:`sharded` — single-axis TP / EP shard. ``replicate>1`` covers
  GQA, where every ``replicate`` ranks on the named axis read the same
  source slot.
* :func:`fused` — multiple HF tensors fuse into one param along
  ``fuse_dim``. Each leg has its own ``_Leg`` describing its
  destination offset/size and shard. Covers fused QKV (with GQA
  replication) and gate/up.
* :func:`vocab` — vocab-parallel embedding with right-edge zero padding
  on the trailing rank. Padding is written inline; no full-padded
  source is materialised.
* :func:`moe_expert` — MoE expert weights with EP-then-TP composition.
  Adds a second ``narrow`` to the basic ``sharded`` flow plus optional
  fuse-dim handling for gate/up-fused experts.
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
    "fused",
    "replicated",
    "sharded",
    "vocab",
]
