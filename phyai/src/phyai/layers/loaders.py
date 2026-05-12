"""Per-parameter weight loaders.

Every nn.Parameter that comes out of phyai.layers has a ``param.loader``
hanging off it. Checkpoint-loading code reaches for that attribute and
calls one of three methods:

* ``load_full(param, loaded)``: a single, un-fused tensor. Always available.
  May still slice along dim 0 or dim 1 to grab this rank's slice.
* ``load_shard(param, loaded, shard_id: int)``: write one logical
  sub-matrix into a fused param (e.g. gate vs. up of a fused MLP).
* ``load_qkv(param, loaded, shard_id: str)``: same idea for the Q/K/V
  fused projection, with named shards instead of integers.

Loaders only carry their TP layout (``tp_rank``, ``tp_size``, partition
sizes), not per-tensor state. So a column- or qkv-parallel layer hands the
same instance to both its weight and its bias.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ReplicatedLoader:
    """Trivial loader for parameters that are bit-identical across ranks.

    Used by RMSNorm / LayerNorm weights, ReplicatedLinear, and the bias on
    RowParallelLinear (only rank 0 actually adds it at forward time, but
    every rank still owns the full tensor).

    The 1-element fast path covers checkpoints that store a learned scalar
    as ``shape=()`` even though the in-memory parameter is ``shape=(1,)``.
    A strict ``copy_`` would reject that. Anything else gets a size-checked
    copy with no slicing.
    """

    def load_full(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        if param.numel() == 1 and loaded.numel() == 1:
            param.data.fill_(loaded.item())
            return
        if param.shape != loaded.shape:
            raise ValueError(
                f"ReplicatedLoader.load_full: param shape {tuple(param.shape)} "
                f"!= loaded shape {tuple(loaded.shape)}"
            )
        param.data.copy_(loaded)


class ColumnShardLoader:
    """Slice along dim 0 and write into a (possibly fused) per-rank parameter.

    ``output_partition_sizes`` is the per-rank output width of each logical
    matrix on this layer. For plain ColumnParallelLinear that's a one-entry
    list ``[out // tp]``; for MergedColumnParallelLinear it's one entry per
    fused sub-matrix, e.g. ``[gate // tp, up // tp]``.

    Both methods narrow on dim 0, which works equally well for the 2-D
    weight and the 1-D bias of the same layer, so a single instance can
    drive both.
    """

    def __init__(
        self,
        *,
        output_partition_sizes: list[int],
        tp_rank: int,
        tp_size: int,
    ) -> None:
        self.output_partition_sizes = output_partition_sizes
        self.tp_rank = tp_rank
        self.tp_size = tp_size

    def load_full(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        """Disk tensor is the global, un-sharded fused matrix."""
        global_out = sum(self.output_partition_sizes) * self.tp_size
        if loaded.shape[0] != global_out:
            raise ValueError(
                f"ColumnShardLoader.load_full: loaded.shape[0]={loaded.shape[0]} "
                f"!= global_out={global_out}"
            )
        per_rank = sum(self.output_partition_sizes)
        sliced = loaded.narrow(0, self.tp_rank * per_rank, per_rank)
        param.data.copy_(sliced)

    def load_shard(
        self,
        param: nn.Parameter,
        loaded: torch.Tensor,
        shard_id: int,
    ) -> None:
        """Disk tensor is one logical sub-matrix; write it into its slot.

        ``loaded`` carries only sub-matrix ``shard_id``, in its full global
        width. We take this rank's slice of it and place it at offset
        ``sum(partition_sizes[:shard_id])`` inside the fused parameter.
        """
        if shard_id < 0 or shard_id >= len(self.output_partition_sizes):
            raise IndexError(
                f"shard_id={shard_id} out of range for "
                f"output_partition_sizes={self.output_partition_sizes}"
            )
        per_rank = self.output_partition_sizes[shard_id]
        global_size = per_rank * self.tp_size
        if loaded.shape[0] != global_size:
            raise ValueError(
                f"ColumnShardLoader.load_shard({shard_id}): loaded.shape[0]="
                f"{loaded.shape[0]} != global_size={global_size}"
            )
        offset = sum(self.output_partition_sizes[:shard_id])
        sliced = loaded.narrow(0, self.tp_rank * per_rank, per_rank)
        param.data.narrow(0, offset, per_rank).copy_(sliced)


class RowShardLoader:
    """Slice along dim 1 (the input dim) of a 2-D weight.

    The bias on a row-parallel layer is global, not sharded, so it gets a
    ReplicatedLoader instead. This class only ever drives the 2-D weight.
    """

    def __init__(self, *, tp_rank: int, tp_size: int) -> None:
        self.tp_rank = tp_rank
        self.tp_size = tp_size

    def load_full(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        shard = loaded.shape[1] // self.tp_size
        if shard * self.tp_size != loaded.shape[1]:
            raise ValueError(
                f"RowShardLoader.load_full: in_dim={loaded.shape[1]} "
                f"not divisible by tp_size={self.tp_size}"
            )
        sliced = loaded.narrow(1, self.tp_rank * shard, shard)
        param.data.copy_(sliced)


class QKVShardLoader(ColumnShardLoader):
    """Q/K/V fused-projection loader with GQA / MQA support.

    Q is column-sharded and behaves like a normal sub-matrix. K and V are
    different: when ``tp_size`` is a multiple of ``num_kv_heads`` we don't
    slice K/V any further. Instead each KV slice is replicated
    ``num_kv_replicas = tp_size // num_kv_heads`` times across ranks, so
    the disk tensor's outer dim is the un-replicated width and adjacent
    ranks within a replica group read the same slot.

    ``load_full`` and ``load_shard`` are inherited unchanged. Their dim-0
    narrow happens to work on both the 2-D fused weight and the 1-D fused
    bias, so we don't need a separate bias path.
    """

    _QKV_IDX = {"q": 0, "k": 1, "v": 2}

    def __init__(
        self,
        *,
        q_size: int,
        kv_size: int,
        num_kv_replicas: int,
        tp_rank: int,
        tp_size: int,
    ) -> None:
        super().__init__(
            output_partition_sizes=[q_size, kv_size, kv_size],
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        if num_kv_replicas < 1:
            raise ValueError(f"num_kv_replicas must be ≥1, got {num_kv_replicas}")
        self.num_kv_replicas = num_kv_replicas

    def load_qkv(
        self,
        param: nn.Parameter,
        loaded: torch.Tensor,
        shard_id: str,
    ) -> None:
        if shard_id not in self._QKV_IDX:
            raise ValueError(f"shard_id must be one of q/k/v, got {shard_id!r}")
        idx = self._QKV_IDX[shard_id]
        per_rank = self.output_partition_sizes[idx]

        if idx == 0:
            # Q is a plain column shard. Disk has the full global Q width.
            global_size = per_rank * self.tp_size
            if loaded.shape[0] != global_size:
                raise ValueError(
                    f"QKVShardLoader.load_qkv('q'): loaded.shape[0]="
                    f"{loaded.shape[0]} != global_q={global_size}"
                )
            offset = 0
            sliced = loaded.narrow(0, self.tp_rank * per_rank, per_rank)
        else:
            # K and V live on the un-replicated width on disk. This rank
            # reads from partition (tp_rank // num_kv_replicas), which is
            # how the same slice gets shared across the replica group.
            disk_width = per_rank * self.tp_size // self.num_kv_replicas
            if loaded.shape[0] != disk_width:
                raise ValueError(
                    f"QKVShardLoader.load_qkv({shard_id!r}): loaded.shape[0]="
                    f"{loaded.shape[0]} != disk_width={disk_width}"
                )
            offset = sum(self.output_partition_sizes[:idx])
            kv_rank = self.tp_rank // self.num_kv_replicas
            sliced = loaded.narrow(0, kv_rank * per_rank, per_rank)

        param.data.narrow(0, offset, per_rank).copy_(sliced)


__all__ = [
    "ReplicatedLoader",
    "ColumnShardLoader",
    "RowShardLoader",
    "QKVShardLoader",
]
