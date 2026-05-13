"""Per-parameter weight loaders.

Every nn.Parameter that comes out of phyai.layers has a ``param.loader``
hanging off it. The modeling-level checkpoint loop reaches for that
attribute and calls a single method::

    param.loader.load_weight(param, loaded, shard_id=...)

The contract on ``loaded`` is one rule: it always carries the full,
natural, un-sharded weight tensor for whatever it represents. There is
no "pre-fused on disk" or "pre-replicated KV" path — those layouts never
worked under GQA and are not supported.

What ``shard_id`` selects:

* ``None`` — ``loaded`` is the whole param. Valid for non-fused loaders
  (:class:`ReplicatedLoader`, :class:`RowShardLoader`,
  :class:`VocabShardLoader`, and :class:`ColumnShardLoader` configured
  with a single output partition).
* ``int`` — ``loaded`` is one full sub-matrix of a fused
  ``MergedColumnParallelLinear``-style param. Used by
  :class:`ColumnShardLoader` when ``output_partition_sizes`` has more
  than one entry.
* ``"q" | "k" | "v"`` — ``loaded`` is the full Q, K, or V matrix.
  Used by :class:`QKVShardLoader`. K and V take the GQA replication
  path on top of the column-shard slicing.

Each loader rejects shapes its protocol does not handle, so a buggy
checkpoint mapping fails at the first mismatched tensor instead of
silently writing garbage.

Loaders only carry their TP layout (``tp_rank``, ``tp_size``, partition
sizes), not per-tensor state, so a column- or qkv-parallel layer hands
the same instance to both its weight and its bias.
"""

from __future__ import annotations

import torch
import torch.nn as nn

ShardId = int | str | None


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

    def load_weight(
        self,
        param: nn.Parameter,
        loaded: torch.Tensor,
        shard_id: ShardId = None,
    ) -> None:
        if shard_id is not None:
            raise ValueError(
                f"ReplicatedLoader does not support sharded loads "
                f"(got shard_id={shard_id!r})"
            )
        if param.numel() == 1 and loaded.numel() == 1:
            param.data.fill_(loaded.item())
            return
        if param.shape != loaded.shape:
            raise ValueError(
                f"ReplicatedLoader.load_weight: param shape {tuple(param.shape)} "
                f"!= loaded shape {tuple(loaded.shape)}"
            )
        param.data.copy_(loaded)


class ColumnShardLoader:
    """Slice along dim 0 and write into a (possibly fused) per-rank parameter.

    ``output_partition_sizes`` is the per-rank output width of each logical
    matrix on this layer. For plain ColumnParallelLinear that's a one-entry
    list ``[out // tp]``; for MergedColumnParallelLinear it's one entry per
    fused sub-matrix, e.g. ``[gate // tp, up // tp]``.

    Both code paths narrow on dim 0, which works equally well for the 2-D
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

    def load_weight(
        self,
        param: nn.Parameter,
        loaded: torch.Tensor,
        shard_id: ShardId = None,
    ) -> None:
        n_slots = len(self.output_partition_sizes)

        if shard_id is None:
            # Non-fused layer: ``loaded`` is the whole un-sharded weight.
            if n_slots != 1:
                raise ValueError(
                    f"ColumnShardLoader.load_weight: param is fused "
                    f"({n_slots} slots, output_partition_sizes="
                    f"{self.output_partition_sizes}); shard_id must be int"
                )
            per_rank = self.output_partition_sizes[0]
            global_out = per_rank * self.tp_size
            if loaded.shape[0] != global_out:
                raise ValueError(
                    f"ColumnShardLoader.load_weight: loaded.shape[0]="
                    f"{loaded.shape[0]} != global_out={global_out}"
                )
            sliced = loaded.narrow(0, self.tp_rank * per_rank, per_rank)
            param.data.copy_(sliced)
            return

        if not isinstance(shard_id, int) or isinstance(shard_id, bool):
            raise TypeError(
                f"ColumnShardLoader.load_weight: shard_id must be None or int, "
                f"got {shard_id!r}"
            )
        if shard_id < 0 or shard_id >= n_slots:
            raise IndexError(
                f"shard_id={shard_id} out of range for "
                f"output_partition_sizes={self.output_partition_sizes}"
            )
        per_rank = self.output_partition_sizes[shard_id]
        global_size = per_rank * self.tp_size
        if loaded.shape[0] != global_size:
            raise ValueError(
                f"ColumnShardLoader.load_weight({shard_id}): loaded.shape[0]="
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

    def load_weight(
        self,
        param: nn.Parameter,
        loaded: torch.Tensor,
        shard_id: ShardId = None,
    ) -> None:
        if shard_id is not None:
            raise ValueError(
                f"RowShardLoader does not support sharded loads "
                f"(got shard_id={shard_id!r})"
            )
        shard = loaded.shape[1] // self.tp_size
        if shard * self.tp_size != loaded.shape[1]:
            raise ValueError(
                f"RowShardLoader.load_weight: in_dim={loaded.shape[1]} "
                f"not divisible by tp_size={self.tp_size}"
            )
        sliced = loaded.narrow(1, self.tp_rank * shard, shard)
        param.data.copy_(sliced)


class VocabShardLoader:
    """Load a vocab embedding / LM head weight whose disk shape is ``(V_real, D)``
    into a per-rank parameter of shape ``(V_padded // tp_size, D)``.

    The layer pads ``num_embeddings`` up to ``num_embeddings_padded`` (a multiple
    of ``tp_size``) so every rank holds the same chunk size. Padding rows that
    extend past ``num_embeddings`` exist on at most one rank — the trailing
    rank — and must read as zero so they contribute nothing to the lookup
    output (after masked all-reduce) or to the LM-head logits.

    Unlike :class:`ColumnShardLoader`, this loader cannot share its dim-0
    ``narrow`` directly because the disk tensor's outer dim is ``V_real``, not
    ``V_padded``. We compute the overlap of ``[start, end)`` with
    ``[0, V_real)`` and only copy that intersection; the rest is zeroed.
    """

    def __init__(
        self,
        *,
        num_embeddings: int,
        num_embeddings_padded: int,
        tp_rank: int,
        tp_size: int,
    ) -> None:
        if num_embeddings_padded % tp_size != 0:
            raise ValueError(
                f"VocabShardLoader: num_embeddings_padded={num_embeddings_padded} "
                f"not divisible by tp_size={tp_size}"
            )
        if num_embeddings_padded < num_embeddings:
            raise ValueError(
                f"VocabShardLoader: num_embeddings_padded={num_embeddings_padded} "
                f"< num_embeddings={num_embeddings}"
            )
        self.num_embeddings = num_embeddings
        self.num_embeddings_padded = num_embeddings_padded
        self.tp_rank = tp_rank
        self.tp_size = tp_size

    def load_weight(
        self,
        param: nn.Parameter,
        loaded: torch.Tensor,
        shard_id: ShardId = None,
    ) -> None:
        """Copy this rank's slice of the full ``(V_real, ...)`` table and zero
        any padding overhang.

        Works for any rank where dim 0 is the vocab dimension: 2-D embedding
        weights ``(V_real, D)`` and 1-D LM-head biases ``(V_real,)`` both flow
        through unchanged.
        """
        if shard_id is not None:
            raise ValueError(
                f"VocabShardLoader does not support sharded loads "
                f"(got shard_id={shard_id!r})"
            )
        per_rank = self.num_embeddings_padded // self.tp_size
        if param.shape[0] != per_rank:
            raise ValueError(
                f"VocabShardLoader.load_weight: param.shape[0]={param.shape[0]} "
                f"!= per_rank={per_rank}"
            )
        if loaded.shape[0] != self.num_embeddings:
            raise ValueError(
                f"VocabShardLoader.load_weight: loaded.shape[0]={loaded.shape[0]} "
                f"!= num_embeddings={self.num_embeddings}"
            )
        if loaded.shape[1:] != param.shape[1:]:
            raise ValueError(
                f"VocabShardLoader.load_weight: trailing dims mismatch "
                f"loaded={tuple(loaded.shape)} param={tuple(param.shape)}"
            )

        start = self.tp_rank * per_rank
        end = start + per_rank
        real_start = min(max(start, 0), self.num_embeddings)
        real_end = min(end, self.num_embeddings)
        n_real = max(0, real_end - real_start)

        if n_real > 0:
            sliced = loaded.narrow(0, real_start, n_real)
            param.data.narrow(0, 0, n_real).copy_(sliced)
        if n_real < per_rank:
            # Padding rows on this rank must be exactly zero.
            param.data.narrow(0, n_real, per_rank - n_real).zero_()


class QKVShardLoader(ColumnShardLoader):
    """Q/K/V fused-projection loader with GQA / MQA support.

    Q is column-sharded and behaves like a normal sub-matrix. K and V are
    different: when ``tp_size`` is a multiple of ``num_kv_heads`` we don't
    slice K/V any further. Instead each KV slice is replicated
    ``num_kv_replicas = tp_size // num_kv_heads`` times across ranks, so
    the disk tensor's outer dim is the un-replicated width and adjacent
    ranks within a replica group read the same slot.

    The fused full-disk path (``shard_id=None``) is unsupported — under
    GQA the disk fused width would not equal ``sum(output_partition_sizes)
    * tp_size`` because K/V on disk are un-replicated. Always pass
    ``shard_id="q"|"k"|"v"`` and feed each natural sub-matrix separately.
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

    def load_weight(
        self,
        param: nn.Parameter,
        loaded: torch.Tensor,
        shard_id: ShardId = None,
    ) -> None:
        if not isinstance(shard_id, str) or shard_id not in self._QKV_IDX:
            raise ValueError(
                f"QKVShardLoader.load_weight: shard_id must be one of q/k/v, "
                f"got {shard_id!r}"
            )
        idx = self._QKV_IDX[shard_id]
        per_rank = self.output_partition_sizes[idx]

        # Natural full-tensor width on disk: num_q_heads*head_dim for Q,
        # num_kv_heads*head_dim for K/V. With GQA replication the K/V
        # natural width is smaller than per_rank*tp_size — every replica
        # group of ranks shares the same KV head, so dividing out the
        # replica count recovers the original head count.
        replica_factor = 1 if idx == 0 else self.num_kv_replicas
        natural_width = per_rank * self.tp_size // replica_factor
        if loaded.shape[0] != natural_width:
            raise ValueError(
                f"QKVShardLoader.load_weight({shard_id!r}): loaded.shape[0]="
                f"{loaded.shape[0]} != natural_width={natural_width}"
            )
        # tp_rank picks the slot for Q. For K/V the same slot is shared
        # across num_kv_replicas adjacent ranks, so we floor-divide.
        slot_rank = self.tp_rank // replica_factor
        offset = sum(self.output_partition_sizes[:idx])
        sliced = loaded.narrow(0, slot_rank * per_rank, per_rank)

        param.data.narrow(0, offset, per_rank).copy_(sliced)


__all__ = [
    "ColumnShardLoader",
    "QKVShardLoader",
    "ReplicatedLoader",
    "RowShardLoader",
    "ShardId",
    "VocabShardLoader",
]
