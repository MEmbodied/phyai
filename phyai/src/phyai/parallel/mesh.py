"""Mesh + Topology inference.

Thin wrapper over ``torch.distributed.DeviceMesh`` that:
  - caches per-axis ProcessGroup lookups,
  - infers a coarse ``Topology`` once and caches it.

The Mesh has no Dispatcher reference — the Dispatcher is a process-level
singleton; see :mod:`phyai.parallel.dispatch`.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from phyai.parallel.backend import Topology


class Mesh:
    """Named multi-axis mesh wrapper."""

    def __init__(self, torch_mesh: DeviceMesh, *, name: str = "model") -> None:
        self.torch_mesh = torch_mesh
        self.name = name
        self._pg_cache: dict[str, dist.ProcessGroup] = {}
        self._topo: Topology | None = None

    def axis_group(self, axis: str) -> dist.ProcessGroup:
        if axis not in self._pg_cache:
            self._pg_cache[axis] = self.torch_mesh.get_group(axis)
        return self._pg_cache[axis]

    def axis_size(self, axis: str) -> int:
        # Torch 2.10's ``DeviceMesh.size(name)`` has a regression: the layout
        # indexer rejects non-int keys, so we resolve the name → index
        # ourselves. ``axis_local_rank`` already uses the name path and works.
        names = self.torch_mesh.mesh_dim_names
        if names is None:
            return self.torch_mesh.size()
        return self.torch_mesh.size(names.index(axis))

    def axis_local_rank(self, axis: str) -> int:
        return self.torch_mesh.get_local_rank(axis)

    def topology(self) -> Topology:
        if self._topo is None:
            self._topo = _infer_topology(self.torch_mesh)
        return self._topo

    def __repr__(self) -> str:
        names = self.torch_mesh.mesh_dim_names
        if not names:
            return f"Mesh(name={self.name!r}, size={self.torch_mesh.size()})"
        # ``rank/size`` per axis: where this process sits, plus the axis width.
        # Reads as "tp axis: rank 3 out of 8".
        parts = [
            f"{ax}={self.axis_local_rank(ax)}/{self.axis_size(ax)}" for ax in names
        ]
        return f"Mesh(name={self.name!r}, {', '.join(parts)})"


def _infer_topology(torch_mesh: DeviceMesh) -> Topology:
    """Coarse topology inference. Conservative defaults are fine here —
    backends that need precise info should refine their own probes.
    """
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    # Best-effort GPUs-per-node: assume the local CUDA device count.
    try:
        n_per_node = max(torch.cuda.device_count(), 1)
    except Exception:
        n_per_node = 1
    n_nodes = max(1, world_size // n_per_node) if n_per_node else 1
    return Topology(
        is_full_nvlink=(n_nodes == 1),
        is_single_node=(n_nodes == 1),
        n_nodes=n_nodes,
        n_gpus_per_node=n_per_node,
    )
