"""Op enum + Backend Protocol + Topology dataclass.

Backend authors implement this Protocol. Capability is a pure predicate
(``can_handle(...) -> bool``) — there is no ``score()`` method; priority
is the Registry's job.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
import torch.distributed as dist

from phyai.parallel.state import Mode


class Op(Enum):
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    BROADCAST = "broadcast"
    ALL_TO_ALL = "all_to_all"
    SEND = "send"
    RECV = "recv"
    BARRIER = "barrier"


@dataclass(frozen=True)
class Topology:
    """Static topology hints visible to ``Backend.can_handle``.

    Coarse-grained on purpose; backends that need finer detail (e.g.
    custom all-reduce kernels) should run their own probes.
    """

    is_full_nvlink: bool
    is_single_node: bool
    n_nodes: int
    n_gpus_per_node: int


@runtime_checkable
class Backend(Protocol):
    """Backend protocol. Pure capability + execute, no internal mode flags.

    ``can_handle`` is called only on Dispatcher cache miss; the hot path
    is the cache lookup.
    """

    name: str

    def can_handle(
        self,
        *,
        op: Op,
        mode: Mode,
        nbytes: int,
        dtype: torch.dtype,
        world_size: int,
        topology: Topology,
        **extra: object,
    ) -> bool: ...

    def supports_capture(self) -> bool: ...

    def execute(
        self,
        *,
        op: Op,
        pg: dist.ProcessGroup,
        **kwargs: object,
    ) -> torch.Tensor | None: ...
