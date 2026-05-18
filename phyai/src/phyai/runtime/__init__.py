"""phyai.runtime — runtime / inference plumbing shared by every model.

Modules:

* :mod:`cuda_graph_manager` — :class:`CudaGraph` (single captured
  graph) and :class:`CudaGraphRegistry` (multi-shape dispatch).
* :mod:`model_runner` — :class:`ModelRunner` ABC.
* :mod:`schedule` — scheduler ABC and primitives.

Per-model runners and schedulers live alongside the model definitions
under :mod:`phyai.models`. The runtime package only owns the shared
base classes.
"""

from __future__ import annotations

from phyai.runtime.cuda_graph_manager import (
    CudaGraph,
    CudaGraphError,
    CudaGraphRegistry,
)
from phyai.runtime.model_runner import ModelRunner
from phyai.runtime.schedule import Scheduler


__all__ = [
    "CudaGraph",
    "CudaGraphError",
    "CudaGraphRegistry",
    "ModelRunner",
    "Scheduler",
]
