"""phyai.payload — runner-input dataclasses.

Each :class:`~phyai.runtime.model_runner.ModelRunner` accepts a
single forward-batch payload that bundles every tensor and metadata
field required to do exactly one model pass. The payloads are intentionally
small and per-runner; the
:class:`~phyai.runtime.schedule.base.Scheduler` plans a request, builds
the appropriate forward-batch, hands it to a runner, packs the runner's
output into the next runner's forward-batch, and so on.

* :class:`VisionForwardBatch` — vision-tower runner (one image stack
  per call).
* :class:`LLMForwardBatch` — text/prefix LLM backbone runner (prefix
  phase, K/V written to the cache pool).
* :class:`ExpertForwardBatch` — action-expert runner (one denoise step
  against the cached prefix K/V).
"""

from __future__ import annotations

from phyai.payload.expert_forward_batch import ExpertForwardBatch
from phyai.payload.llm_forward_batch import LLMForwardBatch
from phyai.payload.vision_forward_batch import VisionForwardBatch


__all__ = [
    "ExpertForwardBatch",
    "LLMForwardBatch",
    "VisionForwardBatch",
]
