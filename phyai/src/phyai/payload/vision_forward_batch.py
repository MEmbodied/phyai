"""Vision runner forward batch."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class VisionForwardBatch:
    """Payload for one vision-tower call.

    A vision runner is captured once at fixed shape ``(N, C, H, W)`` —
    ``N`` images of ``C`` channels at the configured spatial size — and
    replayed once per request batch. The scheduler builds one
    :class:`VisionForwardBatch` per replay and concatenates the
    runner's outputs along the token axis afterwards.

    Fields
    ------
    pixel_values:
        Float tensor of shape ``(N, C, H, W)``: ``N`` images per call,
        ``C`` channels, spatial dims ``H == W == image_size``. dtype is
        the engine's ``params_dtype`` so the captured graph never
        inserts a runtime cast.
    """

    pixel_values: torch.Tensor


__all__ = ["VisionForwardBatch"]
