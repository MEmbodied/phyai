"""``rename_observations_processor`` — remap transition keys by a dict.

lerobot's pi05 preprocessor includes this step (with ``rename_map={}`` in the
pi05_base checkpoint, i.e. a passthrough) to align observation keys with the
pretrained pipeline. phyai implements it so the lerobot json loads without an
"unknown step" error; with a non-empty ``rename_map`` it renames the matching
transition keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from phyai_utils_tools.processing.pipeline import (
    ProcessorStep,
    ProcessorStepRegistry,
)
from phyai_utils_tools.processing.transition import Transition


@ProcessorStepRegistry.register("rename_observations_processor")
@dataclass
class RenameObservationsStep(ProcessorStep):
    """Rename transition keys per ``rename_map`` (``{}`` ⇒ passthrough)."""

    rename_map: dict[str, str] = field(default_factory=dict)

    def __call__(self, transition: Transition) -> Transition:
        if not self.rename_map:
            return transition
        return {self.rename_map.get(k, k): v for k, v in transition.items()}

    def get_config(self) -> dict[str, Any]:
        return {"rename_map": dict(self.rename_map)}


__all__ = ["RenameObservationsStep"]
