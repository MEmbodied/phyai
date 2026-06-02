"""WALL-OSS-FLOW plugin entry for PhyAI.

First-version scope:
- FLOW only, not FAST;
- wall-oss-0.5 is a later migration target;
- reuse wall-x Qwen2_5_VLMoEForAction instead of reimplementing model internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import torch

from phyai.engine import Engine, Entry, EntryArgs
from phyai.engine_config import get_engine_config
from phyai.models.walloss.configuration_walloss import WallOSSFlowConfig
from phyai.models.walloss.model_runner_walloss import WallOSSFlowRunner
from phyai.models.walloss.scheduler_ws1_walloss import (
    WallOSSFlowPredictRequest,
    WallOSSFlowRequest,
    WallOSSFlowWS1Scheduler,
)


@dataclass
class WallOSSArgs(EntryArgs):
    """Args bundle for the first-version WALL-OSS-FLOW plugin.

    Scope note:
    - FLOW is the current target;
    - FAST is intentionally out of scope for this first version;
    - wall-oss-0.5 should be handled later by adapting this wrapper, not by
      hard-coding old FLOW dimensions into future code.
    """

    checkpoint_dir: str | Path
    max_batch_size: int = 1
    action_horizon: int = 32
    dataset_name: str = "x2_normal"
    precision_policy: str = "full_bf16"


@Engine.register
class WallOSSEntry(Entry):
    """PhyAI entry for WALL-OSS-FLOW."""

    name: ClassVar[str] = "walloss"
    args_cls: ClassVar[type[EntryArgs]] = WallOSSArgs

    def __init__(self) -> None:
        self.config: WallOSSFlowConfig | None = None
        self.runner: WallOSSFlowRunner | None = None
        self.scheduler: WallOSSFlowWS1Scheduler | None = None

    def setup(self, args: WallOSSArgs) -> None:  # type: ignore[override]
        eng = get_engine_config()
        dtype = eng.device.params_dtype
        if dtype is None:
            dtype = torch.bfloat16

        self.config = WallOSSFlowConfig.from_checkpoint(
            args.checkpoint_dir,
            action_horizon=args.action_horizon,
            dataset_name=args.dataset_name,
        )
        self.runner = WallOSSFlowRunner(
            self.config.checkpoint_dir,
            device=eng.device.target,
            dtype=dtype,
            precision_policy=args.precision_policy,
        )
        self.scheduler = WallOSSFlowWS1Scheduler(
            self.runner,
            self.config,
            max_batch_size=args.max_batch_size,
            device=eng.device.target,
            dtype=dtype,
        )
        self.scheduler.setup()

    def step(self, request: WallOSSFlowRequest | WallOSSFlowPredictRequest):  # type: ignore[override]
        if self.scheduler is None:
            raise RuntimeError("WallOSSEntry.step called before setup().")
        return self.scheduler.step(request)

    def close(self) -> None:
        if self.scheduler is not None:
            self.scheduler.close()
            self.scheduler = None
        self.runner = None
        self.config = None


__all__ = ["WallOSSArgs", "WallOSSEntry"]
