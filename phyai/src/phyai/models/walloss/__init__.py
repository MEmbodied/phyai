"""phyai.models.walloss — first-version WALL-OSS-FLOW plugin."""

from __future__ import annotations

from phyai.models.walloss.configuration_walloss import WallOSSFlowConfig
from phyai.models.walloss.main_walloss import WallOSSArgs, WallOSSEntry
from phyai.models.walloss.scheduler_ws1_walloss import (
    WallOSSFlowPredictRequest,
    WallOSSFlowRequest,
    WallOSSFlowWS1Scheduler,
)

__all__ = [
    "WallOSSArgs",
    "WallOSSEntry",
    "WallOSSFlowConfig",
    "WallOSSFlowPredictRequest",
    "WallOSSFlowRequest",
    "WallOSSFlowWS1Scheduler",
]
