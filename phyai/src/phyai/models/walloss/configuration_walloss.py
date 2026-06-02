"""Minimal configuration helpers for WALL-OSS-FLOW.

This file deliberately does not re-implement the wall-x model config.
The first PhyAI plugin version treats wall-x as the source of truth and
only extracts the dimensions needed for request validation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


def _sum_dim(config: dict[str, int] | None, *, name: str) -> int:
    if not isinstance(config, dict) or not config:
        raise ValueError(f"{name} must be a non-empty dict, got {type(config).__name__}.")
    try:
        return sum(int(v) for v in config.values())
    except Exception as exc:
        raise ValueError(f"failed to sum {name}: {config!r}") from exc


@dataclass(frozen=True)
class WallOSSFlowConfig:
    """Small runtime config for the first WALL-OSS-FLOW PhyAI plugin.

    The true model architecture remains owned by wall-x's config.json and
    Qwen2_5_VLMoEForAction.from_pretrained(). This dataclass only records
    the fields the scheduler must validate.
    """

    checkpoint_dir: Path
    model_type: str
    vocab_size: int
    action_dim: int
    proprio_dim: int
    action_horizon: int = 32
    dataset_name: str = "x2_normal"

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        *,
        action_horizon: int = 32,
        dataset_name: str = "x2_normal",
    ) -> "WallOSSFlowConfig":
        checkpoint_dir = Path(checkpoint_dir)
        cfg_path = checkpoint_dir / "config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"missing wall-oss config.json: {cfg_path}")

        cfg = json.loads(cfg_path.read_text())

        model_type = str(cfg.get("model_type", ""))
        if model_type != "qwen2_5_vl":
            raise ValueError(
                f"expected WALL-OSS-FLOW model_type='qwen2_5_vl', got {model_type!r}."
            )
        if "vocab_size" not in cfg:
            raise ValueError(f"missing required key 'vocab_size' in {cfg_path}.")

        action_dim = _sum_dim(cfg.get("dof_config"), name="dof_config")
        proprio_dim = _sum_dim(cfg.get("agent_pos_config"), name="agent_pos_config")

        return cls(
            checkpoint_dir=checkpoint_dir,
            model_type=model_type,
            vocab_size=int(cfg["vocab_size"]),
            action_dim=action_dim,
            proprio_dim=proprio_dim,
            action_horizon=int(action_horizon),
            dataset_name=str(dataset_name),
        )
