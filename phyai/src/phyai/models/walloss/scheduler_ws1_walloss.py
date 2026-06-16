"""Single-card scheduler for the first WALL-OSS-FLOW PhyAI plugin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from phyai.models.walloss.configuration_walloss import WallOSSFlowConfig
from phyai.models.walloss.model_runner_walloss import WallOSSFlowRunner
from phyai.runtime.schedule import Scheduler


@dataclass
class WallOSSFlowRequest:
    """One WALL-OSS-FLOW validate request.

    This mirrors the already-validated wall-x fake inference path:
    input_ids, attention_mask, moe_token_types, position_ids,
    proprioception, agent_pos_mask, dof_mask, dataset_names.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    moe_token_types: torch.Tensor
    position_ids: torch.Tensor
    proprioception: torch.Tensor
    agent_pos_mask: torch.Tensor
    dof_mask: torch.Tensor
    dataset_names: Sequence[str] | str = "x2_normal"


@dataclass
class WallOSSFlowPredictRequest:
    """One WALL-OSS-FLOW policy predict request.

    This request is intentionally separate from ``WallOSSFlowRequest`` because
    the real policy path consumes processor outputs such as pixel_values and
    image_grid_thw, calls mode='predict', and returns predict_action instead of
    validate logits.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    moe_token_types: torch.Tensor
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    proprioception: torch.Tensor
    agent_pos_mask: torch.Tensor
    dof_mask: torch.Tensor
    dataset_names: Sequence[str] | str = "x2_normal"
    predict_mode: str = "diffusion"
    action_dim: int | None = None
    action_horizon: int | None = None


class WallOSSFlowWS1Scheduler(Scheduler):
    """Minimal single-GPU scheduler for WALL-OSS-FLOW.

    It owns one runner and performs shape / dtype / device normalization before
    calling the official wall-x model.
    """

    def __init__(
        self,
        runner: WallOSSFlowRunner,
        config: WallOSSFlowConfig,
        *,
        max_batch_size: int = 1,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.runner = runner
        self.config = config
        self.max_batch_size = int(max_batch_size)
        self.device = torch.device(device)
        self.dtype = dtype
        if self.max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {max_batch_size}.")

    def setup(self) -> None:
        self.runner.setup()

    @torch.no_grad()
    def step(self, request: WallOSSFlowRequest | WallOSSFlowPredictRequest):
        if isinstance(request, WallOSSFlowPredictRequest):
            self._validate_predict(request)
            batch = self._to_wallx_predict_batch(request)
            return self.runner.predict(
                batch,
                predict_mode=request.predict_mode,
                action_dim=request.action_dim or self.config.action_dim,
                action_horizon=request.action_horizon or self.config.action_horizon,
            )

        if isinstance(request, WallOSSFlowRequest):
            self._validate(request)
            batch = self._to_wallx_batch(request)
            return self.runner.forward(batch)

        raise TypeError(f"unsupported WALL-OSS request type: {type(request)!r}")

    def _dataset_names(self, dataset_names: Sequence[str] | str, batch_size: int) -> list[str]:
        if isinstance(dataset_names, str):
            return [dataset_names] * batch_size
        dataset_names = list(dataset_names)
        if len(dataset_names) != batch_size:
            raise ValueError(
                f"dataset_names length {len(dataset_names)} != batch size {batch_size}."
            )
        return dataset_names

    def _to_wallx_batch(self, req: WallOSSFlowRequest) -> dict:
        dataset_names = self._dataset_names(req.dataset_names, int(req.input_ids.shape[0]))

        return {
            "input_ids": req.input_ids.to(device=self.device, dtype=torch.long),
            "attention_mask": req.attention_mask.to(device=self.device, dtype=torch.long),
            "moe_token_types": req.moe_token_types.to(device=self.device, dtype=torch.long),
            "position_ids": req.position_ids.to(device=self.device, dtype=torch.long),
            "proprioception": req.proprioception.to(device=self.device, dtype=self.dtype),
            "agent_pos_mask": req.agent_pos_mask.to(device=self.device, dtype=self.dtype),
            "dof_mask": req.dof_mask.to(device=self.device, dtype=self.dtype),
            "dataset_names": dataset_names,
        }

    def _to_wallx_predict_batch(self, req: WallOSSFlowPredictRequest) -> dict:
        dataset_names = self._dataset_names(req.dataset_names, int(req.input_ids.shape[0]))

        # The policy path keeps action/proprio tensors in fp32 because wall-x's
        # selected-bf16 policy keeps action_preprocessor parameters in fp32.
        return {
            "input_ids": req.input_ids.to(device=self.device, dtype=torch.long),
            "attention_mask": req.attention_mask.to(device=self.device, dtype=torch.long),
            "moe_token_types": req.moe_token_types.to(device=self.device, dtype=torch.bool),
            "pixel_values": req.pixel_values.to(device=self.device),
            "image_grid_thw": req.image_grid_thw.to(device=self.device, dtype=torch.long),
            "proprioception": req.proprioception.to(device=self.device, dtype=torch.float32),
            "agent_pos_mask": req.agent_pos_mask.to(device=self.device, dtype=torch.float32),
            "dof_mask": req.dof_mask.to(device=self.device, dtype=torch.float32),
            "dataset_names": dataset_names,
        }

    def _validate(self, req: WallOSSFlowRequest) -> None:
        B, T = req.input_ids.shape
        cfg = self.config

        if not 1 <= B <= self.max_batch_size:
            raise ValueError(f"batch size {B} not in [1, {self.max_batch_size}].")

        expected_2d = (B, T)
        for name in ("attention_mask", "moe_token_types", "position_ids"):
            value = getattr(req, name)
            if tuple(value.shape) != expected_2d:
                raise ValueError(f"{name} shape {tuple(value.shape)} != {expected_2d}.")

        if tuple(req.proprioception.shape) != (B, 1, cfg.proprio_dim):
            raise ValueError(
                f"proprioception shape {tuple(req.proprioception.shape)} != "
                f"({B}, 1, {cfg.proprio_dim})."
            )

        if tuple(req.agent_pos_mask.shape) != (B, 1, cfg.proprio_dim):
            raise ValueError(
                f"agent_pos_mask shape {tuple(req.agent_pos_mask.shape)} != "
                f"({B}, 1, {cfg.proprio_dim})."
            )

        if tuple(req.dof_mask.shape) != (B, cfg.action_horizon, cfg.action_dim):
            raise ValueError(
                f"dof_mask shape {tuple(req.dof_mask.shape)} != "
                f"({B}, {cfg.action_horizon}, {cfg.action_dim})."
            )

        self._dataset_names(req.dataset_names, B)

    def _validate_predict(self, req: WallOSSFlowPredictRequest) -> None:
        B, T = req.input_ids.shape
        cfg = self.config

        if not 1 <= B <= self.max_batch_size:
            raise ValueError(f"batch size {B} not in [1, {self.max_batch_size}].")

        expected_2d = (B, T)
        for name in ("attention_mask", "moe_token_types"):
            value = getattr(req, name)
            if tuple(value.shape) != expected_2d:
                raise ValueError(f"{name} shape {tuple(value.shape)} != {expected_2d}.")

        if req.pixel_values.ndim != 2:
            raise ValueError(f"pixel_values must be 2-D, got shape {tuple(req.pixel_values.shape)}.")

        if req.image_grid_thw.ndim != 2 or req.image_grid_thw.shape[-1] != 3:
            raise ValueError(
                "image_grid_thw must have shape (num_images, 3), "
                f"got {tuple(req.image_grid_thw.shape)}."
            )

        if tuple(req.proprioception.shape) != (B, 1, cfg.proprio_dim):
            raise ValueError(
                f"proprioception shape {tuple(req.proprioception.shape)} != "
                f"({B}, 1, {cfg.proprio_dim})."
            )

        if tuple(req.agent_pos_mask.shape) != (B, 1, cfg.proprio_dim):
            raise ValueError(
                f"agent_pos_mask shape {tuple(req.agent_pos_mask.shape)} != "
                f"({B}, 1, {cfg.proprio_dim})."
            )

        action_dim = req.action_dim or cfg.action_dim
        action_horizon = req.action_horizon or cfg.action_horizon
        if tuple(req.dof_mask.shape) != (B, action_horizon, action_dim):
            raise ValueError(
                f"dof_mask shape {tuple(req.dof_mask.shape)} != "
                f"({B}, {action_horizon}, {action_dim})."
            )

        self._dataset_names(req.dataset_names, B)

    def close(self) -> None:
        self.runner.close()


__all__ = [
    "WallOSSFlowPredictRequest",
    "WallOSSFlowRequest",
    "WallOSSFlowWS1Scheduler",
]
