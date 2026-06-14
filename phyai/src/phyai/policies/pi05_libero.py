"""High-level pi0.5 LIBERO inference wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from phyai.engine import Engine, EngineArgs
from phyai.engine_config import BackendConfig, DeviceConfig, EngineConfig, RuntimeConfig
from phyai.models.pi05.main_pi05 import PI05Args
from phyai.models.pi05.scheduler_ws1_pi05 import PI05Request
from phyai_utils_tools.pipeline import PI05LiberoPipeline


def _lerobot_pi05_weight_remap(key: str) -> str | None:
    """Handle LeRobot checkpoints that wrap keys with an extra model. prefix."""
    if key.startswith("model."):
        key = key[len("model.") :]
    if key == "paligemma_with_expert.gemma_expert.lm_head.weight":
        return None
    return key


class PI05LiberoPolicy:
    """Wrap vla-eval/LIBERO observations for PhyAI Engine inference."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        device: str = "cuda",
        params_dtype: torch.dtype = torch.bfloat16,
        max_batch_size: int = 1,
        use_cuda_graph: bool = True,
        attn_backend: str = "flashinfer",
        norm_backend: str = "flashinfer",
        linear_backend: str | None = None,
        flashinfer_workspace_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self.params_dtype = params_dtype
        self.max_batch_size = int(max_batch_size)
        self.pipeline = PI05LiberoPipeline(self.checkpoint_dir, device=device)
        self.engine = Engine(
            EngineArgs(
                plugin="pi05",
                plugin_args=PI05Args(
                    checkpoint_dir=self.checkpoint_dir,
                    max_batch_size=self.max_batch_size,
                    weight_remap=_lerobot_pi05_weight_remap,
                    inputs_image_shape=[
                        [self.pipeline.image_size, self.pipeline.image_size, 3]
                        for _ in self.pipeline.camera_names
                    ],
                ),
                config=EngineConfig(
                    backends=BackendConfig(
                        attn=attn_backend,
                        norm=norm_backend,
                        linear=linear_backend,
                    ),
                    device=DeviceConfig(target=device, params_dtype=params_dtype),
                    runtime=RuntimeConfig(
                        use_cuda_graph=use_cuda_graph,
                        flashinfer_workspace_bytes=flashinfer_workspace_bytes,
                        force_linear_kernel=linear_backend,
                    ),
                ),
            )
        )

    @property
    def chunk_size(self) -> int:
        return self.pipeline.chunk_size

    @property
    def action_dim(self) -> int:
        return self.pipeline.action_dim

    def infer(
        self,
        obs: dict[str, Any],
        *,
        noise: torch.Tensor | np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        request_inputs = self.pipeline.observation_to_inputs(obs)
        if noise is not None:
            request_inputs["noise"] = torch.as_tensor(noise, device=self.device)
        request = PI05Request(**request_inputs)
        with torch.inference_mode():
            raw_actions = self.engine.step(request)
        actions = self.pipeline.postprocess_actions(raw_actions)
        return {"actions": actions}

    def close(self) -> None:
        self.engine.close()
