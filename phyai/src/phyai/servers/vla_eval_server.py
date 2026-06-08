"""vla-evaluation-harness 适配服务。"""

from __future__ import annotations

from pathlib import Path
import os
from typing import Any

import numpy as np
import torch

from phyai.policies import PI05LiberoPolicy
from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.specs import IMAGE_RGB, LANGUAGE, RAW, DimSpec
from vla_eval.types import Action, Observation


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


class PhyAIModelServer(PredictModelServer):
    """PhyAI pi0.5 的 vla-eval model server。"""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        params_dtype: str = "bfloat16",
        use_cuda_graph: bool = True,
        attn_backend: str = "flashinfer",
        norm_backend: str = "flashinfer",
        linear_backend: str | None = None,
        **kwargs: Any,
    ) -> None:
        # pi0.5 一次返回 50 步 action chunk；让 PredictModelServer 管理 chunk buffer。
        kwargs.setdefault("chunk_size", 50)
        super().__init__(**kwargs)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.params_dtype = _DTYPE_MAP[params_dtype]
        self.use_cuda_graph = bool(use_cuda_graph)
        self.attn_backend = attn_backend
        self.norm_backend = norm_backend
        self.linear_backend = linear_backend
        self._policy: PI05LiberoPolicy | None = None
        dump_dir = os.environ.get("PHYAI_DEBUG_DUMP_DIR")
        self._debug_dump_dir = Path(dump_dir) if dump_dir else None
        self._debug_dump_count = 0

    def _load_policy(self) -> PI05LiberoPolicy:
        if self._policy is None:
            self._policy = PI05LiberoPolicy(
                self.checkpoint_path,
                device=self.device,
                params_dtype=self.params_dtype,
                max_batch_size=1,
                use_cuda_graph=self.use_cuda_graph,
                attn_backend=self.attn_backend,
                norm_backend=self.norm_backend,
                linear_backend=self.linear_backend,
            )
            # 与 checkpoint 配置保持一致，避免 hard-code 50 后遇到其他 checkpoint 出错。
            self.chunk_size = self._policy.chunk_size
        return self._policy

    def _load_model(self) -> None:
        # vla_eval.model_servers.serve.run_server 会在监听前调用该钩子，
        # 避免第一个 benchmark observation 被 checkpoint 冷启动耗时拖到超时。
        self._load_policy()

    def get_observation_params(self) -> dict[str, Any]:
        return {"send_wrist_image": True, "send_state": True}

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"actions": RAW}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {"image": IMAGE_RGB, "state": RAW, "language": LANGUAGE}

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        policy = self._load_policy()
        result = policy.infer(obs)
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if self._debug_dump_dir is not None and self._debug_dump_count == 0:
            self._debug_dump_dir.mkdir(parents=True, exist_ok=True)
            images = obs.get("images", {}) if isinstance(obs, dict) else {}
            payload: dict[str, Any] = {
                "task_description": np.array(str(obs.get("task_description", obs.get("task", "")))),
                "states": np.asarray(obs.get("states", obs.get("state", [])), dtype=np.float32),
                "controller_states": np.asarray(obs.get("controller_states", []), dtype=np.float32),
                "actions": actions.astype(np.float32),
            }
            if isinstance(images, dict):
                for name, image in images.items():
                    payload[f"image_{name}"] = np.asarray(image)
            np.savez_compressed(
                self._debug_dump_dir / f"predict_{self._debug_dump_count:04d}.npz",
                **payload,
            )
            self._debug_dump_count += 1
        return {"actions": actions}

    def close(self) -> None:
        if self._policy is not None:
            self._policy.close()
            self._policy = None
        close = getattr(super(), "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(PhyAIModelServer)
