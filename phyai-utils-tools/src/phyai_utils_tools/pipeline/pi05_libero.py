"""pi0.5 LIBERO 输入输出处理。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file


class PI05LiberoPipeline:
    """把 LIBERO/vla-eval observation 转成 PhyAI PI05Request 所需张量。"""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        device: str | torch.device = "cuda",
        image_size: int | None = None,
        tokenizer_name: str | None = None,
        tokenizer_max_length: int | None = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device(device)
        self.config = self._read_json(self.checkpoint_dir / "config.json")
        self.preprocessor_config = self._read_json(
            self.checkpoint_dir / "policy_preprocessor.json"
        )
        self.postprocessor_config = self._read_json(
            self.checkpoint_dir / "policy_postprocessor.json"
        )

        self.image_size = int(
            image_size
            or self.config.get("image_resolution", [224, 224])[0]
            or 224
        )
        self.tokenizer_max_length = int(
            tokenizer_max_length
            or self.config.get("tokenizer_max_length")
            or self._tokenizer_config().get("max_length")
            or 200
        )
        configured_tokenizer = tokenizer_name or self._tokenizer_config().get(
            "tokenizer_name", "google/paligemma-3b-pt-224"
        )
        self.tokenizer_name = self._resolve_tokenizer_path(configured_tokenizer)
        self.empty_cameras = int(self.config.get("empty_cameras", 1))
        self.unflip_images = os.environ.get("PHYAI_LIBERO_UNFLIP_IMAGES", "0") in {"1", "true", "True", "yes"}
        self.prompt_mode = str(self.config.get("phyai_prompt_mode", "lerobot_state_bins"))
        self.normalization_mode = str(self.config.get("phyai_normalization_mode", "mean_std"))
        self.camera_mode = str(
            os.environ.get("PHYAI_CAMERA_MODE")
            or self.config.get("phyai_camera_mode", "two_camera")
        )
        if self.camera_mode not in {"two_camera", "three_camera"}:
            raise ValueError(
                f"Unsupported phyai_camera_mode={self.camera_mode!r}; "
                "expected 'two_camera' or 'three_camera'."
            )
        default_camera_names = (
            ["agentview", "wrist"]
            if self.camera_mode == "two_camera"
            else ["agentview", "wrist", "empty"]
        )
        self.camera_names = list(self.config.get("phyai_camera_names", default_camera_names))
        expected_cameras = 2 if self.camera_mode == "two_camera" else 3
        if len(self.camera_names) != expected_cameras:
            raise ValueError(
                f"phyai_camera_names must contain {expected_cameras} entries for "
                f"{self.camera_mode}, got {self.camera_names!r}."
            )
        self.action_dim = int(self.config.get("output_features", {}).get("action", {}).get("shape", [7])[0])
        self.max_action_dim = int(self.config.get("max_action_dim", 32))
        self.chunk_size = int(self.config.get("chunk_size", 50))

        self._normalizer_stats = self._load_step_state(
            self.preprocessor_config, "normalizer_processor"
        )
        self._unnormalizer_stats = self._load_step_state(
            self.postprocessor_config, "unnormalizer_processor"
        )
        self._tokenizer = None

    def _read_json(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a JSON object")
        return data


    def _resolve_tokenizer_path(self, name_or_path: str) -> str:
        local_candidates = {
            "google/paligemma-3b-pt-224": [
                Path("/mnt/data2/shared_models/paligemma-3b-pt-224"),
                Path("/mnt/data2/shared_models/google_paligemma-3b-pt-224"),
            ]
        }
        path = Path(name_or_path)
        if path.exists():
            return str(path)
        for candidate in local_candidates.get(name_or_path, []):
            if (candidate / "tokenizer.json").exists() or (candidate / "tokenizer.model").exists():
                return str(candidate)
        return name_or_path

    def _tokenizer_config(self) -> dict[str, Any]:
        for step in self.preprocessor_config.get("steps", []):
            if step.get("registry_name") == "tokenizer_processor":
                return dict(step.get("config") or {})
        return {}

    def _load_step_state(self, config: dict[str, Any], registry_name: str) -> dict[str, torch.Tensor]:
        for step in config.get("steps", []):
            if step.get("registry_name") != registry_name:
                continue
            state_file = step.get("state_file")
            if not state_file:
                return {}
            return load_file(str(self.checkpoint_dir / state_file))
        return {}

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        return self._tokenizer

    def observation_to_inputs(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """转换 vla-eval observation 为 PI05Request 所需 tensor 字段。"""
        empty = torch.full((3, self.image_size, self.image_size), -1.0, dtype=torch.float32)
        cameras: list[torch.Tensor] = []
        for idx, name in enumerate(self.camera_names):
            if name in {"empty", "empty_camera", "empty_camera_0"}:
                cameras.append(empty)
            else:
                image = self._extract_image(obs, name, fallback_index=idx)
                cameras.append(self._image_to_tensor(image))

        pixel_values = torch.stack(cameras, dim=0)
        pixel_values = pixel_values.unsqueeze(0).to(self.device)

        state = self._extract_state(obs)
        state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        state_t = self._normalize_state(state_t)
        input_ids, lang_lens = self._tokenize_inputs(
            [str(obs.get("task_description", obs.get("task", "")))],
            state_t,
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids.to(self.device),
            "lang_lens": lang_lens.to(self.device),
        }

    def observation_to_request(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """向后兼容别名：返回 PI05Request 所需字段，而不是具体请求类。"""
        return self.observation_to_inputs(obs)


    def _tokenize_inputs(
        self, tasks: list[str], states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states.dim() != 2:
            raise ValueError(f"states must be (B, state_dim), got {tuple(states.shape)}")
        if states.shape[0] != len(tasks):
            raise ValueError(f"states batch dim {states.shape[0]} != len(tasks) {len(tasks)}")
        prompts = []
        if self.prompt_mode == "openpi_task":
            prompts = [task.strip().replace("_", " ").replace("\n", " ") + "\n" for task in tasks]
        else:
            state_np = states.detach().cpu().numpy()
            bins = np.linspace(-1.0, 1.0, 257)[:-1]
            discretized = np.digitize(state_np, bins=bins) - 1
            for task, state_bins in zip(tasks, discretized):
                cleaned = task.strip().replace("_", " ").replace("\n", " ")
                state_str = " ".join(map(str, state_bins))
                prompts.append(f"Task: {cleaned}, State: {state_str};\nAction: ")
        encoded = self.tokenizer(
            prompts,
            max_length=self.tokenizer_max_length,
            padding="max_length",
            padding_side="right",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(torch.int64)
        lang_lens = encoded["attention_mask"].sum(dim=-1).to(torch.int64)
        return input_ids, lang_lens

    def postprocess_actions(self, actions: torch.Tensor | np.ndarray) -> np.ndarray:
        """取前 7 维并执行 action 反归一化。"""
        if not isinstance(actions, torch.Tensor):
            action_t = torch.as_tensor(actions)
        else:
            action_t = actions.detach()
        action_t = action_t[..., : self.action_dim].float().cpu()
        action_t = self._unnormalize_action(action_t)
        return action_t.numpy().astype(np.float32)

    def _extract_image(self, obs: dict[str, Any], key: str, *, fallback_index: int) -> np.ndarray:
        images = obs.get("images", {})
        img = images.get(key) if isinstance(images, dict) else None
        if img is None and isinstance(images, dict):
            values = list(images.values())
            if len(values) > fallback_index:
                img = values[fallback_index]
        if img is None:
            img = np.zeros((256, 256, 3), dtype=np.uint8)
        arr = np.asarray(img)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"Expected HWC RGB image for {key}, got shape {arr.shape}")
        if self.unflip_images:
            arr = np.ascontiguousarray(arr[::-1, ::-1])
        return arr.astype(np.uint8, copy=False)

    def _extract_state(self, obs: dict[str, Any]) -> np.ndarray:
        state = obs.get("states", obs.get("state"))
        if state is None:
            return np.zeros(8, dtype=np.float32)
        arr = np.asarray(state, dtype=np.float32)
        if arr.shape != (8,):
            raise ValueError(f"Expected LIBERO state shape (8,), got {arr.shape}")
        return arr

    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0
        if tensor.shape[1:3] != (self.image_size, self.image_size):
            tensor = self._resize_with_pad(tensor, self.image_size, self.image_size)
        tensor = tensor * 2.0 - 1.0
        return tensor.squeeze(0).permute(2, 0, 1).contiguous()

    def _resize_with_pad(self, images: torch.Tensor, height: int, width: int) -> torch.Tensor:
        if images.dim() != 4 or images.shape[-1] != 3:
            raise ValueError(f"Expected BHWC RGB images, got shape {tuple(images.shape)}")
        images = images.permute(0, 3, 1, 2)
        _, _, cur_height, cur_width = images.shape
        ratio = max(cur_width / width, cur_height / height)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)
        resized = F.interpolate(
            images,
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        )
        resized = resized.clamp(0.0, 1.0)
        pad_h0, rem_h = divmod(height - resized_height, 2)
        pad_w0, rem_w = divmod(width - resized_width, 2)
        padded = F.pad(
            resized,
            (pad_w0, pad_w0 + rem_w, pad_h0, pad_h0 + rem_h),
            mode="constant",
            value=0.0,
        )
        return padded.permute(0, 2, 3, 1)

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        if self.normalization_mode == "openpi_quantile":
            q01 = self._normalizer_stats.get("observation.state.min")
            q99 = self._normalizer_stats.get("observation.state.max")
            if q01 is None or q99 is None:
                return state
            q01_t = q01.to(state)[..., : state.shape[-1]]
            q99_t = q99.to(state)[..., : state.shape[-1]]
            return (state - q01_t) / (q99_t - q01_t + 1e-6) * 2.0 - 1.0
        mean = self._normalizer_stats.get("observation.state.mean")
        std = self._normalizer_stats.get("observation.state.std")
        if mean is None or std is None:
            return state
        return (state - mean.to(state)) / torch.clamp(std.to(state), min=1e-8)

    def _unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        if self.normalization_mode == "openpi_quantile":
            q01 = self._unnormalizer_stats.get("action.min")
            q99 = self._unnormalizer_stats.get("action.max")
            if q01 is None or q99 is None:
                return action
            q01_t = q01.to(action)
            q99_t = q99.to(action)
            dim = q01_t.shape[-1]
            if dim < action.shape[-1]:
                head = (action[..., :dim] + 1.0) / 2.0 * (q99_t - q01_t + 1e-6) + q01_t
                return torch.cat([head, action[..., dim:]], dim=-1)
            q01_t = q01_t[..., : action.shape[-1]]
            q99_t = q99_t[..., : action.shape[-1]]
            return (action + 1.0) / 2.0 * (q99_t - q01_t + 1e-6) + q01_t
        mean = self._unnormalizer_stats.get("action.mean")
        std = self._unnormalizer_stats.get("action.std")
        if mean is None or std is None:
            return action
        return action * torch.clamp(std.to(action), min=1e-8) + mean.to(action)
