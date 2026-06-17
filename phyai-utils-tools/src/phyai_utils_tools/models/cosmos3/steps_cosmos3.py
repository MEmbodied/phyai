"""Cosmos3 policy pipeline steps — image, text, action, and domain processing."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from phyai_utils_tools.processing.pipeline import (
    ProcessorStep,
    ProcessorStepRegistry,
)
from phyai_utils_tools.processing.transition import (
    IMAGES,
    PIXEL_VALUES,
    TASK,
    Transition,
)

# Cosmos3 policy-specific transition keys (local, not shared).
TEXT_IDS = "text_ids"
TEXT_MASK = "text_mask"
NEG_TEXT_IDS = "neg_text_ids"
NEG_TEXT_MASK = "neg_text_mask"
COND_ACTION = "cond_action"
DOMAIN_ID = "domain_id"
MODE = "mode"
ACTION_CHUNK = "action_chunk"
RAW_ACTION_DIM = "raw_action_dim"
VIDEO_SHAPE = "video_shape"


EMBODIMENT_TO_DOMAIN_ID: dict[str, int] = {
    "no_action": 0,
    "av": 1,
    "camera_pose": 2,
    "hand_pose": 3,
    "pusht": 4,
    "libero": 5,
    "umi": 6,
    "bridge_orig_lerobot": 7,
    "droid_lerobot": 8,
    "robomind-franka": 8,
    "galbot": 9,
    "robomind-franka-dual": 12,
    "robomind-ur": 13,
    "agibotworld": 15,
    "agibot_gear_gripper": 15,
    "agibot_gear_gripper_ext": 15,
    "fractal": 20,
}


def resolve_domain_id(domain: str | int) -> int:
    if isinstance(domain, int):
        if domain < 0:
            raise ValueError(f"domain_id must be non-negative, got {domain}.")
        return domain
    key = str(domain).strip().lower()
    if key not in EMBODIMENT_TO_DOMAIN_ID:
        raise ValueError(
            f"Unknown domain_name={domain!r}; expected one of "
            f"{sorted(EMBODIMENT_TO_DOMAIN_ID)} or pass an int domain_id."
        )
    return EMBODIMENT_TO_DOMAIN_ID[key]


def _to_pil_rgb(value: Any):
    """Convert various image formats to PIL RGB."""
    import PIL.Image

    if isinstance(value, str):
        return PIL.Image.open(value).convert("RGB")
    if isinstance(value, PIL.Image.Image):
        return value.convert("RGB")
    if isinstance(value, np.ndarray):
        array = value
        if (
            array.ndim == 3
            and array.shape[0] in (3, 4)
            and array.shape[-1] not in (3, 4)
        ):
            array = np.transpose(array, (1, 2, 0))
        if np.issubdtype(array.dtype, np.floating):
            if array.min() < 0.0 or array.max() > 1.0:
                array = np.clip(array, -1.0, 1.0) * 0.5 + 0.5
            array = (np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        return PIL.Image.fromarray(array).convert("RGB")
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.ndim == 3 and tensor.shape[0] in (3, 4):
            tensor = tensor.permute(1, 2, 0)
        if tensor.is_floating_point():
            if tensor.min().item() < 0.0 or tensor.max().item() > 1.0:
                tensor = tensor.clamp(-1.0, 1.0) * 0.5 + 0.5
            tensor = (tensor.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        return PIL.Image.fromarray(tensor.numpy()).convert("RGB")
    raise TypeError(
        f"Expected PIL image, numpy array, torch tensor, or path, got {type(value)!r}."
    )


def _resize_and_pad_action_image(image, target_h: int, target_w: int) -> np.ndarray:
    """Action-mode resize: scale DOWN with BICUBIC + reflect/edge pad."""
    import PIL.Image

    img = _to_pil_rgb(image)
    scale = min(target_w / img.width, target_h / img.height, 1.0)
    resize_w = max(1, int(scale * img.width + 0.5))
    resize_h = max(1, int(scale * img.height + 0.5))
    if (resize_w, resize_h) != img.size:
        img = img.resize((resize_w, resize_h), PIL.Image.Resampling.BICUBIC)

    array = np.asarray(img)
    pad_h = target_h - resize_h
    pad_w = target_w - resize_w
    if pad_h == 0 and pad_w == 0:
        return array
    pad_mode = "reflect" if pad_h < resize_h and pad_w < resize_w else "edge"
    return np.pad(array, ((0, pad_h), (0, pad_w), (0, 0)), mode=pad_mode)


@ProcessorStepRegistry.register("cosmos3_image_preprocess_step")
class Cosmos3ImagePreprocessStep(ProcessorStep):
    """Load/resize/normalize observation image(s) to pixel tensor.

    For action modes (policy/forward_dynamics): single frame,
    scale-down + reflect-pad.
    For inverse_dynamics: all provided frames.
    Output: [1, 3, T, H, W] in [-1, 1].
    """

    def __init__(self, *, height: int, width: int, mode: str) -> None:
        self.height = int(height)
        self.width = int(width)
        self.mode = mode

    def __call__(self, transition: Transition) -> Transition:
        raw = transition.get(IMAGES)
        if raw is None:
            raise ValueError("Cosmos3ImagePreprocessStep requires an IMAGES entry.")

        if isinstance(raw, list):
            frames = raw
        else:
            frames = [raw]

        if self.mode == "inverse_dynamics":
            processed_frames = []
            for frame in frames:
                arr = _resize_and_pad_action_image(frame, self.height, self.width)
                processed_frames.append(arr)
        else:
            arr = _resize_and_pad_action_image(frames[0], self.height, self.width)
            processed_frames = [arr]

        tensors = []
        for arr in processed_frames:
            t = torch.from_numpy(arr).permute(2, 0, 1).float()
            t = t / 127.5 - 1.0
            tensors.append(t)

        pixel_values = torch.stack(tensors, dim=1).unsqueeze(0)

        out = transition.copy()
        out[PIXEL_VALUES] = pixel_values
        out[VIDEO_SHAPE] = (len(processed_frames), self.height, self.width)
        return out

    def get_config(self) -> dict[str, Any]:
        return {"height": self.height, "width": self.width, "mode": self.mode}


@ProcessorStepRegistry.register("cosmos3_text_tokenize_step")
class Cosmos3TextTokenizeStep(ProcessorStep):
    """Tokenize prompt + negative prompt via Cosmos3Processor."""

    def __init__(
        self, *, tokenizer_name_or_path: str, negative_prompt: str = ""
    ) -> None:
        from phyai_utils_tools.models.cosmos3.processor_cosmos3 import Cosmos3Processor

        self._proc = Cosmos3Processor(tokenizer_name_or_path)
        self._negative_prompt = negative_prompt

    def __call__(self, transition: Transition) -> Transition:
        prompt = transition.get(TASK)
        if prompt is None:
            raise ValueError("Cosmos3TextTokenizeStep requires a TASK entry.")
        if isinstance(prompt, list):
            prompt = prompt[0]

        cond, uncond = self._proc.tokenize_pair(
            prompt, self._negative_prompt, device="cpu"
        )
        out = transition.copy()
        out[TEXT_IDS] = cond.text_ids
        out[TEXT_MASK] = cond.text_mask
        out[NEG_TEXT_IDS] = uncond.text_ids
        out[NEG_TEXT_MASK] = uncond.text_mask
        return out

    def get_config(self) -> dict[str, Any]:
        return {"negative_prompt": self._negative_prompt}


@ProcessorStepRegistry.register("cosmos3_action_pad_step")
class Cosmos3ActionPadStep(ProcessorStep):
    """Pad/truncate action tensor for forward_dynamics, or set None for other modes."""

    def __init__(
        self,
        *,
        action_chunk_size: int,
        raw_action_dim: int,
        action_dim: int = 64,
        mode: str = "policy",
    ) -> None:
        self.action_chunk_size = int(action_chunk_size)
        self.raw_action_dim = int(raw_action_dim)
        self.action_dim = int(action_dim)
        self.mode = mode

    def __call__(self, transition: Transition) -> Transition:
        out = transition.copy()
        out[ACTION_CHUNK] = self.action_chunk_size
        out[RAW_ACTION_DIM] = self.raw_action_dim

        if self.mode != "forward_dynamics":
            out[COND_ACTION] = None
            return out

        raw_action = transition.get(COND_ACTION)
        if raw_action is None:
            raise ValueError(
                "forward_dynamics mode requires a 'cond_action' entry with the "
                "conditioning action tensor."
            )
        if isinstance(raw_action, (list, np.ndarray)):
            raw_action = torch.as_tensor(raw_action, dtype=torch.float32)
        if raw_action.ndim == 3:
            raw_action = raw_action.squeeze(0)

        if raw_action.shape[0] < self.action_chunk_size:
            pad = raw_action[-1:].repeat(
                self.action_chunk_size - raw_action.shape[0], 1
            )
            raw_action = torch.cat([raw_action, pad], dim=0)
        elif raw_action.shape[0] > self.action_chunk_size:
            raw_action = raw_action[: self.action_chunk_size]

        padded = torch.zeros(
            self.action_chunk_size, self.action_dim, dtype=torch.float32
        )
        dim = min(raw_action.shape[-1], self.action_dim)
        padded[:, :dim] = raw_action[:, :dim]

        out[COND_ACTION] = padded.unsqueeze(0)
        return out

    def get_config(self) -> dict[str, Any]:
        return {
            "action_chunk_size": self.action_chunk_size,
            "raw_action_dim": self.raw_action_dim,
            "action_dim": self.action_dim,
            "mode": self.mode,
        }


@ProcessorStepRegistry.register("cosmos3_domain_resolve_step")
class Cosmos3DomainResolveStep(ProcessorStep):
    """Resolve domain_name string to integer domain_id."""

    def __call__(self, transition: Transition) -> Transition:
        domain = transition.get(DOMAIN_ID)
        if domain is None:
            raise ValueError("Cosmos3DomainResolveStep requires a DOMAIN_ID entry.")
        out = transition.copy()
        out[DOMAIN_ID] = resolve_domain_id(domain)
        return out


__all__ = [
    "ACTION_CHUNK",
    "COND_ACTION",
    "Cosmos3ActionPadStep",
    "Cosmos3DomainResolveStep",
    "Cosmos3ImagePreprocessStep",
    "Cosmos3TextTokenizeStep",
    "DOMAIN_ID",
    "EMBODIMENT_TO_DOMAIN_ID",
    "MODE",
    "NEG_TEXT_IDS",
    "NEG_TEXT_MASK",
    "RAW_ACTION_DIM",
    "TEXT_IDS",
    "TEXT_MASK",
    "VIDEO_SHAPE",
    "resolve_domain_id",
]
