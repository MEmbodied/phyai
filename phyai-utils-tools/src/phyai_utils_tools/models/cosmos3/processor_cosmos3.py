"""Cosmos3 processors — text-to-video tokenizer + action/policy processor"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from phyai_utils_tools.processing.base_processor import BaseModelProcessor
from phyai_utils_tools.processing.pipeline import ProcessorPipeline
from phyai_utils_tools.processing.transition import IMAGES, TASK, Transition
from phyai_utils_tools.tokenizer import get_tokenizer

from phyai_utils_tools.models.cosmos3.steps_cosmos3 import (
    ACTION_CHUNK,
    COND_ACTION,
    DOMAIN_ID,
    EMBODIMENT_TO_DOMAIN_ID,
    MODE,
    NEG_TEXT_IDS,
    NEG_TEXT_MASK,
    RAW_ACTION_DIM,
    TEXT_IDS,
    TEXT_MASK,
    VIDEO_SHAPE,
    Cosmos3ActionPadStep,
    Cosmos3DomainResolveStep,
    Cosmos3ImagePreprocessStep,
    Cosmos3TextTokenizeStep,
    resolve_domain_id,
)
from phyai_utils_tools.processing.transition import PIXEL_VALUES


COSMOS3_VISION_START_TOKEN = "<|vision_start|>"


def _flatten_chat_ids(out) -> list[int]:
    """Normalize ``apply_chat_template(tokenize=True)`` output to ``list[int]``.

    Different transformers/tokenizers versions return a ``list[int]``, a nested
    ``[[int, ...]]``, or a ``BatchEncoding`` of ``tokenizers.Encoding`` objects.
    """
    # BatchEncoding / list whose first element exposes ``.ids`` (Encoding).
    first = out[0] if len(out) > 0 else None
    if hasattr(first, "ids"):
        return list(first.ids)
    if isinstance(first, (list, tuple)):
        return [int(x) for x in first]
    # Flat list of ints.
    return [int(x) for x in out]


@dataclass
class Cosmos3TokenizedPrompt:
    """Batch-1 tokenized prompt tensors."""

    text_ids: torch.Tensor  # [1, S] int64
    text_mask: torch.Tensor  # [1, S] int64 (all ones — no padding)


class Cosmos3Processor:
    """Qwen2 chat-template tokenizer for Cosmos3 T2V prompts."""

    def __init__(
        self, tokenizer_name_or_path: str, *, use_system_prompt: bool = False
    ) -> None:
        self.tokenizer = get_tokenizer(tokenizer_name_or_path)
        self.use_system_prompt = use_system_prompt
        self.eos_token_id = int(self.tokenizer.eos_token_id)
        self.vision_start_token_id = int(
            self.tokenizer.convert_tokens_to_ids(COSMOS3_VISION_START_TOKEN)
        )

    def tokenize(
        self, prompt: str, *, device: torch.device | str = "cpu"
    ) -> Cosmos3TokenizedPrompt:
        """Tokenize one prompt -> ``[1, S]`` ids + all-ones mask."""
        conversation = []
        if self.use_system_prompt:
            conversation.append(
                {
                    "role": "system",
                    "content": "You are a helpful assistant who will generate videos from a given prompt.",
                }
            )
        conversation.append({"role": "user", "content": prompt})
        out = self.tokenizer.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True
        )
        ids = _flatten_chat_ids(out)
        ids = ids + [self.eos_token_id, self.vision_start_token_id]
        text_ids = torch.tensor([ids], dtype=torch.long, device=device)
        text_mask = torch.ones_like(text_ids)
        return Cosmos3TokenizedPrompt(text_ids=text_ids, text_mask=text_mask)

    def tokenize_pair(
        self,
        prompt: str,
        negative_prompt: str = "",
        *,
        device: torch.device | str = "cpu",
    ) -> tuple[Cosmos3TokenizedPrompt, Cosmos3TokenizedPrompt]:
        """Tokenize the conditional + unconditional (negative/empty) prompts."""
        return self.tokenize(prompt, device=device), self.tokenize(
            negative_prompt, device=device
        )


@dataclass
class Cosmos3PolicyProcessedInputs:
    """Preprocessed inputs for the Cosmos3 action/policy path."""

    pixel_values: torch.Tensor
    text_ids: torch.Tensor
    text_mask: torch.Tensor
    neg_text_ids: torch.Tensor
    neg_text_mask: torch.Tensor
    cond_action: torch.Tensor | None
    domain_id: int
    mode: str
    action_chunk: int
    raw_action_dim: int
    video_shape: tuple[int, int, int]


class Cosmos3PolicyProcessor(BaseModelProcessor):
    """Cosmos3 action/policy pre/post processor.

    Preprocessing: image resize/normalize, text tokenize, action pad, domain resolve.
    Postprocessing: slice action to raw_action_dim, move to CPU.
    """

    def __init__(
        self,
        *,
        tokenizer_name_or_path: str,
        height: int = 480,
        width: int = 832,
        num_frames: int = 17,
        mode: str = "policy",
        domain_name: str | int = "agibotworld",
        action_chunk_size: int = 16,
        raw_action_dim: int = 29,
        action_dim: int = 64,
        negative_prompt: str = "",
        device: torch.device | str = "cpu",
        params_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.height = int(height)
        self.width = int(width)
        self.num_frames = int(num_frames)
        self.mode = mode
        self.domain_name = domain_name
        self.action_chunk_size = int(action_chunk_size)
        self.raw_action_dim = int(raw_action_dim)
        self.action_dim = int(action_dim)
        self.negative_prompt = negative_prompt
        self.device = device
        self.params_dtype = params_dtype
        super().__init__()

    def _to_transition(self, raw: dict[str, Any]) -> Transition:
        """Adapt caller's raw dict into the canonical transition."""
        t: Transition = {}
        t[IMAGES] = raw.get("images")
        t[TASK] = raw.get("task", raw.get("prompt", ""))
        t[COND_ACTION] = raw.get("action") or raw.get("cond_action")
        t[DOMAIN_ID] = raw.get("domain_name", raw.get("domain_id", self.domain_name))
        t[MODE] = raw.get("mode", self.mode)
        return t

    def _to_output(self, transition: Transition) -> Cosmos3PolicyProcessedInputs:
        """Extract typed output from the final transition."""
        return Cosmos3PolicyProcessedInputs(
            pixel_values=transition[PIXEL_VALUES],
            text_ids=transition[TEXT_IDS],
            text_mask=transition[TEXT_MASK],
            neg_text_ids=transition[NEG_TEXT_IDS],
            neg_text_mask=transition[NEG_TEXT_MASK],
            cond_action=transition.get(COND_ACTION),
            domain_id=transition[DOMAIN_ID],
            mode=transition[MODE],
            action_chunk=transition[ACTION_CHUNK],
            raw_action_dim=transition[RAW_ACTION_DIM],
            video_shape=transition[VIDEO_SHAPE],
        )

    def build_preprocessor(self) -> ProcessorPipeline:
        steps = [
            Cosmos3ImagePreprocessStep(
                height=self.height, width=self.width, mode=self.mode
            ),
            Cosmos3TextTokenizeStep(
                tokenizer_name_or_path=self.tokenizer_name_or_path,
                negative_prompt=self.negative_prompt,
            ),
            Cosmos3ActionPadStep(
                action_chunk_size=self.action_chunk_size,
                raw_action_dim=self.raw_action_dim,
                action_dim=self.action_dim,
                mode=self.mode,
            ),
            Cosmos3DomainResolveStep(),
        ]
        return ProcessorPipeline(
            steps=steps,
            name="cosmos3_policy_preprocessor",
            to_transition=self._to_transition,
            to_output=self._to_output,
        )

    def build_postprocessor(self) -> ProcessorPipeline:
        return ProcessorPipeline(
            steps=[],
            name="cosmos3_policy_postprocessor",
            to_transition=lambda x: x,
            to_output=lambda x: x,
        )

    def postprocess(self, output: dict[str, Any] | torch.Tensor) -> dict[str, Any]:
        """Slice action to raw_action_dim and move tensors to CPU."""
        if isinstance(output, torch.Tensor):
            return {"action": output[:, :, : self.raw_action_dim].cpu()}
        result: dict[str, Any] = {}
        if "action" in output:
            result["action"] = output["action"][:, :, : self.raw_action_dim].cpu()
        if "pixels" in output:
            result["pixels"] = output["pixels"].cpu()
        if "video" in output:
            result["video"] = output["video"].cpu()
        return result


__all__ = [
    "Cosmos3PolicyProcessedInputs",
    "Cosmos3PolicyProcessor",
    "Cosmos3Processor",
    "Cosmos3TokenizedPrompt",
    "COSMOS3_VISION_START_TOKEN",
    "EMBODIMENT_TO_DOMAIN_ID",
    "resolve_domain_id",
]
