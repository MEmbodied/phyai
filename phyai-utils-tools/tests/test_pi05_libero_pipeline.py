from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from phyai_utils_tools.pipeline import PI05LiberoPipeline


class _FakeTokenizer:
    def __call__(self, prompts, **kwargs):
        max_length = kwargs["max_length"]
        batch = len(prompts)
        input_ids = torch.zeros(batch, max_length, dtype=torch.int64)
        attention_mask = torch.zeros(batch, max_length, dtype=torch.int64)
        input_ids[:, :3] = torch.tensor([1, 2, 3], dtype=torch.int64)
        attention_mask[:, :3] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_pi05_libero_pipeline_shapes():
    checkpoint = Path("/mnt/data2/shared_models/pi05_libero_finetuned_v044")
    pipeline = PI05LiberoPipeline(checkpoint, device="cpu")
    pipeline._tokenizer = _FakeTokenizer()

    obs = {
        "images": {
            "agentview": np.zeros((256, 256, 3), dtype=np.uint8),
            "wrist": np.ones((256, 256, 3), dtype=np.uint8),
        },
        "states": np.zeros(8, dtype=np.float32),
        "task_description": "pick up the black bowl",
    }

    inputs = pipeline.observation_to_inputs(obs)

    assert tuple(inputs["pixel_values"].shape) == (1, 2, 3, 224, 224)
    assert "image_masks" not in inputs
    assert tuple(inputs["input_ids"].shape) == (1, 200)
    assert tuple(inputs["lang_lens"].shape) == (1,)
    assert inputs["lang_lens"].item() == 3

    raw_actions = torch.zeros(1, pipeline.chunk_size, pipeline.max_action_dim)
    actions = pipeline.postprocess_actions(raw_actions)
    assert actions.shape == (1, pipeline.chunk_size, 7)
    assert actions.dtype == np.float32
