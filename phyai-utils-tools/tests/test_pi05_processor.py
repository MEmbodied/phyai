"""Tests for PI05Processor — pre/post pipelines with a stub tokenizer.

Avoids a network/HF dependency by monkeypatching the tokenizer loader so the
processor builds offline. Image/state/prompt logic is exercised for real; the
tokenizer call is the only stub.
"""

from __future__ import annotations

import torch

import phyai_utils_tools.models.pi05.processor_pi05 as proc_mod
from phyai_utils_tools.models.pi05 import PI05ProcessedInputs, PI05Processor


class _StubTokenizer:
    """Minimal HF-tokenizer-like callable: fixed-length ids + attention mask."""

    def __init__(self, real_len: int = 4):
        self.real_len = real_len

    def __call__(
        self, prompts, max_length, padding, padding_side, truncation, return_tensors
    ):
        b = len(prompts)
        ids = torch.zeros(b, max_length, dtype=torch.long)
        ids[:, 0] = 2
        mask = torch.zeros(b, max_length, dtype=torch.long)
        mask[:, : self.real_len] = 1
        return {"input_ids": ids, "attention_mask": mask}


def _make_processor(monkeypatch, **kwargs) -> PI05Processor:
    monkeypatch.setattr(proc_mod, "get_tokenizer", lambda name: _StubTokenizer())
    defaults = dict(
        image_size=224,
        num_channels=3,
        num_images=2,
        tokenizer_max_length=200,
        action_dim=7,
        device="cpu",
        params_dtype=torch.float32,
    )
    defaults.update(kwargs)
    return PI05Processor(**defaults)


def test_preprocess_shapes(monkeypatch):
    proc = _make_processor(monkeypatch)
    b = 3
    raw = {
        "images": [torch.rand(b, 3, 480, 640), torch.rand(b, 3, 224, 224)],
        "task": ["pick up the cup"] * b,
        "state": torch.rand(b, 7) * 2 - 1,
    }
    out = proc.preprocess(raw)
    assert isinstance(out, PI05ProcessedInputs)
    assert out.pixel_values.shape == (b, 2, 3, 224, 224)
    assert out.input_ids.shape == (b, 200) and out.input_ids.dtype == torch.int64
    assert out.lang_lens.shape == (b,) and out.lang_lens.tolist() == [4, 4, 4]


def test_preprocess_stacked_tensor_input(monkeypatch):
    """Accepts a single stacked (B, n, C, H, W) tensor too."""
    proc = _make_processor(monkeypatch)
    b = 2
    raw = {
        "images": torch.rand(b, 2, 3, 224, 224),
        "task": ["do a thing"] * b,
        "state": torch.rand(b, 5) * 2 - 1,
    }
    out = proc.preprocess(raw)
    assert out.pixel_values.shape == (b, 2, 3, 224, 224)


def test_postprocess_trims_action_dim(monkeypatch):
    proc = _make_processor(monkeypatch, action_dim=7)
    action = torch.rand(2, 50, 32)  # max_action_dim padded
    out = proc.postprocess(action)
    assert out.shape == (2, 50, 7)
    assert out.device.type == "cpu"


def test_make_pi05_processors_factory(monkeypatch):
    monkeypatch.setattr(proc_mod, "get_tokenizer", lambda name: _StubTokenizer())
    pre, post = proc_mod.make_pi05_processors(num_images=1, action_dim=7, device="cpu")
    assert pre.name == "pi05_preprocessor"
    assert post.name == "pi05_postprocessor"


def test_save_then_from_pretrained_roundtrip(monkeypatch, tmp_path):
    """save_pretrained writes lerobot json; from_pretrained rebuilds an equivalent processor."""
    proc = _make_processor(monkeypatch, num_images=2, action_dim=7)
    proc.save_pretrained(tmp_path)
    # Both lerobot-format files written, vision glue + slice excluded.
    assert (tmp_path / "policy_preprocessor.json").exists()
    assert (tmp_path / "policy_postprocessor.json").exists()

    loaded = PI05Processor.from_pretrained(
        tmp_path,
        tokenizer=_StubTokenizer(),
        num_images=2,
        image_size=224,
        action_dim=7,
        device="cpu",
        params_dtype=torch.float32,
    )
    b = 2
    raw = {
        "images": torch.rand(b, 2, 3, 224, 224),
        "task": ["do a thing"] * b,
        "state": torch.rand(b, 5) * 2 - 1,
    }
    out = loaded.preprocess(raw)
    assert isinstance(out, PI05ProcessedInputs)
    assert out.pixel_values.shape == (b, 2, 3, 224, 224)
    assert out.input_ids.shape == (b, 200)
    # postprocess still trims to action_dim
    act = loaded.postprocess(torch.rand(b, 50, 32))
    assert act.shape == (b, 50, 7)
