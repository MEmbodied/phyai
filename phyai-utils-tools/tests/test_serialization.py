"""Tests for lerobot-format serialization round-trip + a pi05-shaped config load.

Self-contained: no external checkpoint. The pi05-shaped test writes a json that
mirrors the exact step list lerobot's ``policy_*processor.json`` ships (empty
normalizer stats, like pi05_base), into ``tmp_path``, then loads it back.
"""

from __future__ import annotations

import json

import pytest
import torch

from phyai_utils_tools.processing import (
    ProcessorPipeline,
    ProcessorStepError,
    ProcessorStepRegistry,
)
from phyai_utils_tools.processing.steps import NormalizerStep
from phyai_utils_tools.processing.transition import STATE

_STATE_FEAT = "observation.state"

# The exact step list lerobot's pi05 policy_*processor.json ships (empty stats).
_PI05_PRE_CONFIG = {
    "name": "policy_preprocessor",
    "steps": [
        {
            "registry_name": "rename_observations_processor",
            "config": {"rename_map": {}},
        },
        {"registry_name": "to_batch_processor", "config": {}},
        {
            "registry_name": "normalizer_processor",
            "config": {
                "eps": 1e-08,
                "features": {},
                "norm_map": {
                    "VISUAL": "IDENTITY",
                    "STATE": "QUANTILES",
                    "ACTION": "QUANTILES",
                },
            },
        },
        {"registry_name": "pi05_prepare_state_tokenizer_processor_step", "config": {}},
        {
            "registry_name": "tokenizer_processor",
            "config": {
                "max_length": 200,
                "task_key": "task",
                "padding_side": "right",
                "padding": "max_length",
                "truncation": True,
                "tokenizer_name": "google/paligemma-3b-pt-224",
            },
        },
        {
            "registry_name": "device_processor",
            "config": {"device": "cpu", "float_dtype": None},
        },
    ],
}
_PI05_POST_CONFIG = {
    "name": "policy_postprocessor",
    "steps": [
        {
            "registry_name": "unnormalizer_processor",
            "config": {
                "eps": 1e-08,
                "features": {},
                "norm_map": {
                    "VISUAL": "IDENTITY",
                    "STATE": "QUANTILES",
                    "ACTION": "QUANTILES",
                },
            },
        },
        {
            "registry_name": "device_processor",
            "config": {"device": "cpu", "float_dtype": None},
        },
    ],
}


class _StubTokenizer:
    def __call__(
        self, prompts, max_length, padding, padding_side, truncation, return_tensors
    ):
        b = len(prompts)
        ids = torch.zeros(b, max_length, dtype=torch.long)
        mask = torch.zeros(b, max_length, dtype=torch.long)
        mask[:, 0] = 1
        return {"input_ids": ids, "attention_mask": mask}


def test_unknown_step_raises(tmp_path):
    """A config with an unregistered step name raises a clear ProcessorStepError."""
    cfg = {"name": "p", "steps": [{"registry_name": "does_not_exist", "config": {}}]}
    (tmp_path / "p.json").write_text(json.dumps(cfg))
    with pytest.raises(ProcessorStepError, match="Unknown processor step"):
        ProcessorPipeline.from_pretrained(tmp_path, "p.json")


def test_missing_local_sidecar_raises_filenotfound(tmp_path):
    """A local config pointing at a missing state_file fails clearly (not via Hub)."""
    cfg = {
        "name": "p",
        "steps": [
            {
                "registry_name": "normalizer_processor",
                "config": {
                    "eps": 1e-08,
                    "features": {"observation.state": {"type": "STATE", "shape": [2]}},
                    "norm_map": {"STATE": "MEAN_STD"},
                },
                "state_file": "p_step_0_normalizer_processor.safetensors",  # not written
            }
        ],
    }
    (tmp_path / "p.json").write_text(json.dumps(cfg))
    with pytest.raises(FileNotFoundError, match="state_file"):
        ProcessorPipeline.from_pretrained(tmp_path, "p.json")


def test_save_load_roundtrip_with_stats(tmp_path):
    """save_pretrained -> from_pretrained reproduces config + stats sidecar."""
    feats = {_STATE_FEAT: {"type": "STATE", "shape": [3]}}
    stats = {_STATE_FEAT: {"mean": [1.0, 2.0, 3.0], "std": [0.5, 0.5, 0.5]}}
    pipe = ProcessorPipeline(
        steps=[
            NormalizerStep(features=feats, norm_map={"STATE": "MEAN_STD"}, stats=stats)
        ],
        name="policy_preprocessor",
    )
    json_path = pipe.save_pretrained(
        tmp_path, config_filename="policy_preprocessor.json"
    )
    assert json_path.exists()

    # A stats sidecar .safetensors was written.
    sidecars = list(tmp_path.glob("*.safetensors"))
    assert len(sidecars) == 1

    loaded = ProcessorPipeline.from_pretrained(tmp_path, "policy_preprocessor.json")
    assert len(loaded) == 1
    assert loaded.get_config() == pipe.get_config()
    # Stats survived the round-trip: normalize gives the same result.
    x = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.allclose(loaded[0]({STATE: x})[STATE], pipe[0]({STATE: x})[STATE])
    # mean == x so normalized == 0.
    assert torch.allclose(loaded[0]({STATE: x})[STATE], torch.zeros_like(x), atol=1e-6)


def test_config_json_schema(tmp_path):
    """Emitted json matches lerobot's {name, steps:[{registry_name, config}]}."""
    pipe = ProcessorPipeline(
        steps=[
            NormalizerStep(
                features={_STATE_FEAT: {"type": "STATE", "shape": [2]}},
                norm_map={"STATE": "QUANTILES"},
                stats=None,
            )
        ],
        name="policy_preprocessor",
    )
    pipe.save_pretrained(tmp_path, config_filename="policy_preprocessor.json")
    cfg = json.loads((tmp_path / "policy_preprocessor.json").read_text())
    assert cfg["name"] == "policy_preprocessor"
    step = cfg["steps"][0]
    assert step["registry_name"] == "normalizer_processor"
    assert set(step["config"]) == {"eps", "features", "norm_map"}
    assert step["config"]["norm_map"] == {"STATE": "QUANTILES"}
    assert "state_file" not in step  # no stats => no sidecar


def test_load_pi05_shaped_config(tmp_path):
    """Load a json mirroring the exact pi05 lerobot policy step list (empty stats).

    Self-contained: writes the fixture into ``tmp_path`` (no external ckpt).
    Importing the pi05 model package registers its model-specific step.
    """
    import phyai_utils_tools.models.pi05  # noqa: F401  (registers pi05 step)

    (tmp_path / "policy_preprocessor.json").write_text(json.dumps(_PI05_PRE_CONFIG))
    (tmp_path / "policy_postprocessor.json").write_text(json.dumps(_PI05_POST_CONFIG))

    step_kwargs = {
        "tokenizer_processor": {"tokenizer": _StubTokenizer()},
        "device_processor": {"device": "cpu"},
    }
    pre = ProcessorPipeline.from_pretrained(
        tmp_path, "policy_preprocessor.json", step_kwargs=step_kwargs
    )
    post = ProcessorPipeline.from_pretrained(
        tmp_path, "policy_postprocessor.json", step_kwargs=step_kwargs
    )
    assert [type(s)._registry_name for s in pre.steps] == [
        "rename_observations_processor",
        "to_batch_processor",
        "normalizer_processor",
        "pi05_prepare_state_tokenizer_processor_step",
        "tokenizer_processor",
        "device_processor",
    ]
    assert [type(s)._registry_name for s in post.steps] == [
        "unnormalizer_processor",
        "device_processor",
    ]
    # Empty features in the config => identity normalize (no sidecar).
    norm = pre.steps[2]
    x = torch.tensor([[0.5, -0.5]])
    assert torch.equal(norm({STATE: x})[STATE], x)


def test_registry_has_lerobot_names():
    """All lerobot pi05 step names are registered.

    The generic steps register on ``import ...processing.steps``; pi0.5's
    model-specific step registers on ``import ...models.pi05``.
    """
    import phyai_utils_tools.models.pi05  # noqa: F401
    import phyai_utils_tools.processing.steps  # noqa: F401

    names = set(ProcessorStepRegistry.list())
    for n in (
        "rename_observations_processor",
        "to_batch_processor",
        "normalizer_processor",
        "unnormalizer_processor",
        "pi05_prepare_state_tokenizer_processor_step",
        "tokenizer_processor",
        "device_processor",
    ):
        assert n in names, n
