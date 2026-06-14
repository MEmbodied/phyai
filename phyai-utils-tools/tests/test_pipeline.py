"""Tests for the pipeline framework — registry, ordering, adapters."""

from __future__ import annotations

import pytest

from phyai_utils_tools.processing import (
    ProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
)


class _AddStep(ProcessorStep):
    def __init__(self, key, delta):
        self.key = key
        self.delta = delta

    def __call__(self, transition):
        out = transition.copy()
        out[self.key] = out.get(self.key, 0) + self.delta
        return out

    def get_config(self):
        return {"key": self.key, "delta": self.delta}


def test_registry_register_get_roundtrip():
    name = "_test_add_step_unique"
    ProcessorStepRegistry.register(name)(_AddStep)
    try:
        assert ProcessorStepRegistry.get(name) is _AddStep
        assert _AddStep._registry_name == name
        assert name in ProcessorStepRegistry.list()
    finally:
        ProcessorStepRegistry.unregister(name)


def test_registry_duplicate_raises():
    name = "_test_dup_step"
    ProcessorStepRegistry.register(name)(_AddStep)
    try:
        with pytest.raises(ValueError, match="already registered"):
            ProcessorStepRegistry.register(name)(_AddStep)
    finally:
        ProcessorStepRegistry.unregister(name)


def test_registry_unknown_get_raises():
    with pytest.raises(KeyError, match="Unknown processor step"):
        ProcessorStepRegistry.get("does_not_exist_xyz")


def test_pipeline_runs_steps_in_order():
    pipe = ProcessorPipeline(steps=[_AddStep("v", 1), _AddStep("v", 10)])
    out = pipe({"v": 0})
    assert out["v"] == 11


def test_pipeline_adapters():
    """to_transition / to_output wrap raw input / output shapes."""
    pipe = ProcessorPipeline(
        steps=[_AddStep("v", 5)],
        to_transition=lambda x: {"v": x},
        to_output=lambda t: t["v"],
    )
    assert pipe(100) == 105


def test_pipeline_len_and_index():
    s0, s1 = _AddStep("v", 1), _AddStep("v", 2)
    pipe = ProcessorPipeline(steps=[s0, s1])
    assert len(pipe) == 2
    assert pipe[0] is s0 and pipe[1] is s1


def test_pipeline_get_config():
    pipe = ProcessorPipeline(steps=[_AddStep("v", 3)], name="p")
    cfg = pipe.get_config()
    assert cfg["name"] == "p"
    assert cfg["steps"][0]["config"] == {"key": "v", "delta": 3}


def test_step_through_yields_each_stage():
    pipe = ProcessorPipeline(steps=[_AddStep("v", 1), _AddStep("v", 1)])
    stages = list(pipe.step_through({"v": 0}))
    assert [s["v"] for s in stages] == [0, 1, 2]
