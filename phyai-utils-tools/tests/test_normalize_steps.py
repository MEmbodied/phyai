"""Tests for the normalize / unnormalize steps (lerobot features+norm_map schema)."""

from __future__ import annotations

import torch

from phyai_utils_tools.processing.steps.normalize_steps import (
    NormalizationMode,
    NormalizerStep,
    UnnormalizerStep,
)
from phyai_utils_tools.processing.transition import ACTION, STATE

_STATE_FEAT = "observation.state"


def _roundtrip(mode_str, stats):
    """Normalize then unnormalize STATE; return (x, normalized, recovered)."""
    feats = {_STATE_FEAT: {"type": "STATE", "shape": [3]}}
    norm_map = {"STATE": mode_str}
    x = torch.tensor([[0.2, -0.4, 0.9]])
    n = NormalizerStep(features=feats, norm_map=norm_map, stats=stats)
    u = UnnormalizerStep(features=feats, norm_map=norm_map, stats=stats)
    y = n({STATE: x})[STATE]
    back = u({STATE: y})[STATE]
    return x, y, back


def test_mean_std_roundtrip():
    stats = {_STATE_FEAT: {"mean": [0.0, 0.0, 0.0], "std": [1.0, 2.0, 0.5]}}
    x, _, back = _roundtrip("MEAN_STD", stats)
    assert torch.allclose(back, x, atol=1e-5)


def test_min_max_roundtrip():
    stats = {_STATE_FEAT: {"min": [-1.0, -2.0, 0.0], "max": [1.0, 2.0, 1.0]}}
    x, _, back = _roundtrip("MIN_MAX", stats)
    assert torch.allclose(back, x, atol=1e-5)


def test_quantiles_roundtrip():
    stats = {_STATE_FEAT: {"q01": [-1.0, -2.0, 0.0], "q99": [1.0, 2.0, 1.0]}}
    x, _, back = _roundtrip("QUANTILES", stats)
    assert torch.allclose(back, x, atol=1e-5)


def test_quantile10_roundtrip():
    stats = {_STATE_FEAT: {"q10": [-1.0, -2.0, 0.0], "q90": [1.0, 2.0, 1.0]}}
    x, _, back = _roundtrip("QUANTILE10", stats)
    assert torch.allclose(back, x, atol=1e-5)


def test_identity_when_no_stats():
    """No stats => pass-through (preserves pi05_base default numerics)."""
    feats = {_STATE_FEAT: {"type": "STATE", "shape": [3]}}
    n = NormalizerStep(features=feats, norm_map={"STATE": "QUANTILES"}, stats=None)
    x = torch.tensor([[0.2, -0.4, 0.9]])
    assert torch.equal(n({STATE: x})[STATE], x)


def test_identity_when_empty_features():
    """Empty features (pi05_base) => no-op even with a norm_map."""
    n = NormalizerStep(features={}, norm_map={"STATE": "QUANTILES"}, stats=None)
    x = torch.tensor([[0.2, -0.4]])
    assert torch.equal(n({STATE: x})[STATE], x)


def test_identity_mode_passthrough():
    feats = {_STATE_FEAT: {"type": "STATE", "shape": [2]}}
    stats = {_STATE_FEAT: {"mean": [9.0, 9.0], "std": [9.0, 9.0]}}
    n = NormalizerStep(features=feats, norm_map={"STATE": "IDENTITY"}, stats=stats)
    x = torch.tensor([[0.2, -0.4]])
    assert torch.equal(n({STATE: x})[STATE], x)


def test_only_mapped_field_touched():
    """A feature bucket maps to exactly its transition field; others untouched."""
    feats = {_STATE_FEAT: {"type": "STATE", "shape": [1]}}
    stats = {_STATE_FEAT: {"mean": [0.0], "std": [2.0]}}
    n = NormalizerStep(features=feats, norm_map={"STATE": "MEAN_STD"}, stats=stats)
    out = n({STATE: torch.tensor([[4.0]]), ACTION: torch.tensor([[7.0]])})
    assert torch.allclose(out[STATE], torch.tensor([[2.0]]))
    assert torch.equal(out[ACTION], torch.tensor([[7.0]]))  # untouched


def test_enum_values_match_lerobot():
    """NormalizationMode strings must be byte-exact for json round-trip."""
    assert {m.value for m in NormalizationMode} == {
        "MIN_MAX",
        "MEAN_STD",
        "IDENTITY",
        "QUANTILES",
        "QUANTILE10",
    }
