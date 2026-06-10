"""Tests for image ops — resize_with_pad + normalize_pixels."""

from __future__ import annotations

import torch

from phyai_utils_tools.processing.ops.image_ops import normalize_pixels, resize_with_pad


def test_resize_fast_path_identity():
    """Already-target-size input is returned unchanged (same object)."""
    x = torch.rand(2, 3, 224, 224)
    assert resize_with_pad(x, 224, 224) is x


def test_resize_nonsquare_letterbox():
    """480x640 -> 224x224: aspect preserved, symmetric black bars."""
    x = torch.rand(2, 3, 480, 640)
    r = resize_with_pad(x, 224, 224)
    assert r.shape == (2, 3, 224, 224)
    # 640 -> 224 (ratio 640/224); 480 / (640/224) = 168 content rows, centered.
    assert torch.count_nonzero(r[:, :, :28, :]) == 0
    assert torch.count_nonzero(r[:, :, 196:, :]) == 0
    assert torch.count_nonzero(r[:, :, 28:196, :]) > 0


def test_resize_pad_value_minus_one():
    """pad_value=-1 letterboxes with -1 (correct for [-1, 1] inputs)."""
    x = torch.rand(1, 3, 480, 640) * 2 - 1
    r = resize_with_pad(x, 224, 224, pad_value=-1.0)
    assert torch.allclose(r[:, :, :28, :], torch.full_like(r[:, :, :28, :], -1.0))


def test_resize_no_float_clamp():
    """Float path must not clamp to [0, 1] (would destroy [-1, 1] data)."""
    x = torch.full((1, 3, 480, 640), -0.5)
    r = resize_with_pad(x, 224, 224, pad_value=-1.0)
    # interior content stays -0.5 (not clamped to 0)
    assert torch.isclose(r[0, 0, 112, 112], torch.tensor(-0.5), atol=1e-4)


def test_normalize_pixels():
    """[0, 1] -> [-1, 1]."""
    x = torch.tensor([0.0, 0.5, 1.0])
    assert torch.allclose(normalize_pixels(x), torch.tensor([-1.0, 0.0, 1.0]))
