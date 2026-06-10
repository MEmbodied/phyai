"""Image preprocessing ops — resize-with-pad and pixel normalization.

``resize_with_pad`` is the openpi / lerobot ``resize_with_pad_torch`` port
(channels-first ``(B, C, H, W)`` only), moved out of ``phyai`` so all image
preprocessing lives in one place. ``normalize_pixels`` is the SigLIP
``[0, 1] -> [-1, 1]`` map that the model expects but ``phyai`` never applied
(callers used to pre-normalize); it is provided here so the processor can own
the full raw-image -> model-ready path.

These are pure tensor functions; the :class:`~phyai_utils_tools.processing.pipeline.ProcessorStep`
wrappers live in :mod:`phyai_utils_tools.processing.steps.image_steps`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def resize_with_pad(
    images: torch.Tensor,
    target_h: int,
    target_w: int,
    *,
    mode: str = "bilinear",
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Aspect-preserving resize of ``(B, C, H, W)`` images, padded to target.

    Port of openpi / lerobot ``resize_with_pad_torch`` (channels-first only).
    The image is downscaled by the larger of the two axis ratios so it fits
    inside ``target_h`` x ``target_w`` without distortion, then symmetrically
    padded with ``pad_value`` to exactly the target size. Inputs already at the
    target size are returned unchanged (fast path).

    The float path does **not** clamp to ``[0, 1]`` (pixels may already be in
    the model's ``[-1, 1]`` range); bilinear interpolation is a convex
    combination so it cannot overshoot the input range. ``uint8`` inputs are
    rounded and clamped to ``[0, 255]`` (range-agnostic).
    """
    if images.dim() != 4:
        raise ValueError(
            f"resize_with_pad expects 4-D (B, C, H, W); got shape "
            f"{tuple(images.shape)}."
        )
    _, _, cur_h, cur_w = images.shape
    if cur_h == target_h and cur_w == target_w:
        return images

    ratio = max(cur_w / target_w, cur_h / target_h)
    resized_h = int(cur_h / ratio)
    resized_w = int(cur_w / ratio)

    resized = F.interpolate(
        images,
        size=(resized_h, resized_w),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    if images.dtype == torch.uint8:
        resized = torch.round(resized).clamp(0, 255).to(torch.uint8)
    elif not images.dtype.is_floating_point:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    pad_h0, rem_h = divmod(target_h - resized_h, 2)
    pad_h1 = pad_h0 + rem_h
    pad_w0, rem_w = divmod(target_w - resized_w, 2)
    pad_w1 = pad_w0 + rem_w
    # F.pad order for the last two dims: (left, right, top, bottom).
    return F.pad(
        resized, (pad_w0, pad_w1, pad_h0, pad_h1), mode="constant", value=pad_value
    )


def normalize_pixels(images: torch.Tensor) -> torch.Tensor:
    """Map ``[0, 1]`` pixels to ``[-1, 1]`` (SigLIP's expected range).

    ``images * 2 - 1``. Mirrors lerobot's ``img * 2.0 - 1.0`` step. Apply only
    to images that are actually in ``[0, 1]``; images already in ``[-1, 1]``
    should skip this (the processor exposes it as an optional step).
    """
    return images * 2.0 - 1.0


__all__ = ["normalize_pixels", "resize_with_pad"]
