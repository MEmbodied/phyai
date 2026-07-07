"""Guarded access to the optional ``humming-kernels`` library."""

from __future__ import annotations

try:  # pragma: no cover - depends on optional install + CUDA toolchain
    import humming  # noqa: F401
    from humming import dtypes as humming_dtypes
    from humming.layer import HummingMethod
    from humming.schema import BaseInputSchema, BaseWeightSchema
    from humming.schema.humming import HummingInputSchema, HummingWeightSchema

    _HAS_HUMMING = True
except Exception:  # pragma: no cover - the common CPU/dev path
    humming = None  # type: ignore[assignment]
    humming_dtypes = None  # type: ignore[assignment]
    HummingMethod = None  # type: ignore[assignment]
    BaseInputSchema = None  # type: ignore[assignment]
    BaseWeightSchema = None  # type: ignore[assignment]
    HummingInputSchema = None  # type: ignore[assignment]
    HummingWeightSchema = None  # type: ignore[assignment]
    _HAS_HUMMING = False


def has_humming() -> bool:
    """True when ``humming-kernels`` is importable in this process."""
    return _HAS_HUMMING


def require_humming() -> None:
    """Raise a clear error if humming is needed but unavailable."""
    if not _HAS_HUMMING:
        raise RuntimeError(
            "humming-kernels is not installed. Install the optional 'humming' "
            "extra on a CUDA host (e.g. `uv sync --extra humming`) to use "
            "humming quantization specs."
        )


__all__ = [
    "has_humming",
    "require_humming",
    "humming_dtypes",
    "HummingMethod",
    "BaseInputSchema",
    "BaseWeightSchema",
    "HummingInputSchema",
    "HummingWeightSchema",
]
