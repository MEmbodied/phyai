"""Interop shims. ``ct`` wraps compressed-tensors; ``phyai_model`` (imported on
demand) wraps phyai model construction/calibration and is intentionally NOT
imported here so the package stays importable without phyai installed."""

from phyai_model_optimizer.compat import ct

__all__ = ["ct"]
