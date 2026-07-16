"""phyai-model-optimizer — post-training quantization (PTQ) toolkit.

Two flows behind one seam (modifiers declare what; pipelines decide how data flows):

* data-free PTQ — ``RTNModifier`` (round-to-nearest int/fp8/MXFP4/NVFP4).
* calibration PTQ — ``GPTQModifier`` (+ AWQ/SmoothQuant), over a sequential pipeline.

Entrypoints :func:`oneshot` and :func:`model_free_ptq`; output is compressed-tensors
by default (``pack_format``), consumable by phyai.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from phyai_model_optimizer.entrypoints import model_free_ptq, oneshot
from phyai_model_optimizer.modifiers import (
    AWQModifier,
    GPTQModifier,
    Modifier,
    RTNModifier,
    SmoothQuantModifier,
)
from phyai_model_optimizer.quant_math import FP8Scheme, QuantDType, WeightQuant
from phyai_model_optimizer.recipes import Recipe

try:
    __version__ = _pkg_version("phyai-model-optimizer")
except PackageNotFoundError:  # note(chenghua): raw source tree, not installed
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    "oneshot",
    "model_free_ptq",
    "Modifier",
    "RTNModifier",
    "GPTQModifier",
    "AWQModifier",
    "SmoothQuantModifier",
    "WeightQuant",
    "FP8Scheme",
    "QuantDType",
    "Recipe",
]
