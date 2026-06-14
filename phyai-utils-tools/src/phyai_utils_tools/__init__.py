"""phyai-utils-tools — pre/post-processing for phyai models.

Holds all image and tokenizer pre/post-processing moved out of the main
``phyai`` package, organized as a composable step-pipeline framework
(:mod:`phyai_utils_tools.processing`) with per-model processors
(:mod:`phyai_utils_tools.models`). This package is a workspace *leaf*: it never
imports ``phyai``, so ``phyai`` can depend on it (or, in the strict setup, the
caller wires both siblings) without a cycle.

Quick start (pi0.5)::

    from phyai_utils_tools.models.pi05 import PI05Processor

    proc = PI05Processor(image_size=224, num_images=3, tokenizer_max_length=200)
    inputs = proc.preprocess({"images": cams, "task": tasks, "state": state})
    # ... build a phyai PI05Request from `inputs` and run engine.step ...
    action = proc.postprocess(raw_action_chunk)
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from phyai_utils_tools.processing import (
    BaseModelProcessor,
    ProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
)

try:
    __version__ = _pkg_version("phyai-utils-tools")
except PackageNotFoundError:  # raw source tree, not installed
    __version__ = "0.0.0+unknown"

__all__ = [
    "BaseModelProcessor",
    "ProcessorPipeline",
    "ProcessorStep",
    "ProcessorStepRegistry",
    "__version__",
]
