"""phyai.layers.mlp — feed-forward blocks.

So far this exposes :class:`DenseMLP`, a generic 2-layer FFN with
optional gating that covers SwiGLU/GeGLU (Llama, Gemma, Qwen) on the
gated path and plain ``fc1->act->fc2`` (BERT, SigLIP, ViT) on the
non-gated path.
"""

from __future__ import annotations

from phyai.layers.mlp.dense_mlp import DenseMLP

__all__ = ["DenseMLP"]
