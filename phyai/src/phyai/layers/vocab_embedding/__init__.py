"""phyai.layers.vocab_embedding — V-sharded input embedding and tied LM head.

Quick start::

    import phyai.parallel as P
    import phyai.layers.linear as L
    from phyai.layers.vocab_embedding import VocabParallelEmbedding, ParallelLMHead

    P.init(layout=(8,), mesh_dim_names=("tp",))
    L.init()

    embed = VocabParallelEmbedding(num_embeddings=151936, embedding_dim=4096)
    lm_head = ParallelLMHead(
        embedding_dim=4096,
        num_embeddings=151936,
        tied_weight=embed.weight,   # share the parameter; no post-hoc mutation
    )

    h = embed(input_ids)             # (..., 4096)
    logits = lm_head(h)              # (..., V_padded // tp_size)

The fused masked-lookup Triton kernel registers itself on import via the
``phyai::masked_embedding_lookup`` custom op. There is no separate
dispatcher to prime; the layer talks to the linear-kernel dispatcher
directly when running the LM-head matmul.
"""

from __future__ import annotations

# Importing ``ops`` registers the ``phyai::masked_embedding_lookup`` custom op
# so callers don't have to do anything to make Dynamo / torch.compile see it.
from phyai.layers.vocab_embedding import ops as _ops  # noqa: F401
from phyai.layers.vocab_embedding.layers import (
    ParallelLMHead,
    VocabParallelEmbedding,
    pad_vocab_to,
)


__all__ = [
    "VocabParallelEmbedding",
    "ParallelLMHead",
    "pad_vocab_to",
]
