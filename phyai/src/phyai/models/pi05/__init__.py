"""phyai.models.pi05 — pi0.5 inference (vision + text + action expert + flow-matching).

The package ships the full pi0.5 inference path:

* :mod:`configuration_pi05` — frozen dataclass configs.
* :mod:`modeling_pi05` — every ``nn.Module`` (vision tower, paligemma
  language model, gemma_300m action expert, joint stack with
  :class:`~phyai.layers.attention.StaticCachedAttention`, action/time heads,
  and the parameter-only :class:`PI05Model` container).
* :mod:`batch_layout_pi05` — index / packing helpers the scheduler invokes
  once per inference (cu_seqlens, write-indices, padded prefix layout,
  joint paged_kv_indices interleave).
* :mod:`model_runner_pi05` — the three runners (vision / LLM / expert)
  that wrap captured CUDA graphs around the modeling code.
* :mod:`scheduler_single_batch_pi05` — single-batch end-to-end inference
  orchestrator.

Training is not in scope here; this package is inference-only.
"""

from __future__ import annotations

from phyai.models.pi05.configuration_pi05 import (
    GemmaExpertConfig,
    PaliGemmaTextConfig,
    PI05Config,
    SiglipVisionConfig,
)
from phyai.models.pi05.modeling_pi05 import (
    SIGLIP_NORM_HF_NAMES,
    ActionTimeHeads,
    MultiModalProjector,
    PaliGemmaDecoderLayer,
    PaliGemmaEmbedTokens,
    PaliGemmaLanguageModel,
    PI05ExpertLayer,
    PI05ExpertStack,
    PI05Model,
    PI05VisionTower,
    PositionEmbedding,
    SiglipVisionEmbeddings,
    SiglipVisionEncoder,
    SiglipVisionModel,
    VisionTowerWrapper,
    create_sinusoidal_pos_embedding,
)
from phyai.models.pi05.batch_layout_pi05 import (
    broadcast_cond_to_tokens,
    build_full_indptrs,
    build_joint_last_page_len,
    build_joint_paged_kv_indices,
    build_pos_ids_from_indptr,
    build_prefix_indptrs,
    build_prefix_last_page_len,
    build_prefix_padded_pos_ids,
    build_prefix_padded_write_indices,
    build_prefix_paged_kv_indices,
    build_suffix_indptrs,
    build_suffix_pos_ids,
    build_suffix_write_indices,
    pack_prefix_per_sample_padded,
)


__all__ = [
    # Configuration
    "GemmaExpertConfig",
    "PaliGemmaTextConfig",
    "PI05Config",
    "SiglipVisionConfig",
    # Modeling
    "ActionTimeHeads",
    "MultiModalProjector",
    "PaliGemmaDecoderLayer",
    "PaliGemmaEmbedTokens",
    "PaliGemmaLanguageModel",
    "PI05ExpertLayer",
    "PI05ExpertStack",
    "PI05Model",
    "PI05VisionTower",
    "PositionEmbedding",
    "SIGLIP_NORM_HF_NAMES",
    "SiglipVisionEmbeddings",
    "SiglipVisionEncoder",
    "SiglipVisionModel",
    "VisionTowerWrapper",
    "create_sinusoidal_pos_embedding",
    # Batch layout helpers
    "broadcast_cond_to_tokens",
    "build_full_indptrs",
    "build_joint_last_page_len",
    "build_joint_paged_kv_indices",
    "build_pos_ids_from_indptr",
    "build_prefix_indptrs",
    "build_prefix_last_page_len",
    "build_prefix_padded_pos_ids",
    "build_prefix_padded_write_indices",
    "build_prefix_paged_kv_indices",
    "build_suffix_indptrs",
    "build_suffix_pos_ids",
    "build_suffix_write_indices",
    "pack_prefix_per_sample_padded",
]
