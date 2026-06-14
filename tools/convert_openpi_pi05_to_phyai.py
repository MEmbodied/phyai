# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "openpi @ git+https://github.com/Physical-Intelligence/openpi.git",
#   "jax[cpu]",
#   "orbax-checkpoint",
#   "numpy",
#   "torch",
#   "safetensors",
# ]
# ///
"""Convert an OpenPI JAX/Orbax pi0.5 checkpoint to PhyAI native safetensors.

Source : an OpenPI Orbax/OCDBT ``pi05_libero`` checkpoint directory.
Target : a PhyAI checkpoint directory containing safetensors plus sidecar
         config/processor files that ``phyai.weights.load_pretrained``
         consumes for the native pi0.5 engine (see ``modeling_pi05.py``).

This is a faithful rewrite of the original (lost) converter, reconstructed
from the existing product header (812 keys) + the source param tree
(``params/_METADATA``) + ``modeling_pi05.py`` docstring layout.

Run with uv so the PEP 723 header pulls OpenPI/jax/orbax into a throwaway env::

    uv run tools/convert_openpi_pi05_to_phyai.py            # dry-run (default)
    uv run tools/convert_openpi_pi05_to_phyai.py --write    # write to a NEW dir

The default mode is **dry-run**: it loads the source State, builds the full
target tensor dict, and diffs SHAPES against an existing reference safetensors
(the product). It writes nothing. ``--write`` is an explicit opt-in and, by
default, targets a NEW directory — it never overwrites the reference product
unless you pass ``--out <product> --allow-overwrite`` together.

NOTE on OpenPI pin: the PEP 723 header tracks ``openpi`` git main (unpinned).
If main has drifted away from the ``pi05_libero`` checkpoint format and the
loader/State shapes no longer match, pin ``openpi`` to the commit that shipped
``pi05_libero`` (the release used to produce the reference product) by editing
the dependency line above to ``... .git@<commit>``.

Conventions confirmed against PhyAI source (read-only) — do NOT change without
re-verifying:
  * All PyTorch linears are ``[out, in]`` -> every JAX ``kernel [in, out]`` is
    transposed on copy.
  * The checkpoint stores SEPARATE q/k/v/o and gate/up/down keys; PhyAI fuses
    them at load time. Do NOT pre-fuse here.
  * Gemma RMSNorm / AdaRMS use the ``(1 + scale)`` runtime convention on BOTH
    sides (JAX stores raw scale, PhyAI's kernel adds 1) -> copy norm scale
    verbatim, no ``+1``.
  * ``gemma_expert.lm_head.weight`` is a dead weight PhyAI drops at load; the
    reference product stores it as ``paligemma.lm_head.weight[:, :1024]``. We
    reproduce that for byte-for-byte structural parity.

Open correctness points that SHAPE validation cannot catch (verify with a
fixed-noise single-step OpenPI-vs-PhyAI action diff after a successful dry-run):
  1. q/k/v einsum axis order (head-major vs hidden-major).
  2. AdaRMS Dense_0 chunk order -> (scale, shift, gate).
  3. gating_einsum gate/up split order (index 0 = gate, 1 = up).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Static dimensions (from modeling_pi05 / configuration_pi05, confirmed against
# the reference product header). Used only to assert shapes — never to fabricate
# data; every tensor's bytes come from the loaded OpenPI State.
# ---------------------------------------------------------------------------
VISION_LAYERS = 27
GEMMA_LAYERS = 18  # shared count for paligemma text and the action expert

VIS_HIDDEN = 1152
VIS_HEADS = 16
VIS_HEAD_DIM = 72  # 16 * 72 == 1152
VIS_MLP = 4304
VIS_PATCH = 14
VIS_POS = 256

TXT_HIDDEN = 2048
TXT_HEADS = 8
TXT_HEAD_DIM = 256
TXT_KV_HEADS = 1
TXT_MLP = 16384
VOCAB = 257152

EXP_HIDDEN = 1024
EXP_MLP = 4096
EXP_ADARMS = 3072  # 3 * 1024 (scale|shift|gate)
JOINT_ATTN = TXT_HEADS * TXT_HEAD_DIM  # 2048, shared q/o joint space
KV_DIM = TXT_KV_HEADS * TXT_HEAD_DIM  # 256

# Target key prefixes (every product key is prefixed with ``model.``).
ROOT = "model."
PALI = ROOT + "paligemma_with_expert.paligemma.model"
EXPERT = ROOT + "paligemma_with_expert.gemma_expert.model"
VIS = PALI + ".vision_tower.vision_model"
PALI_LM_HEAD = ROOT + "paligemma_with_expert.paligemma.lm_head.weight"
EXPERT_LM_HEAD = ROOT + "paligemma_with_expert.gemma_expert.lm_head.weight"


# ===========================================================================
# Reference safetensors header reader (shapes only, no tensor bytes).
# ===========================================================================
def read_safetensors_header(path: Path) -> dict[str, dict]:
    """Return ``{key: {"shape": [...], "dtype": "..."}}`` without mmapping data."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr


# ===========================================================================
# OpenPI source loader. Returns a flat ``{jax_path_tuple: np.ndarray}`` map of
# the FULL (unsharded) parameter tree. Imported lazily so ``--help`` and the
# header reader work without OpenPI installed.
# ===========================================================================
def load_openpi_state(checkpoint_dir: Path) -> dict[tuple, np.ndarray]:
    """Load the OpenPI Orbax checkpoint into a flat path->ndarray dict.

    ``checkpoint_dir`` points at the checkpoint root that contains ``params/``.
    We restore the full State
    via orbax so every tensor is materialised at its true (unsharded) shape.

    OpenPI's public ``pi05_libero`` checkpoint was saved from an 8-device mesh.
    A plain Orbax restore may fail on a 1-device conversion machine with
    ``sharding ... Got None``.  To make the converter portable, we inspect the
    checkpoint metadata and request every array to be restored onto the local
    first device with ``SingleDeviceSharding`` while preserving each array's
    global shape.
    """
    import jax
    import orbax.checkpoint as ocp

    params_dir = checkpoint_dir / "params"
    if not params_dir.exists():
        raise FileNotFoundError(f"no params/ under {checkpoint_dir}")

    ckptr = ocp.PyTreeCheckpointer()
    restored = ckptr.restore(
        params_dir.resolve(),
        args=ocp.args.PyTreeRestore(
            restore_args=_single_device_restore_args(ckptr, params_dir, jax, ocp)
        ),
    )

    flat: dict[tuple, np.ndarray] = {}

    def _walk(node, prefix: tuple) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                _walk(v, prefix + (k,))
        else:
            flat[prefix] = np.asarray(node)

    _walk(restored, ())
    return flat


def _single_device_restore_args(ckptr, params_dir: Path, jax, ocp):
    """Build a restore_args tree that ignores the saved 8-device mesh.

    Orbax metadata leaves expose global ``shape`` / ``dtype``.  Mirroring the
    metadata tree with ``ArrayRestoreArgs`` lets TensorStore assemble the full
    array onto one local device instead of requiring the original save mesh.
    """

    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    metadata_obj = ckptr.metadata(params_dir.resolve())
    item_metadata = getattr(metadata_obj, "item_metadata", metadata_obj)
    metadata = item_metadata.tree

    def _convert(node):
        if isinstance(node, dict):
            return {k: _convert(v) for k, v in node.items()}
        shape = tuple(getattr(node, "shape"))
        dtype = getattr(node, "dtype", None)
        return ocp.ArrayRestoreArgs(
            restore_type=jax.Array,
            dtype=dtype,
            sharding=sharding,
            global_shape=shape,
            shape=shape,
            strict=False,
        )

    return _convert(metadata)


class Source:
    """Path-addressable accessor over the flat OpenPI State.

    Keys in the JAX tree end with a trailing ``value`` leaf (as seen in
    ``_METADATA``). We normalise lookups so callers pass the human path
    (e.g. ``"PaliGemma/img/embedding/kernel"``) and we try both with and
    without the trailing ``value``.
    """

    def __init__(self, flat: dict[tuple, np.ndarray]) -> None:
        self._flat = flat
        # Index by "/"-joined path for both the raw and value-stripped forms.
        self._by_str: dict[str, np.ndarray] = {}
        for path, arr in flat.items():
            joined = "/".join(str(p) for p in path)
            self._by_str[joined] = arr
            if path and path[-1] == "value":
                self._by_str["/".join(str(p) for p in path[:-1])] = arr

    def get(self, path: str) -> np.ndarray:
        key = path.strip("/")
        if key in self._by_str:
            return self._by_str[key]
        # Try the ``params/`` prefix and a trailing ``value`` leaf.
        for cand in (f"params/{key}", f"{key}/value", f"params/{key}/value"):
            if cand in self._by_str:
                return self._by_str[cand]
        raise KeyError(
            f"source leaf not found: {path!r}\n"
            f"available (sample): {sorted(self._by_str)[:8]}"
        )


# ===========================================================================
# Transform helpers.
# ===========================================================================
def kT(arr: np.ndarray) -> np.ndarray:
    """JAX Linear kernel [in, out] -> PyTorch weight [out, in]."""
    return np.ascontiguousarray(np.swapaxes(arr, -1, -2))


# ===========================================================================
# Target tensor-dict builder. Produces ``{target_key: np.ndarray}`` for all
# 812 keys. Layer tensors are unstacked from the leading (stacked) axis.
# ===========================================================================
def build_target(src: Source) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}

    # ---- action / time heads (root) --------------------------------------
    out[ROOT + "action_in_proj.weight"] = kT(src.get("action_in_proj/kernel"))
    out[ROOT + "action_in_proj.bias"] = src.get("action_in_proj/bias")
    out[ROOT + "action_out_proj.weight"] = kT(src.get("action_out_proj/kernel"))
    out[ROOT + "action_out_proj.bias"] = src.get("action_out_proj/bias")
    out[ROOT + "time_mlp_in.weight"] = kT(src.get("time_mlp_in/kernel"))
    out[ROOT + "time_mlp_in.bias"] = src.get("time_mlp_in/bias")
    out[ROOT + "time_mlp_out.weight"] = kT(src.get("time_mlp_out/kernel"))
    out[ROOT + "time_mlp_out.bias"] = src.get("time_mlp_out/bias")

    # ---- vision tower (SigLIP) -------------------------------------------
    _build_vision(src, out)

    # ---- paligemma text LM + tied lm_head --------------------------------
    _build_text(src, out)

    # ---- action expert ---------------------------------------------------
    _build_expert(src, out)

    return out


def _build_vision(src: Source, out: dict[str, np.ndarray]) -> None:
    # patch embedding conv: JAX HWIO [14,14,3,1152] -> torch OIHW [1152,3,14,14]
    conv = src.get("PaliGemma/img/embedding/kernel")
    out[VIS + ".embeddings.patch_embedding.weight"] = np.ascontiguousarray(
        np.transpose(conv, (3, 2, 0, 1))
    )
    out[VIS + ".embeddings.patch_embedding.bias"] = src.get(
        "PaliGemma/img/embedding/bias"
    )
    # position embedding: JAX [1,256,1152] -> [256,1152]
    pos = src.get("PaliGemma/img/pos_embedding")
    out[VIS + ".embeddings.position_embedding.weight"] = np.ascontiguousarray(pos[0])
    # final encoder norm -> post_layernorm
    out[VIS + ".post_layernorm.weight"] = src.get(
        "PaliGemma/img/Transformer/encoder_norm/scale"
    )
    out[VIS + ".post_layernorm.bias"] = src.get(
        "PaliGemma/img/Transformer/encoder_norm/bias"
    )
    # multi_modal_projector (JAX img/head)
    out[PALI + ".multi_modal_projector.linear.weight"] = kT(
        src.get("PaliGemma/img/head/kernel")
    )
    out[PALI + ".multi_modal_projector.linear.bias"] = src.get(
        "PaliGemma/img/head/bias"
    )

    eb = "PaliGemma/img/Transformer/encoderblock"
    ln0_s = src.get(f"{eb}/LayerNorm_0/scale")
    ln0_b = src.get(f"{eb}/LayerNorm_0/bias")
    ln1_s = src.get(f"{eb}/LayerNorm_1/scale")
    ln1_b = src.get(f"{eb}/LayerNorm_1/bias")
    q_k = src.get(f"{eb}/MultiHeadDotProductAttention_0/query/kernel")
    q_b = src.get(f"{eb}/MultiHeadDotProductAttention_0/query/bias")
    k_k = src.get(f"{eb}/MultiHeadDotProductAttention_0/key/kernel")
    k_b = src.get(f"{eb}/MultiHeadDotProductAttention_0/key/bias")
    v_k = src.get(f"{eb}/MultiHeadDotProductAttention_0/value/kernel")
    v_b = src.get(f"{eb}/MultiHeadDotProductAttention_0/value/bias")
    o_k = src.get(f"{eb}/MultiHeadDotProductAttention_0/out/kernel")
    o_b = src.get(f"{eb}/MultiHeadDotProductAttention_0/out/bias")
    fc1_k = src.get(f"{eb}/MlpBlock_0/Dense_0/kernel")
    fc1_b = src.get(f"{eb}/MlpBlock_0/Dense_0/bias")
    fc2_k = src.get(f"{eb}/MlpBlock_0/Dense_1/kernel")
    fc2_b = src.get(f"{eb}/MlpBlock_0/Dense_1/bias")

    for i in range(VISION_LAYERS):
        p = f"{VIS}.encoder.layers.{i}."
        out[p + "layer_norm1.weight"] = ln0_s[i]
        out[p + "layer_norm1.bias"] = ln0_b[i]
        out[p + "layer_norm2.weight"] = ln1_s[i]
        out[p + "layer_norm2.bias"] = ln1_b[i]
        # attention: per-head kernel [in=1152, heads=16, head_dim=72] ->
        # flatten head-major to [1152, 1152] then transpose to [out, in].
        out[p + "self_attn.q_proj.weight"] = _vis_qkv_w(q_k[i])
        out[p + "self_attn.q_proj.bias"] = q_b[i].reshape(VIS_HIDDEN)
        out[p + "self_attn.k_proj.weight"] = _vis_qkv_w(k_k[i])
        out[p + "self_attn.k_proj.bias"] = k_b[i].reshape(VIS_HIDDEN)
        out[p + "self_attn.v_proj.weight"] = _vis_qkv_w(v_k[i])
        out[p + "self_attn.v_proj.bias"] = v_b[i].reshape(VIS_HIDDEN)
        # out: JAX [heads=16, head_dim=72, out=1152] -> [in=1152, out=1152] -> T
        out[p + "self_attn.out_proj.weight"] = np.ascontiguousarray(
            o_k[i].reshape(VIS_HEADS * VIS_HEAD_DIM, VIS_HIDDEN).T
        )
        out[p + "self_attn.out_proj.bias"] = o_b[i]
        out[p + "mlp.fc1.weight"] = kT(fc1_k[i])
        out[p + "mlp.fc1.bias"] = fc1_b[i]
        out[p + "mlp.fc2.weight"] = kT(fc2_k[i])
        out[p + "mlp.fc2.bias"] = fc2_b[i]


def _vis_qkv_w(per_layer: np.ndarray) -> np.ndarray:
    """SigLIP q/k/v kernel [in=1152, heads=16, head_dim=72] -> weight [1152,1152]."""
    flat = per_layer.reshape(VIS_HIDDEN, VIS_HEADS * VIS_HEAD_DIM)  # [in, out]
    return np.ascontiguousarray(flat.T)  # [out, in]


def _build_text(src: Source, out: dict[str, np.ndarray]) -> None:
    # tied embedder -> paligemma.lm_head (full vocab)
    embed = src.get("PaliGemma/llm/embedder/input_embedding")  # [257152, 2048]
    out[PALI_LM_HEAD] = embed
    # final norm
    out[PALI + ".language_model.norm.weight"] = src.get(
        "PaliGemma/llm/final_norm/scale"
    )

    q = src.get("PaliGemma/llm/layers/attn/q_einsum/w")  # [18, 8, 256, 2048]
    kv = src.get("PaliGemma/llm/layers/attn/kv_einsum/w")  # [18, 2, 1, 256, 2048]
    o = src.get("PaliGemma/llm/layers/attn/attn_vec_einsum/w")  # [18, 8, 256, 2048]
    gate_up = src.get("PaliGemma/llm/layers/mlp/gating_einsum")  # [18, 2, 2048, 16384]
    down = src.get("PaliGemma/llm/layers/mlp/linear")  # [18, 16384, 2048]
    pre_attn = src.get("PaliGemma/llm/layers/pre_attention_norm/scale")  # [18, 2048]
    pre_ffw = src.get("PaliGemma/llm/layers/pre_ffw_norm/scale")  # [18, 2048]

    for i in range(GEMMA_LAYERS):
        p = f"{PALI}.language_model.layers.{i}."
        out[p + "input_layernorm.weight"] = pre_attn[i]
        out[p + "post_attention_layernorm.weight"] = pre_ffw[i]
        # q: [heads=8, head_dim=256, hidden=2048] -> [2048, 2048] head-major
        out[p + "self_attn.q_proj.weight"] = _gemma_q_w(q[i], TXT_HIDDEN)
        # kv: [2, 1, head_dim=256, hidden] -> k=idx0, v=idx1 -> [256, hidden]
        out[p + "self_attn.k_proj.weight"] = _gemma_kv_w(kv[i, 0], TXT_HIDDEN)
        out[p + "self_attn.v_proj.weight"] = _gemma_kv_w(kv[i, 1], TXT_HIDDEN)
        # o: [heads=8, head_dim=256, out=2048] -> [in=2048, out=2048] -> T
        out[p + "self_attn.o_proj.weight"] = _gemma_o_w(o[i], TXT_HIDDEN)
        # mlp gate/up: gating_einsum [2, in, inter] -> gate=idx0, up=idx1
        out[p + "mlp.gate_proj.weight"] = kT(gate_up[i, 0])
        out[p + "mlp.up_proj.weight"] = kT(gate_up[i, 1])
        out[p + "mlp.down_proj.weight"] = kT(down[i])


def _build_expert(src: Source, out: dict[str, np.ndarray]) -> None:
    # expert lm_head: dead weight PhyAI drops; product stores paligemma[:, :1024]
    embed = src.get("PaliGemma/llm/embedder/input_embedding")  # [257152, 2048]
    out[EXPERT_LM_HEAD] = np.ascontiguousarray(embed[:, :EXP_HIDDEN])

    # final norm is AdaRMS (Dense_0)
    fn_k = src.get("PaliGemma/llm/final_norm_1/Dense_0/kernel")  # [1024, 3072]
    fn_b = src.get("PaliGemma/llm/final_norm_1/Dense_0/bias")  # [3072]
    out[EXPERT + ".norm.dense.weight"] = kT(fn_k)
    out[EXPERT + ".norm.dense.bias"] = fn_b

    q = src.get("PaliGemma/llm/layers/attn/q_einsum_1/w")  # [18, 8, 256, 1024]
    kv = src.get("PaliGemma/llm/layers/attn/kv_einsum_1/w")  # [18, 2, 1, 256, 1024]
    o = src.get("PaliGemma/llm/layers/attn/attn_vec_einsum_1/w")  # [18, 8, 256, 1024]
    gate_up = src.get("PaliGemma/llm/layers/mlp_1/gating_einsum")  # [18, 2, 1024, 4096]
    down = src.get("PaliGemma/llm/layers/mlp_1/linear")  # [18, 4096, 1024]
    pre_attn_k = src.get(
        "PaliGemma/llm/layers/pre_attention_norm_1/Dense_0/kernel"
    )  # [18, 1024, 3072]
    pre_attn_b = src.get("PaliGemma/llm/layers/pre_attention_norm_1/Dense_0/bias")
    pre_ffw_k = src.get("PaliGemma/llm/layers/pre_ffw_norm_1/Dense_0/kernel")
    pre_ffw_b = src.get("PaliGemma/llm/layers/pre_ffw_norm_1/Dense_0/bias")

    for i in range(GEMMA_LAYERS):
        p = f"{EXPERT}.layers.{i}."
        out[p + "input_layernorm.dense.weight"] = kT(pre_attn_k[i])
        out[p + "input_layernorm.dense.bias"] = pre_attn_b[i]
        out[p + "post_attention_layernorm.dense.weight"] = kT(pre_ffw_k[i])
        out[p + "post_attention_layernorm.dense.bias"] = pre_ffw_b[i]
        # q: [heads=8, head_dim=256, hidden=1024] -> [2048, 1024] head-major
        out[p + "self_attn.q_proj.weight"] = _gemma_q_w(q[i], EXP_HIDDEN)
        out[p + "self_attn.k_proj.weight"] = _gemma_kv_w(kv[i, 0], EXP_HIDDEN)
        out[p + "self_attn.v_proj.weight"] = _gemma_kv_w(kv[i, 1], EXP_HIDDEN)
        # o: ASYMMETRIC [heads=8, head_dim=256, out=1024] -> [in=2048, out=1024] -> T
        out[p + "self_attn.o_proj.weight"] = _gemma_o_w(o[i], EXP_HIDDEN)
        out[p + "mlp.gate_proj.weight"] = kT(gate_up[i, 0])
        out[p + "mlp.up_proj.weight"] = kT(gate_up[i, 1])
        out[p + "mlp.down_proj.weight"] = kT(down[i])


def _gemma_q_w(per_layer: np.ndarray, in_dim: int) -> np.ndarray:
    """q_einsum -> PyTorch weight ``[heads * head_dim, in]``.

    OpenPI checkpoints have appeared with both ``[heads, head_dim, in]`` and
    ``[heads, in, head_dim]`` metadata layouts across versions.  Normalize to
    head-major ``[heads, head_dim, in]`` before flattening.
    """
    heads, a, b = per_layer.shape
    if b == in_dim:
        head_dim = a
        normalized = per_layer
    elif a == in_dim:
        head_dim = b
        normalized = np.swapaxes(per_layer, 1, 2)
    else:
        raise AssertionError(
            f"q shape {per_layer.shape} incompatible with in_dim={in_dim}"
        )
    return np.ascontiguousarray(normalized.reshape(heads * head_dim, in_dim))


def _gemma_kv_w(per_kv: np.ndarray, in_dim: int) -> np.ndarray:
    """kv_einsum slice -> PyTorch weight ``[head_dim, in]``.

    Accept both ``[kv_head, head_dim, in]`` and ``[kv_head, in, head_dim]``.
    """
    kv_head, a, b = per_kv.shape
    assert kv_head == TXT_KV_HEADS, per_kv.shape
    if b == in_dim:
        normalized = per_kv
        head_dim = a
    elif a == in_dim:
        normalized = np.swapaxes(per_kv, 1, 2)
        head_dim = b
    else:
        raise AssertionError(
            f"kv shape {per_kv.shape} incompatible with in_dim={in_dim}"
        )
    return np.ascontiguousarray(normalized.reshape(head_dim, in_dim))


def _gemma_o_w(per_layer: np.ndarray, out_dim: int) -> np.ndarray:
    """attn_vec_einsum [heads, head_dim, out] -> weight [out, heads*head_dim].

    JAX o-projection contracts over (heads, head_dim) producing ``out``; the
    leaf is ``[heads, head_dim, out_dim]``. PyTorch o_proj is ``[out, in]`` with
    ``in = heads*head_dim`` -> reshape to ``[in, out]`` then transpose.
    """
    heads, head_dim, jout = per_layer.shape
    assert jout == out_dim, f"o out-dim {jout} != {out_dim}"
    flat = per_layer.reshape(heads * head_dim, out_dim)  # [in, out]
    return np.ascontiguousarray(flat.T)  # [out, in]


# ===========================================================================
# Diff (dry-run) and write.
# ===========================================================================
NP_TO_ST = {
    "float32": "F32",
    "float16": "F16",
    "bfloat16": "BF16",  # only reached after torch cast; numpy has no bf16
}


def diff_against_reference(built: dict[str, np.ndarray], reference: Path) -> int:
    ref = read_safetensors_header(reference)
    ref_keys = set(ref)
    built_keys = set(built)

    missing = sorted(ref_keys - built_keys)  # in product, not produced
    extra = sorted(built_keys - ref_keys)  # produced, not in product
    mismatched = []
    for k in sorted(ref_keys & built_keys):
        rshape = list(ref[k]["shape"])
        bshape = list(built[k].shape)
        if rshape != bshape:
            mismatched.append((k, bshape, rshape))

    print(f"reference keys : {len(ref_keys)}")
    print(f"built keys     : {len(built_keys)}")
    print(f"missing        : {len(missing)}")
    print(f"extra          : {len(extra)}")
    print(f"mismatched     : {len(mismatched)}")

    def _show(title, items, fmt):
        if items:
            print(f"\n-- {title} (first 20) --")
            for it in items[:20]:
                print("  " + fmt(it))

    _show("MISSING", missing, lambda k: k)
    _show("EXTRA", extra, lambda k: k)
    _show(
        "MISMATCHED",
        mismatched,
        lambda t: f"{t[0]}  built={t[1]} ref={t[2]}",
    )

    ok = not missing and not extra and not mismatched
    print(f"\nRESULT: {'OK — shapes match' if ok else 'FAIL — see above'}")
    return 0 if ok else 1


def write_safetensors(built: dict[str, np.ndarray], out_path: Path, dtype: str) -> None:
    import torch
    from safetensors.torch import save_file

    torch_dtype = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }[dtype]

    tensors: dict[str, torch.Tensor] = {}
    for k, arr in built.items():
        t = torch.from_numpy(np.ascontiguousarray(arr))
        # F32 params (norms, action/time heads) stay float32 to match product;
        # everything else casts to the requested dtype.
        if t.dtype == torch.float32 and _keep_fp32(k):
            tensors[k] = t.contiguous()
        else:
            tensors[k] = t.to(torch_dtype).contiguous()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(out_path.parent), suffix=".tmp")
    os.close(fd)
    save_file(tensors, tmp, metadata={"format": "pt"})
    os.replace(tmp, out_path)
    print(f"wrote {len(tensors)} tensors -> {out_path}")


def _keep_fp32(key: str) -> bool:
    """Keys the reference product stores as float32 (norms + action/time heads)."""
    return (
        key.endswith(".dense.weight")
        or key.endswith(".dense.bias")
        or "patch_embedding" in key
        or "position_embedding" in key
        or "language_model.norm.weight" in key
        or "action_in_proj" in key
        or "action_out_proj" in key
        or "time_mlp_in" in key
        or "time_mlp_out" in key
    )


# ===========================================================================
# Complete checkpoint sidecars (config + processor stats).
# ===========================================================================
SIDECAR_FILES = (
    "config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor_step_2_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)


def write_checkpoint_sidecars(out_dir: Path, metadata_source: Path) -> None:
    """Copy config / processor metadata needed by ``PI05LiberoPipeline``.

    The OpenPI Orbax checkpoint only stores model parameters.  PhyAI also needs
    the LeRobot-style config, pre/postprocessor JSON, and normalizer stats.  The
    default metadata source is the verified converted checkpoint directory, so a
    reconversion produces a complete runnable checkpoint directory.
    """

    missing = [name for name in SIDECAR_FILES if not (metadata_source / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"metadata source {metadata_source} is missing sidecar files: {missing}. "
            "Pass --metadata-source pointing at a complete converted checkpoint "
            "directory that contains config/preprocessor/postprocessor files."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in SIDECAR_FILES:
        shutil.copy2(metadata_source / name, out_dir / name)
    print(f"copied {len(SIDECAR_FILES)} sidecar files -> {out_dir}")


# ===========================================================================
# CLI.
# ===========================================================================
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="OpenPI checkpoint root containing params/",
    )
    ap.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="reference product safetensors (read-only, shapes only)",
    )
    ap.add_argument(
        "--metadata-source",
        type=Path,
        help=(
            "complete converted checkpoint directory to copy config/processor "
            "sidecars from when --write is used"
        ),
    )
    ap.add_argument(
        "--write",
        action="store_true",
        help="actually write the converted checkpoint (default: dry-run only)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        help="output safetensors path for --write",
    )
    ap.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="required to let --out point at an existing file (e.g. the product)",
    )
    ap.add_argument("--dtype", default="bf16", help="write dtype for non-fp32 params")
    args = ap.parse_args(argv)

    if args.write and args.out is None:
        ap.error("--out is required when --write is used")
    if args.write and args.metadata_source is None:
        ap.error("--metadata-source is required when --write is used")

    # Header reader works without OpenPI; load source only when needed (always,
    # since both dry-run and write build the target dict).
    print(f"loading OpenPI State from {args.checkpoint} ...", file=sys.stderr)
    flat = load_openpi_state(args.checkpoint)
    src = Source(flat)
    print(f"  loaded {len(flat)} source leaves", file=sys.stderr)

    built = build_target(src)

    # Sanity: embedder must be full vocab, not a shard.
    embed_shape = built[PALI_LM_HEAD].shape
    assert tuple(embed_shape) == (VOCAB, TXT_HIDDEN), (
        f"embedder shape {embed_shape} != ({VOCAB}, {TXT_HIDDEN}); "
        "the OpenPI State looks sharded — check the loader / OpenPI pin."
    )

    rc = diff_against_reference(built, args.reference)

    if not args.write:
        print("\n(dry-run — nothing written; pass --write to produce a checkpoint)")
        return rc

    if rc != 0:
        print("\nrefusing to --write: shape diff is non-empty (fix mapping first)")
        return rc

    out = args.out
    assert out is not None
    if out.resolve() == args.reference.resolve() and not args.allow_overwrite:
        print(
            f"\nrefusing to overwrite the reference product at {out};\n"
            "pass --allow-overwrite to override (NOT recommended)."
        )
        return 2
    if out.exists() and not args.allow_overwrite:
        print(
            f"\nrefusing to overwrite existing {out}; "
            "choose a new --out or pass --allow-overwrite."
        )
        return 2

    write_safetensors(built, out, args.dtype)
    assert args.metadata_source is not None
    write_checkpoint_sidecars(out.parent, args.metadata_source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
