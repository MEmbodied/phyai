"""Index / packing helpers for the pi0.5 inference batch layout.

The scheduler invokes these once per inference to build every metadata
tensor the runner forward batches need:

* :func:`build_prefix_indptrs` — cumulative real prefix lengths.
* :func:`build_suffix_indptrs` — cumulative suffix lengths
  (constant ``chunk_size`` per sample).
* :func:`build_full_indptrs` — joint (prefix + suffix) cumulative lengths.
* :func:`build_pos_ids_from_indptr` — per-segment ``arange`` flattened.
* :func:`broadcast_cond_to_tokens` — broadcast a per-sample
  AdaRMS condition into per-token form.
* :func:`pack_prefix_per_sample_padded` — assemble the padded
  ``(B * n_per_sample, hidden)`` prefix buffer the LLM runner consumes.
* :func:`build_prefix_padded_pos_ids` — RoPE positions for the padded
  prefix buffer (one per token, padding rows reuse local positions).
* :func:`build_prefix_padded_write_indices` — KV-pool slot index per
  prefix token; padding rows write to the sentinel slot.
* :func:`build_prefix_paged_kv_indices` — flat real-prefix slot list
  consumed by the prefix attention wrapper.
* :func:`build_suffix_pos_ids` — RoPE positions for suffix tokens
  (each sample's chunk sits at ``[real_len_b .. real_len_b + chunk-1]``).
* :func:`build_suffix_write_indices` — KV-pool slot indices for suffix
  K/V writes (one per suffix token).
* :func:`build_joint_paged_kv_indices` — interleaved
  ``[prefix_b0, suffix_b0, prefix_b1, suffix_b1, ...]`` slot list for
  the joint-attention wrapper.

All helpers return tensors on the same device as their primary input;
the scheduler is the sole owner of the metadata tensors and pre-builds
them once per inference (cuda graph captures the index reads against
fixed-shape buffers).
"""

from __future__ import annotations

import torch


# ============================================================================ #
# 1. Indptrs and position ids — shape primitives                               #
# ============================================================================ #


def build_prefix_indptrs(
    image_token_count: int,
    lang_lens: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Cumulative real-prefix lengths.

    Each sample contributes ``image_token_count + lang_lens[b]`` real
    tokens to the prefix.

    Returns ``(cu_prefix (B+1,) int32, total_real_prefix_len int)``.
    """
    if lang_lens.dim() != 1:
        raise ValueError(f"lang_lens must be 1-D, got shape {tuple(lang_lens.shape)}.")
    device = lang_lens.device
    seg_lens = lang_lens.to(torch.int64) + int(image_token_count)
    if int(seg_lens.min()) < int(image_token_count):
        raise ValueError("lang_lens must be non-negative.")
    cu = torch.zeros(seg_lens.numel() + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(seg_lens, dim=0).to(torch.int32)
    return cu, int(cu[-1])


def build_suffix_indptrs(
    batch_size: int,
    chunk_size: int,
    *,
    device: torch.device | str,
) -> tuple[torch.Tensor, int]:
    """Cumulative suffix lengths — constant ``chunk_size`` per sample.

    Returns ``(cu_suffix (B+1,) int32, total_suffix_len int)`` where
    ``cu_suffix == [0, chunk_size, 2*chunk_size, ...]``.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    cu = torch.arange(
        0,
        (batch_size + 1) * chunk_size,
        chunk_size,
        dtype=torch.int32,
        device=torch.device(device),
    )
    return cu, batch_size * chunk_size


def build_full_indptrs(
    cu_prefix: torch.Tensor, cu_suffix: torch.Tensor
) -> torch.Tensor:
    """Joint indptrs: ``cu_full[b] = cu_prefix[b] + cu_suffix[b]``.

    Returns int32 ``(B+1,)``.
    """
    if cu_prefix.shape != cu_suffix.shape:
        raise ValueError(
            f"cu_prefix.shape={tuple(cu_prefix.shape)} must equal "
            f"cu_suffix.shape={tuple(cu_suffix.shape)}."
        )
    if cu_prefix.device != cu_suffix.device:
        raise ValueError(
            f"cu_prefix on {cu_prefix.device} but cu_suffix on "
            f"{cu_suffix.device}; place them on the same device first."
        )
    return (cu_prefix.to(torch.int64) + cu_suffix.to(torch.int64)).to(torch.int32)


def build_pos_ids_from_indptr(
    cu_seqlens: torch.Tensor,
    *,
    offset: torch.Tensor | int = 0,
) -> torch.Tensor:
    """Per-segment ``arange(L_b)`` flattened to ``(N_total,)`` int32.

    ``offset`` may be a scalar or a ``(B,)`` per-sample offset tensor.
    """
    if cu_seqlens.dim() != 1 or cu_seqlens.numel() < 2:
        raise ValueError(
            f"cu_seqlens must be 1-D with length >= 2, got shape "
            f"{tuple(cu_seqlens.shape)}."
        )
    device = cu_seqlens.device
    cu64 = cu_seqlens.to(torch.int64)
    n_total = int(cu64[-1])
    arange_total = torch.arange(n_total, dtype=torch.int64, device=device)
    seg_id = torch.searchsorted(cu64[1:], arange_total, right=True)
    pos_within = arange_total - cu64[seg_id]
    if isinstance(offset, torch.Tensor):
        if offset.shape != (cu64.numel() - 1,):
            raise ValueError(
                f"offset tensor shape {tuple(offset.shape)} != (B,)="
                f"({cu64.numel() - 1},)."
            )
        per_seg_offset = offset.to(device=device, dtype=torch.int64)[seg_id]
    else:
        per_seg_offset = int(offset)
    return (pos_within + per_seg_offset).to(torch.int32)


def broadcast_cond_to_tokens(
    cond_per_sample: torch.Tensor,
    seg_lens: torch.Tensor,
) -> torch.Tensor:
    """Broadcast a ``(B, D_cond)`` condition to ``(N_total, D_cond)``.

    ``seg_lens`` is the per-sample token count; the returned tensor
    repeats each ``cond_per_sample[b]`` exactly ``seg_lens[b]`` times.
    """
    if cond_per_sample.dim() != 2:
        raise ValueError(
            f"cond_per_sample must be 2-D (B, D_cond), got shape "
            f"{tuple(cond_per_sample.shape)}."
        )
    if seg_lens.dim() != 1 or seg_lens.shape[0] != cond_per_sample.shape[0]:
        raise ValueError(
            f"seg_lens shape {tuple(seg_lens.shape)} not compatible with "
            f"cond_per_sample shape {tuple(cond_per_sample.shape)}."
        )
    return cond_per_sample.repeat_interleave(seg_lens.to(torch.int64), dim=0)


# ============================================================================ #
# 2. Per-sample padded prefix layout                                           #
# ============================================================================ #
#
# The LLM runner is captured at fixed shape ``(B * n_per_sample, hidden)``
# so the prefix is padded per-sample to ``n_per_sample`` rows. Real
# tokens occupy the leading ``image_token_count + lang_lens[b]`` rows;
# padding rows hold zeros (or stale embed-step output) and write their
# K/V to the cache pool's sentinel slot where it is harmless.


def pack_prefix_per_sample_padded(
    image_embs: torch.Tensor,
    lang_embs: torch.Tensor,
    lang_lens: torch.Tensor,
    *,
    n_per_sample: int,
) -> torch.Tensor:
    """Assemble per-sample-padded ``(B * n_per_sample, D)`` prefix buffer.

    Per sample ``b`` the layout is::

        [image_b (N_img), lang_b[:lang_lens[b]] (real_lang_b), padding (rest)]

    where ``N_img + tokenizer_max_length == n_per_sample`` and the
    padding region holds zeros.
    """
    if image_embs.dim() != 3 or lang_embs.dim() != 3:
        raise ValueError(
            f"image_embs and lang_embs must be 3-D (B, N, D); got "
            f"image_embs={tuple(image_embs.shape)}, "
            f"lang_embs={tuple(lang_embs.shape)}."
        )
    B = image_embs.shape[0]
    if lang_embs.shape[0] != B:
        raise ValueError(f"image_embs B={B} != lang_embs B={lang_embs.shape[0]}.")
    if image_embs.shape[-1] != lang_embs.shape[-1]:
        raise ValueError(
            f"image_embs last dim {image_embs.shape[-1]} != lang_embs last "
            f"dim {lang_embs.shape[-1]}."
        )
    if lang_lens.dim() != 1 or lang_lens.shape[0] != B:
        raise ValueError(
            f"lang_lens must be (B,)=({B},), got shape {tuple(lang_lens.shape)}."
        )
    n_img = image_embs.shape[1]
    if n_img > n_per_sample:
        raise ValueError(f"image_token_count {n_img} > n_per_sample {n_per_sample}.")
    D = image_embs.shape[-1]
    device = image_embs.device
    dtype = image_embs.dtype

    packed = torch.zeros(B * n_per_sample, D, dtype=dtype, device=device)
    lang_lens_list = lang_lens.tolist()
    for b in range(B):
        L_lang = int(lang_lens_list[b])
        if L_lang + n_img > n_per_sample:
            raise ValueError(
                f"sample {b}: image({n_img}) + lang({L_lang}) = "
                f"{L_lang + n_img} exceeds n_per_sample {n_per_sample}."
            )
        base = b * n_per_sample
        packed[base : base + n_img] = image_embs[b]
        if L_lang > 0:
            packed[base + n_img : base + n_img + L_lang] = lang_embs[b, :L_lang]
    return packed


def build_prefix_padded_pos_ids(
    batch_size: int,
    n_per_sample: int,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """RoPE positions for a per-sample-padded prefix buffer.

    Per sample ``b``, position ``j`` is just ``j`` (local index) for
    every row including padding rows. Real tokens get the right RoPE
    rotation; padding rows produce K rotations that go to the sentinel
    slot and are never read by attention.

    Returns ``(B * n_per_sample,)`` int32.
    """
    return torch.arange(
        n_per_sample, dtype=torch.int32, device=torch.device(device)
    ).repeat(batch_size)


def build_prefix_padded_write_indices(
    real_lens: torch.Tensor,
    *,
    n_per_sample: int,
    prefix_slot_base: int,
    sentinel_slot: int = 0,
) -> torch.Tensor:
    """KV-pool slot index per padded prefix token.

    Real token ``b * n_per_sample + j`` (with ``j < real_lens[b]``) writes
    to ``prefix_slot_base + cu_real[b] + j``. Padding rows write to
    ``sentinel_slot`` (typically 0).

    Returns ``(B * n_per_sample,)`` int64 — directly consumable by
    :meth:`KVCachePool.write_kv`.
    """
    if real_lens.dim() != 1:
        raise ValueError(f"real_lens must be 1-D, got shape {tuple(real_lens.shape)}.")
    device = real_lens.device
    B = int(real_lens.shape[0])
    real64 = real_lens.to(torch.int64)
    cu_real = torch.zeros(B + 1, dtype=torch.int64, device=device)
    cu_real[1:] = torch.cumsum(real64, 0)

    j = torch.arange(n_per_sample, dtype=torch.int64, device=device).unsqueeze(0)
    real_at_b = real64.unsqueeze(1)  # (B, 1)
    cu_at_b = cu_real[:-1].unsqueeze(1)  # (B, 1)
    is_real = j < real_at_b  # (B, n_per_sample)
    real_slot = prefix_slot_base + cu_at_b + j
    write = torch.where(
        is_real,
        real_slot,
        torch.full_like(real_slot, int(sentinel_slot)),
    )
    return write.flatten().to(torch.int64)


def build_prefix_paged_kv_indices(
    total_real_prefix_len: int,
    *,
    prefix_slot_base: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Flat ``(N_real,)`` int32 — contiguous prefix slot ids.

    The real prefix tokens are written contiguously starting at
    ``prefix_slot_base`` (see :func:`build_prefix_padded_write_indices`),
    so the paged-kv-indices for the prefix self-attention wrapper is
    simply the contiguous range.
    """
    return torch.arange(
        prefix_slot_base,
        prefix_slot_base + int(total_real_prefix_len),
        dtype=torch.int32,
        device=torch.device(device),
    )


def build_prefix_last_page_len(
    real_lens: torch.Tensor,
) -> torch.Tensor:
    """Per-sample last-page length, ``page_size=1`` convention.

    Empty samples return 0; non-empty return 1.

    Returns ``(B,)`` int32.
    """
    return (real_lens > 0).to(torch.int32)


# ============================================================================ #
# 3. Suffix layout (joint attention input)                                     #
# ============================================================================ #


def build_suffix_pos_ids(
    real_lens: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """RoPE positions for suffix tokens.

    Sample ``b``'s chunk sits at positions
    ``[real_len_b, real_len_b + 1, ..., real_len_b + chunk_size - 1]``
    so the joint attention K layout (cached prefix + fresh suffix) sees
    one coherent ``[0..real_len_b + chunk_size - 1]`` per sample.

    Returns ``(B * chunk_size,)`` int32.
    """
    if real_lens.dim() != 1:
        raise ValueError(f"real_lens must be 1-D, got shape {tuple(real_lens.shape)}.")
    device = real_lens.device
    base = real_lens.to(torch.int64).unsqueeze(1)
    j = torch.arange(chunk_size, dtype=torch.int64, device=device).unsqueeze(0)
    return (base + j).flatten().to(torch.int32)


def build_suffix_write_indices(
    batch_size: int,
    chunk_size: int,
    *,
    suffix_slot_base: int,
    device: torch.device | str,
) -> torch.Tensor:
    """KV-pool slot indices for the suffix K/V writes.

    Sample ``b``'s ``chunk_size`` tokens write to
    ``[suffix_slot_base + b*chunk_size, suffix_slot_base + (b+1)*chunk_size)``.

    Returns ``(B * chunk_size,)`` int64.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    return torch.arange(
        suffix_slot_base,
        suffix_slot_base + batch_size * chunk_size,
        dtype=torch.int64,
        device=torch.device(device),
    )


def build_joint_paged_kv_indices(
    real_lens: torch.Tensor,
    chunk_size: int,
    *,
    prefix_slot_base: int,
    suffix_slot_base: int,
) -> torch.Tensor:
    """Per-sample interleaved slot list for the joint-attention wrapper.

    Output layout per sample::

        [prefix_b0_slots (real_len_0), suffix_b0_slots (chunk_size),
         prefix_b1_slots (real_len_1), suffix_b1_slots (chunk_size), ...]

    Concatenated end-to-end, returned as ``(N_full,)`` int32 where
    ``N_full = sum(real_lens) + B * chunk_size``.
    """
    if real_lens.dim() != 1:
        raise ValueError(f"real_lens must be 1-D, got shape {tuple(real_lens.shape)}.")
    device = real_lens.device
    B = int(real_lens.shape[0])
    real64 = real_lens.to(torch.int64)

    cu_p = torch.zeros(B + 1, dtype=torch.int64, device=device)
    cu_p[1:] = torch.cumsum(real64, 0)
    full_lens = real64 + chunk_size
    cu_full = torch.zeros(B + 1, dtype=torch.int64, device=device)
    cu_full[1:] = torch.cumsum(full_lens, 0)
    n_full = int(cu_full[-1])

    arange_full = torch.arange(n_full, dtype=torch.int64, device=device)
    seg_id = torch.searchsorted(cu_full[1:], arange_full, right=True)
    pos_within = arange_full - cu_full[seg_id]
    real_at_seg = real64[seg_id]
    is_prefix = pos_within < real_at_seg

    prefix_slot = prefix_slot_base + cu_p[seg_id] + pos_within
    suffix_slot = suffix_slot_base + seg_id * chunk_size + (pos_within - real_at_seg)
    return torch.where(is_prefix, prefix_slot, suffix_slot).to(torch.int32)


def build_joint_last_page_len(
    real_lens: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Joint-attention last-page length per sample.

    Real samples always have ``chunk_size > 0`` suffix tokens, so the
    last page is full of one token (``page_size=1``) → 1.

    Returns ``(B,)`` int32.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    return torch.ones(real_lens.shape[0], dtype=torch.int32, device=real_lens.device)


__all__ = [
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
