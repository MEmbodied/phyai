"""Atom encoding helpers for multimodal LLM mixed prompts and EAGLE bigram.

The cache itself is modality-blind; it only sees byte streams. These helpers
turn typed input (token ids + image hashes) into the right byte layout.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ImageSegment:
    """One image-token span in a multimodal prompt.

    Attributes
    ----------
    image_hash_lo, image_hash_hi
        128-bit content hash split into two 64-bit halves
        (low, high). Use a hash with high-bit entropy (BLAKE3 / UUIDv4).
    insert_position
        Token-stream index where the image segment begins.
    length
        Number of image tokens this segment covers.
    """

    image_hash_lo: int
    image_hash_hi: int
    insert_position: int
    length: int


def page_align(atoms: bytes, page_bytes: int) -> tuple[bytes, int]:
    """Truncate ``atoms`` to the largest page-aligned prefix.

    Returns ``(aligned_bytes, dropped_bytes_count)``.
    """
    if page_bytes <= 0:
        raise ValueError("page_bytes must be positive")
    keep = (len(atoms) // page_bytes) * page_bytes
    return atoms[:keep], len(atoms) - keep


def encode_text_atoms_int32(token_ids: Sequence[int]) -> bytes:
    """Pack int32 token ids into a contiguous byte stream (atom_bytes=4)."""
    return struct.pack(f"{len(token_ids)}i", *token_ids)


def encode_eagle_bigram_atoms(token_ids: Sequence[int]) -> bytes:
    """Sliding bigram encoding: N tokens -> N-1 atoms of 8 bytes each.

    Each atom packs ``(curr_token, next_token)`` as a little-endian int32 pair.
    """
    if len(token_ids) < 2:
        return b""
    parts = []
    for i in range(len(token_ids) - 1):
        parts.append(struct.pack("ii", int(token_ids[i]), int(token_ids[i + 1])))
    return b"".join(parts)


def encode_chunk_hash_atoms_u64(hashes: Iterable[int]) -> bytes:
    """Pack 64-bit chunk hashes (video / audio) as little-endian u64 atoms."""
    hashes = list(hashes)
    return struct.pack(
        f"{len(hashes)}Q", *(int(h) & 0xFFFFFFFFFFFFFFFF for h in hashes)
    )


def encode_image_hash_atom_sha256(digest: bytes) -> bytes:
    """Image content hash as a 32-byte atom (SHA-256 raw digest)."""
    if len(digest) != 32:
        raise ValueError("digest must be 32 bytes")
    return bytes(digest)


def encode_prompt_to_atoms(
    text_token_ids: Sequence[int],
    image_segments: Sequence[ImageSegment],
) -> bytes:
    """Encode a multimodal prompt to 16-byte mixed atoms.

    Layout per atom (little-endian, two u64 limbs):

    - text token ``i``:
      - lo = ``u64(token_id)`` (high bits zero)
      - hi = ``u64(0)``                (text marker)
    - image token ``i`` in segment ``k``:
      - lo = ``u64(image_hash_lo)``
      - hi = ``u64(image_hash_hi) ^ u64(token_pos_in_image)``

    Text and image atoms never collide: text atoms have ``hi == 0`` while
    image atoms have ``hi == image_hash_hi ^ pos``; as long as
    ``image_hash_hi`` is non-zero (BLAKE3 / UUIDv4 satisfies this with
    overwhelming probability) the encoding is collision-free between text and
    image segments.
    """
    n_total = len(text_token_ids) + sum(s.length for s in image_segments)
    atoms: list[tuple[int, int]] = [(0, 0)] * n_total

    image_slots: set[int] = set()
    for seg in image_segments:
        if seg.insert_position < 0 or seg.insert_position + seg.length > n_total:
            raise ValueError(
                f"ImageSegment(at={seg.insert_position}, len={seg.length}) "
                f"out of bounds for total length {n_total}"
            )
        for k in range(seg.length):
            idx = seg.insert_position + k
            atoms[idx] = (
                int(seg.image_hash_lo) & 0xFFFFFFFFFFFFFFFF,
                (int(seg.image_hash_hi) ^ k) & 0xFFFFFFFFFFFFFFFF,
            )
            image_slots.add(idx)

    text_iter = iter(text_token_ids)
    for i in range(n_total):
        if i in image_slots:
            continue
        try:
            tok = next(text_iter)
        except StopIteration as e:
            raise ValueError("Not enough text tokens to fill non-image slots") from e
        atoms[i] = (int(tok) & 0xFFFFFFFFFFFFFFFF, 0)

    return b"".join(struct.pack("<QQ", lo, hi) for (lo, hi) in atoms)
