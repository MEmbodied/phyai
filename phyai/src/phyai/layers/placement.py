"""Declarative weight placements — HF→phyai key remap and TP slicing in one.

A :class:`Placement` describes "this region of source X goes into that
region of dest Y". It replaces the four loader classes that used to live
in :mod:`phyai.layers.loaders` (Replicated / Column / Row / Vocab / QKV)
plus the ``param.loader`` indirection on every parameter. Every
parameter-bearing layer now exposes::

    def placements(self) -> list[Placement]: ...

returning the triples a model loader needs to land an HF state dict onto
this layer's local parameters. The TP / GQA / vocab-padding math happens
when the layer builds the placement (using its already-stored
``tp_rank`` / ``tp_size`` / ``output_partition_sizes`` / replica factors),
so callers see a flat data structure with no further branching.

The placement is purely declarative: building one never touches a
tensor. :func:`apply_placements` is the only function that copies. This
keeps ``placements()`` cheap to unit-test — assert the structure of the
list without spinning up CUDA.

Two placement variants:

* :class:`CopyPlacement` — narrow the HF tensor by ``src_slices``,
  narrow the phyai param by ``dst_slices``, and ``copy_``.
* :class:`ZeroPlacement` — narrow the phyai param by ``dst_slices`` and
  ``zero_``. Used for vocab-padding overhang.

Both narrow chains are applied left-to-right: ``narrow(d0).narrow(d1)``.
``Slice1D`` is one ``narrow(dim, start, size)`` step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import torch


@dataclass(frozen=True)
class Slice1D:
    """A 1-axis ``narrow(dim, start, size)`` step."""

    dim: int
    start: int
    size: int


@dataclass(frozen=True)
class CopyPlacement:
    """Copy a (sliced) HF tensor into a (sliced) phyai parameter region.

    Application order::

        src = hf_get(hf_key)
        for s in src_slices: src = src.narrow(s.dim, s.start, s.size)
        dst = dest_state[phyai_key]
        for s in dst_slices: dst = dst.narrow(s.dim, s.start, s.size)
        dst.copy_(src)
    """

    hf_key: str
    phyai_key: str
    src_slices: tuple[Slice1D, ...] = ()
    dst_slices: tuple[Slice1D, ...] = ()


@dataclass(frozen=True)
class ZeroPlacement:
    """Zero-fill a (sliced) phyai parameter region.

    Used today by :class:`phyai.layers.vocab_embedding.VocabParallelEmbedding`
    to clear padding rows on the trailing rank. Reserved for future
    quant-scale init paths that need to zero a sub-block of a scale
    tensor without an HF source.
    """

    phyai_key: str
    dst_slices: tuple[Slice1D, ...] = ()


Placement = CopyPlacement | ZeroPlacement


def apply_placements(
    placements: list[Placement],
    hf_get: Callable[[str], torch.Tensor],
    dest_state: Mapping[str, torch.Tensor],
) -> None:
    """Execute every placement, mutating ``dest_state`` tensors in place.

    Parameters
    ----------
    placements:
        The list of :class:`CopyPlacement` / :class:`ZeroPlacement`
        produced by ``module.placements()`` (typically aggregated across
        the whole model).
    hf_get:
        ``str -> Tensor`` lookup for HF source tensors. A callable is
        used (rather than a plain dict) so a checkpoint streaming
        across multiple safetensors shards can lazily fetch one tensor
        at a time. ``functools.partial(state_dict.__getitem__)`` works
        for the in-memory case.
    dest_state:
        Mapping from phyai parameter name to the writable param tensor
        (typically ``{n: p.data for n, p in module.named_parameters()}``).

    Notes
    -----
    The 1-element scalar fast path covers checkpoints that store a
    learned scalar with ``shape=()`` even though the in-memory parameter
    is ``shape=(1,)``. A strict ``copy_`` would reject that; we
    ``fill_`` instead.
    """

    for p in placements:
        if isinstance(p, CopyPlacement):
            src = hf_get(p.hf_key)
            for s in p.src_slices:
                src = src.narrow(s.dim, s.start, s.size)
            dst = dest_state[p.phyai_key]
            for s in p.dst_slices:
                dst = dst.narrow(s.dim, s.start, s.size)
            if src.dim() == 0 and dst.numel() == 1:
                dst.fill_(src.item())
                continue
            if dst.shape != src.shape:
                raise ValueError(
                    f"placement shape mismatch at {p.phyai_key!r}: "
                    f"dst={tuple(dst.shape)} src={tuple(src.shape)} "
                    f"(hf_key={p.hf_key!r})"
                )
            dst.copy_(src)
        else:
            dst = dest_state[p.phyai_key]
            for s in p.dst_slices:
                dst = dst.narrow(s.dim, s.start, s.size)
            dst.zero_()


def split_prefix(prefix: str) -> tuple[str, str]:
    """Split a dotted layer prefix into ``(parent, own)``.

    ``"model.layers.3.mlp.gate_up_proj"`` → ``("model.layers.3.mlp",
    "gate_up_proj")``. ``"gate_up_proj"`` → ``("", "gate_up_proj")``.
    Empty input is rejected — every layer must declare its location for
    HF-key construction to be unambiguous.
    """

    if not prefix:
        raise ValueError(
            "placement requires a non-empty prefix; set the layer's "
            "`prefix=` so its HF-key parent path is well-defined."
        )
    parent, _, own = prefix.rpartition(".")
    return parent, own


__all__ = [
    "CopyPlacement",
    "Placement",
    "Slice1D",
    "ZeroPlacement",
    "apply_placements",
    "split_prefix",
]
