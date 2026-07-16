"""Orchestrator resolve targets, pick the pipeline, quantize, serialize."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

from phyai_model_optimizer import serialize
from phyai_model_optimizer.modifiers.base import Modifier
from phyai_model_optimizer.pipelines.base import (
    SequentialDriver,
    Target,
    select_pipeline,
)


def resolve_targets(model: nn.Module, modifiers: list[Modifier]) -> list[Target]:
    """First-match assignment of quantizable (2-D ``weight``) Linears to modifiers."""
    targets: list[Target] = []
    matched: set[int] = set()
    for name, module in model.named_modules():
        w = getattr(module, "weight", None)
        if not isinstance(w, nn.Parameter) or w.ndim != 2:
            continue
        for modifier in modifiers:
            if modifier.matches(name, module):
                targets.append((name, module, modifier))
                matched.add(id(modifier))
                break
    # note(chenghua): a modifier matching nothing is almost always a mis-specified
    # target — fail loudly instead of silently emitting an unquantized checkpoint.
    unmatched = [m for m in modifiers if id(m) not in matched]
    if unmatched:
        raise RuntimeError(
            f"{len(unmatched)} modifier(s) matched no quantizable module; "
            f"check their targets={[m.targets() for m in unmatched]}"
        )
    return targets


def run_oneshot(
    model: nn.Module,
    modifiers: list[Modifier],
    *,
    dataloader: Iterable | None = None,
    driver: SequentialDriver | None = None,
    pack_format: str = "compressed-tensors",
    save_dir: str | None = None,
    base_checkpoint: str | None = None,
) -> list[Target]:
    if not modifiers:
        raise ValueError("run_oneshot needs at least one modifier")
    targets = resolve_targets(model, modifiers)
    if not targets:
        raise RuntimeError("no quantizable Linear modules matched the modifiers")

    pipeline = select_pipeline(modifiers)
    with torch.no_grad():
        pipeline.run(model, targets, dataloader=dataloader, driver=driver)

    if save_dir is not None:
        serialize.save(model, targets, save_dir, pack_format, base_checkpoint)
    return targets


__all__ = ["resolve_targets", "run_oneshot"]
