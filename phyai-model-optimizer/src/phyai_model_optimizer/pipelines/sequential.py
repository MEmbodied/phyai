"""SequentialCalibrationPipeline. Per-sample sequential replay.

Correctness requires TRUE sequential replay: quantize block ``i``, then re-run it
so block ``i+1`` sees post-quant inputs. The single-pass "collect all bf16 inputs,
quantize independently" shortcut drops inter-layer error propagation and is rejected.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

from phyai_model_optimizer.observers.base import Observer
from phyai_model_optimizer.pipelines.base import Pipeline, SequentialDriver, Target


def _pre_hook(observer: Observer):
    def hook(module, args, kwargs):  # noqa: ANN001
        if args:
            observer.observe(args[0])

    return hook


class SequentialCalibrationPipeline(Pipeline):
    def run(
        self,
        model: nn.Module,
        targets: list[Target],
        *,
        dataloader: Iterable | None = None,
        driver: SequentialDriver | None = None,
    ) -> None:
        calib = [(n, m, mod) for (n, m, mod) in targets if mod.requires_calibration]
        datafree = [
            (n, m, mod) for (n, m, mod) in targets if not mod.requires_calibration
        ]

        with torch.no_grad():
            # note(chenghua): non-calibration targets (e.g. RTN on vision/head) need no forward.
            for _n, module, modifier in datafree:
                modifier.quantize_layer(module, None)

            if not calib:
                return
            if driver is None or dataloader is None:
                raise ValueError(
                    "SequentialCalibrationPipeline needs a driver + dataloader for "
                    "calibration-requiring modifiers"
                )
            self._run_sequential(calib, dataloader, driver)

    def _run_sequential(
        self, calib: list[Target], dataloader: Iterable, driver: SequentialDriver
    ) -> None:
        blocks = driver.blocks()
        by_block: list[tuple[str, nn.Module, list[Target]]] = []
        claimed: set[str] = set()
        for bname, bmod in blocks:
            inside = [
                (n, m, mod)
                for (n, m, mod) in calib
                if n == bname or n.startswith(bname + ".")
            ]
            claimed.update(n for n, _, _ in inside)
            by_block.append((bname, bmod, inside))
        orphan = [n for (n, _, _) in calib if n not in claimed]
        if orphan:
            raise RuntimeError(
                f"calibration targets outside any sequential block: {orphan[:5]}... "
                "(extend the driver's block list or route them to RTN)"
            )

        cached = driver.seed(dataloader)
        for bname, bmod, inside in by_block:
            if not inside:
                # note(chenghua): no targets here, but still must propagate to keep inputs correct.
                cached = [
                    driver.advance(a, k, driver.replay(bmod, a, k)) for (a, k) in cached
                ]
                continue

            observers: dict[str, Observer] = {}
            handles = []
            for name, module, modifier in inside:
                obs = modifier.make_observer(module)
                if obs is None:
                    continue
                observers[name] = obs
                handles.append(
                    module.register_forward_pre_hook(_pre_hook(obs), with_kwargs=True)
                )

            # note(chenghua): pass 1 collects stats while weights are still bf16.
            for a, k in cached:
                driver.replay(bmod, a, k)
            for h in handles:
                h.remove()

            for name, module, modifier in inside:
                modifier.quantize_layer(module, observers.get(name))

            # note(chenghua): pass 2 re-runs with quantized weights so the next block sees post-quant inputs.
            cached = [
                driver.advance(a, k, driver.replay(bmod, a, k)) for (a, k) in cached
            ]


class GenericSequentialDriver:
    """Driver for a plain chain of ``x -> x`` blocks (used by CPU tests).

    Assumes ``dataloader`` yields tensors that are the input to the FIRST block,
    and each block is called as ``block(x) -> x`` (or returns ``(x, ...)``).

    Runtime checkable with SequentialDriver:

    ``isinstance(GenericSequentialDriver([...]), SequentialDriver) == True``
    """

    def __init__(self, blocks: list[tuple[str, nn.Module]]) -> None:
        self._blocks = blocks

    def blocks(self) -> list[tuple[str, nn.Module]]:
        return self._blocks

    def seed(self, dataloader: Iterable) -> list[tuple[tuple, dict]]:
        return [((x,), {}) for x in dataloader]

    @staticmethod
    def _out(y):
        return y[0] if isinstance(y, tuple) else y

    def replay(self, block: nn.Module, args: tuple, kwargs: dict):
        return self._out(block(*args, **kwargs))

    def advance(self, args: tuple, kwargs: dict, output) -> tuple[tuple, dict]:
        return ((output,), {})


__all__ = ["SequentialCalibrationPipeline", "GenericSequentialDriver"]
