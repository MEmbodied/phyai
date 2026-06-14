"""Lightweight configurable processor pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ProcessorFn = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class ProcessorStep:
    """Single processing step."""

    name: str
    fn: ProcessorFn

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        return self.fn(data)


class ProcessorPipeline:
    """Simple pipeline that runs processors in order."""

    def __init__(self, steps: list[ProcessorStep]) -> None:
        self.steps = steps

    @classmethod
    def from_steps(cls, steps: list[tuple[str, ProcessorFn]]) -> "ProcessorPipeline":
        return cls([ProcessorStep(name=name, fn=fn) for name, fn in steps])

    @classmethod
    def load_config(cls, config_path: str | Path) -> dict[str, Any]:
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as f:
            if path.suffix.lower() in {".json", ""}:
                return json.load(f)
        raise ValueError(f"Unsupported pipeline config format: {path}")

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        for step in self.steps:
            data = step(data)
        return data
