"""Recipe ordered list of modifiers, buildable from python or yaml/json.

Kept deliberately small (no Session/stage machinery): a recipe is just the set
of modifiers ``oneshot`` / ``model_free_ptq`` apply.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from phyai_model_optimizer.modifiers.awq import AWQModifier
from phyai_model_optimizer.modifiers.base import Modifier
from phyai_model_optimizer.modifiers.gptq import GPTQModifier
from phyai_model_optimizer.modifiers.rtn import RTNModifier
from phyai_model_optimizer.modifiers.smoothquant import SmoothQuantModifier

_REGISTRY: dict[str, type[Modifier]] = {
    "RTN": RTNModifier,
    "GPTQ": GPTQModifier,
    "AWQ": AWQModifier,
    "SmoothQuant": SmoothQuantModifier,
}


def build_modifier(spec: dict) -> Modifier:
    spec = dict(spec)
    kind = spec.pop("type", None) or spec.pop("modifier", None)
    if kind not in _REGISTRY:
        raise ValueError(f"unknown modifier type {kind!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[kind](**spec)


@dataclass
class Recipe:
    modifiers: list[Modifier]

    @classmethod
    def from_modifiers(cls, modifiers: list[Modifier] | Modifier) -> "Recipe":
        if isinstance(modifiers, Modifier):
            modifiers = [modifiers]
        return cls(list(modifiers))

    @classmethod
    def from_dict(cls, data: dict) -> "Recipe":
        specs = data.get("modifiers", [])
        return cls([build_modifier(s) for s in specs])

    @classmethod
    def load(cls, path: str) -> "Recipe":
        with open(path) as f:
            text = f.read()
        if os.path.splitext(path)[1] in (".yaml", ".yml"):
            import yaml  # note(chenghua): optional dep

            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls.from_dict(data)


__all__ = ["Recipe", "build_modifier"]
