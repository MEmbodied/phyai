"""Base config for phyai.models.

A :class:`PretrainedConfig` is a frozen dataclass with three jobs:

* be constructable from a JSON file or dict, silently dropping keys
  the dataclass doesn't declare (so phyai configs can ride along on
  upstream ``config.json`` files that carry many unrelated knobs);
* expose every declared field via mapping-style access
  (``cfg["hidden_size"]``, ``for k, v in cfg.items()``) so generic
  builders can inspect a config without knowing the concrete subclass;
* be hashable and immutable, so configs can travel through
  ``functools.lru_cache`` and graph-capture machinery.

Concrete subclasses just declare their fields with
``@dataclass(frozen=True)`` — no extra plumbing required.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterator, TypeVar


T = TypeVar("T", bound="PretrainedConfig")


@dataclass(frozen=True)
class PretrainedConfig:
    """Base for every model config in phyai.models.

    Subclass with ``@dataclass(frozen=True)``. Every field must declare
    a default (or rely on the subclass' own ``__init__``); the JSON
    loader filters the input dict to declared field names so unknown
    keys are dropped instead of raising.
    """

    @classmethod
    def field_names(cls) -> set[str]:
        return {f.name for f in fields(cls)}

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """Build an instance from a dict; unknown keys are silently dropped.

        This makes it safe to feed a full upstream ``config.json``
        (with optimizer / scheduler / device / dataset knobs) directly
        into a narrow phyai config: only the fields the subclass
        declares survive.
        """
        known = cls.field_names()
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    @classmethod
    def from_json(cls: type[T], path: str | Path) -> T:
        """Read JSON from ``path`` and dispatch through :meth:`from_dict`."""
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(
                f"{path}: expected a JSON object at the top level, got "
                f"{type(data).__name__}."
            )
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # ------------------------------------------------------------------ #
    # Mapping-like read access                                           #
    # ------------------------------------------------------------------ #

    def __getitem__(self, key: str) -> Any:
        if key not in self.field_names():
            raise KeyError(
                f"{type(self).__name__} has no field {key!r}; "
                f"valid fields: {sorted(self.field_names())!r}."
            )
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self.field_names()

    def keys(self) -> Iterator[str]:
        return iter(self.field_names())

    def items(self) -> Iterator[tuple[str, Any]]:
        return ((f.name, getattr(self, f.name)) for f in fields(self))


__all__ = ["PretrainedConfig"]
