"""Registry for `phyai.layers.attention.ar` backends.

Module-level dict keyed by canonical name. Each concrete AR backend
in :mod:`phyai.layers.attention.ar.backends` registers itself at
import time. The dict is **private to this subpackage**; the
no-cache and Diffusion stacks have their own independent registries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, TypeVar

from phyai.layers.attention.ar.base import ARAttentionBackend


if TYPE_CHECKING:
    from phyai.runtime.model_runner import ModelRunner


BackendFactory = Callable[["ModelRunner | None"], ARAttentionBackend]
_FactoryT = TypeVar("_FactoryT", bound=BackendFactory)


_BACKENDS: dict[str, BackendFactory] = {}


def _canonical(name: str) -> str:
    return name.lower().replace("_", "-")


def register_backend(name: str) -> Callable[[_FactoryT], _FactoryT]:
    """Decorator: register an AR backend factory under ``name``."""
    canonical = _canonical(name)

    def deco(factory: _FactoryT) -> _FactoryT:
        if isinstance(factory, type) and issubclass(factory, ARAttentionBackend):
            factory.name = canonical
        if canonical in _BACKENDS:
            raise ValueError(f"@register_backend({name!r}): already registered in ar/.")
        _BACKENDS[canonical] = factory
        return factory

    return deco


def get_backend_factory(name: str) -> BackendFactory:
    """Look up an AR backend factory by name."""
    canonical = _canonical(name)
    if canonical not in _BACKENDS:
        raise ValueError(
            f"AR attention backend {name!r} is not registered. Available: "
            f"{list_backends()}"
        )
    return _BACKENDS[canonical]


def list_backends() -> list[str]:
    """Return registered AR backend names."""
    return sorted(_BACKENDS)


__all__ = [
    "BackendFactory",
    "get_backend_factory",
    "list_backends",
    "register_backend",
]
