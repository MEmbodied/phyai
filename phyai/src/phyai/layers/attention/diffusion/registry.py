"""Registry for `phyai.layers.attention.diffusion` backends.

Module-level dict keyed by canonical name. Independent of the AR and
no-cache registries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, TypeVar

from phyai.layers.attention.diffusion.base import DiffusionAttentionBackend


if TYPE_CHECKING:
    from phyai.runtime.model_runner import ModelRunner


BackendFactory = Callable[["ModelRunner | None"], DiffusionAttentionBackend]
_FactoryT = TypeVar("_FactoryT", bound=BackendFactory)


_BACKENDS: dict[str, BackendFactory] = {}


def _canonical(name: str) -> str:
    return name.lower().replace("_", "-")


def register_backend(name: str) -> Callable[[_FactoryT], _FactoryT]:
    """Decorator: register a diffusion backend factory under ``name``."""
    canonical = _canonical(name)

    def deco(factory: _FactoryT) -> _FactoryT:
        if isinstance(factory, type) and issubclass(factory, DiffusionAttentionBackend):
            factory.name = canonical
        if canonical in _BACKENDS:
            raise ValueError(
                f"@register_backend({name!r}): already registered in diffusion/."
            )
        _BACKENDS[canonical] = factory
        return factory

    return deco


def get_backend_factory(name: str) -> BackendFactory:
    """Look up a diffusion backend factory by name."""
    canonical = _canonical(name)
    if canonical not in _BACKENDS:
        raise ValueError(
            f"Diffusion attention backend {name!r} is not registered. Available: "
            f"{list_backends()}"
        )
    return _BACKENDS[canonical]


def list_backends() -> list[str]:
    """Return registered diffusion backend names."""
    return sorted(_BACKENDS)


__all__ = [
    "BackendFactory",
    "get_backend_factory",
    "list_backends",
    "register_backend",
]
