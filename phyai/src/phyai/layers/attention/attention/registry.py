"""Registry for `phyai.layers.attention.attention` backends.

Module-level dict keyed by canonical name. Each concrete backend in
:mod:`phyai.layers.attention.attention.backends` registers itself at
import time via :func:`register_backend`. The dict is **private to
this subpackage**; AR and Diffusion stacks have their own independent
registries.

The registry stores **factories** — a factory takes an optional runner
reference and returns an :class:`AttentionBackend`:

    BackendFactory = Callable[[ModelRunner | None], AttentionBackend]

Receiving the runner lets the backend size buffers if it has any. For
no-cache backends the runner is typically unused and ``runner=None``
is the convenience path.

When you decorate a class with :func:`register_backend`, the class
itself is stored as the factory — so its ``__init__`` MUST accept a
single positional ``runner`` argument (default ``None``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, TypeVar

from phyai.layers.attention.attention.base import AttentionBackend


if TYPE_CHECKING:
    from phyai.runtime.model_runner import ModelRunner


BackendFactory = Callable[["ModelRunner | None"], AttentionBackend]
_FactoryT = TypeVar("_FactoryT", bound=BackendFactory)


_BACKENDS: dict[str, BackendFactory] = {}


def _canonical(name: str) -> str:
    return name.lower().replace("_", "-")


def register_backend(name: str) -> Callable[[_FactoryT], _FactoryT]:
    """Decorator: register a backend factory under ``name``."""
    canonical = _canonical(name)

    def deco(factory: _FactoryT) -> _FactoryT:
        if isinstance(factory, type) and issubclass(factory, AttentionBackend):
            factory.name = canonical
        if canonical in _BACKENDS:
            raise ValueError(
                f"@register_backend({name!r}): already registered in attention/."
            )
        _BACKENDS[canonical] = factory
        return factory

    return deco


def get_backend_factory(name: str) -> BackendFactory:
    """Look up a backend factory by name."""
    canonical = _canonical(name)
    if canonical not in _BACKENDS:
        raise ValueError(
            f"Attention backend {name!r} is not registered. Available: "
            f"{list_backends()}"
        )
    return _BACKENDS[canonical]


def list_backends() -> list[str]:
    """Return registered backend names."""
    return sorted(_BACKENDS)


__all__ = [
    "BackendFactory",
    "get_backend_factory",
    "list_backends",
    "register_backend",
]
