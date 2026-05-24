"""Registry of linear kernels + selection Policy.

A kernel is registered once with an optional
``prefer_for={(spec_id, regime)}`` hint; the Registry orders candidates
by (preferred-for-this-probe, registration-order, capture-safe) and
``Policy.select`` returns one.

Concrete kernels in :mod:`phyai.layers.linear.backends` use the
:func:`register_linear_kernel` decorator to declare themselves at
import time; :func:`phyai.layers.linear.init` then materialises a fresh
:class:`LinearKernelRegistry` from those declarations.
"""

from __future__ import annotations

from typing import Callable, Protocol, TypeVar, runtime_checkable

import torch

from phyai.layers.linear.backend import KernelProbe, LinearKernel
from phyai.parallel.exceptions import NoBackendError
from phyai.parallel.state import Mode


_LinearKernelT = TypeVar("_LinearKernelT", bound=type[LinearKernel])


# Kernel declarations gathered at import time. Each entry is
# ``(cls, prefer_for)``; :func:`phyai.layers.linear.init` walks the list
# in declaration order to build a :class:`LinearKernelRegistry`.
_KERNEL_REGISTRATIONS: list[tuple[type, frozenset[tuple[str, str]]]] = []


def register_linear_kernel(
    *,
    prefer_for: set[tuple[str, str]] | None = None,
) -> Callable[[_LinearKernelT], _LinearKernelT]:
    """Class decorator: declare a :class:`LinearKernel` for registration.

    The class itself is recorded — it is *not* instantiated until
    :func:`phyai.layers.linear.init` runs, so importing a backend
    module is cheap and free of CUDA / driver side effects. Use the
    optional ``prefer_for`` keyword to declare ``(spec_id, regime)``
    tuples this kernel should win on (e.g. ``{("bf16", "prefill")}``);
    everything else falls through to registration order.
    """
    captured: frozenset[tuple[str, str]] = frozenset(prefer_for or ())

    def deco(cls: _LinearKernelT) -> _LinearKernelT:
        _KERNEL_REGISTRATIONS.append((cls, captured))
        return cls

    return deco


def list_registered_linear_kernels() -> list[tuple[type, frozenset[tuple[str, str]]]]:
    """Return the kernel declarations gathered so far, in import order."""
    return list(_KERNEL_REGISTRATIONS)


def _regime_of(M_bucket: int) -> str:
    """``M_bucket <= 3`` (M ≤ 4) is the decode regime; everything else is prefill."""
    return "decode" if M_bucket <= 3 else "prefill"


class LinearKernelRegistry:
    """Process-level registry of :class:`LinearKernel` instances."""

    def __init__(self) -> None:
        self._kernels: list[LinearKernel] = []
        self._prefer: dict[tuple[str, str], list[str]] = {}

    def register(
        self,
        kernel: LinearKernel,
        *,
        prefer_for: set[tuple[str, str]] | None = None,
    ) -> None:
        self._kernels.append(kernel)
        if prefer_for:
            for k in prefer_for:
                self._prefer.setdefault(k, []).append(kernel.name)

    def candidates(self, probe: KernelProbe) -> list[LinearKernel]:
        regime = _regime_of(probe.M_bucket)
        prefer_names = self._prefer.get((probe.spec_id, regime), [])

        out: list[LinearKernel] = []
        # 1. preferred first, in declaration order
        for name in prefer_names:
            for k in self._kernels:
                if k.name != name:
                    continue
                if probe.mode == Mode.GRAPH_CAPTURING and not k.supports_capture():
                    continue
                if k.can_handle(probe):
                    out.append(k)
                break

        # 2. remaining kernels in registration order
        for k in self._kernels:
            if k.name in prefer_names:
                continue
            if probe.mode == Mode.GRAPH_CAPTURING and not k.supports_capture():
                continue
            if k.can_handle(probe):
                out.append(k)
        return out

    def has(self, probe: KernelProbe) -> bool:
        return bool(self.candidates(probe))

    def all(self) -> list[LinearKernel]:
        return list(self._kernels)

    def validate(
        self,
        *,
        sample_specs: list[str],
        sm: int,
        N: int = 4096,
        K: int = 4096,
    ) -> None:
        """Sanity-check: every ``(spec_id, regime)`` pair has at least one
        eager candidate, and every preferred name actually maps to a
        registered kernel.

        ``Mode.GRAPH_CAPTURING`` coverage is only asserted when at least one
        registered kernel claims capture-safety — setups that deliberately
        run eager-only should still pass.
        """
        names = {b.name for b in self._kernels}
        for (spec_id, _regime), prefs in self._prefer.items():
            for n in prefs:
                if n not in names:
                    raise NoBackendError(
                        f"prefer_for[({spec_id!r}, {_regime!r})]={n!r} but no "
                        f"kernel with that name is registered "
                        f"(have: {sorted(names)})"
                    )

        has_capture_kernel = any(b.supports_capture() for b in self._kernels)
        modes: list[Mode] = [Mode.EAGER]
        if has_capture_kernel:
            modes.append(Mode.GRAPH_CAPTURING)

        # Probe one decode bucket (M=1) and one prefill bucket (M=512) per spec.
        for spec_id in sample_specs:
            for M_bucket in (1, 10):
                for mode in modes:
                    probe = KernelProbe(
                        spec_id=spec_id,
                        M_bucket=M_bucket,
                        N=N,
                        K=K,
                        in_dtype=torch.bfloat16,
                        out_dtype=torch.bfloat16,
                        sm=sm,
                        mode=mode,
                    )
                    if not self.has(probe):
                        raise NoBackendError(
                            f"no LinearKernel for spec={spec_id!r} "
                            f"regime={_regime_of(M_bucket)} mode={mode.value} "
                            f"sm={sm}"
                        )


@runtime_checkable
class Policy(Protocol):
    def select(self, candidates: list[LinearKernel]) -> LinearKernel: ...


class DefaultPolicy:
    """Pick the first candidate. Registry has already ordered them."""

    def select(self, candidates: list[LinearKernel]) -> LinearKernel:
        if not candidates:
            raise NoBackendError("no LinearKernel candidate")
        return candidates[0]


class ForcedPolicy:
    """Pick the kernel registered under ``name``.

    Sourced from
    :attr:`~phyai.engine_config.RuntimeConfig.force_linear_kernel`
    (which absorbs ``PHYAI_FORCE_LINEAR_KERNEL`` in
    :meth:`EngineConfig.from_env`); applied by
    :class:`~phyai.layers.linear.dispatch.KernelDispatcher` when set.
    """

    def __init__(
        self,
        name: str,
        fallback: Policy | None = None,
    ) -> None:
        self.name = name
        self.fallback = fallback if fallback is not None else DefaultPolicy()

    def select(self, candidates: list[LinearKernel]) -> LinearKernel:
        for k in candidates:
            if k.name == self.name:
                return k
        return self.fallback.select(candidates)
