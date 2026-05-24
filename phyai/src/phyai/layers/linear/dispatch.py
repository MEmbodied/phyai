"""KernelDispatcher with hashmap cache.

The cache key includes an ``M_bucket`` so decode (M≤8) and prefill
(M≥1024) naturally land on different kernels when ``prefer_for`` is
declared.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from phyai.engine_config import get_engine_config
from phyai.layers.linear.backend import KernelProbe, LinearKernel
from phyai.layers.linear.registry import (
    DefaultPolicy,
    ForcedPolicy,
    LinearKernelRegistry,
    Policy,
)
from phyai.parallel.exceptions import NoBackendError
from phyai.parallel.state import current_mode
from phyai.utils.cuda import sm_arch


def _M_bucket(M: int) -> int:
    """Logarithmic bucket over token count. ``bit_length`` gives 1,2,4,8,…"""
    return M.bit_length() if M > 0 else 0


class KernelDispatcher:
    """Process-level dispatcher for linear matmul kernels."""

    def __init__(
        self,
        registry: LinearKernelRegistry,
        policy: Policy | None = None,
    ) -> None:
        self.registry = registry
        # Env override takes precedence over the engine-config singleton —
        # the singleton is populated once at first read and otherwise
        # ignores subsequent ``PHYAI_FORCE_LINEAR_KERNEL`` changes
        # (e.g. mid-test ``monkeypatch.setenv``). Falling back to the
        # engine config preserves the centralised-config intent for
        # callers that set ``force_linear_kernel`` programmatically.
        forced = (
            os.environ.get("PHYAI_FORCE_LINEAR_KERNEL")
            or get_engine_config().runtime.force_linear_kernel
        )
        if policy is not None:
            self.policy = policy
        elif forced:
            self.policy = ForcedPolicy(forced)
        else:
            self.policy = DefaultPolicy()
        self._cache: dict[tuple[Any, ...], LinearKernel] = {}
        self._sm = sm_arch()

    def select(
        self,
        *,
        spec_id: str,
        M: int,
        N: int,
        K: int,
        in_dtype: torch.dtype,
        out_dtype: torch.dtype,
    ) -> LinearKernel:
        mode = current_mode()
        Mb = _M_bucket(M)
        key = (spec_id, Mb, N, K, in_dtype, out_dtype, self._sm, mode)
        k = self._cache.get(key)
        if k is None:
            probe = KernelProbe(
                spec_id=spec_id,
                M_bucket=Mb,
                N=N,
                K=K,
                in_dtype=in_dtype,
                out_dtype=out_dtype,
                sm=self._sm,
                mode=mode,
            )
            cands = self.registry.candidates(probe)
            if not cands:
                raise NoBackendError(
                    f"no LinearKernel for spec={spec_id} M_bucket={Mb} "
                    f"N={N} K={K} in_dtype={in_dtype} out_dtype={out_dtype} "
                    f"mode={mode.value} sm={self._sm}"
                )
            k = self.policy.select(cands)
            self._cache[key] = k
        return k

    def clear_cache(self) -> None:
        self._cache.clear()


# Process-level singleton; populated in :func:`phyai.layers.linear.init`.
_dispatcher: KernelDispatcher | None = None


def get_linear_dispatcher() -> KernelDispatcher:
    if _dispatcher is None:
        raise RuntimeError("phyai.layers.linear.init(...) not called yet")
    return _dispatcher


def _set_linear_dispatcher(d: KernelDispatcher | None) -> None:
    global _dispatcher
    _dispatcher = d
