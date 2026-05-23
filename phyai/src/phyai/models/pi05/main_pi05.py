"""pi0.5 plugin entry — the engine's pi0.5 hook.

Two pieces, both consumed by :class:`~phyai.engine.Engine`:

* :class:`PI05Args` — typed arg bundle. Carries the safetensors weight
  paths, an optional :class:`PI05Config` (defaults to ``pi05_base``),
  and the model-specific scheduler knob ``max_batch_size``.
* :class:`PI05Entry` — :class:`~phyai.engine.Entry` subclass that
  builds a :class:`PI05Model`, runs :func:`load_pretrained`,
  constructs and warms a :class:`PI05WS1Scheduler`, then
  forwards :meth:`step` to it.

Importing this module registers ``PI05Entry`` with the engine via
``@Engine.register`` at class-definition time. The engine's own
``engine.py`` imports this module at the bottom of its file, so the
plugin is available the moment the engine module is loaded.

``device`` / ``params_dtype`` / ``*_backend`` / ``use_cuda_graph`` are
*not* fields on :class:`PI05Args`; they live on
:class:`~phyai.engine.EngineArgs` and are propagated to
:class:`PI05Model` and the scheduler via the
:class:`~phyai.engine_config.EngineConfig` singleton (which the
engine seeds in its ``__init__``). Adding them here would just create
a second source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, ClassVar

import torch

from phyai.engine import Engine, Entry, EntryArgs
from phyai.engine_config import get_engine_config
from phyai.models.pi05.configuration_pi05 import PI05Config
from phyai.models.pi05.modeling_pi05 import PI05Model
from phyai.models.pi05.scheduler_ws1_pi05 import (
    PI05Request,
    PI05WS1Scheduler,
)
from phyai.weights import load_pretrained


# Keys present in the upstream pi0.5 base safetensors that the inference
# model never consumes. The expert was trained with a lm_head sibling to
# the language model's, but at inference the expert produces flow-matching
# vectors (not tokens), so its lm_head weight has no parameter to land in.
# Dropping it silently keeps `weight_strict=True` honest for everything else.
_PI05_DROP_KEYS: frozenset[str] = frozenset(
    {"paligemma_with_expert.gemma_expert.lm_head.weight"}
)


def _compose_remap(
    user_remap: Callable[[str], str | None] | dict[str, str] | None,
) -> Callable[[str], str | None]:
    """Combine the pi0.5-specific drop set with the caller's remap.

    The drop set runs first; any key it returns ``None`` for is removed
    before the user remap sees it.
    """
    if user_remap is None:
        return lambda k: None if k in _PI05_DROP_KEYS else k
    if callable(user_remap):

        def _chained(k: str) -> str | None:
            if k in _PI05_DROP_KEYS:
                return None
            return user_remap(k)

        return _chained
    if isinstance(user_remap, dict):
        rules = list(user_remap.items())

        def _chained_dict(k: str) -> str | None:
            if k in _PI05_DROP_KEYS:
                return None
            for src, dst in rules:
                if src in k:
                    k = k.replace(src, dst)
            return k

        return _chained_dict
    raise TypeError(
        f"weight_remap must be callable, dict, or None; got {type(user_remap).__name__}"
    )


@dataclass
class PI05Args(EntryArgs):
    """Args bundle for the pi0.5 plugin.

    ``weights_paths`` is empty by default for unit-test / debug paths —
    :meth:`PI05Entry.setup` then constructs the model with random init
    and skips :func:`load_pretrained`. Production callers always supply
    at least one safetensors path.

    ``config`` defaults to the public ``pi05_base`` geometry; override
    with ``PI05Config.from_json(...)`` (or a hand-built
    :class:`PI05Config`) when running a non-base variant.

    ``weight_remap`` and ``weight_strict`` pass straight through to
    :func:`load_pretrained` for checkpoints whose key names diverge
    from upstream pi0.5 (HF rewrites, mid-training renames, etc.).
    """

    weights_paths: list[str | Path] = field(default_factory=list)
    config: PI05Config = field(default_factory=PI05Config)
    max_batch_size: int = 1
    weight_remap: Callable[[str], str | None] | dict[str, str] | None = None
    weight_strict: bool = True


@Engine.register
class PI05Entry(Entry):
    """pi0.5 inference plugin entry."""

    name: ClassVar[str] = "pi05"
    args_cls: ClassVar[type[EntryArgs]] = PI05Args

    def __init__(self) -> None:
        # Default-init the slots so :meth:`step` / :meth:`close` can
        # check for "setup not yet run" without an attr-exists guard.
        self.model: PI05Model | None = None
        self.scheduler: PI05WS1Scheduler | None = None

    def setup(self, args: PI05Args) -> None:  # type: ignore[override]
        """Build model, load weights, construct + warm the scheduler."""
        eng = get_engine_config()
        self.model = PI05Model(args.config, device=eng.device.target)

        if args.weights_paths:
            load_pretrained(
                self.model,
                args.weights_paths,
                remap=_compose_remap(args.weight_remap),
                strict=args.weight_strict,
            )

        self.scheduler = PI05WS1Scheduler(
            self.model,
            max_batch_size=args.max_batch_size,
            device=eng.device.target,
            use_cuda_graph=eng.runtime.use_cuda_graph,
        )
        self.scheduler.setup()

    def step(self, request: PI05Request) -> torch.Tensor:  # type: ignore[override]
        """Run one pi0.5 inference; return the action chunk ``(B, chunk, action_dim)``."""
        if self.scheduler is None:
            raise RuntimeError(
                "PI05Entry.step called before setup; the scheduler is None."
            )
        return self.scheduler.step(request)

    def close(self) -> None:
        if self.scheduler is not None:
            self.scheduler.close()
            self.scheduler = None
        self.model = None


__all__ = ["PI05Args", "PI05Entry"]
