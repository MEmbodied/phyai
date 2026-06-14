"""Composable data-processing pipeline — the unified pre/post-process core.

A self-contained port of the lerobot processor framework (``ProcessorStep`` ABC
+ a string registry + a ``ProcessorPipeline`` that chains steps over a canonical
transition dict), including the **lerobot-format serialization round-trip** so a
pipeline can load / save HuggingFace ``policy_*processor.json`` checkpoints. See
:mod:`phyai_utils_tools.processing.base_processor` for the per-model ABC and
:mod:`phyai_utils_tools.models.pi05.processor_pi05` for the first concrete
subclass.

Design:

* :class:`ProcessorStep` — one transform. Implements ``__call__(transition)``;
  optionally overrides ``get_config()`` (JSON hyperparams) and
  ``state_dict()`` / ``load_state_dict()`` (tensor state persisted to a sidecar
  ``.safetensors``, e.g. the normalizer's dataset stats).
* :class:`ProcessorStepRegistry` — maps a string name to a step class so a
  pipeline can be rebuilt from a config json without hardcoding imports. This is
  exactly the mechanism lerobot uses for ``policy_{pre,post}processor.json``.
* :class:`ProcessorPipeline` — an ordered list of steps plus pluggable
  ``to_transition`` / ``to_output`` adapters, with
  :meth:`~ProcessorPipeline.from_pretrained` /
  :meth:`~ProcessorPipeline.save_pretrained` for the lerobot round-trip.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from phyai_utils_tools.processing.transition import Transition, identity_adapter


class ProcessorStepError(Exception):
    """Raised when a step in a config json can't be resolved or instantiated."""


# --------------------------------------------------------------------------- #
# Registry                                                                    #
# --------------------------------------------------------------------------- #


class ProcessorStepRegistry:
    """Maps a string name to a :class:`ProcessorStep` subclass.

    Lets a pipeline be described by ``(name, config)`` pairs and rebuilt
    without hardcoding imports — the same mechanism lerobot uses to load a
    ``policy_{pre,post}processor.json``. Registration stamps
    ``_registry_name`` on the class for the reverse (serialization) direction.
    """

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str | None = None) -> Callable[[type], type]:
        """Class decorator registering a step under ``name`` (or its class name)."""

        def decorator(step_class: type) -> type:
            registration_name = name if name is not None else step_class.__name__
            if registration_name in cls._registry:
                raise ValueError(
                    f"Processor step '{registration_name}' is already registered. "
                    f"Use a different name or unregister the existing one first."
                )
            cls._registry[registration_name] = step_class
            step_class._registry_name = registration_name
            return step_class

        return decorator

    @classmethod
    def get(cls, name: str) -> type:
        """Return the step class registered under ``name``."""
        if name not in cls._registry:
            raise KeyError(
                f"Unknown processor step {name!r}; registered: {sorted(cls._registry)}."
            )
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        """Return every registered step name."""
        return sorted(cls._registry)

    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove ``name`` from the registry (mainly for tests)."""
        cls._registry.pop(name, None)


# --------------------------------------------------------------------------- #
# Step ABC                                                                    #
# --------------------------------------------------------------------------- #


class ProcessorStep(ABC):
    """One step in a :class:`ProcessorPipeline`.

    A step reads the canonical :data:`~phyai_utils_tools.processing.transition.Transition`
    dict, transforms one or more of its entries, and returns the (updated)
    dict. Steps should treat the transition as owned by the pipeline and may
    mutate-and-return or copy-and-return; the bundled steps copy the top-level
    dict so a step never aliases the caller's input.

    Implement :meth:`__call__`. Override :meth:`get_config` if the step has
    JSON-able hyperparameters worth surfacing for the config json. Override
    :meth:`state_dict` / :meth:`load_state_dict` if the step holds tensor state
    (e.g. dataset normalization stats) that should persist to a sidecar
    ``.safetensors`` — only the normalizer steps do.
    """

    @abstractmethod
    def __call__(self, transition: Transition) -> Transition:
        """Transform and return the transition."""
        raise NotImplementedError

    def get_config(self) -> dict[str, Any]:
        """Return JSON-serializable hyperparameters (default: none)."""
        return {}

    def state_dict(self) -> dict[str, Any]:
        """Return tensor state to persist alongside the config (default: none).

        Non-empty only for stateful steps (the normalizer's stat tensors). The
        pipeline writes this to a sidecar ``.safetensors`` on save and feeds it
        back via :meth:`load_state_dict` on load.
        """
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load tensor state produced by :meth:`state_dict` (default: no-op)."""
        return None

    def reset(self) -> None:
        """Reset any per-episode state (default: no-op)."""
        return None


# --------------------------------------------------------------------------- #
# Pipeline                                                                     #
# --------------------------------------------------------------------------- #


@dataclass
class ProcessorPipeline:
    """An ordered chain of :class:`ProcessorStep` over a transition dict.

    ``to_transition`` adapts the raw input into the canonical transition the
    steps operate on; ``to_output`` adapts the final transition into whatever
    the caller wants back. Both default to identity, so a pipeline can also be
    driven directly with a transition dict. This adapter pair is the single
    point that makes the step list reusable across models with different raw
    input/output shapes.
    """

    steps: Sequence[ProcessorStep] = field(default_factory=list)
    name: str = "ProcessorPipeline"
    to_transition: Callable[[Any], Transition] = field(
        default=identity_adapter, repr=False
    )
    to_output: Callable[[Transition], Any] = field(default=identity_adapter, repr=False)

    def __call__(self, data: Any) -> Any:
        """Run ``data`` through every step in order, with the I/O adapters."""
        transition = self.to_transition(data)
        for step in self.steps:
            transition = step(transition)
        return self.to_output(transition)

    def step_through(self, data: Any) -> Iterable[Transition]:
        """Yield the transition after each step (debugging aid)."""
        transition = self.to_transition(data)
        yield transition
        for step in self.steps:
            transition = step(transition)
            yield transition

    def get_config(self) -> dict[str, Any]:
        """Describe the pipeline as ``{name, steps:[{registry_name, config}]}``."""
        return {
            "name": self.name,
            "steps": [
                {
                    "registry_name": getattr(
                        type(step), "_registry_name", type(step).__name__
                    ),
                    "config": step.get_config(),
                }
                for step in self.steps
            ],
        }

    # -- serialization (lerobot-format round-trip) ---------------------- #

    @classmethod
    def from_pretrained(
        cls,
        src: str | Path,
        config_filename: str,
        *,
        name: str | None = None,
        overrides: dict[str, dict[str, Any]] | None = None,
        step_kwargs: dict[str, dict[str, Any]] | None = None,
        to_transition: Callable[[Any], Transition] | None = None,
        to_output: Callable[[Transition], Any] | None = None,
        **hub_kwargs: Any,
    ) -> ProcessorPipeline:
        """Rebuild a pipeline from a lerobot-format ``config_filename`` json.

        ``src`` is a local directory (reads ``src/config_filename``), a local
        file (read directly), or an HF repo id (``hf_hub_download``). Each step
        entry ``{registry_name, config, state_file?}`` is resolved through
        :class:`ProcessorStepRegistry` and instantiated with its saved config
        merged under ``overrides[name]`` then ``step_kwargs[name]`` — the latter
        injects non-serialized runtime objects (e.g. a tokenizer instance, the
        target device) the json can't carry. A ``state_file`` sidecar is loaded
        via safetensors into the step's :meth:`~ProcessorStep.load_state_dict`.

        An unknown ``registry_name`` raises :class:`ProcessorStepError` (never
        silently skipped). ``to_transition`` / ``to_output`` set the rebuilt
        pipeline's adapters (the json does not carry them).
        """
        overrides = overrides or {}
        step_kwargs = step_kwargs or {}
        config, base_path = cls._load_config(src, config_filename, hub_kwargs)

        steps: list[ProcessorStep] = []
        for entry in config.get("steps", []):
            steps.append(
                cls._build_step(
                    entry, overrides, step_kwargs, src, base_path, hub_kwargs
                )
            )

        return cls(
            steps=steps,
            name=name or config.get("name", "ProcessorPipeline"),
            to_transition=to_transition
            if to_transition is not None
            else identity_adapter,
            to_output=to_output if to_output is not None else identity_adapter,
        )

    @staticmethod
    def _load_config(
        src: str | Path,
        config_filename: str,
        hub_kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], Path]:
        """Locate + read the config json; return ``(config, base_path)``.

        ``base_path`` is the directory used to resolve sidecar ``state_file``s.
        """
        path = Path(src)
        if path.is_dir():
            config_path = path / config_filename
            if not config_path.is_file():
                raise FileNotFoundError(
                    f"{config_filename} not found in directory {path}."
                )
            base_path = path
        elif path.is_file():
            config_path = path
            base_path = path.parent
        else:
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(
                repo_id=str(src),
                filename=config_filename,
                repo_type="model",
                **hub_kwargs,
            )
            config_path = Path(downloaded)
            base_path = config_path.parent
        with open(config_path) as fp:
            return json.load(fp), base_path

    @classmethod
    def _build_step(
        cls,
        entry: dict[str, Any],
        overrides: dict[str, dict[str, Any]],
        step_kwargs: dict[str, dict[str, Any]],
        src: str | Path,
        base_path: Path,
        hub_kwargs: dict[str, Any],
    ) -> ProcessorStep:
        """Resolve, instantiate, and state-load one step entry."""
        registry_name = entry.get("registry_name")
        if registry_name is None:
            raise ProcessorStepError(
                f"Step entry has no 'registry_name' (class-path steps are not "
                f"supported): {entry!r}."
            )
        try:
            step_cls = ProcessorStepRegistry.get(registry_name)
        except KeyError as exc:
            raise ProcessorStepError(
                f"Unknown processor step {registry_name!r} in config. phyai "
                f"implements: {ProcessorStepRegistry.list()}. Implement + "
                f"register it, or remove it from the checkpoint."
            ) from exc

        merged = {
            **entry.get("config", {}),
            **overrides.get(registry_name, {}),
            **step_kwargs.get(registry_name, {}),
        }
        try:
            step = step_cls(**merged)
        except Exception as exc:
            raise ProcessorStepError(
                f"Failed to instantiate step {registry_name!r} with config "
                f"{merged!r}: {exc}"
            ) from exc

        state_file = entry.get("state_file")
        if state_file:
            from safetensors.torch import load_file

            local = base_path / state_file
            if local.is_file():
                state_path = str(local)
            elif Path(src).exists():
                raise FileNotFoundError(
                    f"Step {registry_name!r} references state_file "
                    f"{state_file!r}, but it was not found at {local}."
                )
            else:
                from huggingface_hub import hf_hub_download

                state_path = hf_hub_download(
                    repo_id=str(src),
                    filename=state_file,
                    repo_type="model",
                    **hub_kwargs,
                )
            step.load_state_dict(load_file(state_path))
        return step

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        config_filename: str | None = None,
    ) -> Path:
        """Write the pipeline as a lerobot-format json (+ stats sidecars).

        Emits ``{name, steps:[{registry_name, config, state_file?}]}`` (indent
        2). Any step whose :meth:`~ProcessorStep.state_dict` is non-empty gets a
        ``{sanitized_name}_step_{i}_{registry_name}.safetensors`` sidecar and a
        ``state_file`` pointer. ``config_filename`` defaults to
        ``{sanitized_name}.json``. Returns the json path.
        """
        from safetensors.torch import save_file

        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", self.name.lower())
        if config_filename is None:
            config_filename = f"{sanitized}.json"

        config: dict[str, Any] = {"name": self.name, "steps": []}
        for i, step in enumerate(self.steps):
            registry_name = getattr(type(step), "_registry_name", type(step).__name__)
            entry: dict[str, Any] = {
                "registry_name": registry_name,
                "config": step.get_config(),
            }
            state = step.state_dict()
            if state:
                state_filename = f"{sanitized}_step_{i}_{registry_name}.safetensors"
                save_file(
                    {k: v.clone() for k, v in state.items()},
                    os.path.join(str(save_dir), state_filename),
                )
                entry["state_file"] = state_filename
            config["steps"].append(entry)

        json_path = save_dir / config_filename
        with open(json_path, "w") as fp:
            json.dump(config, fp, indent=2)
        return json_path

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, idx: int) -> ProcessorStep:
        return self.steps[idx]


__all__ = [
    "ProcessorPipeline",
    "ProcessorStep",
    "ProcessorStepError",
    "ProcessorStepRegistry",
]
