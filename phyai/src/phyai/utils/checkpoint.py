"""HuggingFace-style checkpoint folder helpers.

A "checkpoint" in phyai follows the same on-disk layout as a
``transformers`` snapshot: one folder containing

* ``config.json`` — the model's architecture/geometry,
* one or more ``*.safetensors`` shards, optionally indexed by
  ``model.safetensors.index.json``.

These helpers resolve a folder into the (config object, shard paths)
pair every plugin needs:

* :func:`find_safetensors` — list the safetensors shards in a folder,
  honouring ``model.safetensors.index.json`` when present.
* :func:`load_config` — parse ``config.json`` into a
  :class:`~phyai.models.configuration.PretrainedConfig` subclass.
* :func:`load_checkpoint` — convenience: do both in one call.

Plugins (:class:`phyai.engine.Entry`) and ``examples/`` consume these
instead of hand-rolling per-model directory parsing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar


if TYPE_CHECKING:
    from phyai.models.configuration import PretrainedConfig


T = TypeVar("T", bound="PretrainedConfig")


_SAFETENSORS_SINGLE = "model.safetensors"
_SAFETENSORS_INDEX = "model.safetensors.index.json"
_DEFAULT_CONFIG_FILENAME = "config.json"


def _ensure_dir(folder: Path) -> Path:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"checkpoint folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(
            f"expected a checkpoint folder, got a file path: {folder}. "
            f"Pass the folder containing config.json + safetensors shard(s)."
        )
    return folder


def find_safetensors(folder: str | Path) -> list[Path]:
    """List safetensors shard files in ``folder``.

    Resolution order, mirroring HuggingFace's snapshot layout:

    1. ``model.safetensors.index.json`` exists → parse its
       ``weight_map`` and return every distinct shard it references,
       sorted by filename.
    2. ``model.safetensors`` exists → return ``[folder/model.safetensors]``.
    3. Otherwise fall back to a glob of ``*.safetensors`` (catches
       non-canonical filenames). Raises if none are found.

    Returned paths are absolute. Files referenced by the index but
    missing on disk raise :class:`FileNotFoundError`.
    """
    folder = _ensure_dir(folder)

    index_path = folder / _SAFETENSORS_INDEX
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(
                f"{index_path}: missing or empty 'weight_map' — cannot resolve shards."
            )
        shards = sorted({str(v) for v in weight_map.values()})
        resolved = [(folder / shard).resolve() for shard in shards]
        for path in resolved:
            if not path.is_file():
                raise FileNotFoundError(
                    f"{index_path} references {path.name} but the file is "
                    f"missing from {folder}."
                )
        return resolved

    single = folder / _SAFETENSORS_SINGLE
    if single.is_file():
        return [single.resolve()]

    fallback = sorted(folder.glob("*.safetensors"))
    if fallback:
        return [p.resolve() for p in fallback]

    raise FileNotFoundError(
        f"no safetensors shards found in {folder}: expected one of "
        f"{_SAFETENSORS_INDEX!r}, {_SAFETENSORS_SINGLE!r}, or any '*.safetensors'."
    )


def load_config(
    folder: str | Path,
    config_cls: type[T],
    *,
    filename: str = _DEFAULT_CONFIG_FILENAME,
) -> T:
    """Read ``filename`` from ``folder`` and parse it via ``config_cls.from_json``.

    ``config_cls`` must be a :class:`~phyai.models.configuration.PretrainedConfig`
    subclass. Unknown JSON keys are silently dropped — see
    :meth:`PretrainedConfig.from_dict`.
    """
    folder = _ensure_dir(folder)
    config_path = folder / filename
    if not config_path.is_file():
        raise FileNotFoundError(
            f"config file not found: {config_path}. "
            f"Pass a checkpoint folder that contains '{filename}'."
        )
    return config_cls.from_json(config_path)


def load_checkpoint(
    folder: str | Path,
    config_cls: type[T],
    *,
    config_filename: str = _DEFAULT_CONFIG_FILENAME,
) -> tuple[T, list[Path]]:
    """One-shot folder resolver: ``(config, [shard_paths])``.

    Equivalent to calling :func:`load_config` and :func:`find_safetensors`
    on the same folder; provided for the common case where both pieces
    are needed at the same call site.
    """
    folder = _ensure_dir(folder)
    config = load_config(folder, config_cls, filename=config_filename)
    shards = find_safetensors(folder)
    return config, shards


__all__ = ["find_safetensors", "load_checkpoint", "load_config"]
