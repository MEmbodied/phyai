"""Forward-hook tensor dumping for whole-model activation capture.

Register a forward hook on every *leaf* operator of one or more model
modules, and during inference record every intermediate tensor the
forward pass produces. Each pass (one :meth:`phyai.engine.Engine.step`)
writes a single ``.pt`` file whose keys are the operator names (the dotted
path from ``named_modules``, e.g. ``model.paligemma_lm.layers.0.o_proj``)
and whose values are the modules' outputs, moved to CPU.

Eager-only by design
---------------------
A captured CUDA graph replays kernels without re-entering Python, so
``register_forward_hook`` callbacks never fire during
:meth:`phyai.runtime.cuda_graph_manager.CudaGraph.replay`. Activation
capture therefore only works when the runners run **eagerly**. The engine
forces ``use_cuda_graph=False`` whenever a dump directory is configured.
Do not expect this module to capture anything while graphs are live.

Why leaf-only
-------------
Hooking only modules with no child modules avoids recording the same
tensor twice: a parent like ``self_attn`` returns what its last child
``o_proj`` already produced. Hooking leaves keeps the dump compact and
each key unambiguous.

Selecting what to dump
----------------------
``filter`` decides which leaf operators are recorded, matched against each
operator's full dotted name (``model.expert_stack.layers.0.o_proj``):

* ``None`` — record every leaf (the default).
* a sequence of regex strings — record a leaf if **any** pattern
  ``re.search``-matches its name (an OR / union). This is the right tool
  for a model with several heterogeneous stacks: ``r"expert_stack\\.layers\\.0\\."``
  isolates one stack's first layer, ``r"o_proj$"`` grabs every output
  projection, ``r"\\.heads\\."`` reaches a component that has no
  ``layers.<int>`` index at all.
* a callable ``(name: str, module: nn.Module) -> bool`` — record a leaf
  when it returns True. The escape hatch for logic a regex can't express
  (e.g. dispatch on ``isinstance(module, ...)``, or "every ``o_proj``
  except the vision tower's"). :func:`load_filter_fn` resolves a
  ``"pkg.module:func"`` / ``"/path/to/file.py:func"`` string to such a
  callable so the predicate can live outside the caller's process setup.

Repeat fires within one pass
----------------------------
phyai decomposes inference into several runners, so a single
:meth:`Engine.step` may invoke one module many times — the vision tower
runs once per camera-stack, and the action-expert stack runs once per
Euler denoise step. Each invocation is preserved: the first fire is keyed
by the bare module name, subsequent fires get a ``"::callN"`` suffix
(``name::call1``, ``name::call2``, ...), so nothing is silently
overwritten.

Usage
-----
::

    from phyai.runtime.tensor_dump import register_tensor_dumper, load_pass

    dumper = register_tensor_dumper(
        {"model": model},
        dump_dir="/tmp/dump",
        filter=[r"expert_stack\\.layers\\.0\\.", r"\\.heads\\."],
    )
    # ... run engine.step() ...
    dumper.flush_pass()          # writes rank0_pid1234/pass00000.pt
    dumper.detach()              # remove hooks

    tensors = load_pass("/tmp/dump/rank0_pid1234/pass00000.pt")
    print(tensors["model.expert_stack.layers.0.o_proj"].shape)
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, Union

import torch
import torch.distributed as dist
from torch import nn

from phyai.utils import all_ranks_log, this_rank_log


logger = logging.getLogger(__name__)

# A leaf-operator predicate over (dotted_name, module). ``filter`` accepts
# this directly, or a sequence of regex strings compiled into one, or None
# (record everything). See :func:`_compile_filter`.
FilterFn = Callable[[str, nn.Module], bool]
FilterSpec = Union[None, Sequence[str], FilterFn]


def _resolve_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _resolve_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def _cpu_tensors(value: Any) -> Any:
    """Recursively detach + move tensors in ``value`` to CPU.

    Tensors are detached and ``.cpu()``-copied so dumped values neither
    pin GPU memory nor keep the autograd graph alive. ``tuple`` / ``list``
    / ``dict`` containers are walked and rebuilt; any leaf that is not a
    tensor is dropped (containers that yield no tensor collapse to
    ``None``). Returns ``None`` when nothing tensor-like is found, so the
    caller can skip the entry entirely.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu")
    if isinstance(value, (tuple, list)):
        items = [_cpu_tensors(v) for v in value]
        items = [it for it in items if it is not None]
        if not items:
            return None
        return items[0] if len(items) == 1 else items
    if isinstance(value, Mapping):
        out = {k: _cpu_tensors(v) for k, v in value.items()}
        out = {k: v for k, v in out.items() if v is not None}
        return out or None
    return None


def _compile_filter(spec: FilterSpec) -> FilterFn:
    """Normalise a :data:`FilterSpec` into one ``(name, module) -> bool`` predicate.

    * ``None`` -> a predicate that is always True (record everything).
    * a callable -> returned as-is (it already has the right shape).
    * a sequence of regex strings -> compiled once; the predicate returns
      True when **any** pattern ``re.search``-matches the operator name (an
      OR / union). An empty sequence matches nothing — a loud no-op rather
      than a silent "match everything". A pattern that fails to compile
      raises :class:`re.error` here, at construction, naming the offender.

    A bare ``str`` is rejected: a lone regex is almost always a mistake
    (the caller meant ``[pattern]``), and accepting it would silently
    iterate the string character-by-character.
    """
    if spec is None:
        return lambda _name, _module: True
    if callable(spec):
        return spec
    if isinstance(spec, str):
        raise TypeError(
            f"filter must be None, a sequence of regex strings, or a callable; "
            f"got a bare str {spec!r}. Wrap a single pattern in a list: [{spec!r}]."
        )
    patterns: list[re.Pattern] = []
    for p in spec:
        try:
            patterns.append(re.compile(p))
        except re.error as e:
            raise re.error(f"invalid filter regex {p!r}: {e}") from e

    def _match(name: str, _module: nn.Module) -> bool:
        return any(pat.search(name) for pat in patterns)

    return _match


def load_filter_fn(path: str) -> FilterFn:
    """Resolve a ``"<module-or-file>:<func>"`` string to a filter callable.

    Two address forms, distinguished by whether the left side ends in
    ``.py``:

    * import path — ``"my_pkg.filters:only_expert"`` imports the module and
      grabs the attribute.
    * file path — ``"/tmp/mydump.py:filter"`` loads the file directly
      (handy for ad-hoc debugging without installing anything).

    The resolved object must be callable; it is used as
    ``(name: str, module: nn.Module) -> bool``.
    """
    module_part, sep, attr = path.rpartition(":")
    if not sep or not module_part or not attr:
        raise ValueError(
            f"filter_fn must look like 'pkg.module:func' or '/path/to/file.py:func'; "
            f"got {path!r}."
        )

    if module_part.endswith(".py"):
        file_path = Path(module_part)
        if not file_path.is_file():
            raise FileNotFoundError(f"filter_fn file not found: {file_path}")
        spec = importlib.util.spec_from_file_location(
            f"_phyai_dump_filter_{file_path.stem}", str(file_path)
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load filter_fn module from {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_part)

    fn = getattr(module, attr, None)
    if fn is None:
        raise AttributeError(f"{module_part!r} has no attribute {attr!r}")
    if not callable(fn):
        raise TypeError(f"filter_fn {path!r} resolved to a non-callable {type(fn)}.")
    return fn


class TensorDumper:
    """Forward-hook activation recorder over one or more model modules.

    Hooks the leaf modules of each target, accumulating their outputs into
    a per-pass dict. :meth:`flush_pass` writes the accumulated dict to a
    ``.pt`` file and starts the next pass. :meth:`detach` removes every
    hook (idempotent).

    Parameters
    ----------
    targets:
        Mapping of *root name* to module. The root name prefixes every
        recorded operator key, so ``{"model": pi05_model}`` yields keys
        like ``model.paligemma_lm.layers.0.mlp.down_proj``. Passing more
        than one target lets a plugin dump several top-level modules under
        distinct namespaces.
    dump_dir:
        Base directory. A per-process subdirectory
        ``rank{rank}_pid{pid}`` is created beneath it so concurrent ranks
        never collide.
    filter:
        Which leaf operators to record, matched against each operator's
        full dotted name. ``None`` (default) records every leaf; a
        sequence of regex strings records a leaf if any pattern
        ``re.search``-matches (a union); a ``(name, module) -> bool``
        callable records a leaf when it returns True. See
        :func:`_compile_filter` and the module docstring.
    rank, world_size:
        Distributed identity used for the subdirectory name and log
        gating. Default to the live ``torch.distributed`` values (or
        ``0`` / ``1`` when not initialised).
    """

    def __init__(
        self,
        targets: Mapping[str, nn.Module],
        *,
        dump_dir: str | Path,
        filter: FilterSpec = None,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> None:
        if not targets:
            raise ValueError("TensorDumper requires at least one target module.")
        self._filter_spec = filter
        self._filter: FilterFn = _compile_filter(filter)
        self._rank = _resolve_rank() if rank is None else int(rank)
        self._world_size = (
            _resolve_world_size() if world_size is None else int(world_size)
        )
        self._pid = os.getpid()
        self._pass_index = 0
        self._current: dict[str, torch.Tensor | list | dict] = {}
        # Per-pass count of how many times each base name has fired, so
        # repeat invocations within one pass get a ``::callN`` suffix
        # instead of clobbering the first.
        self._fire_counts: dict[str, int] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

        self._base_dir = Path(dump_dir)
        self._process_dir = self._base_dir / f"rank{self._rank}_pid{self._pid}"
        self._process_dir.mkdir(parents=True, exist_ok=True)

        n_hooked = 0
        for root_name, module in targets.items():
            n_hooked += self._attach_to_tree(root_name, module)
        all_ranks_log(
            logger,
            logging.INFO,
            "TensorDumper attached %d leaf hooks across %d target(s) "
            "(dump_dir=%s, filter=%s).",
            n_hooked,
            len(targets),
            str(self._process_dir),
            self._describe_filter(),
        )
        if n_hooked == 0:
            all_ranks_log(
                logger,
                logging.WARNING,
                "TensorDumper hooked 0 leaf operators — the filter %s matched "
                "nothing across the targets. No tensors will be recorded.",
                self._describe_filter(),
            )

    def _describe_filter(self) -> str:
        spec = self._filter_spec
        if spec is None:
            return "all"
        if callable(spec):
            return f"callable({getattr(spec, '__name__', repr(spec))})"
        return f"regex{list(spec)}"

    # ------------------------------------------------------------------ #
    # Hook registration                                                  #
    # ------------------------------------------------------------------ #

    def _attach_to_tree(self, root_name: str, root: nn.Module) -> int:
        """Hook every leaf module under ``root`` that the filter selects."""
        count = 0
        for name, module in root.named_modules(prefix=root_name):
            # Leaf-only: a module with children is represented by its last
            # child's output, so hooking it would double-record.
            if next(module.children(), None) is not None:
                continue
            if not self._filter(name, module):
                continue
            handle = module.register_forward_hook(self._make_hook(name))
            self._handles.append(handle)
            count += 1
        return count

    def _make_hook(self, name: str):
        def _hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            self._record(name, output)

        return _hook

    # ------------------------------------------------------------------ #
    # Recording + flush                                                  #
    # ------------------------------------------------------------------ #

    def _record(self, name: str, output: Any) -> None:
        converted = _cpu_tensors(output)
        if converted is None:
            return
        fired = self._fire_counts.get(name, 0)
        self._fire_counts[name] = fired + 1
        key = name if fired == 0 else f"{name}::call{fired}"
        self._current[key] = converted

    def flush_pass(self) -> Path | None:
        """Write the accumulated pass to ``pass{N:05d}.pt`` and reset.

        Returns the written path, or ``None`` when the pass recorded
        nothing (no hook fired since the last flush — e.g. the model ran
        entirely under captured graphs).
        """
        if not self._current:
            return None
        path = self._process_dir / f"pass{self._pass_index:05d}.pt"
        torch.save(self._current, str(path))
        this_rank_log(
            logger,
            logging.INFO,
            "TensorDumper wrote pass %05d (%d tensors) to %s",
            self._pass_index,
            len(self._current),
            str(path),
            rank=self._rank,
        )
        self._pass_index += 1
        self._current = {}
        self._fire_counts = {}
        return path

    def detach(self) -> None:
        """Remove every registered hook. Idempotent."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @property
    def process_dir(self) -> Path:
        """The per-process directory dump files are written to."""
        return self._process_dir

    @property
    def pass_index(self) -> int:
        """Number of passes flushed so far (also the next pass's index)."""
        return self._pass_index


def register_tensor_dumper(
    targets: Mapping[str, nn.Module],
    *,
    dump_dir: str | Path,
    filter: FilterSpec = None,
    rank: int | None = None,
    world_size: int | None = None,
) -> TensorDumper:
    """Construct a :class:`TensorDumper` and attach its hooks.

    Thin convenience wrapper — the constructor already attaches the hooks;
    this is the named, importable entry point.
    """
    return TensorDumper(
        targets,
        dump_dir=dump_dir,
        filter=filter,
        rank=rank,
        world_size=world_size,
    )


def load_pass(path: str | Path) -> dict[str, Any]:
    """Load one dumped pass file written by :meth:`TensorDumper.flush_pass`.

    Returns the ``{operator_name: value}`` dict (tensors land on CPU).
    A thin wrapper over ``torch.load`` so callers don't have to remember
    ``weights_only=False`` / ``map_location``.
    """
    return torch.load(str(path), weights_only=False, map_location="cpu")


__all__ = [
    "FilterFn",
    "FilterSpec",
    "TensorDumper",
    "load_filter_fn",
    "load_pass",
    "register_tensor_dumper",
]
