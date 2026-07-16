"""Read-only environment probes shared by ``phyai doctor`` and ``phyai info``."""

from __future__ import annotations

import importlib
import importlib.metadata as _md
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Hard pins mirrored from phyai/pyproject.toml (see CLAUDE.md "Repository
# shape"). Kept here so `phyai info` can flag a drifted environment without
# re-parsing pyproject at runtime. (distribution, import module, expected).
CORE_DEPS: tuple[tuple[str, str, str], ...] = (
    ("torch", "torch", "2.11"),
    ("flashinfer-python", "flashinfer", "0.6.14"),
    ("transformers", "transformers", "5.8.1"),
)

# Workspace members. (distribution, import module).
EXT_PACKAGES: tuple[tuple[str, str], ...] = (
    ("phyai-kernel", "phyai_kernel"),
    ("phyai-ext", "phyai_ext"),
    ("phyai-model-optimizer", "phyai_model_optimizer"),
    ("phyai-utils-tools", "phyai_utils_tools"),
)


@dataclass
class ImportProbe:
    module: str
    ok: bool
    version: str | None = None
    error: str | None = None


@dataclass
class PackageProbe:
    distribution: str
    module: str
    dist_version: str | None
    imported: ImportProbe


@dataclass
class DepProbe:
    distribution: str
    module: str
    installed: str | None
    expected: str
    matches: bool


@dataclass
class GpuProbe:
    index: int
    name: str
    capability: str
    multi_processor_count: int
    total_memory_gib: float


@dataclass
class TorchProbe:
    imported: ImportProbe
    version: str | None = None
    built_cuda: str | None = None
    cuda_available: bool = False
    device_count: int = 0
    devices: list[GpuProbe] = field(default_factory=list)
    error: str | None = None


@dataclass
class CudaToolProbe:
    cuda_home: str | None
    nvcc_path: str | None
    nvcc_version: str | None
    nvidia_smi_path: str | None
    driver_version: str | None
    driver_cuda_version: str | None


@dataclass
class EnvVarProbe:
    name: str
    is_set: bool
    raw: str | None
    parsed_or_default: str | None
    error: str | None


def dist_version(distribution: str) -> str | None:
    """Installed version of a distribution, or ``None`` if absent."""
    try:
        return _md.version(distribution)
    except _md.PackageNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001 - report, never crash the CLI
        return f"error: {exc}"


def probe_import(module: str) -> ImportProbe:
    try:
        mod = importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001
        return ImportProbe(module, ok=False, error=f"{type(exc).__name__}: {exc}")
    return ImportProbe(module, ok=True, version=getattr(mod, "__version__", None))


def version_matches(installed: str | None, expected: str) -> bool:
    """True when ``installed`` starts with ``expected`` (ignoring a ``v`` prefix)."""
    if not installed or installed.startswith("error:"):
        return False
    norm = installed.lstrip("vV")
    exp = expected.lstrip("vV")
    return (
        norm == exp
        or norm.startswith(exp + ".")
        or norm.startswith(exp + "+")
        or norm == exp
    )


def python_info() -> dict[str, str]:
    return {
        "version": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def core_deps() -> list[DepProbe]:
    out: list[DepProbe] = []
    for distribution, module, expected in CORE_DEPS:
        installed = dist_version(distribution)
        out.append(
            DepProbe(
                distribution=distribution,
                module=module,
                installed=installed,
                expected=expected,
                matches=version_matches(installed, expected),
            )
        )
    return out


def ext_packages() -> list[PackageProbe]:
    out: list[PackageProbe] = []
    for distribution, module in EXT_PACKAGES:
        out.append(
            PackageProbe(
                distribution=distribution,
                module=module,
                dist_version=dist_version(distribution),
                imported=probe_import(module),
            )
        )
    return out


def _run(cmd: list[str], timeout: float = 8.0) -> str | None:
    try:
        done = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return (done.stdout or "") + (done.stderr or "")


def cuda_tools() -> CudaToolProbe:
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    nvcc_path = shutil.which("nvcc")
    smi_path = shutil.which("nvidia-smi")

    nvcc_version = None
    if nvcc_path and (out := _run(["nvcc", "--version"])):
        m = re.search(r"release\s+([0-9]+(?:\.[0-9]+)*)", out)
        nvcc_version = m.group(1) if m else None

    driver_version = driver_cuda = None
    if smi_path and (out := _run(["nvidia-smi"])):
        if m := re.search(r"Driver Version:\s*([0-9.]+)", out):
            driver_version = m.group(1)
        if m := re.search(r"CUDA Version:\s*([0-9.]+)", out):
            driver_cuda = m.group(1)

    return CudaToolProbe(
        cuda_home=cuda_home,
        nvcc_path=nvcc_path,
        nvcc_version=nvcc_version,
        nvidia_smi_path=smi_path,
        driver_version=driver_version,
        driver_cuda_version=driver_cuda,
    )


def torch_info() -> TorchProbe:
    imported = probe_import("torch")
    if not imported.ok:
        return TorchProbe(imported=imported)

    import torch

    probe = TorchProbe(
        imported=imported,
        version=torch.__version__,
        built_cuda=getattr(torch.version, "cuda", None),
    )
    try:
        probe.cuda_available = bool(torch.cuda.is_available())
        probe.device_count = torch.cuda.device_count() if probe.cuda_available else 0
        for idx in range(probe.device_count):
            props = torch.cuda.get_device_properties(idx)
            major, minor = torch.cuda.get_device_capability(idx)
            probe.devices.append(
                GpuProbe(
                    index=idx,
                    name=props.name,
                    capability=f"sm_{major}{minor}",
                    multi_processor_count=props.multi_processor_count,
                    total_memory_gib=round(props.total_memory / (1 << 30), 1),
                )
            )
    except Exception as exc:  # noqa: BLE001
        probe.error = f"{type(exc).__name__}: {exc}"
    return probe


def _repo_root() -> Path | None:
    try:
        import phyai

        start = Path(phyai.__file__).resolve()
    except Exception:  # noqa: BLE001
        start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def git_info() -> dict[str, str | bool | None]:
    root = _repo_root()
    if root is None:
        return {"available": False, "commit": None, "dirty": None}
    commit = _run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], timeout=5)
    status = _run(["git", "-C", str(root), "status", "--porcelain"], timeout=5)
    if commit is None:
        return {"available": False, "commit": None, "dirty": None}
    return {
        "available": True,
        "commit": commit.strip() or None,
        "dirty": bool(status.strip()) if status is not None else None,
        "root": str(root),
    }


# Highest SM tier gate in supported_specs_for_sm (fp4 activation). Passing it
# yields the full set of spec_ids phyai can register on any hardware.
_MAX_SM_TIER = 120


def quant_info() -> dict[str, object]:
    """Enumerate the quant surface phyai actually registers.

    ``dtypes`` is the :class:`QDType` element vocabulary, ``importers`` the
    checkpoint front-ends, and ``specs`` the linear spec_ids reported by
    :func:`phyai.layers.linear.supported_specs_for_sm` for the detected GPU —
    with ``specs_extra`` listing what a higher SM would additionally unlock.
    Reading them from the registry (not a hand-written list) keeps the report
    honest as backends and humming tiers change.
    """
    info: dict[str, object] = {
        "dtypes": [],
        "importers": [],
        "sm": 0,
        "specs": [],
        "specs_extra": [],
        "error": None,
    }
    try:
        from phyai.layers.quant.scheme import QDType

        info["dtypes"] = [d.value for d in QDType]
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"{type(exc).__name__}: {exc}"
        return info

    try:
        from phyai.layers.quant.importers.registry import DEFAULT_IMPORTERS

        info["importers"] = [imp.name for imp in DEFAULT_IMPORTERS]
    except Exception as exc:  # noqa: BLE001
        info["importers"] = [f"error: {exc}"]

    try:
        from phyai.layers.linear import supported_specs_for_sm
        from phyai.utils.cuda import sm_arch

        sm = sm_arch()
        here = supported_specs_for_sm(sm)
        full = supported_specs_for_sm(_MAX_SM_TIER)
        info["sm"] = sm
        info["specs"] = here
        info["specs_extra"] = [s for s in full if s not in here]
    except Exception as exc:  # noqa: BLE001
        info["specs"] = [f"error: {type(exc).__name__}: {exc}"]
    return info


def _safe_list(fn) -> list[str] | str:
    try:
        return list(fn())
    except Exception as exc:  # noqa: BLE001
        return f"error: {type(exc).__name__}: {exc}"


def backend_info() -> dict[str, list[str] | str]:
    """Enumerate registered backends across each subsystem.

    Names populate via side-effect imports, so importing the backend
    packages here is what makes the registries non-empty. flashinfer being
    absent degrades to a shorter list / an error string per row.
    """
    out: dict[str, list[str] | str] = {}

    def attn(stack: str):
        def inner() -> list[str]:
            mod = importlib.import_module(f"phyai.layers.attention.{stack}")
            return mod.list_backends()

        return inner

    out["attention"] = _safe_list(attn("attention"))
    out["ar"] = _safe_list(attn("ar"))
    out["diffusion"] = _safe_list(attn("diffusion"))

    def norm() -> list[str]:
        from phyai.layers.layer_norm import list_norm_backends

        return list_norm_backends()

    def linear() -> list[str]:
        import phyai.layers.linear  # noqa: F401 - triggers kernel registration
        from phyai.layers.linear.registry import list_registered_linear_kernels

        return [cls.name for cls, _ in list_registered_linear_kernels()]

    def vgpu() -> list[str]:
        from phyai.vgpu.backend import known_backends

        return known_backends()

    out["norm"] = _safe_list(norm)
    out["linear"] = _safe_list(linear)
    out["vgpu"] = _safe_list(vgpu)
    return out


def phyai_env() -> tuple[list[EnvVarProbe], dict[str, str]]:
    """Registered PHYAI_* fields (with parse status) plus any extra PHYAI_* set."""
    import inspect

    registered: list[EnvVarProbe] = []
    known_names: set[str] = set()
    try:
        from phyai.env import EnvField, envs

        for name, value in inspect.getmembers(envs):
            if not name.startswith("PHYAI_") or not isinstance(value, EnvField):
                continue
            known_names.add(value.name)
            raw = os.environ.get(value.name)
            parsed: object = None
            error: str | None = None
            if value.is_set():
                try:
                    parsed = value.get()
                except Exception as exc:  # noqa: BLE001
                    error = f"{type(exc).__name__}: {exc}"
            else:
                parsed = value.default
            registered.append(
                EnvVarProbe(
                    name=value.name,
                    is_set=value.is_set(),
                    raw=raw,
                    parsed_or_default=str(parsed) if parsed is not None else None,
                    error=error,
                )
            )
    except Exception:  # noqa: BLE001 - registry import failure handled by caller
        pass

    registered.sort(key=lambda e: e.name)
    extra = {
        k: v
        for k, v in sorted(os.environ.items())
        if k.startswith("PHYAI_") and k not in known_names
    }
    return registered, extra
