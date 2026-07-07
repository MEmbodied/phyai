"""``phyai doctor`` — diagnose a phyai install without touching the GPU heavily.

Runs a fast battery of checks: interpreter/platform, phyai + workspace
imports, core dependency pins, the CUDA toolchain (nvcc / driver), torch's
CUDA view (with a forward-compat hint when torch was built for a newer CUDA
than the driver exposes), visible GPUs, a lightweight phyai config/init smoke
test, and ``PHYAI_*`` parsing. Prints a branded per-section report and exits
non-zero when any check fails so it is usable in CI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.console import Console
from rich.text import Text

from phyai.cli import probe, ui


@dataclass
class Section:
    name: str
    rows: list[tuple[str, str, str]] = field(default_factory=list)

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.rows.append((status, name, detail))


def _render(console: Console, section: Section) -> None:
    console.print(ui.section_title(section.name))
    table = ui.checks_table()
    for status, name, detail in section.rows:
        table.add_row(
            ui.badge(status), name, Text(detail, style="dim") if detail else Text("")
        )
    console.print(table)
    console.print()


def _sec_python() -> Section:
    s = Section("Interpreter & platform")
    info = probe.python_info()
    s.add(ui.INFO, "python", f"{info['version']} ({info['implementation']})")
    s.add(ui.INFO, "executable", info["executable"])
    s.add(ui.INFO, "platform", f"{info['platform']} · {info['machine']}")
    return s


def _sec_phyai() -> Section:
    s = Section("phyai package")
    imp = probe.probe_import("phyai")
    if imp.ok:
        import phyai

        s.add(ui.OK, "import phyai", f"version {phyai.__version__}")
    else:
        s.add(ui.FAIL, "import phyai", imp.error or "import failed")
    return s


# Workspace members phyai can run without are advisory; the rest are required.
_OPTIONAL_EXT = {"phyai-ext", "phyai-model-optimizer"}


def _sec_ext() -> Section:
    s = Section("Workspace extension packages")
    for pkg in probe.ext_packages():
        if pkg.imported.ok:
            s.add(ui.OK, pkg.distribution, f"version {pkg.dist_version or '?'}")
        elif pkg.distribution in _OPTIONAL_EXT:
            s.add(
                ui.WARN,
                pkg.distribution,
                f"optional, not importable: {pkg.imported.error}",
            )
        else:
            s.add(ui.FAIL, pkg.distribution, pkg.imported.error or "import failed")
    return s


def _sec_deps() -> Section:
    s = Section("Core dependencies")
    for dep in probe.core_deps():
        if dep.installed is None:
            s.add(
                ui.FAIL, dep.distribution, f"not installed (expected =={dep.expected})"
            )
        elif dep.matches:
            s.add(ui.OK, dep.distribution, f"{dep.installed}")
        else:
            s.add(
                ui.WARN,
                dep.distribution,
                f"{dep.installed} drifted from pin =={dep.expected}",
            )
    return s


def _sec_cuda_tools(tools: probe.CudaToolProbe) -> Section:
    s = Section("CUDA toolchain")
    if tools.nvidia_smi_path:
        detail = f"driver {tools.driver_version or '?'}"
        if tools.driver_cuda_version:
            detail += f", supports CUDA {tools.driver_cuda_version}"
        s.add(ui.OK, "nvidia-smi", detail)
    else:
        s.add(ui.WARN, "nvidia-smi", "not on PATH (no NVIDIA driver visible?)")
    if tools.nvcc_path:
        s.add(ui.OK, "nvcc", f"CUDA {tools.nvcc_version or '?'}")
    else:
        s.add(
            ui.WARN, "nvcc", "not on PATH (fine for runtime; needed to build kernels)"
        )
    s.add(ui.INFO, "CUDA_HOME", tools.cuda_home or "not set")
    return s


def _major_minor(version: str | None) -> tuple[int, int] | None:
    if not version:
        return None
    parts = version.split(".")
    try:
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        return None


def _sec_torch_cuda(torch: probe.TorchProbe, tools: probe.CudaToolProbe) -> Section:
    s = Section("Torch CUDA")
    if not torch.imported.ok:
        s.add(ui.FAIL, "import torch", torch.imported.error or "import failed")
        return s
    s.add(
        ui.INFO, "torch", f"{torch.version} (built for CUDA {torch.built_cuda or '?'})"
    )

    if torch.cuda_available:
        s.add(
            ui.OK, "torch.cuda.is_available()", f"True · {torch.device_count} device(s)"
        )
    else:
        built = _major_minor(torch.built_cuda)
        driver = _major_minor(tools.driver_cuda_version)
        if built and driver and driver < built:
            s.add(
                ui.WARN,
                "torch.cuda.is_available()",
                f"False — torch needs CUDA {torch.built_cuda} but the driver only "
                f"exposes CUDA {tools.driver_cuda_version}; install a newer driver or a "
                f"forward-compatible libcuda (CUDA compat package).",
            )
        else:
            s.add(
                ui.WARN,
                "torch.cuda.is_available()",
                "False — no usable CUDA device (CPU-only host, or driver/runtime mismatch).",
            )
    if torch.error:
        s.add(ui.WARN, "torch.cuda probe", torch.error)
    return s


def _sec_gpus(torch: probe.TorchProbe) -> Section:
    s = Section("GPUs")
    if not torch.cuda_available or not torch.devices:
        s.add(ui.INFO, "devices", "none visible to torch")
        return s
    for dev in torch.devices:
        s.add(
            ui.OK,
            f"cuda:{dev.index}",
            f"{dev.name} · {dev.capability} · {dev.multi_processor_count} SMs · "
            f"{dev.total_memory_gib} GiB",
        )
    return s


def _sec_init() -> Section:
    s = Section("phyai initialization")

    try:
        from phyai.engine_config import BackendConfig

        bc = BackendConfig()
        s.add(
            ui.OK,
            "backend registries",
            f"attn={bc.attn}, norm={bc.norm} validated",
        )
    except Exception as exc:  # noqa: BLE001
        s.add(ui.FAIL, "backend registries", f"{type(exc).__name__}: {exc}")

    try:
        from phyai.engine_config import EngineConfig

        EngineConfig.from_env()
        s.add(ui.OK, "EngineConfig.from_env()", "config built + validated")
    except Exception as exc:  # noqa: BLE001
        s.add(ui.FAIL, "EngineConfig.from_env()", f"{type(exc).__name__}: {exc}")

    try:
        import torch

        from phyai.utils.cuda import init_cuda

        saved = init_cuda("cpu", torch.bfloat16)
        torch.set_default_dtype(saved)  # restore process default immediately
        s.add(ui.OK, "init_cuda bootstrap", "device/dtype pin runs (cpu, bf16)")
    except Exception as exc:  # noqa: BLE001
        s.add(ui.WARN, "init_cuda bootstrap", f"{type(exc).__name__}: {exc}")
    return s


def _sec_env() -> Section:
    s = Section("PHYAI_* environment")
    registered, extra = probe.phyai_env()
    active = [e for e in registered if e.is_set]
    errored = [e for e in registered if e.error]
    for var in active:
        if var.error:
            s.add(ui.FAIL, var.name, f"{var.raw!r} → {var.error}")
        else:
            s.add(ui.OK, var.name, f"{var.raw}")
    for name, raw in extra.items():
        s.add(ui.WARN, name, f"{raw!r} (set but not a registered PHYAI_* field)")
    if not active and not extra:
        s.add(ui.INFO, "overrides", "none set (all defaults)")
    if errored:
        s.add(ui.INFO, "note", f"{len(errored)} variable(s) failed to parse")
    return s


def _summary(console: Console, sections: list[Section]) -> int:
    counts = {ui.OK: 0, ui.WARN: 0, ui.FAIL: 0, ui.INFO: 0}
    for section in sections:
        for status, _, _ in section.rows:
            counts[status] = counts.get(status, 0) + 1

    line = Text()
    line.append(f"{counts[ui.OK]} ok", style="green")
    line.append("  ·  ")
    line.append(f"{counts[ui.WARN]} warnings", style="yellow")
    line.append("  ·  ")
    line.append(f"{counts[ui.FAIL]} failed", style="red")

    if counts[ui.FAIL]:
        verdict = Text("✘ problems found", style="bold red")
        code = 1
    elif counts[ui.WARN]:
        verdict = Text("▲ healthy, with warnings", style="bold yellow")
        code = 0
    else:
        verdict = Text("✔ all systems go", style="bold green")
        code = 0

    console.rule(style=ui.PHI_BLUE)
    grid = ui.kv_table()
    grid.add_row(ui.phi(), verdict)
    grid.add_row("", line)
    console.print(grid)
    console.print()
    return code


def run_doctor(console: Console) -> int:
    import phyai

    ui.print_header(console, "doctor", version=phyai.__version__)

    tools = probe.cuda_tools()
    torch = probe.torch_info()

    sections = [
        _sec_python(),
        _sec_phyai(),
        _sec_ext(),
        _sec_deps(),
        _sec_cuda_tools(tools),
        _sec_torch_cuda(torch, tools),
        _sec_gpus(torch),
        _sec_init(),
        _sec_env(),
    ]
    for section in sections:
        _render(console, section)
    return _summary(console, sections)
