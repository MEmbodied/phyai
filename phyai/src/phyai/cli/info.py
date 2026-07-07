"""``phyai info`` — a branded snapshot of the installed phyai stack."""

from __future__ import annotations

from rich.console import Console
from rich.text import Text

from phyai.cli import probe, ui


def _version_section(console: Console) -> None:
    git = probe.git_info()
    table = ui.kv_table()

    import phyai

    table.add_row("version", Text(phyai.__version__, style="bold"))
    if git.get("available"):
        commit = str(git.get("commit") or "?")
        text = Text(commit, style=ui.PHI_BLUE)
        if git.get("dirty"):
            text.append("  (dirty)", style="yellow")
        elif git.get("dirty") is False:
            text.append("  (clean)", style="dim green")
        table.add_row("git commit", text)
    console.print(table)
    console.print()


def _ext_section(console: Console) -> None:
    console.print(ui.section_title("Extension packages"))
    table = ui.data_table("Package", "Installed", "Version", "Import")
    for pkg in probe.ext_packages():
        installed = pkg.dist_version is not None
        imp = pkg.imported
        table.add_row(
            pkg.distribution,
            ui.badge(ui.OK) if installed else ui.badge(ui.FAIL),
            ui.value_or_dash(pkg.dist_version),
            ui.badge(ui.OK) if imp.ok else Text(f"✘ {imp.error}", style="red"),
        )
    console.print(table)
    console.print()


def _deps_section(console: Console) -> None:
    console.print(ui.section_title("Core dependencies"))
    table = ui.data_table("Dependency", "Installed", "Expected pin", "Match")
    for dep in probe.core_deps():
        if dep.installed is None:
            match = Text("✘ missing", style="red")
        elif dep.matches:
            match = ui.badge(ui.OK)
        else:
            match = Text("▲ drift", style="yellow")
        table.add_row(
            dep.distribution,
            ui.value_or_dash(dep.installed),
            f"=={dep.expected}",
            match,
        )
    console.print(table)
    console.print()


def _quant_section(console: Console) -> None:
    console.print(ui.section_title("Quantization support"))
    info = probe.quant_info()
    if info.get("error"):
        console.print(Text(f"  unavailable: {info['error']}", style="red"))
        console.print()
        return

    kv = ui.kv_table()
    kv.add_row("element dtypes", Text(", ".join(info["dtypes"])))  # type: ignore[arg-type]
    kv.add_row("checkpoint importers", Text(", ".join(info["importers"])))  # type: ignore[arg-type]

    sm = info["sm"]
    sm_label = f"sm_{sm}" if sm else "cpu / no GPU"
    kv.add_row(
        f"linear specs ({sm_label})",
        Text(", ".join(info["specs"]) or "\u2014"),  # type: ignore[arg-type]
    )
    if info["specs_extra"]:
        kv.add_row(
            "unlocked at higher SM",
            Text(", ".join(info["specs_extra"]), style="dim"),  # type: ignore[arg-type]
        )
    console.print(kv)
    console.print()


def _backends_section(console: Console) -> None:
    console.print(ui.section_title("Registered backends"))
    table = ui.data_table("Subsystem", "Backends")
    labels = {
        "attention": "attention (prefill)",
        "ar": "attention (autoregressive)",
        "diffusion": "attention (diffusion)",
        "norm": "norm",
        "linear": "linear kernels",
        "vgpu": "vgpu",
    }
    info = probe.backend_info()
    for key, label in labels.items():
        value = info.get(key)
        if isinstance(value, str):  # error string
            rendered = Text(value, style="red")
        elif value:
            rendered = Text(", ".join(value))
        else:
            rendered = ui.value_or_dash(None)
        table.add_row(label, rendered)
    console.print(table)
    console.print()


def _gpu_section(console: Console) -> None:
    console.print(ui.section_title("GPU"))
    torch = probe.torch_info()
    if not torch.imported.ok:
        console.print(
            Text(f"  torch import failed: {torch.imported.error}", style="red")
        )
        console.print()
        return
    if not torch.cuda_available or not torch.devices:
        note = Text("  torch.cuda.is_available() = False", style="dim")
        if torch.built_cuda:
            note.append(f"  (torch built for CUDA {torch.built_cuda})", style="dim")
        console.print(note)
        console.print()
        return

    table = ui.data_table("Index", "Name", "Arch", "SMs", "Memory")
    for dev in torch.devices:
        table.add_row(
            str(dev.index),
            dev.name,
            dev.capability,
            str(dev.multi_processor_count),
            f"{dev.total_memory_gib} GiB",
        )
    console.print(table)
    console.print()


def _env_section(console: Console) -> None:
    console.print(ui.section_title("Active PHYAI_* overrides"))
    registered, extra = probe.phyai_env()
    active = [e for e in registered if e.is_set]
    if not active and not extra:
        console.print(Text("  none set (all defaults)", style="dim"))
        console.print()
        return

    table = ui.data_table("Variable", "Value")
    for var in active:
        value = Text(var.raw or "")
        if var.error:
            value = Text(f"{var.raw}  ✘ {var.error}", style="red")
        table.add_row(var.name, value)
    for name, raw in extra.items():
        table.add_row(Text(name, style="dim"), Text(raw))
    console.print(table)
    console.print()


def run_info(console: Console) -> int:
    import phyai

    ui.print_header(console, "info", version=phyai.__version__)
    _version_section(console)
    _ext_section(console)
    _deps_section(console)
    _quant_section(console)
    _backends_section(console)
    _gpu_section(console)
    _env_section(console)
    return 0
