"""Shared rich presentation layer for the phyai CLI."""

from __future__ import annotations

from rich.box import SIMPLE_HEAD
from rich.console import Console
from rich.table import Table
from rich.text import Text

# The phyai mark: lowercase Greek phi (U+03C6), always drawn in brand blue.
PHI = "\u03c6"
PHI_BLUE = "#4C8DFF"
BRAND = f"bold {PHI_BLUE}"

# Status vocabulary shared by the check tables. Each maps to (glyph, style).
OK = "ok"
WARN = "warn"
FAIL = "fail"
INFO = "info"

_BADGES: dict[str, tuple[str, str]] = {
    OK: ("\u2714", "bold green"),  # ✔
    WARN: ("\u25b2", "bold yellow"),  # ▲
    FAIL: ("\u2718", "bold red"),  # ✘
    INFO: ("\u2022", BRAND),  # •
}


def make_console(*, no_color: bool = False) -> Console:
    """Console with syntax highlighting off (we style everything explicitly)."""
    return Console(no_color=no_color, highlight=False, emoji=False)


def phi(size: int = 1) -> Text:
    """The bare φ mark in brand blue (``size`` repeats for a heavier glyph)."""
    return Text(PHI * size, style=BRAND)


def brand_line(subtitle: str | None = None) -> Text:
    """`` φ phyai  <subtitle>`` as a single styled line."""
    line = Text(" ")
    line.append(PHI, style=BRAND)
    line.append("  phyai", style="bold")
    if subtitle:
        line.append("  ")
        line.append(subtitle, style="dim")
    return line


def print_header(
    console: Console, subtitle: str, *, version: str | None = None
) -> None:
    """Compact branded header: brand line on the left, version on the right."""
    grid = Table.grid(expand=True, padding=0)
    grid.add_column(justify="left")
    grid.add_column(justify="right")
    right = Text(f"v{version}", style="dim") if version else Text("")
    console.print()
    grid.add_row(brand_line(subtitle), right)
    console.print(grid)
    console.rule(style=PHI_BLUE)


def badge(status: str) -> Text:
    glyph, style = _BADGES.get(status, _BADGES[INFO])
    return Text(glyph, style=style)


def section_title(name: str) -> Text:
    """A φ-prefixed sub-heading used above each block of output."""
    t = Text()
    t.append(f"{PHI} ", style=BRAND)
    t.append(name, style="bold")
    return t


def kv_table() -> Table:
    """Two-column key/value grid (label dim, value default)."""
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim", justify="left")
    t.add_column(justify="left")
    return t


def data_table(*headers: str) -> Table:
    """Bordered table with brand-blue headers for tabular sections."""
    t = Table(
        box=SIMPLE_HEAD,
        show_edge=False,
        pad_edge=False,
        header_style=BRAND,
        expand=False,
    )
    for h in headers:
        t.add_column(h)
    return t


def checks_table() -> Table:
    """Borderless [badge | name | detail] table for doctor's checks."""
    t = Table.grid(padding=(0, 2))
    t.add_column(justify="center", width=1)
    t.add_column(style="bold", no_wrap=True)
    t.add_column(overflow="fold")
    return t


def value_or_dash(value: object) -> Text:
    """Render ``None``/empty as a dim ``—`` so tables stay aligned."""
    if value is None or value == "":
        return Text("\u2014", style="dim")
    return Text(str(value))
