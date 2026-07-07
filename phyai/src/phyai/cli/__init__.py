"""phyai command-line interface."""

from __future__ import annotations

import argparse


def _common_flags() -> argparse.ArgumentParser:
    # SUPPRESS default so the flag works before OR after the subcommand
    # without a subparser resetting a value set on the top-level parser.
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--no-color",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Disable colored / styled output.",
    )
    return parent


def build_parser() -> argparse.ArgumentParser:
    common = _common_flags()
    parser = argparse.ArgumentParser(
        prog="phyai",
        description="phyai — Physical AI inference engine command-line interface.",
        parents=[common],
    )
    parser.add_argument(
        "-V",
        "--version",
        action="store_true",
        help="Show the phyai version and exit.",
    )
    sub = parser.add_subparsers(dest="command", metavar="{doctor,info}")
    sub.add_parser(
        "doctor",
        parents=[common],
        help="Diagnose the CUDA / phyai installation and initialization.",
        description="Diagnose the CUDA / phyai installation and initialization.",
    )
    sub.add_parser(
        "info",
        parents=[common],
        help="Show phyai version, quantization specs, backends, and packages.",
        description="Show phyai version, quantization specs, backends, and packages.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    from phyai.cli.ui import brand_line, make_console

    console = make_console(no_color=getattr(args, "no_color", False))

    if args.version:
        import phyai

        console.print(brand_line(f"v{phyai.__version__}"))
        return 0

    if args.command == "doctor":
        from phyai.cli.doctor import run_doctor

        return run_doctor(console)

    if args.command == "info":
        from phyai.cli.info import run_info

        return run_info(console)

    parser.print_help()
    return 0


__all__ = ["main", "build_parser"]
