"""``phyai-optimize`` CLI."""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

from phyai_model_optimizer.modifiers.rtn import RTNModifier
from phyai_model_optimizer.quant_math import FP8Scheme, QuantDType
from phyai_model_optimizer.recipes import Recipe


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="phyai-optimize", description="phyai PTQ toolkit")
    sub = p.add_subparsers(dest="command", required=True)

    q = sub.add_parser("quantize", help="data-free quantize a checkpoint (RTN)")
    q.add_argument("--input", required=True, help="safetensors file or directory")
    q.add_argument("--output", required=True, help="output directory")
    q.add_argument("--recipe", help="recipe yaml/json (overrides inline flags)")
    q.add_argument(
        "--weight-dtype",
        choices=tuple(dtype.value for dtype in QuantDType),
        default=QuantDType.INT4.value,
    )
    q.add_argument(
        "--activation-dtype",
        choices=(
            "auto",
            "none",
            *(dtype.value for dtype in QuantDType if dtype.supports_activation),
        ),
        default="auto",
    )
    q.add_argument("--group-size", type=int)
    q.add_argument(
        "--fp8-scheme",
        choices=tuple(scheme.value for scheme in FP8Scheme),
        help="FP8 scale layout (defaults to block-128 for FP8 weights)",
    )
    q.add_argument("--asymmetric", action="store_true")
    q.add_argument(
        "--targets", nargs="*", default=["re:.*"], help="target name/regex patterns"
    )
    q.add_argument("--ignore", nargs="*", default=[])
    q.add_argument(
        "--pack-format",
        choices=("compressed-tensors", "humming"),
        default="compressed-tensors",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    console = Console(highlight=False)
    if args.command == "quantize":
        from phyai_model_optimizer.entrypoints import model_free_ptq

        if args.recipe:
            modifiers = Recipe.load(args.recipe).modifiers
        else:
            weight_dtype = QuantDType(args.weight_dtype)
            if weight_dtype.is_fp8:
                if args.group_size is not None:
                    parser.error(
                        "FP8 weights use --fp8-scheme; do not pass --group-size"
                    )
                if args.asymmetric:
                    parser.error("FP8 weight quantization must be symmetric")
                if (
                    weight_dtype is QuantDType.FP8_E4M3
                    and args.activation_dtype not in ("auto", "fp8_e4m3")
                ):
                    parser.error(
                        "fp8_e4m3 weights require dynamic "
                        "--activation-dtype fp8_e4m3 (or auto)"
                    )
            elif args.fp8_scheme is not None:
                parser.error("--fp8-scheme is only valid for FP8 weights")
            group_size = args.group_size
            if group_size is None and weight_dtype in (
                QuantDType.INT2,
                QuantDType.INT3,
                QuantDType.INT4,
                QuantDType.INT6,
            ):
                group_size = 128
            modifiers = [
                RTNModifier(
                    weight_dtype=weight_dtype,
                    activation_dtype=args.activation_dtype,
                    symmetric=not args.asymmetric,
                    group_size=group_size,
                    fp8_scheme=args.fp8_scheme,
                    targets=args.targets,
                    ignore=args.ignore,
                )
            ]
        if args.pack_format == "humming" and any(
            modifier.weight_quant().is_fp4 for modifier in modifiers
        ):
            parser.error(
                "FP4 only supports pack_format='compressed-tensors'; "
                "Humming consumes the portable checkpoint at runtime"
            )
        with Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            expand=True,
        ) as progress:
            task_id = progress.add_task("Loading checkpoint", total=None)

            def update_progress(completed: int, total: int, _module_name: str) -> None:
                progress.update(
                    task_id,
                    description="Quantizing weights",
                    completed=completed,
                    total=total,
                )

            model_free_ptq(
                args.input,
                modifiers,
                args.output,
                pack_format=args.pack_format,
                progress_callback=update_progress,
            )
            progress.update(task_id, description="Checkpoint ready")

        message = Text("Done", style="bold green")
        message.append("  Quantized checkpoint written to ")
        message.append(args.output, style="bold cyan")
        console.print(message)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
