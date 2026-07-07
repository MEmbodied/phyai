"""Enable ``python -m phyai.cli`` alongside the ``phyai`` console script."""

from __future__ import annotations

from phyai.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
