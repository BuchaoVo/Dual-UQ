"""Compatibility wrapper for ``dual-uq dataset census``."""

from __future__ import annotations

import sys

from dual_uq.cli import main as dual_uq_main


def main() -> int:
    return dual_uq_main(["dataset", "census", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
