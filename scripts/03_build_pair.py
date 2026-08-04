"""Compatibility wrapper for manifest-driven Dataset resolution."""

from __future__ import annotations

import sys

from dual_uq.cli import main as dual_uq_main


def main() -> int:
    return dual_uq_main(["dataset", "resolve", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
