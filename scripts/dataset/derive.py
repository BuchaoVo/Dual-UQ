"""Compatibility wrapper for ``dual-uq dataset derive``."""

import sys

from dual_uq.cli import main as dual_uq_main


def main() -> int:
    return dual_uq_main(["dataset", "derive", *sys.argv[1:]])

if __name__ == "__main__":
    raise SystemExit(main())
