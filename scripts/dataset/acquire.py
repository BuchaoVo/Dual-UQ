"""Thin Dataset acquisition command dispatcher."""

from __future__ import annotations

import sys
from collections.abc import Callable

from dual_uq.dataset.stages import acquisition, acquisition_plan, dependent_assets

COMMANDS: dict[str, Callable[[], int | None]] = {
    "plan": acquisition_plan.main,
    "run": acquisition.main,
    "complete-dependents": dependent_assets.main,
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        choices = ", ".join(sorted(COMMANDS))
        print(f"usage: acquire.py {{{choices}}} [options]", file=sys.stderr)
        return 2
    command = sys.argv.pop(1)
    result = COMMANDS[command]()
    return 0 if result is None else result


if __name__ == "__main__":
    raise SystemExit(main())
