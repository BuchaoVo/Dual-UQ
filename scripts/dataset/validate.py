"""Thin Dataset evidence-audit command dispatcher."""

from __future__ import annotations

import sys
from collections.abc import Callable

from dual_uq.dataset.audits import (
    acquisition_identity,
    fragments,
    metadata,
    rejected_metadata,
    variant_recovery,
)

COMMANDS: dict[str, Callable[[], int | None]] = {
    "acquisition-identity": acquisition_identity.main,
    "fragment-evidence": fragments.main,
    "metadata-records": metadata.main,
    "rejected-metadata": rejected_metadata.main,
    "variant-recovery": variant_recovery.main,
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        choices = ", ".join(sorted(COMMANDS))
        print(f"usage: validate.py {{{choices}}} [options]", file=sys.stderr)
        return 2
    command = sys.argv.pop(1)
    result = COMMANDS[command]()
    return 0 if result is None else result


if __name__ == "__main__":
    raise SystemExit(main())
