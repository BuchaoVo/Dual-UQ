"""Build frozen Stage-0 common masks and backbone-independent fixed probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dual_uq.core.paths import ProjectPaths
from dual_uq.dataset.fixed_probes import materialize_fixed_probes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = ProjectPaths.discover(
        project_root=args.project_root.resolve() if args.project_root else None,
        anchor=Path(__file__),
    )
    config = args.config
    if not config.is_absolute():
        config = paths.repository_root / config
    result = materialize_fixed_probes(config.resolve(), paths.repository_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
