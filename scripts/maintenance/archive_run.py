#!/usr/bin/env python3
"""Move one completed run directory into an archive without copying it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def archive_run(run_directory: Path, archive_root: Path) -> Path:
    run_directory = run_directory.resolve(strict=True)
    if not run_directory.is_dir() or run_directory.is_symlink():
        raise ValueError("run_directory must be a real directory")
    terminal_markers = [name for name in ("SUCCESS", "FAILED") if (run_directory / name).is_file()]
    if len(terminal_markers) != 1:
        raise ValueError("run directory must contain exactly one SUCCESS or FAILED marker")

    archive_root.mkdir(parents=True, exist_ok=True)
    archive_root = archive_root.resolve(strict=True)
    destination = archive_root / run_directory.name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"archive destination already exists: {destination}")
    run_directory.replace(destination)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("archive_root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    destination = archive_run(args.run_directory, args.archive_root)
    print(json.dumps({"archived_to": str(destination)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
