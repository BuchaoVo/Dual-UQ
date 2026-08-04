#!/usr/bin/env python3
"""Create a deterministic, local-only project filesystem inventory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

EXCLUDED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}


def build_inventory(project_root: Path, *, max_depth: int = 5) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    directories: list[str] = []
    files: list[dict[str, Any]] = []
    total_file_bytes = 0

    for current, child_directories, child_files in os.walk(root):
        current_path = Path(current)
        relative = current_path.relative_to(root)
        depth = len(relative.parts)
        child_directories[:] = sorted(
            name for name in child_directories if name not in EXCLUDED_DIRECTORIES
        )
        if depth >= max_depth:
            child_directories[:] = []
        for name in child_directories:
            path = current_path / name
            relative_path = path.relative_to(root).as_posix()
            directories.append(relative_path)
        for name in sorted(child_files):
            path = current_path / name
            relative_path = path.relative_to(root).as_posix()
            size = path.lstat().st_size
            files.append(
                {
                    "path": relative_path,
                    "size_bytes": size,
                    "is_symlink": path.is_symlink(),
                }
            )
            total_file_bytes += size

    return {
        "project_root": str(root),
        "max_depth": max_depth,
        "directory_count": len(directories),
        "file_count": len(files),
        "total_file_bytes": total_file_bytes,
        "directories": sorted(directories),
        "files": sorted(files, key=lambda item: item["path"]),
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inventory = build_inventory(args.project_root, max_depth=args.max_depth)
    write_json_atomic(args.output, inventory)
    print(json.dumps({"output": str(args.output), "file_count": inventory["file_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
