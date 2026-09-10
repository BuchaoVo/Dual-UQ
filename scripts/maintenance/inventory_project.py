#!/usr/bin/env python3
"""Create a deterministic, local-only project filesystem inventory."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
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


def write_tree_stats_atomic(project_root: Path, path: Path) -> dict[str, int]:
    """Write a deterministic inventory containing every filesystem entry."""

    root = project_root.resolve(strict=True)
    destination = path.resolve()
    try:
        self_path = destination.relative_to(root).as_posix()
    except ValueError:
        self_path = None
    destination.parent.mkdir(parents=True, exist_ok=True)
    entries: list[tuple[str, int, str]] = [("d", root.lstat().st_size, ".")]
    for current, directories, files in os.walk(root, followlinks=False):
        directories.sort()
        files.sort()
        current_path = Path(current)
        for name in directories + files:
            item = current_path / name
            relative = item.relative_to(root).as_posix()
            kind = (
                "l"
                if item.is_symlink()
                else "d"
                if item.is_dir()
                else "f"
                if item.is_file()
                else "o"
            )
            entries.append((kind, item.lstat().st_size, relative))
    if self_path is not None and not any(relative == self_path for _, _, relative in entries):
        entries.append(("f", 0, self_path))
    entries.sort(key=lambda item: item[2])

    directory_count = sum(kind == "d" for kind, _, _ in entries)
    regular_file_count = sum(kind == "f" for kind, _, _ in entries)
    symlink_count = sum(kind == "l" for kind, _, _ in entries)
    other_count = sum(kind == "o" for kind, _, _ in entries)
    regular_bytes = sum(
        size for kind, size, relative in entries if kind == "f" and relative != self_path
    )
    git_files = sum(
        kind == "f" and relative.startswith(".git/") for kind, _, relative in entries
    )
    top_level: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for kind, size, relative in entries:
        if kind != "f" or relative == ".":
            continue
        name = relative.split("/", 1)[0]
        top_level[name][0] += 1
        if relative != self_path:
            top_level[name][1] += size

    lines = [
        "PROJECT_FILE_TREE_STATISTICS",
        f"root\t{root}",
        f"generated_utc\t{datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "scope\tAll entries under the project root, including hidden files, Git metadata, runs, artifacts, models, and third_party. Symlinks are listed but not followed.",
        "self_size_policy\tThe report entry uses SELF; aggregate bytes exclude this report to avoid recursive size instability.",
        "",
        "SUMMARY",
        f"directories\t{directory_count}",
        f"regular_files\t{regular_file_count}",
        f"symlinks\t{symlink_count}",
        f"other_entries\t{other_count}",
        f"regular_file_bytes_excluding_this_report\t{regular_bytes}",
        f"git_internal_regular_files\t{git_files}",
        "",
        "TOP_LEVEL_REGULAR_FILES",
        "top_level\tfile_count\tbytes_excluding_this_report",
    ]
    lines.extend(
        f"{name}\t{values[0]}\t{values[1]}" for name, values in sorted(top_level.items())
    )
    lines.extend(["", "ALL_ENTRIES", "type\tsize_bytes\tpath"])
    lines.extend(
        f"{kind}\t{'SELF' if relative == self_path else size}\t./{relative}"
        for kind, size, relative in entries
    )
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return {
        "directories": directory_count,
        "regular_files": regular_file_count,
        "symlinks": symlink_count,
        "regular_file_bytes": regular_bytes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("json", "tree-stats"), default="json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.format == "tree-stats":
        summary = write_tree_stats_atomic(args.project_root, args.output)
        print(json.dumps({"output": str(args.output), **summary}))
        return 0
    inventory = build_inventory(args.project_root, max_depth=args.max_depth)
    write_json_atomic(args.output, inventory)
    print(json.dumps({"output": str(args.output), "file_count": inventory["file_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
