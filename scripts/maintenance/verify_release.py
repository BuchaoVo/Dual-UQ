#!/usr/bin/env python3
"""Verify a curated release manifest against local SHA-256 digests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dual_uq.core.hashing import sha256_file


def verify_release(manifest_path: Path, release_root: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise TypeError("release manifest must contain a files list")

    root = release_root.resolve(strict=True)
    failures: list[dict[str, str]] = []
    checked_file_count = 0
    for entry in entries:
        if not isinstance(entry, dict):
            failures.append({"code": "invalid_entry", "path": ""})
            continue
        relative_text = entry.get("path")
        expected = entry.get("sha256")
        if not isinstance(relative_text, str) or not isinstance(expected, str):
            failures.append({"code": "invalid_entry", "path": str(relative_text or "")})
            continue
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            failures.append({"code": "unsafe_path", "path": relative_text})
            continue
        target = root / relative
        if not target.is_file():
            failures.append({"code": "missing_file", "path": relative_text})
            continue
        checked_file_count += 1
        actual = sha256_file(target)
        if actual != expected.lower():
            failures.append(
                {
                    "code": "sha256_mismatch",
                    "path": relative_text,
                    "expected": expected.lower(),
                    "actual": actual,
                }
            )

    return {
        "ok": not failures,
        "manifest": str(manifest_path),
        "release_root": str(root),
        "declared_file_count": len(entries),
        "checked_file_count": checked_file_count,
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--release-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.release_root or args.manifest.parent
    report = verify_release(args.manifest, root)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
