"""Deterministic pre-migration inventory of legacy dataset namespaces."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

_NAMESPACE_PATTERN = re.compile(r"dataset_a_scale|dataset_a|data_a", re.IGNORECASE)
_IMPORT_PATTERN = re.compile(
    r"dual_uq\.dataset_a_scale|scripts/dataset_a|configs/dataset_a|runs/dataset_a"
)
_REFERENCE_ROOTS = ("src", "scripts", "tests", "configs", "docs")
_REFERENCE_FILES = ("Makefile", "pyproject.toml", "README.md")
_TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
_PRUNED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}
_TRACKED_LEGACY_ROOTS = (
    "reports/dataset_a_census",
    "reports/dataset_a_scale",
    "runs/dataset_a",
    "artifacts/audits/dataset_a",
    "artifacts/reports/dataset_a",
)


def _relative(root: Path, path: Path) -> str:
    return PurePosixPath(*path.relative_to(root).parts).as_posix()


def _run_git(root: Path, *args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ("git", *args),
        cwd=root,
        check=True,
        capture_output=True,
        text=not binary,
    )
    return result.stdout


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _namespace_paths(root: Path) -> tuple[str, ...]:
    matches: list[str] = []
    for directory, names, files in os.walk(root):
        names[:] = sorted(name for name in names if name not in _PRUNED_DIRECTORIES)
        current = Path(directory)
        for name in (*names, *sorted(files)):
            if _NAMESPACE_PATTERN.search(name):
                matches.append(_relative(root, current / name))
    return tuple(sorted(set(matches)))


def _reference_files(root: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    for name in _REFERENCE_ROOTS:
        base = root / name
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if any(part in _PRUNED_DIRECTORIES for part in path.parts):
                continue
            if path.is_file() and path.suffix.lower() in _TEXT_SUFFIXES:
                files.append(path)
    files.extend(
        path for name in _REFERENCE_FILES if (path := root / name).is_file()
    )
    return tuple(sorted(set(files), key=lambda path: _relative(root, path)))


def _references(root: Path, pattern: re.Pattern[str]) -> tuple[str, ...]:
    matches: list[str] = []
    for path in _reference_files(root):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            continue
        relative = _relative(root, path)
        matches.extend(
            f"{relative}:{line_number}:{line}"
            for line_number, line in enumerate(lines, start=1)
            if pattern.search(line)
        )
    return tuple(sorted(matches))


def _dirty_paths(root: Path) -> tuple[str, ...]:
    payload = _run_git(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        binary=True,
    )
    assert isinstance(payload, bytes)
    records = payload.split(b"\0")
    paths: list[str] = []
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        status = record[:2].decode("ascii")
        paths.append(record[3:].decode("utf-8", errors="surrogateescape"))
        if ("R" in status or "C" in status) and index < len(records) and records[index]:
            paths.append(records[index].decode("utf-8", errors="surrogateescape"))
            index += 1
    return tuple(sorted(set(paths)))


def _tracked_legacy_files(root: Path) -> tuple[dict[str, Any], ...]:
    output = _run_git(root, "ls-files", "--", *_TRACKED_LEGACY_ROOTS)
    assert isinstance(output, str)
    rows = []
    for relative in sorted(line for line in output.splitlines() if line):
        path = root / relative
        rows.append(
            {
                "path": PurePosixPath(relative).as_posix(),
                "byte_count": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return tuple(rows)


@dataclass(frozen=True)
class NamespaceInventory:
    git_head: str
    git_branch: str
    namespace_paths: tuple[str, ...]
    import_references: tuple[str, ...]
    path_references: tuple[str, ...]
    preexisting_dirty_paths: tuple[str, ...]
    tracked_legacy_files: tuple[dict[str, Any], ...]
    test_counts: Mapping[str, int]

    def to_json(self) -> dict[str, Any]:
        counts = {
            "namespace_paths": len(self.namespace_paths),
            "import_references": len(self.import_references),
            "path_references": len(self.path_references),
            "preexisting_dirty_paths": len(self.preexisting_dirty_paths),
            "tracked_legacy_files": len(self.tracked_legacy_files),
        }
        return {
            "schema_version": "dataset.namespace-migration-baseline.v1",
            "git": {
                "head": self.git_head,
                "branch": self.git_branch,
                "preexisting_dirty_paths": list(self.preexisting_dirty_paths),
            },
            "test_counts": dict(sorted(self.test_counts.items())),
            "counts": counts,
            "tracked_legacy_files": list(self.tracked_legacy_files),
            "scientific_state_changed": False,
        }


def capture_namespace_inventory(
    project_root: Path,
    *,
    test_counts: Mapping[str, int],
    migration_owned_paths: Sequence[str] = (),
) -> NamespaceInventory:
    """Capture legacy namespace, reference, dirty-state and hash baselines."""
    root = project_root.expanduser().resolve()
    if not (root / ".git").exists() or not (root / "pyproject.toml").is_file():
        raise ValueError("project_root must identify a Git project with pyproject.toml")
    if any(isinstance(value, bool) or int(value) < 0 for value in test_counts.values()):
        raise ValueError("test counts must be non-negative integers")
    owned = {PurePosixPath(path).as_posix() for path in migration_owned_paths}
    dirty = tuple(path for path in _dirty_paths(root) if path not in owned)
    head = _run_git(root, "rev-parse", "HEAD")
    branch = _run_git(root, "branch", "--show-current")
    assert isinstance(head, str) and isinstance(branch, str)
    return NamespaceInventory(
        git_head=head.strip(),
        git_branch=branch.strip(),
        namespace_paths=_namespace_paths(root),
        import_references=_references(root, _IMPORT_PATTERN),
        path_references=_references(root, _NAMESPACE_PATTERN),
        preexisting_dirty_paths=dirty,
        tracked_legacy_files=_tracked_legacy_files(root),
        test_counts={key: int(value) for key, value in test_counts.items()},
    )


def _immutable_atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"refusing to overwrite conflicting inventory: {path.name}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _text_payload(rows: Sequence[str]) -> bytes:
    return (("\n".join(rows) + "\n") if rows else "").encode("utf-8")


def write_namespace_inventory(
    inventory: NamespaceInventory, output_dir: Path
) -> dict[str, Path]:
    """Write the four immutable Phase-0 inventory views."""
    outputs = {
        "namespace": output_dir / "dataset_namespace_inventory_before.txt",
        "imports": output_dir / "import_references_before.txt",
        "paths": output_dir / "path_references_before.txt",
        "baseline": output_dir / "migration_baseline.json",
    }
    payloads = {
        "namespace": _text_payload(inventory.namespace_paths),
        "imports": _text_payload(inventory.import_references),
        "paths": _text_payload(inventory.path_references),
        "baseline": (
            json.dumps(
                inventory.to_json(), indent=2, sort_keys=True, ensure_ascii=False
            )
            + "\n"
        ).encode("utf-8"),
    }
    for name, path in outputs.items():
        _immutable_atomic_write(path, payloads[name])
    return outputs
