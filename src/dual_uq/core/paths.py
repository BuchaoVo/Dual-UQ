"""Central, portable filesystem resolution for Dual-UQ."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


class ProjectPathError(ValueError):
    """A project root or logical path could not be resolved safely."""


def _is_repository_root(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() and (path / "src/dual_uq").is_dir()


def _absolute(path: Path, *, label: str) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        raise ProjectPathError(f"{label} must be an explicit absolute path")
    return expanded.resolve()


def _repository_from_anchor(anchor: Path) -> Path | None:
    current = anchor.expanduser().resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if _is_repository_root(candidate):
            return candidate
    return None


def _runtime_root(
    explicit: Path | None,
    *,
    environment_name: str,
    default: Path,
) -> Path:
    value = explicit
    if value is None and (environment_value := os.environ.get(environment_name)):
        value = Path(environment_value)
    if value is None:
        return default.resolve()
    if value.is_absolute():
        return value.expanduser().resolve()
    return (default.parent / value).resolve()


def _portable_parts(value: str) -> tuple[str, ...]:
    if "\\" in value:
        raise ProjectPathError("logical references must use portable POSIX separators")
    logical = PurePosixPath(value)
    if logical.is_absolute():
        raise ProjectPathError("logical references must be relative")
    if not logical.parts or any(part in {"", ".", ".."} for part in logical.parts):
        raise ProjectPathError("logical reference contains parent traversal or empty parts")
    return logical.parts


@dataclass(frozen=True)
class ProjectPaths:
    """Physical roots plus portable logical-reference conversion."""

    repository_root: Path
    data_root: Path
    raw_root: Path
    processed_root: Path
    reports_root: Path
    runs_root: Path
    artifacts_root: Path

    @classmethod
    def discover(
        cls,
        *,
        project_root: Path | None = None,
        anchor: Path | None = None,
        data_root: Path | None = None,
        reports_root: Path | None = None,
        runs_root: Path | None = None,
        artifacts_root: Path | None = None,
    ) -> ProjectPaths:
        """Resolve roots without consulting the current working directory implicitly."""
        root: Path | None = None
        if project_root is not None:
            root = _absolute(project_root, label="project_root")
            if not _is_repository_root(root):
                raise ProjectPathError(f"repository root markers are missing: {root}")
        elif environment_root := os.environ.get("DUAL_UQ_PROJECT_ROOT"):
            root = _absolute(Path(environment_root), label="DUAL_UQ_PROJECT_ROOT")
            if not _is_repository_root(root):
                raise ProjectPathError(f"repository root markers are missing: {root}")
        else:
            if anchor is not None:
                root = _repository_from_anchor(anchor)
            if root is None:
                root = _repository_from_anchor(Path(__file__))
            if root is None:
                raise ProjectPathError("repository_root_unresolved")

        resolved_data = _runtime_root(
            data_root,
            environment_name="DUAL_UQ_DATA_ROOT",
            default=root / "data",
        )
        resolved_reports = _runtime_root(
            reports_root,
            environment_name="DUAL_UQ_REPORTS_ROOT",
            default=root / "reports",
        )
        resolved_runs = _runtime_root(
            runs_root,
            environment_name="DUAL_UQ_RUNS_ROOT",
            default=root / "runs",
        )
        resolved_artifacts = _runtime_root(
            artifacts_root,
            environment_name="DUAL_UQ_ARTIFACTS_ROOT",
            default=root / "artifacts",
        )
        return cls(
            repository_root=root,
            data_root=resolved_data,
            raw_root=resolved_data / "raw",
            processed_root=resolved_data / "processed",
            reports_root=resolved_reports,
            runs_root=resolved_runs,
            artifacts_root=resolved_artifacts,
        )

    def resolve_logical(self, logical_ref: str) -> Path:
        """Resolve one portable logical reference to its configured physical root."""
        parts = _portable_parts(logical_ref)
        roots = {
            "data": self.data_root,
            "reports": self.reports_root,
            "runs": self.runs_root,
            "artifacts": self.artifacts_root,
        }
        if parts[0] in roots:
            return roots[parts[0]].joinpath(*parts[1:]).resolve()
        return self.repository_root.joinpath(*parts).resolve()

    def logical_ref(self, path: Path) -> str:
        """Return a portable logical reference without serializing an absolute path."""
        resolved = path.expanduser().resolve()
        roots = (
            ("data", self.data_root),
            ("reports", self.reports_root),
            ("runs", self.runs_root),
            ("artifacts", self.artifacts_root),
        )
        for namespace, root in roots:
            try:
                relative = resolved.relative_to(root.resolve())
            except ValueError:
                continue
            return PurePosixPath(namespace, *relative.parts).as_posix()
        try:
            relative = resolved.relative_to(self.repository_root.resolve())
        except ValueError as exc:
            raise ProjectPathError(
                f"path is outside declared logical roots: {resolved.name}"
            ) from exc
        return PurePosixPath(*relative.parts).as_posix()
