"""Portable path projection for dataset-domain resources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dual_uq.core.paths import ProjectPaths


@dataclass(frozen=True)
class DatasetPaths:
    """Canonical dataset data paths derived from configured project roots."""

    root: Path
    sources: Path
    working: Path
    manifests: Path
    fixtures: Path
    releases: Path
    runs: Path
    artifacts: Path
    reports: Path
    audits: Path

    @classmethod
    def from_project(cls, project: ProjectPaths) -> DatasetPaths:
        """Project the configured data root without creating any directories."""
        root = project.data_root / "dataset"
        artifacts = project.artifacts_root / "dataset"
        return cls(
            root=root,
            sources=root / "sources",
            working=root / "working",
            manifests=root / "manifests",
            fixtures=root / "fixtures",
            releases=root / "releases",
            runs=project.runs_root / "dataset",
            artifacts=artifacts,
            reports=artifacts / "reports",
            audits=artifacts / "audits",
        )
