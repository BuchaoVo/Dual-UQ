"""Portable path projection for dataset-domain resources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from dual_uq.core.paths import ProjectPaths


CANONICAL_EXPERIMENT_REFS: Final = MappingProxyType(
    {
        "sampling_frame": "experiments/dataset/construction/sampling_frame",
        "admission": "experiments/dataset/construction/admission",
        "full_frame": "experiments/dataset/construction/full_frame",
        "redundancy": "experiments/dataset/construction/redundancy",
        "expansion_planning": "experiments/dataset/construction/expansion/planning",
        "expansion_wave_1": "experiments/dataset/construction/expansion/wave_1",
        "expansion_wave_2": "experiments/dataset/construction/expansion/wave_2",
        "cohort_release": "experiments/dataset/releases/cohort",
        "scoring_protocol_release": "experiments/dataset/releases/scoring_protocol",
        "confirmatory_release": "experiments/dataset/releases/confirmatory",
        "structural_response": "experiments/dataset/analysis/structural_response",
        "pair_validity": "experiments/dataset/analysis/pair_validity",
    }
)

# These are deliberately descriptive metadata rather than filesystem aliases.
# Existing paths remain readable for historical manifests and frozen artifacts;
# new writes must use the semantic refs above.
HISTORICAL_EXPERIMENT_RELOCATION: Final = (
    {
        "legacy": "experiments/p2_design_baseline/scale1/scale1a0",
        "canonical": CANONICAL_EXPERIMENT_REFS["sampling_frame"],
        "status": "historical_read_only",
        "write_policy": "canonical_only",
    },
    {
        "legacy": "experiments/p2_design_baseline/scale1/scale1a1",
        "canonical": CANONICAL_EXPERIMENT_REFS["admission"],
        "status": "historical_read_only",
        "write_policy": "canonical_only",
    },
    {
        "legacy": "experiments/p2_design_baseline/scale1/scale1a2",
        "canonical": CANONICAL_EXPERIMENT_REFS["full_frame"],
        "status": "historical_read_only",
        "write_policy": "canonical_only",
    },
    {
        "legacy": "experiments/p2_design_baseline/scale1/scale1a3",
        "canonical": CANONICAL_EXPERIMENT_REFS["redundancy"],
        "status": "historical_read_only",
        "write_policy": "canonical_only",
    },
    {
        "legacy": "experiments/p2_design_baseline/scale1/expansion_design",
        "canonical": CANONICAL_EXPERIMENT_REFS["expansion_planning"],
        "status": "historical_read_only",
        "write_policy": "canonical_only",
    },
    {
        "legacy": "experiments/p2_design_baseline/scale1/expansion_wave1",
        "canonical": CANONICAL_EXPERIMENT_REFS["expansion_wave_1"],
        "status": "historical_read_only",
        "write_policy": "canonical_only",
    },
    {
        "legacy": "experiments/p2_design_baseline/scale1/expansion_wave2",
        "canonical": CANONICAL_EXPERIMENT_REFS["expansion_wave_2"],
        "status": "historical_read_only",
        "write_policy": "canonical_only",
    },
    {
        "legacy": "experiments/p2_design_baseline/scale1/scale1b_freeze",
        "canonical": CANONICAL_EXPERIMENT_REFS["cohort_release"],
        "status": "historical_read_only",
        "write_policy": "canonical_only",
    },
    {
        "legacy": "experiments/p2_design_baseline/scale1/scale1b_protocol",
        "canonical": CANONICAL_EXPERIMENT_REFS["scoring_protocol_release"],
        "status": "historical_read_only",
        "write_policy": "canonical_only",
    },
    {
        "legacy": "experiments/p2_design_baseline/scale1/scale1b_v2",
        "canonical": CANONICAL_EXPERIMENT_REFS["confirmatory_release"],
        "status": "historical_read_only",
        "write_policy": "canonical_only",
    },
    {
        "legacy": "experiments/p2_design_baseline/scale1b-v2/structural_response",
        "canonical": CANONICAL_EXPERIMENT_REFS["structural_response"],
        "status": "historical_read_only",
        "write_policy": "canonical_only",
    },
    {
        "legacy": "experiments/p2_design_baseline/scale1b-v2/pair_validity",
        "canonical": CANONICAL_EXPERIMENT_REFS["pair_validity"],
        "status": "historical_read_only",
        "write_policy": "canonical_only",
    },
)


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


@dataclass(frozen=True)
class DatasetExperimentPaths:
    """Semantic experiment paths for Dataset construction and evaluation.

    The paths are projections only: constructing this object never creates
    directories and never moves historical artifacts.  Frozen historical
    inputs remain bound to their recorded paths; active writers use these
    semantic locations instead of development-stage labels.
    """

    root: Path

    @classmethod
    def from_project(cls, project: ProjectPaths) -> "DatasetExperimentPaths":
        return cls(project.repository_root / "experiments/dataset")

    @property
    def construction(self) -> Path:
        return self.root / "construction"

    @property
    def sampling_frame(self) -> Path:
        return self.construction / "sampling_frame"

    @property
    def admission(self) -> Path:
        return self.construction / "admission"

    @property
    def full_frame(self) -> Path:
        return self.construction / "full_frame"

    @property
    def redundancy(self) -> Path:
        return self.construction / "redundancy"

    @property
    def expansion_planning(self) -> Path:
        return self.construction / "expansion/planning"

    @property
    def expansion_wave_1(self) -> Path:
        return self.construction / "expansion/wave_1"

    @property
    def expansion_wave_2(self) -> Path:
        return self.construction / "expansion/wave_2"

    @property
    def releases(self) -> Path:
        return self.root / "releases"

    @property
    def cohort_release(self) -> Path:
        return self.releases / "cohort"

    @property
    def scoring_protocol_release(self) -> Path:
        return self.releases / "scoring_protocol"

    @property
    def confirmatory_release(self) -> Path:
        return self.releases / "confirmatory"

    @property
    def analysis(self) -> Path:
        return self.root / "analysis"

    @property
    def structural_response(self) -> Path:
        return self.analysis / "structural_response"

    @property
    def pair_validity(self) -> Path:
        return self.analysis / "pair_validity"

    def logical_ref(self, name: str) -> str:
        """Return a canonical repository-relative ref by semantic name."""
        try:
            return CANONICAL_EXPERIMENT_REFS[name]
        except KeyError as exc:
            raise KeyError(f"unknown dataset experiment path: {name}") from exc
