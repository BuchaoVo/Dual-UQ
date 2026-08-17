from __future__ import annotations

from pathlib import Path

import yaml

from dual_uq.core.paths import ProjectPaths
from dual_uq.dataset.paths import (
    CANONICAL_EXPERIMENT_REFS,
    DatasetExperimentPaths,
    HISTORICAL_EXPERIMENT_RELOCATION,
)


def _repository(root: Path) -> Path:
    (root / "src/dual_uq").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        "[project]\nname='fixture'\n", encoding="utf-8"
    )
    return root


def test_dataset_experiment_paths_are_semantic_and_repository_relative(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path / "clone")
    project = ProjectPaths.discover(project_root=repository)

    paths = DatasetExperimentPaths.from_project(project)

    assert paths.root == repository / "experiments/dataset"
    assert paths.construction == repository / "experiments/dataset/construction"
    assert paths.sampling_frame == paths.construction / "sampling_frame"
    assert paths.admission == paths.construction / "admission"
    assert paths.full_frame == paths.construction / "full_frame"
    assert paths.redundancy == paths.construction / "redundancy"
    assert paths.expansion_planning == paths.construction / "expansion/planning"
    assert paths.expansion_wave_1 == paths.construction / "expansion/wave_1"
    assert paths.expansion_wave_2 == paths.construction / "expansion/wave_2"
    assert paths.cohort_release == repository / "experiments/dataset/releases/cohort"
    assert paths.scoring_protocol_release == (
        repository / "experiments/dataset/releases/scoring_protocol"
    )
    assert paths.confirmatory_release == (
        repository / "experiments/dataset/releases/confirmatory"
    )
    assert paths.structural_response == (
        repository / "experiments/dataset/analysis/structural_response"
    )
    assert paths.pair_validity == (
        repository / "experiments/dataset/analysis/pair_validity"
    )
    assert not paths.root.exists()


def test_semantic_refs_and_relocation_map_are_deterministic() -> None:
    assert set(CANONICAL_EXPERIMENT_REFS) == {
        "sampling_frame",
        "admission",
        "full_frame",
        "redundancy",
        "expansion_planning",
        "expansion_wave_1",
        "expansion_wave_2",
        "cohort_release",
        "scoring_protocol_release",
        "confirmatory_release",
        "structural_response",
        "pair_validity",
    }
    assert all(ref.startswith("experiments/dataset/") for ref in CANONICAL_EXPERIMENT_REFS.values())
    assert all(
        token not in ref
        for ref in CANONICAL_EXPERIMENT_REFS.values()
        for token in ("scale1", "scale1a", "scale1b", "v1", "v2")
    )
    assert all(row["status"] == "historical_read_only" for row in HISTORICAL_EXPERIMENT_RELOCATION)
    assert all(row["write_policy"] == "canonical_only" for row in HISTORICAL_EXPERIMENT_RELOCATION)
    assert {row["legacy"] for row in HISTORICAL_EXPERIMENT_RELOCATION} == {
        "experiments/p2_design_baseline/scale1/scale1a0",
        "experiments/p2_design_baseline/scale1/scale1a1",
        "experiments/p2_design_baseline/scale1/scale1a2",
        "experiments/p2_design_baseline/scale1/scale1a3",
        "experiments/p2_design_baseline/scale1/expansion_design",
        "experiments/p2_design_baseline/scale1/expansion_wave1",
        "experiments/p2_design_baseline/scale1/expansion_wave2",
        "experiments/p2_design_baseline/scale1/scale1b_freeze",
        "experiments/p2_design_baseline/scale1/scale1b_protocol",
        "experiments/p2_design_baseline/scale1/scale1b_v2",
        "experiments/p2_design_baseline/scale1b-v2/structural_response",
        "experiments/p2_design_baseline/scale1b-v2/pair_validity",
    }


def test_repository_path_map_matches_python_contract() -> None:
    path = Path(__file__).resolve().parents[3] / "configs/dataset/experiment_path_map.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "dataset.experiment-path-map.v1"
    assert payload["canonical_paths"] == dict(CANONICAL_EXPERIMENT_REFS)
    assert payload["historical_paths"] == list(HISTORICAL_EXPERIMENT_RELOCATION)
