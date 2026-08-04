from __future__ import annotations

from pathlib import Path

import pytest

from dual_uq.core.hashing import sha256_file
from dual_uq.core.paths import ProjectPaths
from dual_uq.dataset.paths import DatasetPaths
from dual_uq.dataset.pipeline import DatasetTaskError, PlanningOptions, plan_tasks
from dual_uq.dataset.stages.manifest import build_manifest_stage_handler


def _paths(tmp_path: Path) -> ProjectPaths:
    (tmp_path / "src/dual_uq").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    return ProjectPaths.discover(project_root=tmp_path)


@pytest.mark.parametrize(
    "stage",
    ["census", "resolution", "acquisition", "derivation", "validation", "release"],
)
def test_every_canonical_stage_has_a_manifest_adapter(
    tmp_path: Path, stage: str
) -> None:
    project = _paths(tmp_path)

    handler = build_manifest_stage_handler(
        stage, project=project, dataset_paths=DatasetPaths.from_project(project)
    )

    assert callable(handler)


def test_validation_adapter_checks_portable_resource_hashes(tmp_path: Path) -> None:
    project = _paths(tmp_path)
    logical_path = "artifacts/dataset/fixture.txt"
    path = project.resolve_logical(logical_path)
    path.parent.mkdir(parents=True)
    path.write_text("fixture\n")
    record = {
        "record_id": "protein-001",
        "resources": [
            {
                "logical_path": logical_path,
                "sha256": sha256_file(path),
            }
        ],
        "disposition": "eligible",
    }
    task = plan_tasks(
        [record],
        PlanningOptions(stage="validation", stage_version="v1", base_seed=1),
    )[0]
    handler = build_manifest_stage_handler(
        "validation",
        project=project,
        dataset_paths=DatasetPaths.from_project(project),
    )

    output = handler(task)

    assert output.metrics == {"checked_resource_count": 1}
    assert output.scientific_disposition == "eligible"


def test_validation_adapter_rejects_corrupt_resource(tmp_path: Path) -> None:
    project = _paths(tmp_path)
    logical_path = "artifacts/dataset/fixture.txt"
    path = project.resolve_logical(logical_path)
    path.parent.mkdir(parents=True)
    path.write_text("corrupt\n")
    task = plan_tasks(
        [
            {
                "record_id": "protein-001",
                "resources": [{"logical_path": logical_path, "sha256": "a" * 64}],
            }
        ],
        PlanningOptions(stage="validation", stage_version="v1", base_seed=1),
    )[0]
    handler = build_manifest_stage_handler(
        "validation",
        project=project,
        dataset_paths=DatasetPaths.from_project(project),
    )

    with pytest.raises(DatasetTaskError) as error:
        handler(task)

    assert error.value.failure.code == "resource_hash_mismatch"


def test_release_adapter_preserves_dataset_disposition(tmp_path: Path) -> None:
    project = _paths(tmp_path)
    task = plan_tasks(
        [
            {
                "record_id": "protein-001",
                "disposition": "eligible",
                "release_id": "dataset-a-dev-v1",
            }
        ],
        PlanningOptions(stage="release", stage_version="v1", base_seed=1),
    )[0]
    handler = build_manifest_stage_handler(
        "release",
        project=project,
        dataset_paths=DatasetPaths.from_project(project),
    )

    output = handler(task)

    assert output.scientific_disposition == "eligible"
    assert output.metrics["release_id"] == "dataset-a-dev-v1"
