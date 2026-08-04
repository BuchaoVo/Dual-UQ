from __future__ import annotations

import os
from pathlib import Path

import pytest

from dual_uq.core.paths import ProjectPathError, ProjectPaths


def _repository(root: Path) -> Path:
    (root / "src/dual_uq").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    return root


def test_explicit_repository_root_uses_portable_defaults(tmp_path: Path) -> None:
    root = _repository(tmp_path / "clone")

    paths = ProjectPaths.discover(project_root=root)

    assert paths.repository_root == root.resolve()
    assert paths.data_root == root.resolve() / "data"
    assert paths.raw_root == root.resolve() / "data/raw"
    assert paths.processed_root == root.resolve() / "data/processed"
    assert paths.reports_root == root.resolve() / "reports"
    assert paths.runs_root == root.resolve() / "runs"
    assert paths.artifacts_root == root.resolve() / "artifacts"


def test_runtime_roots_can_be_overridden_without_changing_logical_refs(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path / "clone")
    data = tmp_path / "runtime-data"
    reports = tmp_path / "runtime-reports"
    runs = tmp_path / "runtime-runs"
    artifacts = tmp_path / "runtime-artifacts"
    paths = ProjectPaths.discover(
        project_root=root,
        data_root=data,
        reports_root=reports,
        runs_root=runs,
        artifacts_root=artifacts,
    )
    model = data / "raw/afdb/P12345/model.cif"

    assert paths.resolve_logical("data/raw/afdb/P12345/model.cif") == model.resolve()
    assert paths.logical_ref(model) == "data/raw/afdb/P12345/model.cif"
    assert paths.logical_ref(reports / "pilot.json") == "reports/pilot.json"
    assert paths.logical_ref(runs / "run-1/status.json") == "runs/run-1/status.json"
    assert paths.logical_ref(artifacts / "audit.json") == "artifacts/audit.json"


def test_discovery_is_cwd_independent_and_uses_nearest_anchor_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repository(tmp_path / "clone")
    anchor = root / "scripts/dataset/entrypoint.py"
    anchor.parent.mkdir(parents=True)
    anchor.write_text("", encoding="utf-8")
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)

    paths = ProjectPaths.discover(anchor=anchor)

    assert paths.repository_root == root.resolve()
    assert Path.cwd() == unrelated


def test_documented_environment_overrides_are_respected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repository(tmp_path / "clone")
    data = tmp_path / "data-override"
    monkeypatch.setenv("DUAL_UQ_PROJECT_ROOT", os.fspath(root))
    monkeypatch.setenv("DUAL_UQ_DATA_ROOT", os.fspath(data))

    paths = ProjectPaths.discover()

    assert paths.repository_root == root.resolve()
    assert paths.data_root == data.resolve()


def test_out_of_root_path_and_parent_traversal_are_rejected(tmp_path: Path) -> None:
    root = _repository(tmp_path / "clone")
    paths = ProjectPaths.discover(project_root=root)

    with pytest.raises(ProjectPathError, match="outside declared logical roots"):
        paths.logical_ref(tmp_path / "elsewhere/file.json")
    with pytest.raises(ProjectPathError, match="parent traversal"):
        paths.resolve_logical("data/../outside.json")
