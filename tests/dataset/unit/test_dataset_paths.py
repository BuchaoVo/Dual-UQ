from __future__ import annotations

from pathlib import Path

import pytest

from dual_uq.core.paths import ProjectPaths
from dual_uq.dataset import DatasetPaths


def _repository(root: Path) -> Path:
    (root / "src/dual_uq").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    return root


def test_dataset_paths_project_configured_data_root_without_using_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path / "clone")
    data_root = tmp_path / "external-data"
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    project = ProjectPaths.discover(project_root=repository, data_root=data_root)

    paths = DatasetPaths.from_project(project)

    assert paths.root == data_root.resolve() / "dataset"
    assert paths.sources == data_root.resolve() / "dataset/sources"
    assert paths.working == data_root.resolve() / "dataset/working"
    assert paths.manifests == data_root.resolve() / "dataset/manifests"
    assert paths.fixtures == data_root.resolve() / "dataset/fixtures"
    assert paths.releases == data_root.resolve() / "dataset/releases"
    assert Path.cwd() == unrelated


def test_dataset_paths_are_a_projection_and_do_not_create_directories(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path / "clone")
    project = ProjectPaths.discover(project_root=repository)

    paths = DatasetPaths.from_project(project)

    assert paths.root == repository.resolve() / "data/dataset"
    assert not paths.root.exists()
