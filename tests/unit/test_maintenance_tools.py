from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = ROOT / "scripts" / "maintenance" / f"{name}.py"
    assert path.is_file(), f"missing maintenance script: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inventory_is_deterministic_and_excludes_generated_trees(tmp_path: Path) -> None:
    module = _load_script("inventory_project")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.txt").write_text("bb", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "secret").write_text("x", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "cached.pyc").write_bytes(b"x")

    first = module.build_inventory(tmp_path, max_depth=3)
    second = module.build_inventory(tmp_path, max_depth=3)

    assert first == second
    assert [item["path"] for item in first["files"]] == ["a.txt", "nested/b.txt"]
    assert first["total_file_bytes"] == 3


def test_archive_run_moves_only_terminal_run_and_rejects_collision(tmp_path: Path) -> None:
    module = _load_script("archive_run")
    run_dir = tmp_path / "runs" / "track" / "stage" / "run-1"
    run_dir.mkdir(parents=True)
    archive_root = tmp_path / "archive"

    with pytest.raises(ValueError, match="SUCCESS or FAILED"):
        module.archive_run(run_dir, archive_root)

    (run_dir / "SUCCESS").touch()
    destination = module.archive_run(run_dir, archive_root)
    assert destination == archive_root / "run-1"
    assert destination.is_dir()
    assert not run_dir.exists()

    run_dir.mkdir(parents=True)
    (run_dir / "FAILED").touch()
    with pytest.raises(FileExistsError):
        module.archive_run(run_dir, archive_root)


def test_verify_release_reports_hash_match_and_mismatch(tmp_path: Path) -> None:
    module = _load_script("verify_release")
    payload = tmp_path / "payload.txt"
    payload.write_text("frozen", encoding="utf-8")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"files": [{"path": "payload.txt", "sha256": digest}]}),
        encoding="utf-8",
    )

    valid = module.verify_release(manifest, tmp_path)
    assert valid["ok"] is True
    assert valid["checked_file_count"] == 1
    assert valid["failures"] == []

    payload.write_text("changed", encoding="utf-8")
    invalid = module.verify_release(manifest, tmp_path)
    assert invalid["ok"] is False
    assert invalid["failures"][0]["code"] == "sha256_mismatch"
