from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from dual_uq.dataset.audits.namespace import (
    capture_namespace_inventory,
    write_namespace_inventory,
)


def _run(*args: str, cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)


def _fixture_repository(tmp_path: Path) -> Path:
    root = tmp_path / "portable-project"
    files = {
        "src/dual_uq/dataset_a_scale/legacy.py": (
            "from dual_uq.dataset_a_scale.hashing import sha256_file\n"
        ),
        "scripts/dataset_a/tool.py": "OUTPUT = 'runs/dataset_a/acquisition'\n",
        "reports/dataset_a_scale/report.json": '{"value": 7}\n',
        "docs/scientific_identity.md": "dataset_id: dataset_a\n",
        "pyproject.toml": "[project]\nname = 'fixture'\n",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _run("git", "init", "-q", cwd=root)
    _run("git", "config", "user.email", "fixture@example.test", cwd=root)
    _run("git", "config", "user.name", "Fixture", cwd=root)
    _run("git", "add", ".", cwd=root)
    _run("git", "commit", "-qm", "fixture", cwd=root)
    (root / "docs/scientific_identity.md").write_text(
        "dataset_id: dataset_a\nchanged: true\n", encoding="utf-8"
    )
    migration_owned = root / "tests/dataset/audits/test_namespace_inventory.py"
    migration_owned.parent.mkdir(parents=True, exist_ok=True)
    migration_owned.write_text("migration fixture\n", encoding="utf-8")
    return root


def test_inventory_is_sorted_portable_and_hash_bound(tmp_path: Path) -> None:
    root = _fixture_repository(tmp_path)

    inventory = capture_namespace_inventory(
        root,
        test_counts={"dataset": 50, "dataset_a_scale": 484, "full": 858},
        migration_owned_paths=(
            "tests/dataset/audits/test_namespace_inventory.py",
        ),
    )

    assert inventory.namespace_paths == tuple(sorted(inventory.namespace_paths))
    assert "src/dual_uq/dataset_a_scale" in inventory.namespace_paths
    assert "scripts/dataset_a" in inventory.namespace_paths
    assert inventory.import_references == (
        "scripts/dataset_a/tool.py:1:OUTPUT = 'runs/dataset_a/acquisition'",
        "src/dual_uq/dataset_a_scale/legacy.py:1:from dual_uq.dataset_a_scale.hashing import sha256_file",
    )
    assert any(
        row == "docs/scientific_identity.md:1:dataset_id: dataset_a"
        for row in inventory.path_references
    )
    assert inventory.preexisting_dirty_paths == ("docs/scientific_identity.md",)
    assert inventory.test_counts == {
        "dataset": 50,
        "dataset_a_scale": 484,
        "full": 858,
    }
    legacy = {row["path"]: row for row in inventory.tracked_legacy_files}
    assert legacy["reports/dataset_a_scale/report.json"]["sha256"] == hashlib.sha256(
        b'{"value": 7}\n'
    ).hexdigest()
    assert str(root) not in json.dumps(inventory.to_json(), sort_keys=True)


def test_inventory_outputs_are_deterministic_and_byte_identical(tmp_path: Path) -> None:
    root = _fixture_repository(tmp_path)
    inventory = capture_namespace_inventory(
        root,
        test_counts={"dataset": 50, "dataset_a_scale": 484, "full": 858},
        migration_owned_paths=(
            "tests/dataset/audits/test_namespace_inventory.py",
        ),
    )
    output = root / "artifacts/dataset/reports/migration"

    first = write_namespace_inventory(inventory, output)
    first_bytes = {name: path.read_bytes() for name, path in first.items()}
    second = write_namespace_inventory(inventory, output)

    assert first.keys() == second.keys()
    assert first_bytes == {name: path.read_bytes() for name, path in second.items()}
    baseline = json.loads(first["baseline"].read_text(encoding="utf-8"))
    assert baseline["schema_version"] == "dataset.namespace-migration-baseline.v1"
    assert baseline["scientific_state_changed"] is False
    assert baseline["counts"] == {
        "import_references": len(inventory.import_references),
        "namespace_paths": len(inventory.namespace_paths),
        "path_references": len(inventory.path_references),
        "preexisting_dirty_paths": 1,
        "tracked_legacy_files": len(inventory.tracked_legacy_files),
    }
