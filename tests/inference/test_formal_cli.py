from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = REPOSITORY_ROOT / "scripts/dataset/run_formal_scoring.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("run_formal_scoring", CLI_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_dry_inventory_closes_all_requests_without_model_forward(
    tmp_path: Path,
    capsys,
) -> None:
    cli = _load_cli()
    artifact_root = tmp_path / "artifacts"
    inventory_root = tmp_path / "inventory"

    exit_code = cli.main(
        [
            "--project-root",
            str(REPOSITORY_ROOT),
            "--artifact-root",
            str(artifact_root),
            "--inventory-root",
            str(inventory_root),
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "FORMAL_INVENTORY_SNAPSHOT_COMPLETE"
    assert output["proteinmpnn_forward_executions"] == 0
    assert output["counts"]["total_requests"] == 7_620
    assert output["counts"]["expected_wt_rows"] == 7_620
    assert output["counts"]["expected_probe_rows"] == 31_111_740
    assert output["counts"]["states"] == {
        "FRESH_EXECUTION_REQUIRED": 7_380,
        "VALID_CANONICAL_COMPLETE": 240,
    }
    assert output["counts"]["inventory_gaps"] == 0
    assert output["counts"]["binding_collisions"] == 0
    assert output["counts"]["unresolved_conflicts"] == 0
    inventory_path = Path(output["inventory_path"])
    manifest_path = Path(output["manifest_path"])
    inventory = pd.read_parquet(inventory_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(inventory) == 7_620
    assert inventory["orchestration_id"].nunique() == 7_620
    assert inventory["artifact_reference"].nunique() == 7_620
    assert len(tuple((artifact_root / "shards").glob("*/*.json"))) == 240
    assert manifest["snapshot_id"] == output["snapshot_id"]
    assert manifest["provenance"]["proteinmpnn_forward_executions"] == 0
    assert output["historical_reuse_materialization"] == {
        "accepted": 240,
        "created": 240,
        "reused_identical": 0,
    }
