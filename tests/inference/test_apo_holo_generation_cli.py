from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts/analysis/generate_apo_holo_sequences.py"
    spec = importlib.util.spec_from_file_location("generate_apo_holo_sequences", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generation_projection_failure_is_a_whole_protein_exclusion() -> None:
    module = _module()
    value = module._generation_exclusion(
        model="ProteinMPNN",
        protein_id="P00001",
        pair_id="apo_holo__P00001",
        failure_state="APO",
        error=ValueError("generation projection does not preserve full-chain order"),
    )

    assert value["reason"] == "generation_projection_order_incompatible"
    assert value["states_attempted"] == ["APO", "HOLO"]
    assert value["protein_id"] == "P00001"


def test_generation_exclusion_artifact_is_immutable(tmp_path: Path) -> None:
    module = _module()
    exclusions = (
        {
            "model": "ProteinMPNN",
            "protein_id": "P00001",
            "pair_id": "apo_holo__P00001",
            "states_attempted": ["APO", "HOLO"],
            "failure_state": "APO",
            "reason": "generation_projection_order_incompatible",
            "detail": "order",
        },
    )
    module._write_worker_exclusions(
        tmp_path,
        model="ProteinMPNN",
        worker_index=0,
        exclusions=exclusions,
    )
    module._write_worker_exclusions(
        tmp_path,
        model="ProteinMPNN",
        worker_index=0,
        exclusions=exclusions,
    )
