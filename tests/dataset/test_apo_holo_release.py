from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dual_uq.dataset.apo_holo import (
    ApoHoloResult,
    diagnose_apo_holo_attrition,
    render_apo_holo_release,
    summarize_apo_holo,
)


def test_truncated_discovery_is_limited_even_without_primary_pair() -> None:
    census = pd.DataFrame({"uniprot_id": ["P1"]})
    census.attrs["census_truncated"] = True
    summary = summarize_apo_holo(
        census,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    assert summary["decision"] == "LIMITED"


def test_release_renderer_writes_required_tables_and_portable_manifest(tmp_path: Path) -> None:
    empty = pd.DataFrame({"protein_id": ["P1"]})
    result = ApoHoloResult(
        census=empty,
        pairs=empty,
        admission=empty,
        primary_pairs=empty,
        alternative_pairs=empty,
        residue_mappings=empty,
        ligand_sites=empty,
        pair_descriptors=empty,
        residue_descriptors=empty,
        summary={"decision": "BLOCKED"},
        provenance={"source": {"relative_path": "data/raw/example.json", "sha256": "a" * 64}},
    )
    render_apo_holo_release(result, tmp_path)
    expected = {
        "candidate_structures.parquet",
        "candidate_pairs.parquet",
        "admission_decisions.parquet",
        "primary_pairs.parquet",
        "alternative_pairs.parquet",
        "residue_mappings.parquet",
        "ligand_sites.parquet",
        "pair_structural_descriptors.parquet",
        "residue_structural_descriptors.parquet",
        "summary.json",
        "manifest.json",
        "report.md",
    }
    assert expected.issubset({path.name for path in tmp_path.iterdir()})
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert not any(
        str(value).startswith("/")
        for value in manifest.get("provenance", {}).values()
        if isinstance(value, str)
    )


def test_attrition_diagnosis_preserves_pairs_and_primary_secondary_reasons() -> None:
    pairs = pd.DataFrame(
        [
            {
                "pair_id": "p1",
                "state_pair_status": "unresolved",
                "apo_state": "unresolved",
                "holo_state": "unresolved",
                "sequence_identity": 0.99,
                "construct_mismatch_count": 0,
                "common_fraction": 0.98,
                "assembly_comparability": "comparable",
                "experimental_method_comparable": True,
                "admitted": False,
                "exclusion_reasons": "unresolved_ligand_state",
                "uniprot_id": "P1",
                "apo_polymer_entity_id": "1abc_1",
                "holo_polymer_entity_id": "2abc_1",
            },
            {
                "pair_id": "p2",
                "state_pair_status": "resolved",
                "apo_state": "apo",
                "holo_state": "holo",
                "sequence_identity": 0.91,
                "construct_mismatch_count": 1,
                "common_fraction": 0.80,
                "assembly_comparability": "not_comparable",
                "experimental_method_comparable": True,
                "admitted": False,
                "exclusion_reasons": "sequence_identity_below_threshold;common_coverage_below_threshold;construct_mismatch_exceeds_threshold;assembly_context_not_comparable",
                "uniprot_id": "P1",
                "apo_polymer_entity_id": "3abc_1",
                "holo_polymer_entity_id": "4abc_1",
            },
        ]
    )
    diagnosis, summary = diagnose_apo_holo_attrition(pairs)
    assert len(diagnosis) == 2
    assert diagnosis.set_index("pair_id").loc["p1", "primary_exclusion_reason"] == "unresolved_ligand_state"
    assert diagnosis.set_index("pair_id").loc["p2", "primary_exclusion_reason"] == "sequence_mismatch"
    assert "construct_domain_mismatch" in diagnosis.set_index("pair_id").loc["p2", "secondary_exclusion_reasons"]
    assert summary["funnel"]["candidate_pair"] == 2
    assert summary["funnel"]["final_admitted"] == 0
