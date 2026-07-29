from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dual_uq.lifecycle import build_candidate_lifecycle


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_candidate_lifecycle_uses_stage_artifacts_and_preflight(tmp_path: Path) -> None:
    pool = pd.DataFrame(
        [
            {
                "screening_index": 1,
                "pdb_id": "1abc",
                "chain_id": "A",
                "uniprot_id": "P00001",
                "provisional_stratum": "high_global_confidence",
            },
            {
                "screening_index": 2,
                "pdb_id": "2abc",
                "chain_id": "B",
                "uniprot_id": "P00002",
                "provisional_stratum": "lower_global_confidence",
            },
            {
                "screening_index": 3,
                "pdb_id": "3abc",
                "chain_id": "A",
                "uniprot_id": "P00003",
                "provisional_stratum": "balanced_background",
            },
        ]
    )
    preflight = pd.DataFrame(
        [
            {
                "screening_index": 1,
                "preflight_status": "pass_full_length",
                "preflight_reason": "pass",
                "sequence_identity": 1.0,
                "observed_ca_fraction_of_mapped": 1.0,
                "full_length_mapping_coverage": 1.0,
            },
            {
                "screening_index": 2,
                "preflight_status": "fail_preflight",
                "preflight_reason": "low_full_length_mapping_coverage",
                "sequence_identity": 1.0,
                "observed_ca_fraction_of_mapped": 1.0,
                "full_length_mapping_coverage": 0.4,
            },
            {
                "screening_index": 3,
                "preflight_status": "pass_full_length",
                "preflight_reason": "pass",
                "sequence_identity": 0.99,
                "observed_ca_fraction_of_mapped": 0.95,
                "full_length_mapping_coverage": 0.95,
            },
        ]
    )
    status = pd.DataFrame(
        [
            {
                "screening_index": 2,
                "pair_name": "2abc_B__P00002",
                "status": "skipped_preflight",
            }
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "pair_name": "1abc_A__P00001",
                "candidate_tags": "easy_control",
                "primary_category": "easy_control",
            }
        ]
    )

    complete = tmp_path / "pairs/1abc_A__P00001"
    _write_json(
        complete / "pair_qc.json",
        {"mapping_coverage": 1.0, "sequence_identity": 1.0, "quality_flag": "pass"},
    )
    _write_json(
        complete / "pair_geometry_qc.json",
        {"pdb_afdb_aa_match_fraction": 1.0, "mapped_ca_coverage": 1.0},
    )
    _write_json(complete / "robust_pair_diagnostics.json", {"residue_count": 100})
    _write_json(complete / "segment_context.json", [{"start_position": 10}])

    no_segments = tmp_path / "pairs/3abc_A__P00003"
    _write_json(
        no_segments / "pair_qc.json",
        {"mapping_coverage": 0.95, "sequence_identity": 0.99, "quality_flag": "pass"},
    )
    _write_json(
        no_segments / "pair_geometry_qc.json",
        {"pdb_afdb_aa_match_fraction": 0.99, "mapped_ca_coverage": 0.95},
    )
    _write_json(no_segments / "robust_pair_diagnostics.json", {"residue_count": 90})
    (no_segments / "disagreement_segments.csv").write_text("\n", encoding="utf-8")

    lifecycle, audit = build_candidate_lifecycle(
        pool=pool,
        preflight=preflight,
        status=status,
        pair_root=tmp_path / "pairs",
        candidate_summary=summary,
    )
    by_index = lifecycle.set_index("screening_index")

    assert by_index.loc[1, "segment_context_status"] == "complete"
    assert bool(by_index.loc[1, "quality_pass"])
    assert by_index.loc[1, "candidate_tags"] == "easy_control"
    assert by_index.loc[2, "pair_status"] == "skipped_preflight"
    assert by_index.loc[2, "exclusion_reason"] == "low_full_length_mapping_coverage"
    assert not bool(by_index.loc[2, "quality_pass"])
    assert by_index.loc[3, "segment_context_status"] == "successful_no_segments"
    assert bool(by_index.loc[3, "complete_diagnostics"])
    assert audit["total_candidates"] == 3
    assert audit["complete_diagnostics"] == 2
    assert audit["quality_pass"] == 2
