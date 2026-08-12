"""Contracts for the frozen Stage0 H1 confirmatory audit."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from dual_uq.core.hashing import sha256_file
from dual_uq.dataset import h1_confirmatory_audit as h1
from dual_uq.dataset.h1_confirmatory_audit import (
    H1AuditError,
    assess_h1_verdict,
    assign_frozen_tertile,
    build_position_audit,
    deterministic_greedy_matches,
    load_frozen_h1_inputs,
    materialize_conditional_cells,
    materialize_h1_confirmatory_audit,
    run_h1_confirmatory_audit,
    within_protein_rank_pct,
)


def _mechanism_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "protein_id": ["p"] * 6,
            "position": [1, 2, 3, 4, 5, 6],
            "position_mean_abs_interaction": [1.0, 1.0, 3.0, 4.0, 5.0, 6.0],
            "margin_mean_both": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "perturbation_to_margin": [1 / 6, 1 / 5, 3 / 4, 4 / 3, 2.5, 6.0],
            "top1_disagreement_fraction": [0.0, 0.1, 0.2, 0.4, 0.7, 1.0],
            "symmetric_regret_mean": [0.0, 0.01, 0.02, 0.04, 0.07, 0.10],
        }
    )


def _technical_fixture() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for position in range(1, 7):
        for condition, flip, regret in (
            ("PDB", 0.2, 0.02),
            ("AFDB", 0.4, 0.04),
        ):
            rows.append(
                {
                    "protein_id": "p",
                    "position": position,
                    "backbone_condition": condition,
                    "pairwise_top1_disagreement_rate": flip,
                    "pairwise_symmetric_regret_mean": regret,
                }
            )
    return pd.DataFrame(rows)


def test_within_protein_rank_pct_uses_average_ties_and_frozen_formula() -> None:
    values = pd.Series([1.0, 1.0, 3.0, 4.0])
    observed = within_protein_rank_pct(values)
    assert observed.tolist() == pytest.approx([0.25, 0.25, 0.625, 0.875])


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.1, "LOW"), (1 / 3, "MID"), (0.5, "MID"), (2 / 3, "HIGH")],
)
def test_tertile_boundaries_are_not_adaptive(value: float, expected: str) -> None:
    assert assign_frozen_tertile(value) == expected


def test_position_audit_uses_existing_null_and_preserves_negative_excess() -> None:
    result = build_position_audit(
        _mechanism_fixture(), _technical_fixture(), protein_order=("p",)
    )
    first = result.iloc[0]
    assert first["y_flip_tech_pdb"] == pytest.approx(0.2)
    assert first["y_flip_tech_afdb"] == pytest.approx(0.4)
    assert first["e_flip"] == pytest.approx(-0.3)
    assert first["e_regret"] == pytest.approx(-0.03)
    assert (result["m"] > 0).all()
    assert result["sdfi"].to_numpy() == pytest.approx(
        result["p"].to_numpy() / result["m"].to_numpy()
    )


def test_position_audit_rejects_missing_technical_condition() -> None:
    technical = _technical_fixture().query(
        "not (position == 1 and backbone_condition == 'AFDB')"
    )
    with pytest.raises(H1AuditError, match="technical-reference"):
        build_position_audit(_mechanism_fixture(), technical, protein_order=("p",))


def test_position_audit_preserves_canonical_order_after_technical_pivot() -> None:
    first = _mechanism_fixture().assign(protein_id="first")
    second = _mechanism_fixture().assign(protein_id="second")
    mechanism = pd.concat([second, first], ignore_index=True)
    technical = pd.concat(
        [
            _technical_fixture().assign(protein_id="first"),
            _technical_fixture().assign(protein_id="second"),
        ],
        ignore_index=True,
    ).sample(frac=1, random_state=7)

    result = build_position_audit(
        mechanism,
        technical,
        protein_order=("second", "first"),
    )

    assert tuple(dict.fromkeys(result["protein_id"])) == ("second", "first")


def test_conditional_cells_materialize_all_nine_cells() -> None:
    positions = build_position_audit(
        _mechanism_fixture(), _technical_fixture(), protein_order=("p",)
    )
    cells = materialize_conditional_cells(positions, protein_order=("p",))
    assert len(cells) == 9
    assert set(cells["p_tertile"]) == {"LOW", "MID", "HIGH"}
    assert set(cells["m_tertile"]) == {"LOW", "MID", "HIGH"}
    assert cells["n_positions"].sum() == 6


def test_c1_matching_is_deterministic_one_to_one_and_oriented() -> None:
    positions = pd.DataFrame(
        {
            "protein_id": ["p"] * 6,
            "position": [10, 20, 30, 40, 50, 60],
            "p": [1.0, 2.0, 3.0, 1.1, 2.1, 3.1],
            "m": [1.0, 1.0, 1.0, 9.0, 9.0, 9.0],
            "p_rank_pct": [0.1, 0.3, 0.5, 0.12, 0.31, 0.9],
            "m_rank_pct": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
            "p_tertile": ["LOW", "LOW", "MID", "LOW", "MID", "HIGH"],
            "m_tertile": ["LOW", "LOW", "LOW", "HIGH", "HIGH", "HIGH"],
            "y_flip_struct": [0.8, 0.7, 0.6, 0.2, 0.3, 0.4],
            "y_regret_struct": [0.08, 0.07, 0.06, 0.02, 0.03, 0.04],
            "e_flip": [0.7, 0.6, 0.5, 0.1, 0.2, 0.3],
            "e_regret": [0.07, 0.06, 0.05, 0.01, 0.02, 0.03],
        }
    )
    first = deterministic_greedy_matches(positions, match_type="C1", caliper=0.10)
    second = deterministic_greedy_matches(
        positions.sample(frac=1, random_state=4), match_type="C1", caliper=0.10
    )
    pd.testing.assert_frame_equal(first, second)
    assert not first["exposed_position"].duplicated().any()
    assert not first["comparator_position"].duplicated().any()
    assert (first["delta_flip"] > 0).all()
    assert (first["matching_rank_distance"] <= 0.10).all()


def test_c2_matching_is_deterministic_one_to_one_and_oriented() -> None:
    positions = pd.DataFrame(
        {
            "protein_id": ["second"] * 4 + ["first"] * 4,
            "position": [10, 20, 30, 40] * 2,
            "p": [9.0, 8.0, 2.0, 1.0] * 2,
            "m": [1.0, 2.0, 1.1, 2.1] * 2,
            "p_rank_pct": [0.9, 0.8, 0.2, 0.1] * 2,
            "m_rank_pct": [0.10, 0.30, 0.12, 0.31] * 2,
            "p_tertile": ["HIGH", "HIGH", "LOW", "LOW"] * 2,
            "m_tertile": ["LOW", "LOW", "LOW", "LOW"] * 2,
            "y_flip_struct": [0.9, 0.8, 0.2, 0.1] * 2,
            "y_regret_struct": [0.09, 0.08, 0.02, 0.01] * 2,
            "e_flip": [0.8, 0.7, 0.1, 0.0] * 2,
            "e_regret": [0.08, 0.07, 0.01, 0.0] * 2,
        }
    ).sample(frac=1, random_state=13)

    result = deterministic_greedy_matches(
        positions,
        match_type="C2",
        protein_order=("second", "first"),
    )

    assert tuple(dict.fromkeys(result["protein_id"])) == ("second", "first")
    assert not result.duplicated(["protein_id", "exposed_position"]).any()
    assert not result.duplicated(["protein_id", "comparator_position"]).any()
    assert (result["delta_flip"] > 0).all()
    assert (result["delta_regret"] > 0).all()
    assert (result["matching_rank_distance"] <= 0.10).all()


def test_verdict_uses_five_of_eight_and_leave_one_out() -> None:
    strong = {name: [True] * 5 + [False] * 3 for name in ("audit_a", "audit_b", "audit_c")}
    verdict, evidence = assess_h1_verdict(strong, existing_evidence_supported=True)
    assert verdict == "H1_STRONGLY_SUPPORTED_IN_STAGE0"
    assert all(evidence[name]["leave_one_out_stable"] for name in strong)

    partial = {**strong, "audit_b": [True] * 4 + [False] * 4}
    assert assess_h1_verdict(partial, existing_evidence_supported=True)[0] == "H1_PARTIALLY_SUPPORTED"

    unstable = {**strong, "audit_c": [False] * 5 + [True] * 3}
    assert assess_h1_verdict(unstable, existing_evidence_supported=True)[0] == "H1_NOT_STABLE"


def test_frozen_h1_release_has_exact_cohort_and_stop_rule() -> None:
    result = run_h1_confirmatory_audit(Path(__file__).parents[3])
    assert len(result.position_audit) == 1790
    assert len(result.cells) == 72
    assert result.cells["n_positions"].sum() == 1790
    assert len(result.protein_summary) == 8
    assert result.manifest_fields["final_counts"] == {
        "position_audit_rows": 1790,
        "conditional_cell_rows": 72,
        "matched_pair_rows": len(result.matches),
        "protein_summary_rows": 8,
    }
    assert result.manifest_fields["stage0_local_analysis_stop"] is True
    assert result.manifest_fields["scope_declarations"]["proteinmpnn_executed"] is False


def test_frozen_duplicate_technical_evidence_must_reconcile() -> None:
    inputs = load_frozen_h1_inputs(Path(__file__).parents[3])
    positions = build_position_audit(
        inputs.mechanism_positions,
        inputs.technical_null,
        protein_order=inputs.protein_order,
    )
    decision = inputs.decision_positions.copy()
    decision.loc[0, "pdb_same_state_pairwise_top1_disagreement_rate"] += 0.01

    with pytest.raises(H1AuditError, match="frozen evidence"):
        h1._validate_position_evidence(
            replace(inputs, decision_positions=decision),
            positions,
        )


def test_immutable_materialization_reuses_identical_and_rejects_conflict(
    tmp_path: Path,
) -> None:
    result = run_h1_confirmatory_audit(Path(__file__).parents[3])
    portable = replace(result, project_root=tmp_path, input_provenance={})

    first = materialize_h1_confirmatory_audit(portable, tmp_path / "stage0")
    second = materialize_h1_confirmatory_audit(portable, tmp_path / "stage0")

    assert set(first["write_status"].values()) == {"created"}
    assert set(second["write_status"].values()) == {"reused_identical"}
    altered_manifest = dict(portable.manifest_fields)
    altered_manifest["scientific_scope"] = "conflicting_scope"
    with pytest.raises(H1AuditError, match="immutable"):
        materialize_h1_confirmatory_audit(
            replace(portable, manifest_fields=altered_manifest),
            tmp_path / "stage0",
        )


def test_materialization_rejects_duplicate_position_key_before_writing(
    tmp_path: Path,
) -> None:
    result = run_h1_confirmatory_audit(Path(__file__).parents[3])
    duplicated = result.position_audit.copy()
    duplicated.loc[1, ["protein_id", "position"]] = duplicated.loc[
        0, ["protein_id", "position"]
    ].to_numpy()

    with pytest.raises(H1AuditError, match="position audit"):
        materialize_h1_confirmatory_audit(
            replace(
                result,
                project_root=tmp_path,
                input_provenance={},
                position_audit=duplicated,
            ),
            tmp_path / "stage0",
        )


def test_materialization_rejects_invalid_match_contract_before_writing(
    tmp_path: Path,
) -> None:
    result = run_h1_confirmatory_audit(Path(__file__).parents[3])
    invalid = result.matches.copy()
    invalid.loc[0, "matching_rank_distance"] = 0.11

    with pytest.raises(H1AuditError, match="matching"):
        materialize_h1_confirmatory_audit(
            replace(
                result,
                project_root=tmp_path,
                input_provenance={},
                matches=invalid,
            ),
            tmp_path / "stage0",
        )


def test_cli_reports_missing_frozen_inputs_as_structured_blocked(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/dataset/analyze_h1_confirmatory_audit.py",
            "--project-root",
            str(tmp_path),
        ],
        cwd=Path(__file__).parents[3],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    payload = json.loads(completed.stderr)
    assert payload["status"] == "BLOCKED"
    assert payload["failure_code"]


def test_formal_h1_release_is_manifest_bound_and_complete() -> None:
    project_root = Path(__file__).parents[3]
    release_root = project_root / "experiments/p2_design_baseline/stage0"
    manifest_path = release_root / "h1_confirmatory_audit_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "complete"
    assert manifest["h1_verdict"] in {
        "H1_STRONGLY_SUPPORTED_IN_STAGE0",
        "H1_PARTIALLY_SUPPORTED",
        "H1_NOT_STABLE",
    }
    assert manifest["stage0_local_analysis_stop"] is True
    assert manifest["final_counts"]["position_audit_rows"] == 1790
    assert manifest["final_counts"]["conditional_cell_rows"] == 72
    assert manifest["final_counts"]["protein_summary_rows"] == 8
    for record in (*manifest["inputs"].values(), *manifest["outputs"].values()):
        path = Path(record["path"])
        assert not path.is_absolute()
        assert sha256_file(project_root / path) == record["sha256"]
    for record in manifest["outputs"].values():
        assert len(pd.read_parquet(project_root / record["path"])) == record["rows"]
