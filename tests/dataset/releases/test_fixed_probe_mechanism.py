from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _mechanism_module():
    return importlib.import_module("dual_uq.dataset.fixed_probe_mechanism")


def _mechanism_cli_module():
    path = PROJECT_ROOT / "scripts/dataset/analyze_fixed_probe_mechanism.py"
    spec = importlib.util.spec_from_file_location("analyze_fixed_probe_mechanism", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _position_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    continuous = pd.DataFrame(
        {
            "protein_id": ["p1", "p1", "p2"],
            "position": [1, 2, 1],
            "wt_aa": ["A", "C", "D"],
            "position_mean_abs_interaction": [0.1, 0.2, 0.3],
            "position_rms_interaction": [0.11, 0.21, 0.31],
            "position_max_abs_interaction": [0.12, 0.22, 0.32],
        }
    )
    decision = pd.DataFrame(
        {
            "protein_id": ["p1", "p1", "p2"],
            "position": [1, 2, 1],
            "wt_aa": ["A", "C", "D"],
            "top1_disagreement_fraction": [0.2, 0.4, 0.6],
            "symmetric_regret_mean": [0.01, 0.02, 0.03],
            "structural_minus_mean_same_state_symmetric_regret": [
                0.005,
                0.01,
                0.015,
            ],
            "mean_normalized_rank_displacement_mean": [0.1, 0.2, 0.3],
            "pdb_same_state_pairwise_top1_disagreement_rate": [0.0, 0.1, 0.2],
            "afdb_same_state_pairwise_top1_disagreement_rate": [0.0, 0.1, 0.2],
        }
    )
    return continuous, decision


def test_bound_parquet_requires_expected_sha_rows_and_schema(tmp_path: Path) -> None:
    module = _mechanism_module()
    path = tmp_path / "bound.parquet"
    pd.DataFrame({"protein_id": ["p1"], "position": [1]}).to_parquet(
        path, index=False
    )

    frame = module.load_bound_parquet(
        path,
        expected_sha256=_sha256(path),
        expected_rows=1,
        required_columns={"protein_id", "position"},
        label="test",
    )
    assert len(frame) == 1

    with pytest.raises(module.MechanismError) as exc_info:
        module.load_bound_parquet(
            path,
            expected_sha256="0" * 64,
            expected_rows=1,
            required_columns={"protein_id", "position"},
            label="test",
        )
    assert exc_info.value.code == "upstream_hash_mismatch"
    assert exc_info.value.outcome == "BLOCKED"


def test_bound_parquet_missing_is_structured_blocked(tmp_path: Path) -> None:
    module = _mechanism_module()
    with pytest.raises(module.MechanismError) as exc_info:
        module.load_bound_parquet(
            tmp_path / "missing.parquet",
            expected_sha256="0" * 64,
            expected_rows=1,
            required_columns={"protein_id", "position"},
            label="test",
        )
    assert exc_info.value.code == "upstream_artifact_missing"
    assert exc_info.value.outcome == "BLOCKED"


def test_exact_position_join_preserves_frozen_order() -> None:
    module = _mechanism_module()
    continuous, decision = _position_frames()

    joined = module.join_position_mechanism_inputs(
        continuous,
        decision,
        protein_order=("p2", "p1"),
        expected_position_count=3,
    )

    assert list(joined[["protein_id", "position"]].itertuples(index=False, name=None)) == [
        ("p2", 1),
        ("p1", 1),
        ("p1", 2),
    ]
    assert joined["wt_aa"].tolist() == ["D", "A", "C"]
    assert len(joined) == 3


@pytest.mark.parametrize("side", ["continuous", "decision"])
def test_exact_position_join_rejects_duplicate_keys(side: str) -> None:
    module = _mechanism_module()
    continuous, decision = _position_frames()
    if side == "continuous":
        continuous = pd.concat([continuous, continuous.iloc[[0]]], ignore_index=True)
    else:
        decision = pd.concat([decision, decision.iloc[[0]]], ignore_index=True)

    with pytest.raises(module.MechanismError) as exc_info:
        module.join_position_mechanism_inputs(
            continuous,
            decision,
            protein_order=("p1", "p2"),
            expected_position_count=3,
        )
    assert exc_info.value.code == "duplicate_position_key"


def test_exact_position_join_rejects_unmatched_keys() -> None:
    module = _mechanism_module()
    continuous, decision = _position_frames()
    decision.loc[2, "position"] = 2

    with pytest.raises(module.MechanismError) as exc_info:
        module.join_position_mechanism_inputs(
            continuous,
            decision,
            protein_order=("p1", "p2"),
            expected_position_count=3,
        )
    assert exc_info.value.code == "position_key_mismatch"
    assert "left_only=1" in str(exc_info.value)
    assert "right_only=1" in str(exc_info.value)


def test_exact_position_join_rejects_wt_identity_conflict() -> None:
    module = _mechanism_module()
    continuous, decision = _position_frames()
    decision.loc[0, "wt_aa"] = "V"

    with pytest.raises(module.MechanismError) as exc_info:
        module.join_position_mechanism_inputs(
            continuous,
            decision,
            protein_order=("p1", "p2"),
            expected_position_count=3,
        )
    assert exc_info.value.code == "position_identity_mismatch"


def test_frozen_mechanism_loader_binds_checkpoint_and_upstream_release() -> None:
    module = _mechanism_module()

    inputs = module.load_frozen_mechanism_inputs(PROJECT_ROOT)

    assert inputs.checkpoint_commit == (
        "9cf144656b96ef26a14c42d7a7ff6d400e2c8f15"
    )
    assert inputs.decision_manifest_sha256 == (
        "f7e0df9a6636864207a849e659d68aad077dbaf6aaaf5aa1fb2d943480986e98"
    )
    assert len(inputs.continuous_positions) == 1_790
    assert len(inputs.decision_positions) == 1_790
    assert len(inputs.protein_order) == 8


def _ranked_landscapes(
    margins: dict[tuple[str, int], float],
    *,
    practical_ties: set[tuple[str, int]] | None = None,
) -> pd.DataFrame:
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    rows: list[dict[str, object]] = []
    practical_ties = practical_ties or set()
    for (condition, repeat_index), margin in margins.items():
        scores = [1.0, 1.0 - margin] + [0.8 - index * 0.01 for index in range(18)]
        for rank, (aa, score) in enumerate(zip(amino_acids, scores, strict=True), 1):
            rows.append(
                {
                    "protein_id": "p1",
                    "position": 7,
                    "backbone_condition": condition,
                    "repeat_index": repeat_index,
                    "aa": aa,
                    "rank": rank,
                    "score_mean_logp_mask": score,
                    "practical_top_tie": (condition, repeat_index) in practical_ties,
                }
            )
    return pd.DataFrame(rows)


def test_repeat_margin_requires_exact_20_aa_landscape() -> None:
    module = _mechanism_module()
    ranked = _ranked_landscapes({("PDB", 0): 0.2}).iloc[:-1].copy()

    with pytest.raises(module.MechanismError) as exc_info:
        module.build_repeat_margins(ranked)
    assert exc_info.value.code == "local_amino_acid_grid_mismatch"
    assert exc_info.value.outcome == "FAIL"


def test_repeat_margin_uses_frozen_top1_top2_order_and_tie_semantics() -> None:
    module = _mechanism_module()
    ranked = _ranked_landscapes(
        {("PDB", 0): 0.2, ("AFDB", 0): 0.0},
        practical_ties={("AFDB", 0)},
    )

    margins = module.build_repeat_margins(ranked)

    assert margins["top1_top2_margin"].tolist() == pytest.approx([0.2, 0.0])
    assert margins["practical_top_tie"].tolist() == [False, True]
    assert set(margins["top1_aa"]) == {"A"}
    assert set(margins["top2_aa"]) == {"C"}


def test_repeat_margin_rejects_negative_value_beyond_roundoff() -> None:
    module = _mechanism_module()
    ranked = _ranked_landscapes({("PDB", 0): 0.2})
    ranked.loc[ranked["rank"] == 2, "score_mean_logp_mask"] = 1.1

    with pytest.raises(module.MechanismError) as exc_info:
        module.build_repeat_margins(ranked)
    assert exc_info.value.code == "negative_top1_top2_margin"


def test_position_margin_aggregation_uses_all_conditions_and_repeats() -> None:
    module = _mechanism_module()
    ranked = _ranked_landscapes(
        {
            ("PDB", 0): 0.2,
            ("PDB", 1): 0.4,
            ("AFDB", 0): 0.1,
            ("AFDB", 1): 0.3,
        },
        practical_ties={("AFDB", 0)},
    )
    repeats = module.build_repeat_margins(ranked)

    summary = module.summarize_position_margins(
        repeats,
        repeat_count=2,
        protein_order=("p1",),
    )

    row = summary.iloc[0]
    assert row["margin_mean_pdb"] == pytest.approx(0.3)
    assert row["margin_median_pdb"] == pytest.approx(0.3)
    assert row["margin_mean_afdb"] == pytest.approx(0.2)
    assert row["margin_median_afdb"] == pytest.approx(0.2)
    assert row["margin_mean_both"] == pytest.approx(0.25)
    assert row["margin_median_both"] == pytest.approx(0.25)
    assert row["margin_min"] == pytest.approx(0.1)
    assert row["practical_tie_fraction"] == pytest.approx(0.25)


def test_perturbation_to_margin_zero_is_null_without_epsilon() -> None:
    module = _mechanism_module()
    positions = pd.DataFrame(
        {
            "protein_id": ["p1", "p1"],
            "position": [1, 2],
            "position_mean_abs_interaction": [0.2, 0.2],
        }
    )
    margins = pd.DataFrame(
        {
            "protein_id": ["p1", "p1"],
            "position": [1, 2],
            "margin_mean_both": [0.1, 0.0],
        }
    )

    result = module.attach_margin_diagnostics(positions, margins)

    assert result.loc[0, "perturbation_to_margin"] == pytest.approx(2.0)
    assert result.loc[0, "margin_ratio_status"] == "defined"
    assert pd.isna(result.loc[1, "perturbation_to_margin"])
    assert result.loc[1, "margin_ratio_status"] == "zero_margin"


def test_attaching_margins_preserves_canonical_position_order() -> None:
    module = _mechanism_module()
    positions = pd.DataFrame(
        {
            "protein_id": ["p2", "p1", "p1"],
            "position": [1, 1, 2],
            "position_mean_abs_interaction": [0.3, 0.1, 0.2],
        }
    )
    margins = pd.DataFrame(
        {
            "protein_id": ["p1", "p1", "p2"],
            "position": [1, 2, 1],
            "margin_mean_both": [0.1, 0.1, 0.1],
        }
    )

    result = module.attach_margin_diagnostics(positions, margins)

    assert list(result[["protein_id", "position"]].itertuples(index=False, name=None)) == [
        ("p2", 1),
        ("p1", 1),
        ("p1", 2),
    ]


def test_descriptive_correlation_is_finite_and_pairwise_complete() -> None:
    module = _mechanism_module()
    result = module.descriptive_correlation(
        pd.Series([1.0, 2.0, 3.0, float("nan")]),
        pd.Series([2.0, 4.0, 6.0, 8.0]),
    )

    assert result["n"] == 3
    assert result["pearson"] == pytest.approx(1.0)
    assert result["spearman"] == pytest.approx(1.0)
    assert result["pearson_status"] == "defined"
    assert result["spearman_status"] == "defined"
    assert result["pearson_reason"] is None
    assert result["spearman_reason"] is None


def test_descriptive_correlation_constant_vector_is_structured_null() -> None:
    module = _mechanism_module()
    result = module.descriptive_correlation(
        pd.Series([1.0, 1.0, 1.0]),
        pd.Series([1.0, 2.0, 3.0]),
    )

    assert result["n"] == 3
    assert result["pearson"] is None
    assert result["spearman"] is None
    assert result["pearson_status"] == "undefined"
    assert result["spearman_status"] == "undefined"
    assert result["pearson_reason"] == "constant_x"
    assert result["spearman_reason"] == "constant_x"


def _association_positions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for protein_id, sign in [("p1", 1.0), ("p2", -1.0)]:
        for position in range(1, 5):
            perturbation = float(position)
            response = sign * perturbation
            rows.append(
                {
                    "protein_id": protein_id,
                    "position": position,
                    "position_mean_abs_interaction": perturbation,
                    "margin_mean_both": float(5 - position),
                    "perturbation_to_margin": perturbation / float(5 - position),
                    "top1_disagreement_fraction": response,
                    "symmetric_regret_mean": response * 0.1,
                    "mean_normalized_rank_displacement_mean": response * 0.2,
                }
            )
    return pd.DataFrame(rows)


def test_protein_associations_are_isolated_and_use_frozen_matrix() -> None:
    module = _mechanism_module()
    result = module.summarize_protein_mechanism_associations(
        _association_positions(), protein_order=("p2", "p1")
    )

    assert result["protein_id"].tolist() == ["p2", "p1"]
    assert len(result) == 2
    assert result.loc[0, "perturbation_top1_disagreement_pearson"] == pytest.approx(
        -1.0
    )
    assert result.loc[1, "perturbation_top1_disagreement_pearson"] == pytest.approx(
        1.0
    )
    for association_id, _, _ in module.ASSOCIATION_SPECS:
        for suffix in [
            "n",
            "pearson",
            "pearson_status",
            "pearson_reason",
            "spearman",
            "spearman_status",
            "spearman_reason",
        ]:
            assert f"{association_id}_{suffix}" in result.columns
    assert len(module.ASSOCIATION_SPECS) == 8
    assert not any("pvalue" in column or "p_value" in column for column in result)


def _fork_associations(*, coherent: bool) -> pd.DataFrame:
    module = _mechanism_module()
    rows: list[dict[str, object]] = []
    for protein_index in range(8):
        record: dict[str, object] = {
            "protein_id": f"p{protein_index}",
            "position_count": 10,
        }
        direction = 1.0 if coherent else (-1.0 if protein_index % 2 else 1.0)
        for association_id, _, _ in module.ASSOCIATION_SPECS:
            if association_id.startswith("margin_"):
                value = -0.5 if coherent else direction * 0.05
            elif association_id.startswith("ratio_"):
                value = 0.6 if coherent else direction * 0.04
            else:
                value = 0.2 if coherent else direction * 0.05
            record[f"{association_id}_n"] = 10
            record[f"{association_id}_pearson"] = value
            record[f"{association_id}_pearson_status"] = "defined"
            record[f"{association_id}_pearson_reason"] = None
            record[f"{association_id}_spearman"] = value
            record[f"{association_id}_spearman_status"] = "defined"
            record[f"{association_id}_spearman_reason"] = None
        rows.append(record)
    return pd.DataFrame(rows)


def _fork_positions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "protein_id": [f"p{index // 4}" for index in range(32)],
            "position": [index % 4 + 1 for index in range(32)],
            "top1_disagreement_fraction": [0.0, 0.2, 0.8, 1.0] * 8,
            "symmetric_regret_mean": [0.0, 0.001, 0.01, 0.1] * 8,
            "structural_minus_mean_same_state_symmetric_regret": [
                -0.001,
                -0.001,
                0.005,
                0.05,
            ]
            * 8,
            "margin_mean_both": [0.2, 0.1, 0.02, 0.01] * 8,
            "position_mean_abs_interaction": [0.01, 0.02, 0.03, 0.04] * 8,
            "perturbation_to_margin": [0.05, 0.2, 1.5, 4.0] * 8,
        }
    )


def test_project_fork_go_requires_recurrent_multi_criterion_evidence() -> None:
    module = _mechanism_module()
    assessment = module.assess_project_fork(
        _fork_associations(coherent=True), _fork_positions()
    )

    assert assessment["recommendation"] == "GO_STAGE0_2B"
    assert set(assessment["evidence_summary"]) == {"q1", "q2", "q3", "q4"}
    assert assessment["assessment_semantics"] == (
        "multi_criterion_descriptive_gate_without_universal_coefficient_threshold"
    )
    assert assessment["decision_rule"] == {
        "recurrence_target_rule": "floor(protein_count / 2) + 1",
        "recurrence_target": 5,
        "ratio_clearer_minimum_outcomes": 2,
        "ratio_clearer_total_outcomes": 3,
        "single_protein_dominance_threshold": 0.5,
        "go_requires": [
            "continuous_perturbation_coherent",
            "margin_protective",
            "ratio_clearer",
        ],
        "semantics": "descriptive_project_fork_not_calibrated_risk_threshold",
    }
    assert assessment["evidence_summary"]["q4"][
        "substantive_regret_tail_definition"
    ] == "q95_structural_minus_mean_same_state_regret_above_zero"
    assert assessment["evidence_summary"]["q4"][
        "substantive_regret_tail_semantics"
    ] == "descriptive_upper_tail_excess_not_biological_calibration"


def test_project_fork_holds_when_patterns_are_directionally_incoherent() -> None:
    module = _mechanism_module()
    assessment = module.assess_project_fork(
        _fork_associations(coherent=False), _fork_positions()
    )

    assert assessment["recommendation"] == "HOLD_STAGE0_2B"
    assert assessment["evidence_summary"]["q3"]["ratio_clearer"] is False
    assert assessment["stage0_2b_executed"] is False


def test_substantive_regret_tail_uses_frozen_same_state_contrast() -> None:
    module = _mechanism_module()
    positions = _fork_positions()
    positions["structural_minus_mean_same_state_symmetric_regret"] = -0.001

    assessment = module.assess_project_fork(
        _fork_associations(coherent=True), positions
    )

    q4 = assessment["evidence_summary"]["q4"]
    assert q4["substantive_regret_tail_present"] is False
    assert q4["structural_exceeds_mean_same_state_regret_count"] == 0
    assert q4["structural_minus_same_state_regret_quantiles"]["q95"] < 0


def _small_mechanism_result(tmp_path: Path):
    module = _mechanism_module()
    input_path = tmp_path / "input.json"
    input_path.write_text('{"frozen":true}\n', encoding="utf-8")
    positions = pd.DataFrame(
        {
            "protein_id": ["p1", "p1"],
            "position": [1, 2],
            "wt_aa": ["A", "C"],
            "position_mean_abs_interaction": [0.1, 0.2],
            "position_rms_interaction": [0.11, 0.21],
            "position_max_abs_interaction": [0.12, 0.22],
            "top1_disagreement_fraction": [0.2, 0.4],
            "symmetric_regret_mean": [0.01, 0.02],
            "structural_minus_mean_same_state_symmetric_regret": [0.005, 0.01],
            "mean_normalized_rank_displacement_mean": [0.1, 0.2],
            "margin_mean_pdb": [0.2, 0.3],
            "margin_median_pdb": [0.2, 0.3],
            "margin_mean_afdb": [0.1, 0.2],
            "margin_median_afdb": [0.1, 0.2],
            "margin_mean_both": [0.15, 0.25],
            "margin_median_both": [0.15, 0.25],
            "margin_min": [0.1, 0.2],
            "practical_tie_fraction": [0.0, 0.0],
            "perturbation_to_margin": [2.0 / 3.0, 0.8],
            "margin_ratio_status": ["defined", "defined"],
        }
    )
    associations = module.summarize_protein_mechanism_associations(
        positions, protein_order=("p1",)
    )
    fork = module.assess_project_fork(associations, positions)
    return module.FixedProbeMechanismResult(
        project_root=tmp_path,
        checkpoint_commit=module.STAGE0_CHECKPOINT_COMMIT,
        input_provenance={
            "fixture": {
                "path": "input.json",
                "sha256": _sha256(input_path),
                "rows": None,
            }
        },
        position_mechanism=positions,
        protein_associations=associations,
        project_fork=fork,
        repeat_count=2,
        protein_order=("p1",),
    )


def test_mechanism_result_rejects_forbidden_statistical_fields(tmp_path: Path) -> None:
    module = _mechanism_module()
    result = _small_mechanism_result(tmp_path)
    result.position_mechanism["association_p_value"] = 0.01

    with pytest.raises(module.MechanismError) as exc_info:
        module.validate_mechanism_result(
            result, expected_position_rows=2, expected_protein_rows=1
        )
    assert exc_info.value.code == "forbidden_mechanism_field"


def test_immutable_mechanism_outputs_are_reused_identically(tmp_path: Path) -> None:
    module = _mechanism_module()
    result = _small_mechanism_result(tmp_path)
    output_root = tmp_path / "outputs"

    first = module.materialize_fixed_probe_mechanism(
        result,
        output_root,
        expected_position_rows=2,
        expected_protein_rows=1,
    )
    first_hashes = {
        name: _sha256(output_root / name)
        for name in [
            "fixed_probe_position_mechanism.parquet",
            "fixed_probe_protein_mechanism_association.parquet",
            "fixed_probe_mechanism_manifest.json",
        ]
    }
    second = module.materialize_fixed_probe_mechanism(
        result,
        output_root,
        expected_position_rows=2,
        expected_protein_rows=1,
    )

    assert set(first["write_status"].values()) == {"created"}
    assert set(second["write_status"].values()) == {"reused_identical"}
    assert first_hashes == {
        name: _sha256(output_root / name) for name in first_hashes
    }
    manifest = json.loads(
        (output_root / "fixed_probe_mechanism_manifest.json").read_text()
    )
    assert manifest["project_fork"]["recommendation"] in {
        "GO_STAGE0_2B",
        "HOLD_STAGE0_2B",
    }
    assert manifest["scope_declarations"]["proteinmpnn_executed"] is False
    assert manifest["scope_declarations"]["p_values_computed"] is False


def test_materialization_blocks_when_upstream_bytes_change(tmp_path: Path) -> None:
    module = _mechanism_module()
    result = _small_mechanism_result(tmp_path)
    (tmp_path / "input.json").write_text('{"frozen":false}\n', encoding="utf-8")

    with pytest.raises(module.MechanismError) as exc_info:
        module.materialize_fixed_probe_mechanism(
            result,
            tmp_path / "outputs",
            expected_position_rows=2,
            expected_protein_rows=1,
        )
    assert exc_info.value.code == "upstream_input_changed"


def test_cli_missing_project_root_returns_structured_blocked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = _mechanism_cli_module()

    exit_code = cli.main(["--project-root", str(tmp_path)])

    assert exit_code == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["status"] == "BLOCKED"
    assert payload["failure_code"] == "decision_manifest_missing"
