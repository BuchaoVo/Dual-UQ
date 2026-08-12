from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from dual_uq.dataset import fixed_probe_decision_sensitivity as decision

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_loads_exact_frozen_decision_inputs() -> None:
    inputs = decision.load_frozen_decision_inputs(REPOSITORY_ROOT)

    assert inputs.sensitivity_manifest_sha256 == decision.SENSITIVITY_MANIFEST_SHA256
    assert inputs.structural_effects_sha256 == decision.STRUCTURAL_EFFECTS_SHA256
    assert inputs.candidate_sensitivity_sha256 == decision.CANDIDATE_SENSITIVITY_SHA256
    assert inputs.position_sensitivity_sha256 == decision.POSITION_SENSITIVITY_SHA256
    assert inputs.protein_sensitivity_sha256 == decision.PROTEIN_SENSITIVITY_SHA256
    assert len(inputs.scoring.fixed_probes) == 34_010
    assert len(inputs.scoring.wt_scores) == 480
    assert len(inputs.scoring.raw_scores) == 2_040_600
    assert len(inputs.structural_effects) == 1_020_300
    assert len(inputs.candidate_sensitivity) == 34_010
    assert len(inputs.position_sensitivity) == 1_790
    assert len(inputs.protein_sensitivity) == 8
    assert inputs.protein_order == (
        "5gv8_A__P83686",
        "5mn1_A__P00760",
        "1fn8_A__P35049",
        "1pjx_A__Q7SIG4",
        "3pyp_A__P16113",
        "6s2s_A__P02689",
        "4ce8_A__Q9HYN5",
        "5avh_A__P24300",
    )


def test_loader_blocks_sensitivity_manifest_hash_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_hash = decision.sha256_file
    manifest = (
        REPOSITORY_ROOT
        / "experiments/p2_design_baseline/stage0/fixed_probe_sensitivity_manifest.json"
    )
    monkeypatch.setattr(
        decision,
        "sha256_file",
        lambda path: "0" * 64 if path == manifest else real_hash(path),
    )

    with pytest.raises(decision.DecisionSensitivityError) as caught:
        decision.load_frozen_decision_inputs(REPOSITORY_ROOT)

    assert caught.value.code == "sensitivity_manifest_hash_mismatch"
    assert caught.value.outcome == "BLOCKED"


def test_loader_structures_missing_sensitivity_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        decision,
        "SENSITIVITY_MANIFEST_PATH",
        Path("experiments/p2_design_baseline/stage0/missing-sensitivity.json"),
    )

    with pytest.raises(decision.DecisionSensitivityError) as caught:
        decision.load_frozen_decision_inputs(REPOSITORY_ROOT)

    assert caught.value.code == "sensitivity_manifest_missing"
    assert caught.value.outcome == "BLOCKED"


def test_manifest_output_loader_structures_missing_payload(tmp_path: Path) -> None:
    outputs = {
        "paired_effects": {
            "path": "missing.parquet",
            "sha256": "0" * 64,
            "rows": 1,
        }
    }

    with pytest.raises(decision.DecisionSensitivityError) as caught:
        decision._load_manifest_output(
            tmp_path,
            outputs,
            label="paired_effects",
            expected_sha256="0" * 64,
            expected_rows=1,
        )

    assert caught.value.code == "paired_effects_missing"
    assert caught.value.outcome == "BLOCKED"


def test_decision_source_contract_rejects_duplicate_position_key() -> None:
    fixed_probes = pd.DataFrame(
        {
            "protein_id": ["fixture_A__P00001", "fixture_A__P00001"],
            "sequence_hash": ["a" * 64, "b" * 64],
            "position": [2, 2],
            "wt_aa": ["A", "A"],
            "mut_aa": ["C", "C"],
        }
    )

    with pytest.raises(decision.DecisionSensitivityError) as caught:
        decision.validate_fixed_probe_identity_table(
            fixed_probes,
            expected_candidate_count=2,
            expected_position_count=1,
            mutations_per_position=2,
        )

    assert caught.value.code == "duplicate_position_amino_acid"


def _local_score_inputs(*, repeats: int = 2) -> SimpleNamespace:
    protein_id = "fixture_A__P00001"
    wt_aa = "A"
    mutants = [aa for aa in decision.STANDARD_AMINO_ACIDS if aa != wt_aa]
    fixed_rows = [
        {
            "protein_id": protein_id,
            "sequence_hash": f"seq-{mut_aa}".ljust(64, "0"),
            "position": 7,
            "wt_aa": wt_aa,
            "mut_aa": mut_aa,
        }
        for mut_aa in mutants
    ]
    wt_rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []
    for condition in ("PDB", "AFDB"):
        for repeat in range(repeats):
            fingerprint = f"repeat-{repeat}".ljust(64, "0")
            wt_score = -2.0 + repeat * 0.01 + (0.1 if condition == "AFDB" else 0.0)
            wt_rows.append(
                {
                    "protein_id": protein_id,
                    "backbone_condition": condition,
                    "backbone_sha256": condition.lower().ljust(64, "0"),
                    "repeat_index": repeat,
                    "seed": repeat,
                    "decoding_realization_sha256": fingerprint,
                    "score_mean_logp_mask": wt_score,
                }
            )
            for amino_index, fixed in enumerate(fixed_rows, start=1):
                raw_rows.append(
                    {
                        **fixed,
                        "backbone_condition": condition,
                        "backbone_sha256": condition.lower().ljust(64, "0"),
                        "repeat_index": repeat,
                        "seed": repeat,
                        "decoding_realization_sha256": fingerprint,
                        "score_mean_logp_mask": wt_score + amino_index / 100.0,
                    }
                )
    return SimpleNamespace(
        scoring=SimpleNamespace(
            fixed_probes=pd.DataFrame(fixed_rows),
            wt_scores=pd.DataFrame(wt_rows),
            raw_scores=pd.DataFrame(raw_rows),
        ),
        protein_order=(protein_id,),
    )


def test_builds_exact_20_amino_acid_local_sets_with_analytical_wt() -> None:
    inputs = _local_score_inputs(repeats=2)
    frozen_before = inputs.scoring.fixed_probes.copy(deep=True)

    local = decision.build_local_amino_acid_scores(inputs, repeat_count=2)

    keys = ["protein_id", "position", "backbone_condition", "repeat_index"]
    assert len(local) == 80
    assert set(local.groupby(keys).size()) == {20}
    assert set(local.groupby(keys)["aa"].agg(lambda values: "".join(sorted(values)))) == {
        "".join(sorted(decision.STANDARD_AMINO_ACIDS))
    }
    assert set(local.groupby(keys)["is_wt"].sum()) == {1}
    wt_rows = local.loc[local["is_wt"]]
    assert set(wt_rows["aa"]) == {"A"}
    assert wt_rows.loc[
        (wt_rows["backbone_condition"] == "PDB")
        & (wt_rows["repeat_index"] == 0),
        "score_mean_logp_mask",
    ].iloc[0] == pytest.approx(-2.0)
    pd.testing.assert_frame_equal(inputs.scoring.fixed_probes, frozen_before)


def test_local_set_rejects_missing_mutant_amino_acid() -> None:
    inputs = _local_score_inputs(repeats=2)
    inputs.scoring.raw_scores = inputs.scoring.raw_scores.iloc[:-1].reset_index(
        drop=True
    )

    with pytest.raises(decision.DecisionSensitivityError) as caught:
        decision.build_local_amino_acid_scores(inputs, repeat_count=2)

    assert caught.value.code == "local_amino_acid_grid_mismatch"
    assert caught.value.outcome == "FAIL"


def _rank_fixture() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pdb_order = "ACDEFGHIKLMNPQRSTVWY"
    afdb_order = "CEFA DGHIKLMNPQRSTVWY".replace(" ", "")
    assert set(pdb_order) == set(afdb_order) == set(decision.STANDARD_AMINO_ACIDS)
    pdb_scores = {aa: float(20 - index) for index, aa in enumerate(pdb_order)}
    afdb_scores = {aa: float(20 - index) for index, aa in enumerate(afdb_order)}
    for condition, scores in (("PDB", pdb_scores), ("AFDB", afdb_scores)):
        for aa in reversed(decision.STANDARD_AMINO_ACIDS):
            rows.append(
                {
                    "protein_id": "fixture_A__P00001",
                    "position": 7,
                    "wt_aa": "A",
                    "aa": aa,
                    "is_wt": aa == "A",
                    "sequence_hash": pd.NA if aa == "A" else aa.ljust(64, "0"),
                    "backbone_condition": condition,
                    "backbone_sha256": condition.lower().ljust(64, "0"),
                    "repeat_index": 0,
                    "seed": 0,
                    "decoding_realization_sha256": "r0".ljust(64, "0"),
                    "score_mean_logp_mask": scores[aa],
                }
            )
    return pd.DataFrame(rows)


def test_ranking_is_higher_better_and_canonical_for_exact_ties() -> None:
    local = _rank_fixture()
    local.loc[
        local["backbone_condition"] == "PDB", "score_mean_logp_mask"
    ] = 0.0
    pdb_a = (local["backbone_condition"] == "PDB") & (local["aa"] == "A")
    pdb_c = (local["backbone_condition"] == "PDB") & (local["aa"] == "C")
    local.loc[pdb_a | pdb_c, "score_mean_logp_mask"] = 20.0

    ranked = decision.rank_local_amino_acid_scores(
        local.sample(frac=1.0, random_state=7).reset_index(drop=True)
    )
    pdb = ranked.loc[ranked["backbone_condition"] == "PDB"].sort_values("rank")

    assert list(pdb.head(2)["aa"]) == ["A", "C"]
    assert list(pdb.head(2)["rank"]) == [1, 2]
    assert set(pdb["exact_top_tie_count"]) == {2}
    assert set(pdb["practical_top_tie_count"]) == {2}
    assert set(pdb["exact_top_tie"]) == {True}


def test_ranking_records_practical_tie_without_changing_strict_order() -> None:
    local = _rank_fixture()
    local.loc[
        local["backbone_condition"] == "PDB", "score_mean_logp_mask"
    ] = 0.0
    pdb_a = (local["backbone_condition"] == "PDB") & (local["aa"] == "A")
    pdb_c = (local["backbone_condition"] == "PDB") & (local["aa"] == "C")
    local.loc[pdb_a, "score_mean_logp_mask"] = 1.0
    local.loc[pdb_c, "score_mean_logp_mask"] = 1.0 - 5.0e-7

    ranked = decision.rank_local_amino_acid_scores(local)
    pdb = ranked.loc[ranked["backbone_condition"] == "PDB"].sort_values("rank")

    assert list(pdb.head(2)["aa"]) == ["A", "C"]
    assert set(pdb["exact_top_tie_count"]) == {1}
    assert set(pdb["practical_top_tie_count"]) == {2}
    assert set(pdb["practical_top_tie"]) == {True}


def test_structural_decision_metrics_use_paired_local_rankings() -> None:
    ranked = decision.rank_local_amino_acid_scores(_rank_fixture())

    effects = decision.build_repeat_decision_effects(ranked)

    assert len(effects) == 1
    row = effects.iloc[0]
    assert row["top1_pdb"] == "A"
    assert row["top1_afdb"] == "C"
    assert bool(row["top1_disagreement"])
    assert row["top3_instability"] == pytest.approx(2 / 3)
    assert 0.0 <= row["top5_instability"] <= 1.0
    assert -1.0 <= row["spearman_rank"] <= 1.0
    assert 0.0 <= row["mean_normalized_rank_displacement"] <= 1.0
    assert 0.0 <= row["max_normalized_rank_displacement"] <= 1.0
    assert row["regret_pdb_to_afdb"] == pytest.approx(3.0)
    assert row["regret_afdb_to_pdb"] == pytest.approx(1.0)
    assert row["symmetric_regret"] == pytest.approx(2.0)
    assert bool(row["wt_top1_pdb"])
    assert not bool(row["wt_top1_afdb"])
    assert bool(row["wt_top1_changed"])


def test_structural_decisions_reject_realization_fingerprint_mismatch() -> None:
    local = _rank_fixture()
    local.loc[
        local["backbone_condition"] == "AFDB", "decoding_realization_sha256"
    ] = "other".ljust(64, "0")
    ranked = decision.rank_local_amino_acid_scores(local)

    with pytest.raises(decision.DecisionSensitivityError) as caught:
        decision.build_repeat_decision_effects(ranked)

    assert caught.value.code == "decision_realization_fingerprint_mismatch"
    assert caught.value.outcome == "FAIL"


def _same_state_fixture() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for repeat in range(3):
        frame = _rank_fixture()
        frame["repeat_index"] = repeat
        frame["seed"] = repeat
        frame["decoding_realization_sha256"] = f"r{repeat}".ljust(64, "0")
        if repeat == 2:
            pdb = frame["backbone_condition"] == "PDB"
            frame.loc[pdb, "score_mean_logp_mask"] = frame.loc[
                pdb, "score_mean_logp_mask"
            ].map(lambda value: -value)
        frames.append(frame)
    return decision.rank_local_amino_acid_scores(pd.concat(frames, ignore_index=True))


def test_same_state_null_summarizes_all_repeat_pairs_without_emitting_them() -> None:
    ranked = _same_state_fixture()

    null = decision.summarize_same_state_decision_null(ranked, repeat_count=3)

    assert len(null) == 2
    pdb = null.loc[null["backbone_condition"] == "PDB"].iloc[0]
    assert pdb["n_repeats"] == 3
    assert pdb["repeat_pair_count"] == 3
    assert pdb["modal_top1_aa"] == "A"
    assert pdb["top1_modal_concentration"] == pytest.approx(2 / 3)
    assert pdb["distinct_top1_count"] == 2
    assert pdb["pairwise_top1_disagreement_rate"] == pytest.approx(2 / 3)
    assert 0.0 <= pdb["pairwise_top3_instability_mean"] <= 1.0
    assert 0.0 <= pdb["pairwise_top5_instability_mean"] <= 1.0
    assert 0.0 <= pdb["pairwise_mean_normalized_rank_displacement_mean"] <= 1.0
    assert pdb["pairwise_symmetric_regret_mean"] >= 0.0
    assert pdb["pairwise_symmetric_regret_q95"] >= 0.0


def test_same_state_null_rejects_incomplete_repeat_grid() -> None:
    ranked = _same_state_fixture()
    missing = ~(
        (ranked["backbone_condition"] == "PDB")
        & (ranked["repeat_index"] == 2)
    )

    with pytest.raises(decision.DecisionSensitivityError) as caught:
        decision.summarize_same_state_decision_null(
            ranked.loc[missing].reset_index(drop=True), repeat_count=3
        )

    assert caught.value.code == "same_state_repeat_grid_mismatch"
    assert caught.value.outcome == "FAIL"


def test_regret_gate_rejects_material_negative_and_clamps_roundoff() -> None:
    with pytest.raises(decision.DecisionSensitivityError) as caught:
        decision._checked_nonnegative_regrets(
            pd.Series([-1.0e-6, 0.1]).to_numpy(),
            label="fixture regret",
        )

    assert caught.value.code == "negative_local_regret"
    assert caught.value.outcome == "FAIL"
    values = decision._checked_nonnegative_regrets(
        pd.Series([-1.0e-13, 0.1]).to_numpy(),
        label="fixture regret",
    )
    assert values.tolist() == pytest.approx([0.0, 0.1])


def _summary_fixtures() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    effect_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    continuous_rows: list[dict[str, object]] = []
    for position, wt_aa in ((7, "A"), (9, "C")):
        for repeat in range(3):
            effect_rows.append(
                {
                    "protein_id": "fixture_A__P00001",
                    "position": position,
                    "repeat_index": repeat,
                    "seed": repeat,
                    "decoding_realization_sha256": f"r{repeat}".ljust(64, "0"),
                    "wt_aa": wt_aa,
                    "top1_pdb": "A" if repeat < 2 else "C",
                    "top1_afdb": "C" if position == 7 else "A",
                    "top1_disagreement": bool(position == 7 or repeat == 2),
                    "top3_instability": 0.1 * (repeat + 1),
                    "top5_instability": 0.05 * (repeat + 1),
                    "spearman_rank": 0.9 - 0.1 * repeat,
                    "mean_normalized_rank_displacement": 0.02 * (repeat + 1),
                    "max_normalized_rank_displacement": 0.1 * (repeat + 1),
                    "regret_pdb_to_afdb": float(repeat + 1),
                    "regret_afdb_to_pdb": float(repeat + 2),
                    "symmetric_regret": float(repeat + 1.5),
                    "wt_top1_pdb": repeat < 2,
                    "wt_top1_afdb": False,
                    "wt_top1_changed": repeat < 2,
                    "pdb_exact_top_tie": False,
                    "pdb_practical_top_tie": repeat == 2,
                    "afdb_exact_top_tie": False,
                    "afdb_practical_top_tie": False,
                }
            )
        for condition, rate in (("PDB", 0.2), ("AFDB", 0.4)):
            null_rows.append(
                {
                    "protein_id": "fixture_A__P00001",
                    "position": position,
                    "backbone_condition": condition,
                    "n_repeats": 3,
                    "repeat_pair_count": 3,
                    "modal_top1_aa": "A" if condition == "PDB" else "C",
                    "top1_modal_concentration": 2 / 3,
                    "distinct_top1_count": 2,
                    "pairwise_top1_disagreement_rate": rate,
                    "pairwise_top3_instability_mean": rate / 2,
                    "pairwise_top5_instability_mean": rate / 4,
                    "pairwise_mean_normalized_rank_displacement_mean": rate / 5,
                    "pairwise_symmetric_regret_mean": rate * 2,
                    "pairwise_symmetric_regret_median": rate * 2,
                    "pairwise_symmetric_regret_q75": rate * 3,
                    "pairwise_symmetric_regret_q95": rate * 4,
                    "pairwise_symmetric_regret_max": rate * 5,
                    "exact_top_tie_fraction": 0.0,
                    "practical_top_tie_fraction": 1 / 3,
                }
            )
        continuous_rows.append(
            {
                "protein_id": "fixture_A__P00001",
                "position": position,
                "position_mean_abs_interaction": position / 100.0,
                "position_rms_interaction": position / 90.0,
                "position_max_abs_interaction": position / 80.0,
            }
        )
    return (
        pd.DataFrame(effect_rows),
        pd.DataFrame(null_rows),
        pd.DataFrame(continuous_rows),
    )


def test_position_summary_derives_from_repeat_effects_and_joins_null_once() -> None:
    effects, null, continuous = _summary_fixtures()

    positions = decision.summarize_position_decisions(
        effects,
        null,
        continuous,
        repeat_count=3,
        protein_order=("fixture_A__P00001",),
    )

    assert len(positions) == 2
    row = positions.loc[positions["position"] == 7].iloc[0]
    assert row["n_repeats"] == 3
    assert row["top1_disagreement_fraction"] == pytest.approx(1.0)
    assert row["top3_instability_mean"] == pytest.approx(0.2)
    assert row["symmetric_regret_median"] == pytest.approx(2.5)
    assert row["modal_pdb_top1_aa"] == "A"
    assert row["modal_afdb_top1_aa"] == "C"
    assert row["unique_pdb_top1_count"] == 2
    assert row["unique_afdb_top1_count"] == 1
    assert row["pdb_same_state_pairwise_top1_disagreement_rate"] == pytest.approx(
        0.2
    )
    assert row["afdb_same_state_pairwise_top1_disagreement_rate"] == pytest.approx(
        0.4
    )
    assert row[
        "structural_minus_mean_same_state_top1_disagreement"
    ] == pytest.approx(0.7)
    assert row["position_mean_abs_interaction"] == pytest.approx(0.07)
    continuous_columns = [
        column
        for column in positions.columns
        if column.startswith("position_") and column.endswith("interaction")
    ]
    assert continuous_columns == [
        "position_mean_abs_interaction",
        "position_rms_interaction",
        "position_max_abs_interaction",
    ]


def test_position_summary_rejects_missing_same_state_condition() -> None:
    effects, null, continuous = _summary_fixtures()
    missing = ~(
        (null["position"] == 7) & (null["backbone_condition"] == "AFDB")
    )

    with pytest.raises(decision.DecisionSensitivityError) as caught:
        decision.summarize_position_decisions(
            effects,
            null.loc[missing].reset_index(drop=True),
            continuous,
            repeat_count=3,
            protein_order=("fixture_A__P00001",),
        )

    assert caught.value.code == "position_null_join_mismatch"
    assert caught.value.outcome == "FAIL"


def test_protein_summary_uses_position_summary_and_frozen_order() -> None:
    effects, null, continuous = _summary_fixtures()
    positions = decision.summarize_position_decisions(
        effects,
        null,
        continuous,
        repeat_count=3,
        protein_order=("fixture_A__P00001",),
    )

    proteins = decision.summarize_protein_decisions(
        positions,
        protein_order=("fixture_A__P00001",),
    )

    assert list(proteins["protein_id"]) == ["fixture_A__P00001"]
    row = proteins.iloc[0]
    assert row["position_count"] == 2
    assert row["top1_disagreement_fraction_mean"] == pytest.approx(2 / 3)
    assert row["positions_with_any_top1_disagreement_fraction"] == pytest.approx(1.0)
    assert row["pdb_same_state_top1_disagreement_rate_mean"] == pytest.approx(0.2)
    assert row["afdb_same_state_top1_disagreement_rate_mean"] == pytest.approx(0.4)
    assert row[
        "structural_minus_mean_same_state_top1_disagreement_mean"
    ] == pytest.approx(2 / 3 - 0.3)


def _synthetic_result(tmp_path: Path) -> decision.DecisionSensitivityResult:
    effects, null, continuous = _summary_fixtures()
    positions = decision.summarize_position_decisions(
        effects,
        null,
        continuous,
        repeat_count=3,
        protein_order=("fixture_A__P00001",),
    )
    proteins = decision.summarize_protein_decisions(
        positions,
        protein_order=("fixture_A__P00001",),
    )
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    return decision.DecisionSensitivityResult(
        project_root=tmp_path,
        input_provenance={
            "fixture": {
                "path": "source.json",
                "sha256": decision.sha256_file(source),
                "rows": None,
            }
        },
        repeat_effects=effects,
        position_summary=positions,
        technical_null=null,
        protein_summary=proteins,
        repeat_count=3,
        protein_order=("fixture_A__P00001",),
    )


def test_result_validator_enforces_counts_bounds_and_scientific_keys(
    tmp_path: Path,
) -> None:
    result = _synthetic_result(tmp_path)

    decision.validate_decision_result(
        result,
        expected_repeat_effect_rows=6,
        expected_position_rows=2,
        expected_null_rows=4,
        expected_protein_rows=1,
    )

    changed = result.repeat_effects.copy()
    changed.loc[0, "top3_instability"] = 1.1
    invalid = replace(result, repeat_effects=changed)
    with pytest.raises(decision.DecisionSensitivityError) as caught:
        decision.validate_decision_result(
            invalid,
            expected_repeat_effect_rows=6,
            expected_position_rows=2,
            expected_null_rows=4,
            expected_protein_rows=1,
        )
    assert caught.value.code == "decision_metric_out_of_bounds"


def test_result_validator_rejects_same_size_key_and_repeat_substitution(
    tmp_path: Path,
) -> None:
    result = _synthetic_result(tmp_path)
    changed_null = result.technical_null.copy()
    changed_null.loc[changed_null["position"] == 9, "position"] = 11
    with pytest.raises(decision.DecisionSensitivityError) as caught:
        decision.validate_decision_result(
            replace(result, technical_null=changed_null),
            expected_repeat_effect_rows=6,
            expected_position_rows=2,
            expected_null_rows=4,
            expected_protein_rows=1,
        )
    assert caught.value.code == "decision_key_set_mismatch"

    changed_effects = result.repeat_effects.copy()
    changed_effects.loc[
        (changed_effects["position"] == 7)
        & (changed_effects["repeat_index"] == 2),
        "repeat_index",
    ] = 3
    with pytest.raises(decision.DecisionSensitivityError) as caught:
        decision.validate_decision_result(
            replace(result, repeat_effects=changed_effects),
            expected_repeat_effect_rows=6,
            expected_position_rows=2,
            expected_null_rows=4,
            expected_protein_rows=1,
        )
    assert caught.value.code == "decision_repeat_grid_mismatch"


def test_materialization_is_immutable_and_manifest_is_complete(tmp_path: Path) -> None:
    result = _synthetic_result(tmp_path)
    output = tmp_path / "output"
    kwargs = {
        "expected_repeat_effect_rows": 6,
        "expected_position_rows": 2,
        "expected_null_rows": 4,
        "expected_protein_rows": 1,
    }

    first = decision.materialize_fixed_probe_decision_sensitivity(
        result, output, **kwargs
    )
    second = decision.materialize_fixed_probe_decision_sensitivity(
        result, output, **kwargs
    )

    assert first["status"] == "PASS"
    assert set(first["write_status"].values()) == {"created"}
    assert set(second["write_status"].values()) == {"reused_identical"}
    manifest = pd.read_json(first["manifest_path"], typ="series")
    assert manifest["schema_version"] == (
        "stage0_fixed_probe_decision_sensitivity_manifest_v1"
    )
    assert manifest["scope_declarations"]["proteinmpnn_executed"] is False
    assert manifest["scope_declarations"]["hypothesis_testing_performed"] is False

    changed = result.repeat_effects.copy()
    changed.loc[0, "symmetric_regret"] += 0.1
    with pytest.raises(decision.DecisionSensitivityError) as caught:
        decision.materialize_fixed_probe_decision_sensitivity(
            replace(result, repeat_effects=changed), output, **kwargs
        )
    assert caught.value.code == "immutable_analysis_artifact_conflict"


def test_decision_analysis_cli_is_thin_and_available() -> None:
    import subprocess
    import sys

    cli = REPOSITORY_ROOT / "scripts/dataset/analyze_fixed_probe_decisions.py"
    completed = subprocess.run(
        [sys.executable, str(cli), "--help"],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--project-root" in completed.stdout
