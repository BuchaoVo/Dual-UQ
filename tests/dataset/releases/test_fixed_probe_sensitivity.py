from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from dual_uq.dataset import fixed_probe_sensitivity as sensitivity

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _synthetic_input_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    wt_rows = []
    raw_rows = []
    null_rows = []
    for backbone, backbone_sha in (("PDB", "p" * 64), ("AFDB", "a" * 64)):
        for repeat in range(2):
            wt_rows.append(
                {
                    "protein_id": "fixture_A__P00001",
                    "backbone_condition": backbone,
                    "backbone_sha256": backbone_sha,
                    "repeat_index": repeat,
                    "seed": repeat,
                    "decoding_realization_sha256": str(repeat) * 64,
                    "score_sum_logp_mask": -6.0,
                    "score_mean_logp_mask": -2.0,
                    "scored_residue_count": 3,
                    "model_checkpoint_sha256": "c" * 64,
                    "scoring_protocol": "fixture_protocol",
                }
            )
            raw_rows.append(
                {
                    "protein_id": "fixture_A__P00001",
                    "sequence_hash": "s" * 64,
                    "position": 2,
                    "wt_aa": "A",
                    "mut_aa": "C",
                    "backbone_condition": backbone,
                    "backbone_sha256": backbone_sha,
                    "repeat_index": repeat,
                    "seed": repeat,
                    "decoding_realization_sha256": str(repeat) * 64,
                    "score_sum_logp_mask": -5.7,
                    "score_mean_logp_mask": -1.9,
                    "delta_score_vs_wt": 0.1,
                    "scored_residue_count": 3,
                    "model_checkpoint_sha256": "c" * 64,
                    "scoring_protocol": "fixture_protocol",
                }
            )
        null_rows.append(
            {
                "protein_id": "fixture_A__P00001",
                "sequence_hash": "s" * 64,
                "position": 2,
                "wt_aa": "A",
                "mut_aa": "C",
                "backbone_condition": backbone,
                "backbone_sha256": backbone_sha,
                "model_checkpoint_sha256": "c" * 64,
                "scoring_protocol": "fixture_protocol",
                "n_repeats": 2,
                "scoring_null_type": "empirical_repeat_distribution",
                "std_convention": "population_ddof0",
                "score_mean_logp_mask_mean": -1.9,
                "score_mean_logp_mask_std_population": 0.05,
                "score_mean_logp_mask_min": -1.95,
                "score_mean_logp_mask_max": -1.85,
                "delta_score_vs_wt_mean": 0.1,
                "delta_score_vs_wt_std_population": 0.05,
                "delta_score_vs_wt_min": 0.05,
                "delta_score_vs_wt_max": 0.15,
            }
        )
    return pd.DataFrame(wt_rows), pd.DataFrame(raw_rows), pd.DataFrame(null_rows)


def test_loads_exact_frozen_stage0_scoring_release() -> None:
    inputs = sensitivity.load_frozen_sensitivity_inputs(REPOSITORY_ROOT)

    assert inputs.scoring_manifest_sha256 == sensitivity.SCORING_MANIFEST_SHA256
    assert inputs.wt_scores_sha256 == sensitivity.WT_SCORES_SHA256
    assert inputs.raw_scores_sha256 == sensitivity.RAW_SCORES_SHA256
    assert inputs.scoring_null_sha256 == sensitivity.SCORING_NULL_SHA256
    assert len(inputs.wt_scores) == 480
    assert len(inputs.raw_scores) == 2_040_600
    assert len(inputs.scoring_null) == 68_020
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


def test_loader_blocks_scoring_manifest_hash_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_hash = sensitivity.sha256_file
    manifest = (
        REPOSITORY_ROOT
        / "experiments/p2_design_baseline/stage0/fixed_probe_scoring_manifest.json"
    )
    monkeypatch.setattr(
        sensitivity,
        "sha256_file",
        lambda path: "0" * 64 if path == manifest else real_hash(path),
    )

    with pytest.raises(sensitivity.FixedProbeSensitivityError) as caught:
        sensitivity.load_frozen_sensitivity_inputs(REPOSITORY_ROOT)

    assert caught.value.code == "scoring_manifest_hash_mismatch"


def test_input_table_contract_rejects_missing_schema_column() -> None:
    wt, raw, null = _synthetic_input_tables()

    with pytest.raises(sensitivity.FixedProbeSensitivityError) as caught:
        sensitivity.validate_sensitivity_tables(
            wt,
            raw.drop(columns="delta_score_vs_wt"),
            null,
            repeat_count=2,
            expected_protein_count=1,
            expected_candidate_count=1,
        )

    assert caught.value.code == "raw_score_schema_mismatch"


def test_input_table_contract_rejects_duplicate_scientific_key() -> None:
    wt, raw, null = _synthetic_input_tables()
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)

    with pytest.raises(sensitivity.FixedProbeSensitivityError) as caught:
        sensitivity.validate_sensitivity_tables(
            wt,
            raw,
            null,
            repeat_count=2,
            expected_protein_count=1,
            expected_candidate_count=1,
        )

    assert caught.value.code == "duplicate_raw_score_key"


def test_input_table_contract_rejects_incomplete_repeat_grid() -> None:
    wt, raw, null = _synthetic_input_tables()

    with pytest.raises(sensitivity.FixedProbeSensitivityError) as caught:
        sensitivity.validate_sensitivity_tables(
            wt,
            raw.iloc[:-1].reset_index(drop=True),
            null,
            repeat_count=2,
            expected_protein_count=1,
            expected_candidate_count=1,
        )

    assert caught.value.code == "raw_repeat_grid_mismatch"


def _pairing_inputs() -> SimpleNamespace:
    wt_rows = []
    raw_rows = []
    values = {
        0: {
            "PDB": {"wt": -2.0, "delta": 0.1},
            "AFDB": {"wt": -1.8, "delta": 0.4},
        },
        1: {
            "PDB": {"wt": -2.1, "delta": 0.2},
            "AFDB": {"wt": -1.9, "delta": 0.1},
        },
    }
    for repeat in range(2):
        fingerprint = f"r{repeat}".ljust(64, "0")
        for backbone in ("PDB", "AFDB"):
            wt_score = values[repeat][backbone]["wt"]
            delta = values[repeat][backbone]["delta"]
            wt_rows.append(
                {
                    "protein_id": "fixture_A__P00001",
                    "backbone_condition": backbone,
                    "backbone_sha256": backbone.lower().ljust(64, "0"),
                    "repeat_index": repeat,
                    "seed": repeat,
                    "decoding_realization_sha256": fingerprint,
                    "score_mean_logp_mask": wt_score,
                }
            )
            raw_rows.append(
                {
                    "protein_id": "fixture_A__P00001",
                    "sequence_hash": "s" * 64,
                    "position": 2,
                    "wt_aa": "A",
                    "mut_aa": "C",
                    "backbone_condition": backbone,
                    "backbone_sha256": backbone.lower().ljust(64, "0"),
                    "repeat_index": repeat,
                    "seed": repeat,
                    "decoding_realization_sha256": fingerprint,
                    "score_mean_logp_mask": wt_score + delta,
                    "delta_score_vs_wt": delta,
                }
            )
    probes = pd.DataFrame(
        {
            "protein_id": ["fixture_A__P00001"],
            "sequence_hash": ["s" * 64],
            "position": [2],
            "wt_aa": ["A"],
            "mut_aa": ["C"],
        }
    )
    return SimpleNamespace(
        wt_scores=pd.DataFrame(wt_rows),
        raw_scores=pd.DataFrame(raw_rows),
        fixed_probes=probes,
        protein_order=("fixture_A__P00001",),
    )


def test_paired_effects_use_afdb_minus_pdb_and_validate_algebra() -> None:
    paired = sensitivity.build_paired_structural_effects(
        _pairing_inputs(), repeat_count=2
    )

    assert list(paired["repeat_index"]) == [0, 1]
    assert list(paired["raw_structural_shift"]) == pytest.approx([0.5, 0.1])
    assert list(paired["wt_structural_shift"]) == pytest.approx([0.2, 0.2])
    assert list(paired["structural_mutation_interaction"]) == pytest.approx(
        [0.3, -0.1]
    )
    assert np.allclose(
        paired["structural_mutation_interaction"],
        paired["afdb_mutation_effect"] - paired["pdb_mutation_effect"],
        atol=0.0,
        rtol=0.0,
    )
    assert np.allclose(
        paired["structural_mutation_interaction"],
        paired["raw_structural_shift"] - paired["wt_structural_shift"],
        atol=1.0e-12,
        rtol=1.0e-12,
    )


@pytest.mark.parametrize("missing_condition", ["PDB", "AFDB"])
def test_paired_effects_reject_missing_structural_condition(
    missing_condition: str,
) -> None:
    inputs = _pairing_inputs()
    inputs.raw_scores = inputs.raw_scores.loc[
        ~(
            (inputs.raw_scores["backbone_condition"] == missing_condition)
            & (inputs.raw_scores["repeat_index"] == 1)
        )
    ].reset_index(drop=True)

    with pytest.raises(sensitivity.FixedProbeSensitivityError) as caught:
        sensitivity.build_paired_structural_effects(inputs, repeat_count=2)

    assert caught.value.code == "missing_structural_pair"
    assert caught.value.outcome == "FAIL"


def test_paired_effects_reject_different_realization_fingerprints() -> None:
    inputs = _pairing_inputs()
    changed = (
        (inputs.raw_scores["backbone_condition"] == "AFDB")
        & (inputs.raw_scores["repeat_index"] == 1)
    )
    inputs.raw_scores.loc[changed, "decoding_realization_sha256"] = "x" * 64

    with pytest.raises(sensitivity.FixedProbeSensitivityError) as caught:
        sensitivity.build_paired_structural_effects(inputs, repeat_count=2)

    assert caught.value.code == "realization_fingerprint_mismatch"
    assert caught.value.outcome == "FAIL"


def test_paired_effects_use_realization_fingerprint_as_join_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_identities: list[tuple[str, ...]] = []
    original = sensitivity._pair_conditions

    def recording_pair_conditions(
        frame: pd.DataFrame,
        *,
        identity: list[str],
        value_columns: list[str],
        label: str,
    ) -> pd.DataFrame:
        observed_identities.append(tuple(identity))
        return original(
            frame,
            identity=identity,
            value_columns=value_columns,
            label=label,
        )

    monkeypatch.setattr(sensitivity, "_pair_conditions", recording_pair_conditions)

    sensitivity.build_paired_structural_effects(_pairing_inputs(), repeat_count=2)

    assert observed_identities == [
        (
            "protein_id",
            "sequence_hash",
            "repeat_index",
            "decoding_realization_sha256",
        ),
        ("protein_id", "repeat_index", "decoding_realization_sha256"),
    ]


def _candidate_summary_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hashes = ("a" * 64, "b" * 64)
    interactions = {
        hashes[0]: (-2.0e-6, 0.0, 1.0e-6, 3.0e-6),
        hashes[1]: (0.0, 0.0, 0.0, 0.0),
    }
    paired_rows = []
    for candidate_index, sequence_hash in enumerate(hashes):
        for repeat, interaction in enumerate(interactions[sequence_hash]):
            paired_rows.append(
                {
                    "protein_id": "fixture_A__P00001",
                    "sequence_hash": sequence_hash,
                    "position": candidate_index + 1,
                    "wt_aa": "A",
                    "mut_aa": "CD"[candidate_index],
                    "repeat_index": repeat,
                    "structural_mutation_interaction": interaction,
                }
            )
    null_rows = []
    for sequence_hash, position, mut_aa, pdb_sd, afdb_sd in (
        (hashes[0], 1, "C", 3.0, 4.0),
        (hashes[1], 2, "D", 0.0, 0.0),
    ):
        for condition, standard_deviation in (("PDB", pdb_sd), ("AFDB", afdb_sd)):
            null_rows.append(
                {
                    "protein_id": "fixture_A__P00001",
                    "sequence_hash": sequence_hash,
                    "position": position,
                    "wt_aa": "A",
                    "mut_aa": mut_aa,
                    "backbone_condition": condition,
                    "n_repeats": 4,
                    "delta_score_vs_wt_std_population": standard_deviation,
                }
            )
    candidates = pd.DataFrame(
        {
            "protein_id": ["fixture_A__P00001", "fixture_A__P00001"],
            "sequence_hash": list(hashes),
            "position": [1, 2],
            "wt_aa": ["A", "A"],
            "mut_aa": ["C", "D"],
        }
    )
    return pd.DataFrame(paired_rows), pd.DataFrame(null_rows), candidates


def test_candidate_summary_uses_frozen_quantiles_signs_and_technical_scale() -> None:
    paired, scoring_null, candidates = _candidate_summary_inputs()

    summary = sensitivity.summarize_candidate_sensitivity(
        paired,
        scoring_null,
        candidates,
        repeat_count=4,
        zero_tolerance=1.0e-6,
    )

    assert len(summary) == 2
    first = summary.iloc[0]
    values = np.asarray([-2.0e-6, 0.0, 1.0e-6, 3.0e-6])
    assert first["n_repeats"] == 4
    assert first["interaction_mean"] == pytest.approx(values.mean())
    assert first["interaction_std_population"] == pytest.approx(values.std(ddof=0))
    assert first["interaction_q05"] == pytest.approx(
        np.quantile(values, 0.05, method="linear")
    )
    assert first["interaction_q95"] == pytest.approx(
        np.quantile(values, 0.95, method="linear")
    )
    assert first["fraction_positive"] == pytest.approx(0.25)
    assert first["fraction_negative"] == pytest.approx(0.25)
    assert first["fraction_zero_within_tolerance"] == pytest.approx(0.5)
    assert first["technical_sd_pdb"] == pytest.approx(3.0)
    assert first["technical_sd_afdb"] == pytest.approx(4.0)
    assert first["technical_scale"] == pytest.approx(np.sqrt(12.5))
    assert first["interaction_to_technical_scale"] == pytest.approx(
        abs(values.mean()) / np.sqrt(12.5)
    )
    assert first["technical_scale_status"] == "defined"
    assert first["quantile_interval_semantics"] == (
        "decoding_realization_quantile_interval"
    )

    second = summary.iloc[1]
    assert second["technical_scale"] == pytest.approx(0.0)
    assert pd.isna(second["interaction_to_technical_scale"])
    assert second["technical_scale_status"] == "zero_technical_scale"


def test_candidate_summary_rejects_incomplete_repeat_grid() -> None:
    paired, scoring_null, candidates = _candidate_summary_inputs()
    paired = paired.drop(index=0).reset_index(drop=True)

    with pytest.raises(sensitivity.FixedProbeSensitivityError) as caught:
        sensitivity.summarize_candidate_sensitivity(
            paired,
            scoring_null,
            candidates,
            repeat_count=4,
            zero_tolerance=1.0e-6,
        )

    assert caught.value.code == "candidate_repeat_grid_mismatch"
    assert caught.value.outcome == "FAIL"


def _hierarchy_candidate_summary() -> pd.DataFrame:
    rows = []
    alphabet = "CDEFGHIKLMNPQRSTVWY"
    for protein_index, protein_id in enumerate(
        ("first_A__P00001", "second_A__P00002")
    ):
        for mutation_index, mut_aa in enumerate(alphabet):
            interaction = float(mutation_index - 9) / 10.0
            if mutation_index == 0:
                interaction = -2.0
            elif mutation_index == 1:
                interaction = 2.0
            rows.append(
                {
                    "protein_id": protein_id,
                    "sequence_hash": f"{protein_index}{mutation_index}".ljust(64, "0"),
                    "position": 10 + protein_index,
                    "wt_aa": "A",
                    "mut_aa": mut_aa,
                    "n_repeats": 30,
                    "interaction_mean": interaction,
                    "interaction_std_population": 0.2 + protein_index,
                    "technical_scale": 0.5 + protein_index,
                    "interaction_to_technical_scale": abs(interaction)
                    / (0.5 + protein_index),
                    "technical_scale_status": "defined",
                    "zero_tolerance": 1.0e-6,
                }
            )
    return pd.DataFrame(rows)


def test_position_summary_requires_19_mutations_and_uses_frozen_tie_break() -> None:
    candidates = _hierarchy_candidate_summary()

    positions = sensitivity.summarize_position_sensitivity(candidates)

    assert len(positions) == 2
    first = positions.iloc[0]
    assert first["protein_id"] == "first_A__P00001"
    assert first["position"] == 10
    assert first["n_mutations"] == 19
    assert first["position_max_abs_interaction"] == pytest.approx(2.0)
    assert first["max_abs_interaction_substitution"] == "A>C"
    assert first["positive_mutation_count"] == 10
    assert first["negative_mutation_count"] == 8
    assert first["positive_mutation_fraction"] == pytest.approx(10 / 19)
    assert first["negative_mutation_fraction"] == pytest.approx(8 / 19)


def test_position_summary_rejects_incomplete_mutation_set() -> None:
    candidates = _hierarchy_candidate_summary().drop(index=0).reset_index(drop=True)

    with pytest.raises(sensitivity.FixedProbeSensitivityError) as caught:
        sensitivity.summarize_position_sensitivity(candidates)

    assert caught.value.code == "position_mutation_grid_mismatch"
    assert caught.value.outcome == "FAIL"


def test_protein_summary_preserves_membership_and_reconciles_counts() -> None:
    candidates = _hierarchy_candidate_summary()
    positions = sensitivity.summarize_position_sensitivity(candidates)
    protein_order = ("first_A__P00001", "second_A__P00002")

    proteins = sensitivity.summarize_protein_sensitivity(
        candidates, positions, protein_order
    )

    assert list(proteins["protein_id"]) == list(protein_order)
    assert list(proteins["candidate_count"]) == [19, 19]
    assert list(proteins["position_count"]) == [1, 1]
    assert list(proteins["defined_ratio_count"]) == [19, 19]
    assert np.isfinite(
        proteins[
            [
                "candidate_mean_abs_interaction",
                "candidate_rms_interaction",
                "candidate_abs_interaction_q95",
                "position_mean_abs_interaction_mean",
            ]
        ].to_numpy()
    ).all()


def test_protein_summary_rejects_unexpected_membership() -> None:
    candidates = _hierarchy_candidate_summary()
    positions = sensitivity.summarize_position_sensitivity(candidates)

    with pytest.raises(sensitivity.FixedProbeSensitivityError) as caught:
        sensitivity.summarize_protein_sensitivity(
            candidates, positions, ("first_A__P00001",)
        )

    assert caught.value.code == "protein_membership_mismatch"
    assert caught.value.outcome == "FAIL"


def _synthetic_result() -> sensitivity.FixedProbeSensitivityResult:
    paired, scoring_null, candidates = _candidate_summary_inputs()
    candidate_summary = sensitivity.summarize_candidate_sensitivity(
        paired,
        scoring_null,
        candidates,
        repeat_count=4,
        zero_tolerance=1.0e-6,
    )
    position_candidates = _hierarchy_candidate_summary()
    position_summary = sensitivity.summarize_position_sensitivity(
        position_candidates
    )
    protein_summary = sensitivity.summarize_protein_sensitivity(
        position_candidates,
        position_summary,
        ("first_A__P00001", "second_A__P00002"),
    )
    return sensitivity.FixedProbeSensitivityResult(
        project_root=REPOSITORY_ROOT,
        input_provenance={
            "scoring_manifest": {
                "path": "input/scoring.json",
                "sha256": "1" * 64,
                "rows": None,
            },
            "wt_scores": {
                "path": "input/wt.parquet",
                "sha256": "2" * 64,
                "rows": 4,
            },
            "raw_scores": {
                "path": "input/raw.parquet",
                "sha256": "3" * 64,
                "rows": 16,
            },
            "same_state_null": {
                "path": "input/null.parquet",
                "sha256": "4" * 64,
                "rows": 4,
            },
        },
        paired_effects=paired,
        candidate_summary=candidate_summary,
        position_summary=position_summary,
        protein_summary=protein_summary,
        repeat_count=4,
        protein_order=("first_A__P00001", "second_A__P00002"),
    )


def test_result_validator_rejects_forbidden_binary_or_decision_fields() -> None:
    result = _synthetic_result()
    changed = result.candidate_summary.copy()
    changed["uncertainty_label"] = "positive"
    result = replace(result, candidate_summary=changed)

    with pytest.raises(sensitivity.FixedProbeSensitivityError) as caught:
        sensitivity.validate_sensitivity_result(
            result,
            expected_effect_rows=8,
            expected_candidate_rows=2,
            expected_position_rows=2,
            expected_protein_rows=2,
        )

    assert caught.value.code == "forbidden_analysis_field"


def test_immutable_analysis_writers_reuse_identical_and_reject_conflict(
    tmp_path: Path,
) -> None:
    frame_path = tmp_path / "frame.parquet"
    json_path = tmp_path / "manifest.json"
    frame = pd.DataFrame({"value": [1.0, 2.0], "label": ["a", "b"]})
    payload = {"schema_version": "fixture_v1", "status": "complete"}

    assert sensitivity.write_immutable_parquet(frame_path, frame) == "created"
    assert sensitivity.write_immutable_parquet(frame_path, frame) == "reused_identical"
    assert sensitivity.write_immutable_json(json_path, payload) == "created"
    assert sensitivity.write_immutable_json(json_path, payload) == "reused_identical"

    with pytest.raises(sensitivity.FixedProbeSensitivityError) as caught:
        sensitivity.write_immutable_parquet(
            frame_path, frame.iloc[::-1].reset_index(drop=True)
        )
    assert caught.value.code == "immutable_analysis_artifact_conflict"


def test_analysis_cli_is_thin_and_available() -> None:
    import subprocess
    import sys

    cli = REPOSITORY_ROOT / "scripts/dataset/analyze_fixed_probe_sensitivity.py"
    completed = subprocess.run(
        [sys.executable, str(cli), "--help"],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--project-root" in completed.stdout
