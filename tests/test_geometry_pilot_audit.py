from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).parents[1]
MANIFEST_PATH = ROOT / "data/manifests/geometry_pilot.tsv"
MECHANISMS_PATH = ROOT / "reports/geometry_pilot_mechanisms.csv"
AUDIT_PATH = ROOT / "reports/geometry_pilot_audit.json"
PREFLIGHT_PATH = ROOT / "data/manifests/screening_pool_preflight.tsv"
SELECTION_CONFIG_PATH = ROOT / "configs/legacy/a0_screening/a0_selection.yaml"

SCREENING_INDICES = {6, 8, 24, 35, 36}
POSITIVE_PILOTS = {
    "screening-006",
    "screening-008",
    "screening-024",
    "screening-036",
    "reference-1ake",
}
CONTROL_PILOT = "screening-035"

EVIDENCE_COLUMNS = {
    "mapped_plddt_min",
    "mapped_plddt_q10",
    "mapped_plddt_median",
    "ca_disagreement_median",
    "ca_disagreement_p90",
    "ca_disagreement_max",
    "long_range_pae_q90",
    "long_range_pae_above_10_fraction",
    "long_range_pae_above_15_fraction",
    "segment_status",
    "mechanism_label",
    "mechanism_explanation",
    "evidence_complete",
}


def _manifest() -> pd.DataFrame:
    assert MANIFEST_PATH.exists(), f"missing {MANIFEST_PATH}"
    return pd.read_csv(MANIFEST_PATH, sep="\t")


def _mechanisms() -> pd.DataFrame:
    assert MECHANISMS_PATH.exists(), f"missing {MECHANISMS_PATH}"
    return pd.read_csv(MECHANISMS_PATH)


def _audit() -> dict[str, object]:
    assert AUDIT_PATH.exists(), f"missing {AUDIT_PATH}"
    return json.loads(AUDIT_PATH.read_text(encoding="utf-8"))


def _source_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _largest_contiguous_run(positions: list[int]) -> int:
    largest = 0
    current = 0
    previous: int | None = None
    for position in sorted(set(positions)):
        current = current + 1 if previous is not None and position == previous + 1 else 1
        largest = max(largest, current)
        previous = position
    return largest


def test_fixed_pilot_manifest_contains_five_screening_and_reference() -> None:
    manifest = _manifest()

    screening = manifest.loc[manifest["source"] == "screening_pool"]
    reference = manifest.loc[manifest["source"] == "reference_pair"]
    assert set(screening["screening_index"].dropna().astype(int)) == SCREENING_INDICES
    assert set(manifest["pilot_id"]) == POSITIVE_PILOTS | {CONTROL_PILOT}
    assert len(reference) == 1
    assert reference.iloc[0]["pilot_id"] == "reference-1ake"
    assert pd.isna(reference.iloc[0]["screening_index"])


def test_positive_and_construct_control_roles_are_separate() -> None:
    mechanisms = _mechanisms().set_index("pilot_id")

    assert set(mechanisms.loc[list(POSITIVE_PILOTS), "analysis_group"]) == {
        "positive"
    }
    assert mechanisms.loc[CONTROL_PILOT, "analysis_group"] == "negative_control"
    assert (
        mechanisms.loc[CONTROL_PILOT, "pilot_role"]
        == "construct_difference_negative_control"
    )
    assert not bool(mechanisms.loc[CONTROL_PILOT, "main_quality_pass"])


def test_complete_diagnostic_requires_each_independent_stage() -> None:
    mechanisms = _mechanisms()
    complete = mechanisms.loc[mechanisms["full_diagnostic_complete"]]

    assert (complete["pair_status"] == "complete").all()
    assert (complete["geometry_status"] == "complete").all()
    assert (complete["robust_status"] == "complete").all()
    assert complete["segment_context_status"].isin(
        ["complete", "successful_no_segments"]
    ).all()


def test_1ake_is_reference_without_fabricated_screening_index() -> None:
    reference = _mechanisms().set_index("pilot_id").loc["reference-1ake"]

    assert reference["source"] == "reference_pair"
    assert pd.isna(reference["screening_index"])
    assert bool(reference["full_diagnostic_complete"])
    assert reference["mechanism_label"] == "high_confidence_state_disagreement"


def test_confidence_and_geometry_model_pairing_is_consistent() -> None:
    mechanisms = _mechanisms()

    assert mechanisms["selected_afdb_model_entity_id"].notna().all()
    assert mechanisms["selected_afdb_version"].notna().all()
    assert mechanisms["afdb_fragment_start"].notna().all()
    assert mechanisms["afdb_fragment_end"].notna().all()
    assert (
        mechanisms["plddt_model_entity_id"]
        == mechanisms["selected_afdb_model_entity_id"]
    ).all()
    assert (
        mechanisms["pae_model_entity_id"]
        == mechanisms["selected_afdb_model_entity_id"]
    ).all()
    assert (
        mechanisms["geometry_model_entity_id"]
        == mechanisms["selected_afdb_model_entity_id"]
    ).all()
    for source_version in [
        "plddt_model_version",
        "pae_model_version",
        "geometry_model_version",
    ]:
        assert (
            mechanisms[source_version] == mechanisms["selected_afdb_version"]
        ).all()
    assert mechanisms["confidence_model_match"].all()


def test_report_values_match_source_artifacts_and_afdb_metadata() -> None:
    mechanisms = _mechanisms()
    preflight = pd.read_csv(PREFLIGHT_PATH, sep="\t").set_index("screening_index")

    for row in mechanisms.itertuples(index=False):
        pair = json.loads(_source_path(row.pair_qc_path).read_text(encoding="utf-8"))
        geometry = json.loads(
            _source_path(row.geometry_qc_path).read_text(encoding="utf-8")
        )
        robust = json.loads(
            _source_path(row.robust_diagnostics_path).read_text(encoding="utf-8")
        )
        metadata = json.loads(
            _source_path(row.afdb_metadata_path).read_text(encoding="utf-8")
        )

        assert row.mapped_residue_count == pair["mapped_residue_count"]
        assert row.mapped_ca_count == geometry["mapped_ca_count"]
        assert row.selected_afdb_model_entity_id == pair["afdb_model_entity_id"]
        assert row.selected_afdb_model_entity_id == metadata["modelEntityId"]
        assert row.selected_afdb_version == pair["afdb_version"]
        assert row.selected_afdb_version == metadata["latestVersion"]
        assert row.afdb_fragment_start == metadata["uniprotStart"]
        assert row.afdb_fragment_end == metadata["uniprotEnd"]
        assert np.isclose(
            row.mapped_plddt_median, robust["plddt_distribution"]["median"]
        )
        assert np.isclose(
            row.ca_disagreement_p90,
            robust["local_disagreement_distribution"]["q90"],
        )

        model_dir = Path(pair["afdb_model_path"]).parent
        assert Path(pair["plddt_path"]).parent == model_dir
        assert Path(pair["pae_path"]).parent == model_dir
        assert Path(pair["plddt_path"]).exists()
        assert Path(pair["pae_path"]).exists()

        if row.source == "screening_pool":
            source_quality = preflight.loc[int(row.screening_index)]
            assert np.isclose(
                row.full_length_mapping_coverage,
                source_quality["full_length_mapping_coverage"],
            )
            assert np.isclose(row.sequence_identity, source_quality["sequence_identity"])
            assert np.isclose(
                row.observed_ca_fraction,
                source_quality["observed_ca_fraction_of_mapped"],
            )
        else:
            assert np.isclose(
                row.full_length_mapping_coverage, pair["mapping_coverage"]
            )
            assert np.isclose(row.sequence_identity, pair["sequence_identity"])
            assert np.isclose(
                row.observed_ca_fraction,
                geometry["mapped_ca_count"] / pair["mapped_residue_count"],
            )


def test_long_range_pae_and_segment_evidence_match_source_outputs() -> None:
    mechanisms = _mechanisms()

    for row in mechanisms.itertuples(index=False):
        pair_dir = _source_path(row.pair_qc_path).parent
        pairwise = np.load(pair_dir / "pairwise_geometry.npz")
        positions = pairwise["uniprot_positions"]
        pae = pairwise["symmetric_pae"]
        long_range_mask = (
            np.triu(np.ones(pae.shape, dtype=bool), k=1)
            & (np.abs(positions[:, None] - positions[None, :]) >= 48)
        )
        long_range_pae = pae[long_range_mask]
        assert np.isclose(row.long_range_pae_q90, np.quantile(long_range_pae, 0.90))
        assert np.isclose(
            row.long_range_pae_above_10_fraction, np.mean(long_range_pae > 10)
        )
        assert np.isclose(
            row.long_range_pae_above_15_fraction, np.mean(long_range_pae > 15)
        )

        high_confidence_path = _source_path(
            row.high_confidence_disagreement_path
        )
        try:
            high_confidence = pd.read_csv(high_confidence_path)
        except pd.errors.EmptyDataError:
            high_confidence = pd.DataFrame(columns=["uniprot_residue_number"])
        largest_run = _largest_contiguous_run(
            high_confidence["uniprot_residue_number"].astype(int).tolist()
        )
        assert row.largest_high_conf_disagreement_segment == largest_run

        segment_path = pair_dir / "segment_context.json"
        if row.segment_context_status == "complete":
            segments = json.loads(segment_path.read_text(encoding="utf-8"))
            assert row.segment_count == len(segments)
        else:
            assert row.segment_context_status == "successful_no_segments"
            assert not segment_path.exists()
            assert row.segment_count == 0


def test_mechanism_labels_use_existing_selection_thresholds() -> None:
    mechanisms = _mechanisms()
    audit = _audit()
    config = yaml.safe_load(SELECTION_CONFIG_PATH.read_text(encoding="utf-8"))
    thresholds = config["thresholds"]
    rules = audit["mechanism_evidence_rules"]

    assert (
        rules["high_confidence_state_disagreement_min_contiguous_length"]
        == thresholds["high_conf_state_disagreement"]["min_contiguous_length"]
    )
    assert (
        rules["high_pae_long_range_above_10_fraction_min"]
        == thresholds["high_pae_long_range"]["pae_above_10_fraction_min"]
    )
    assert (
        rules["easy_control_high_conf_segment_max_length"]
        == thresholds["easy_control"]["high_conf_segment_max_length"]
    )

    for row in mechanisms.itertuples(index=False):
        if row.analysis_group == "negative_control":
            expected = "construct_difference_control"
        elif (
            row.largest_high_conf_disagreement_segment
            >= thresholds["high_conf_state_disagreement"]["min_contiguous_length"]
        ):
            expected = "high_confidence_state_disagreement"
        elif (
            row.long_range_pae_q90
            >= thresholds["high_pae_long_range"]["pae_q90_min"]
            and row.long_range_pae_above_10_fraction
            >= thresholds["high_pae_long_range"]["pae_above_10_fraction_min"]
        ):
            expected = "high_pae_long_range_candidate"
        elif (
            row.mapped_plddt_median
            >= thresholds["easy_control"]["median_plddt_min"]
            and row.ca_disagreement_p90
            <= thresholds["easy_control"]["p90_disagreement_max"]
            and row.largest_high_conf_disagreement_segment
            <= thresholds["easy_control"]["high_conf_segment_max_length"]
        ):
            expected = "easy_control_candidate"
        else:
            expected = "ordinary_or_unclassified"
        assert row.mechanism_label == expected


def test_positive_main_quality_gate_uses_existing_thresholds() -> None:
    mechanisms = _mechanisms().set_index("pilot_id")
    passing = mechanisms.loc[list(POSITIVE_PILOTS)]
    quality = passing.loc[passing["main_quality_pass"]]

    assert (quality["full_length_mapping_coverage"] >= 0.90).all()
    assert (quality["sequence_identity"] >= 0.95).all()
    assert (quality["observed_ca_fraction"] >= 0.90).all()
    assert not quality["preflight_status"].isin(
        ["unsupported_afdb_fragment", "warn_construct_difference"]
    ).any()


def test_construct_control_cannot_be_mislabeled_low_confidence_local() -> None:
    control = _mechanisms().set_index("pilot_id").loc[CONTROL_PILOT]

    assert control["mechanism_label"] == "construct_difference_control"
    assert "low-confidence local" not in control["mechanism_explanation"].lower()


def test_audit_requires_four_complete_and_quality_pass_positives() -> None:
    audit = _audit()

    assert audit["requested_pilot_count"] == 6
    assert audit["positive_requested_count"] == 5
    assert audit["positive_complete_count"] >= 4
    assert audit["main_quality_pass_count"] >= 4
    assert audit["negative_control_complete_count"] == 1


def test_audit_has_at_least_two_evidence_derived_non_control_mechanisms() -> None:
    mechanisms = _mechanisms()
    positive = mechanisms.loc[mechanisms["analysis_group"] == "positive"]
    audit = _audit()

    assert positive["mechanism_label"].nunique() >= 2
    assert audit["non_control_mechanism_count"] >= 2


def test_index24_no_segment_is_a_complete_terminal_state() -> None:
    index24 = _mechanisms().set_index("pilot_id").loc["screening-024"]

    assert index24["segment_context_status"] == "successful_no_segments"
    assert index24["segment_status"] == "successful_no_segments"
    assert bool(index24["full_diagnostic_complete"])


def test_every_mechanism_row_has_auditable_evidence() -> None:
    mechanisms = _mechanisms()

    assert EVIDENCE_COLUMNS.issubset(mechanisms.columns)
    assert mechanisms[list(EVIDENCE_COLUMNS - {"evidence_complete"})].notna().all().all()
    assert mechanisms["mechanism_explanation"].str.strip().ne("").all()
    assert mechanisms["evidence_complete"].all()


def test_audit_records_no_missing_or_unexplained_numbering_failures() -> None:
    audit = _audit()
    mechanisms = _mechanisms()
    positive = mechanisms.loc[mechanisms["analysis_group"] == "positive"]
    controls = mechanisms.loc[mechanisms["analysis_group"] == "negative_control"]

    assert audit["missing_pairs"] == []
    assert audit["requested_pilot_count"] == len(mechanisms)
    assert audit["screening_candidate_count"] == int(
        (mechanisms["source"] == "screening_pool").sum()
    )
    assert audit["reference_pair_count"] == int(
        (mechanisms["source"] == "reference_pair").sum()
    )
    assert audit["complete_pair_count"] == int(
        (mechanisms["pair_status"] == "complete").sum()
    )
    assert audit["complete_geometry_count"] == int(
        (mechanisms["geometry_status"] == "complete").sum()
    )
    assert audit["complete_robust_count"] == int(
        (mechanisms["robust_status"] == "complete").sum()
    )
    assert audit["complete_segment_count"] == int(
        (mechanisms["segment_context_status"] == "complete").sum()
    )
    assert audit["successful_no_segments_count"] == int(
        (mechanisms["segment_context_status"] == "successful_no_segments").sum()
    )
    assert audit["full_diagnostic_count"] == int(
        mechanisms["full_diagnostic_complete"].sum()
    )
    assert audit["positive_requested_count"] == len(positive)
    assert audit["positive_complete_count"] == int(
        positive["full_diagnostic_complete"].sum()
    )
    assert audit["negative_control_complete_count"] == int(
        controls["full_diagnostic_complete"].sum()
    )
    assert audit["main_quality_pass_count"] == int(positive["main_quality_pass"].sum())
    assert audit["confidence_model_match_count"] == int(
        mechanisms["confidence_model_match"].sum()
    )
    assert audit["confidence_model_mismatch_pairs"] == mechanisms.loc[
        ~mechanisms["confidence_model_match"], "pair_name"
    ].tolist()
    assert audit["mechanism_counts"] == mechanisms["mechanism_label"].value_counts().to_dict()
    assert audit["non_control_mechanism_count"] == positive["mechanism_label"].nunique()
    assert audit["incomplete_pairs"] == mechanisms.loc[
        ~mechanisms["full_diagnostic_complete"], "pair_name"
    ].tolist()
    assert audit["quality_failed_pairs"] == mechanisms.loc[
        ~mechanisms["main_quality_pass"], "pair_name"
    ].tolist()
    assert audit["construct_controls"] == controls["pair_name"].tolist()
    assert audit["unsupported_or_numbering_issue_pairs"] == mechanisms.loc[
        mechanisms["preflight_status"].isin(
            ["unsupported_afdb_fragment", "failed_numbering"]
        ),
        "pair_name",
    ].tolist()
    assert audit["audit_pass"]
    assert audit["blocked_reasons"] == []
