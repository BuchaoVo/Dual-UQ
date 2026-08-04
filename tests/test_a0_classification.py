from __future__ import annotations

from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import yaml

from dual_uq.a0_classification import (
    A0Evidence,
    DisagreementSegment,
    PaeStratumStats,
    classify_candidate,
    compute_pae_strata,
)


@pytest.fixture
def config() -> dict:
    return {
        "selection": {
            "target_per_primary_category": {
                "easy_control": 3,
                "low_confidence_local": 3,
                "high_pae_long_range": 3,
                "high_confidence_state_disagreement": 3,
            }
        },
        "classification": {
            "primary_category_priority": [
                "unsupported_afdb_fragment",
                "construct_difference_control",
                "missing_coordinate_stress",
                "high_confidence_state_disagreement",
                "high_pae_long_range",
                "low_confidence_local",
                "easy_control",
                "ordinary_or_unclassified",
                "insufficient_evidence",
            ]
        },
        "thresholds": {
            "easy_control": {
                "mapped_plddt_median_min": 90.0,
                "ca_disagreement_p90_max": 1.0,
                "high_conf_segment_max_length": 2,
            },
            "low_confidence_local": {
                "internal_below_80_min_length": 5,
            },
            "high_pae_long_range": {
                "minimum_sequence_separation": 48,
                "q90_threshold": 10.0,
                "above_10_fraction_threshold": 0.20,
                "above_15_fraction_threshold": 0.10,
                "rule": "any",
            },
            "high_confidence_state_disagreement": {
                "minimum_segment_length": 5,
                "minimum_segment_median_plddt": 90.0,
                "minimum_segment_disagreement": 1.0,
            },
        },
        "quality": {
            "minimum_full_length_mapping_coverage": 0.90,
            "minimum_entity_mapping_coverage": 0.90,
            "minimum_sequence_identity": 0.95,
            "minimum_observed_ca_fraction": 0.90,
        },
    }


@pytest.fixture
def summary_module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "12_build_a0_candidate_summary.py"
    )
    spec = spec_from_file_location("a0_candidate_summary_script", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def low_pae_stats() -> dict[str, PaeStratumStats]:
    return {
        "48_95": PaeStratumStats(100, 3.0, 5.0, 3.5, 0.01, 0.0),
        "96_plus": PaeStratumStats(100, 3.0, 5.0, 3.5, 0.01, 0.0),
    }


def complete_evidence(**overrides: object) -> A0Evidence:
    values: dict[str, object] = {
        "preflight_status": "pass_full_length",
        "full_length_mapping_coverage": 1.0,
        "entity_mapping_coverage": 1.0,
        "sequence_identity": 1.0,
        "observed_ca_fraction": 1.0,
        "pair_status": "complete",
        "geometry_status": "complete",
        "robust_status": "complete",
        "segment_context_status": "successful_no_segments",
        "confidence_model_match": True,
        "mapped_plddt_median": 98.0,
        "longest_internal_below_80_length": 0,
        "low_conf_evidence_available": True,
        "pae_strata": low_pae_stats(),
        "pae_evidence_available": True,
        "ca_disagreement_p90": 0.5,
        "disagreement_segments": (),
        "state_disagreement_evidence_available": True,
        "pilot_role": None,
        "secondary_missing_coordinate_stress": False,
        "unsupported_afdb_fragment": False,
    }
    values.update(overrides)
    return A0Evidence(**values)


def test_easy_control(config: dict) -> None:
    result = classify_candidate(complete_evidence(), config)

    assert result.is_easy_control is True
    assert result.primary_category == "easy_control"
    assert result.classification_status == "complete_classification"
    assert result.full_diagnostic_complete is True
    assert result.main_quality_pass is True
    assert result.selection_eligible is True


@pytest.mark.parametrize(
    ("override", "expected_label"),
    [
        (
            {
                "pae_strata": {
                    "48_95": PaeStratumStats(
                        100, 8.0, 12.0, 8.5, 0.25, 0.05
                    )
                }
            },
            "is_high_pae_long_range",
        ),
        (
            {
                "disagreement_segments": (
                    DisagreementSegment(10, 14, 5, 95.0, 1.2),
                ),
                "segment_context_status": "complete",
            },
            "is_high_conf_state_disagreement",
        ),
        (
            {"longest_internal_below_80_length": 5},
            "is_low_conf_local",
        ),
    ],
)
def test_easy_control_excludes_major_risk_labels(
    config: dict,
    override: dict,
    expected_label: str,
) -> None:
    result = classify_candidate(complete_evidence(**override), config)

    assert result.is_easy_control is False
    assert getattr(result, expected_label) is True


def test_low_confidence_local_can_be_partial(config: dict) -> None:
    evidence = complete_evidence(
        pair_status="not_started",
        geometry_status="not_started",
        robust_status="not_started",
        segment_context_status="not_started",
        pae_strata=None,
        pae_evidence_available=False,
        ca_disagreement_p90=None,
        disagreement_segments=None,
        state_disagreement_evidence_available=False,
        longest_internal_below_80_length=5,
    )

    result = classify_candidate(evidence, config)

    assert result.is_low_conf_local is True
    assert result.primary_category == "low_confidence_local"
    assert result.classification_status == "partial_classification"
    assert result.full_diagnostic_complete is False
    assert result.selection_eligible is False


def test_terminal_low_confidence_is_not_local(config: dict) -> None:
    evidence = complete_evidence(
        longest_internal_below_80_length=0,
        terminal_only_below_80=True,
    )

    result = classify_candidate(evidence, config)

    assert result.is_low_conf_local is False


def test_high_pae_uses_real_long_range_statistics(config: dict) -> None:
    evidence = complete_evidence(
        pae_strata={
            "48_95": PaeStratumStats(100, 8.0, 12.0, 8.5, 0.15, 0.05),
            "96_plus": PaeStratumStats(100, 8.0, 9.0, 8.5, 0.25, 0.05),
        }
    )

    result = classify_candidate(evidence, config)

    assert result.is_high_pae_long_range is True
    assert result.primary_category == "high_pae_long_range"


def test_length_and_provisional_stratum_cannot_trigger_high_pae(
    config: dict,
) -> None:
    evidence = complete_evidence(
        protein_length=500,
        provisional_stratum="long_backbone_proxy",
        pae_strata=low_pae_stats(),
    )

    result = classify_candidate(evidence, config)

    assert result.is_high_pae_long_range is False


def test_high_pae_rejects_noncanonical_minimum_separation(
    config: dict,
) -> None:
    config["thresholds"]["high_pae_long_range"][
        "minimum_sequence_separation"
    ] = 47

    with pytest.raises(ValueError, match="must be 48"):
        classify_candidate(complete_evidence(), config)


def test_high_confidence_state_disagreement(config: dict) -> None:
    evidence = complete_evidence(
        disagreement_segments=(
            DisagreementSegment(42, 48, 7, 96.5, 2.69),
        ),
        segment_context_status="complete",
    )

    result = classify_candidate(evidence, config)

    assert result.is_high_conf_state_disagreement is True
    assert result.primary_category == "high_confidence_state_disagreement"


def test_low_confidence_disagreement_is_not_high_confidence_state(
    config: dict,
) -> None:
    evidence = complete_evidence(
        disagreement_segments=(
            DisagreementSegment(42, 48, 7, 89.9, 2.69),
        ),
        segment_context_status="complete",
    )

    result = classify_candidate(evidence, config)

    assert result.is_high_conf_state_disagreement is False


def test_construct_control_has_priority_and_is_not_selectable(
    config: dict,
) -> None:
    evidence = complete_evidence(
        preflight_status="warn_construct_difference",
        pilot_role="construct_difference_negative_control",
        full_length_mapping_coverage=0.89,
        pae_strata={
            "48_95": PaeStratumStats(100, 8.0, 12.0, 8.5, 0.25, 0.05)
        },
    )

    result = classify_candidate(evidence, config)

    assert result.is_construct_difference is True
    assert result.is_high_pae_long_range is True
    assert result.primary_category == "construct_difference_control"
    assert result.selection_eligible is False


def test_missing_coordinate_stress_has_priority(config: dict) -> None:
    evidence = complete_evidence(
        observed_ca_fraction=0.89,
        disagreement_segments=(
            DisagreementSegment(42, 48, 7, 96.5, 2.69),
        ),
        segment_context_status="complete",
    )

    result = classify_candidate(evidence, config)

    assert result.is_missing_coordinate_stress is True
    assert result.is_high_conf_state_disagreement is True
    assert result.primary_category == "missing_coordinate_stress"
    assert result.selection_eligible is False


def test_explicit_secondary_missing_coordinate_stress(config: dict) -> None:
    result = classify_candidate(
        complete_evidence(secondary_missing_coordinate_stress=True),
        config,
    )

    assert result.is_missing_coordinate_stress is True
    assert result.primary_category == "missing_coordinate_stress"


def test_unsupported_fragment_has_no_mechanism_labels(config: dict) -> None:
    evidence = complete_evidence(
        preflight_status="unsupported_afdb_fragment",
        unsupported_afdb_fragment=True,
        low_conf_evidence_available=False,
        pae_strata=None,
        pae_evidence_available=False,
        disagreement_segments=None,
        state_disagreement_evidence_available=False,
    )

    result = classify_candidate(evidence, config)

    assert result.primary_category == "unsupported_afdb_fragment"
    assert result.selection_eligible is False
    assert result.is_easy_control is False
    assert result.is_low_conf_local is None
    assert result.is_high_pae_long_range is None
    assert result.is_high_conf_state_disagreement is None


def test_multilabel_overlap_is_preserved(config: dict) -> None:
    evidence = complete_evidence(
        longest_internal_below_80_length=8,
        pae_strata={
            "48_95": PaeStratumStats(100, 8.0, 12.0, 8.5, 0.25, 0.15)
        },
    )

    result = classify_candidate(evidence, config)

    assert result.is_low_conf_local is True
    assert result.is_high_pae_long_range is True
    assert result.primary_category == "high_pae_long_range"


def test_complete_no_mechanism_is_ordinary(config: dict) -> None:
    evidence = complete_evidence(
        mapped_plddt_median=89.0,
        ca_disagreement_p90=1.1,
    )

    result = classify_candidate(evidence, config)

    assert result.primary_category == "ordinary_or_unclassified"
    assert result.classification_status == "complete_classification"
    assert result.selection_eligible is False


def test_missing_pae_is_unknown_not_false(config: dict) -> None:
    evidence = complete_evidence(
        pae_strata=None,
        pae_evidence_available=False,
    )

    result = classify_candidate(evidence, config)

    assert result.is_high_pae_long_range is None
    assert result.high_pae_evidence_available is False
    assert result.full_diagnostic_complete is False


def test_no_safe_primary_mechanism_is_insufficient(config: dict) -> None:
    evidence = complete_evidence(
        pair_status="not_started",
        geometry_status="not_started",
        robust_status="not_started",
        segment_context_status="not_started",
        low_conf_evidence_available=False,
        pae_strata=None,
        pae_evidence_available=False,
        ca_disagreement_p90=None,
        disagreement_segments=None,
        state_disagreement_evidence_available=False,
    )

    result = classify_candidate(evidence, config)

    assert result.primary_category == "insufficient_evidence"
    assert result.classification_status == "insufficient_evidence"
    assert result.selection_eligible is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("full_length_mapping_coverage", 0.899),
        ("entity_mapping_coverage", 0.899),
        ("sequence_identity", 0.949),
        ("observed_ca_fraction", 0.899),
    ],
)
def test_each_hard_quality_threshold_blocks_selection(
    config: dict,
    field: str,
    value: float,
) -> None:
    evidence = complete_evidence(**{field: value})

    result = classify_candidate(evidence, config)

    assert result.main_quality_pass is False
    assert result.selection_eligible is False


def test_primary_priority_is_config_driven_and_deterministic(
    config: dict,
) -> None:
    evidence = complete_evidence(
        longest_internal_below_80_length=8,
        pae_strata={
            "48_95": PaeStratumStats(100, 8.0, 12.0, 8.5, 0.25, 0.15)
        },
    )
    reversed_config = deepcopy(config)
    priority = reversed_config["classification"]["primary_category_priority"]
    priority.remove("low_confidence_local")
    priority.insert(priority.index("high_pae_long_range"), "low_confidence_local")

    first = classify_candidate(evidence, reversed_config)
    second = classify_candidate(evidence, reversed_config)

    assert first == second
    assert first.primary_category == "low_confidence_local"


def test_pae_strata_use_unique_real_pairs() -> None:
    positions = np.array([1, 7, 13, 25, 49, 97])
    pae = np.zeros((6, 6), dtype=float)
    pae[0, 4] = pae[4, 0] = 20.0  # separation 48
    pae[0, 5] = pae[5, 0] = 16.0  # separation 96
    pae[1, 5] = pae[5, 1] = 4.0  # separation 90

    strata = compute_pae_strata(positions, pae)

    assert set(strata) == {"6_11", "12_23", "24_47", "48_95", "96_plus"}
    assert strata["48_95"].pair_count == 5
    assert strata["48_95"].above_10_fraction == pytest.approx(0.20)
    assert strata["96_plus"].pair_count == 1
    assert strata["96_plus"].above_15_fraction == pytest.approx(1.0)


def test_pae_strata_reject_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="square matrix"):
        compute_pae_strata(np.array([1, 2]), np.zeros((2, 3)))


def test_repository_config_uses_canonical_state_threshold_key() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "legacy"
        / "a0_screening"
        / "a0_selection.yaml"
    )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    canonical = config["thresholds"][
        "high_confidence_state_disagreement"
    ]
    legacy = config["thresholds"]["high_conf_state_disagreement"]
    assert (
        canonical["minimum_segment_length"]
        == legacy["min_contiguous_length"]
    )
    assert (
        canonical["minimum_segment_median_plddt"]
        == legacy["plddt_min"]
    )
    assert (
        canonical["minimum_segment_disagreement"]
        == legacy["disagreement_threshold"]
    )


def test_existing_pair_confidence_skips_unobserved_legacy_keys(
    summary_module: ModuleType,
    tmp_path: Path,
) -> None:
    pair_dir = tmp_path / "legacy_pair"
    pair_dir.mkdir()
    positions = list(range(1, 21))
    pd.DataFrame(
        {
            "pdb_chain_id": ["A"] * 20,
            "pdb_residue_number": [
                "null",
                *(str(value) for value in range(2, 20)),
                "null",
            ],
            "uniprot_residue_number": positions,
        }
    ).to_parquet(pair_dir / "residue_mapping.parquet", index=False)
    observed_positions = list(range(2, 20))
    pd.DataFrame(
        {
            "pdb_chain_id": ["A"] * len(observed_positions),
            "pdb_residue_number_norm": [
                str(value) for value in observed_positions
            ],
            "uniprot_residue_number": observed_positions,
            "plddt": [
                70.0 if 8 <= value <= 12 else 95.0
                for value in observed_positions
            ],
        }
    ).to_parquet(pair_dir / "residue_geometry.parquet", index=False)

    result = summary_module._mapped_confidence_from_pair(pair_dir)

    assert result is not None
    assert result["longest_internal_below_80_length"] == 5
    assert result["low_conf_evidence_available"] is True


def test_candidate_assembly_keeps_missing_coordinate_evidence_unknown(
    summary_module: ModuleType,
    config: dict,
    tmp_path: Path,
) -> None:
    candidate = {
        "screening_index": 120,
        "source": "replacement_pool",
        "pdb_id": "1vyr",
        "chain_id": "A",
        "uniprot_id": "P71278",
        "pair_name": "1vyr_A__P71278",
        "provisional_stratum": "lower_global_confidence_replacement",
    }
    replacement = {
        **candidate,
        "preflight_status": "failed_runtime",
    }

    row = summary_module._assemble_candidate(
        candidate,
        root=tmp_path,
        config=config,
        preflight={},
        lifecycle={},
        replacement=replacement,
        pilot={},
        mechanism={},
        mapped_confidence={},
    )

    assert row["is_missing_coordinate_stress"] is None
    assert row["missing_coordinate_evidence_available"] is False


def test_mechanism_identity_is_attached_before_four_key_join(
    summary_module: ModuleType,
) -> None:
    pilot = pd.DataFrame(
        [
            {
                "pilot_id": "screening-006",
                "screening_index": 6,
                "pdb_id": "1gci",
                "chain_id": "A",
                "uniprot_id": "P29600",
                "pair_name": "1gci_A__P29600",
                "source": "screening_pool",
                "pilot_role": "high_confidence_positive",
            },
            {
                "pilot_id": "reference-1ake",
                "screening_index": pd.NA,
                "pdb_id": "1ake",
                "chain_id": "A",
                "uniprot_id": "P69441",
                "pair_name": "1ake_A__P69441",
                "source": "reference_pair",
                "pilot_role": (
                    "high_confidence_state_disagreement_reference"
                ),
            }
        ]
    )
    mechanisms = pd.DataFrame(
        [
            {
                "pilot_id": "screening-006",
                "screening_index": 6.0,
                "source": "screening_pool",
                "pair_name": "1gci_A__P29600",
                "pilot_role": "high_confidence_positive",
                "mechanism_label": "easy_control_candidate",
            },
            {
                "pilot_id": "reference-1ake",
                "screening_index": np.nan,
                "source": "reference_pair",
                "pair_name": "1ake_A__P69441",
                "pilot_role": (
                    "high_confidence_state_disagreement_reference"
                ),
                "mechanism_label": (
                    "high_confidence_state_disagreement"
                ),
            },
        ]
    )

    result = summary_module.enrich_pilot_mechanism_identity(
        mechanisms,
        pilot,
    )

    assert len(result) == len(mechanisms)
    assert result.loc[0, "screening_index"] == 6
    assert result.loc[0, "pdb_id"] == "1gci"
    assert result.loc[0, "chain_id"] == "A"
    assert result.loc[0, "uniprot_id"] == "P29600"
    assert result.loc[0, "source"] == "screening_pool"
    assert pd.isna(result.loc[1, "screening_index"])
    assert result.loc[1, "source"] == "reference_pair"
    assert result.loc[1, "pair_name"] == "1ake_A__P69441"


@pytest.mark.parametrize(
    "case",
    [
        "duplicate_mechanism_pilot_id",
        "unknown_mechanism_pilot_id",
        "duplicate_manifest_pilot_id",
        "pair_name_conflict",
        "preexisting_identity_conflict",
    ],
)
def test_mechanism_identity_rejects_ambiguous_or_conflicting_input(
    summary_module: ModuleType,
    case: str,
) -> None:
    pilot = pd.DataFrame(
        [
            {
                "pilot_id": "screening-006",
                "screening_index": 6,
                "pdb_id": "1gci",
                "chain_id": "A",
                "uniprot_id": "P29600",
                "pair_name": "1gci_A__P29600",
                "source": "screening_pool",
                "pilot_role": "high_confidence_positive",
            }
        ]
    )
    mechanisms = pd.DataFrame(
        [
            {
                "pilot_id": "screening-006",
                "pair_name": "1gci_A__P29600",
            }
        ]
    )
    if case == "duplicate_mechanism_pilot_id":
        mechanisms = pd.concat([mechanisms, mechanisms], ignore_index=True)
    elif case == "unknown_mechanism_pilot_id":
        mechanisms.loc[0, "pilot_id"] = "screening-999"
    elif case == "duplicate_manifest_pilot_id":
        pilot = pd.concat([pilot, pilot], ignore_index=True)
    elif case == "pair_name_conflict":
        mechanisms.loc[0, "pair_name"] = "wrong_A__P29600"
    elif case == "preexisting_identity_conflict":
        mechanisms["pdb_id"] = "wrong"

    with pytest.raises(ValueError):
        summary_module.enrich_pilot_mechanism_identity(
            mechanisms,
            pilot,
        )


def test_mechanism_identity_rejects_join_row_count_change(
    summary_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pilot = pd.DataFrame(
        [
            {
                "pilot_id": "screening-006",
                "screening_index": 6,
                "source": "screening_pool",
                "pdb_id": "1gci",
                "chain_id": "A",
                "uniprot_id": "P29600",
                "pair_name": "1gci_A__P29600",
                "pilot_role": "high_confidence_positive",
            }
        ]
    )
    mechanisms = pd.DataFrame(
        [
            {
                "pilot_id": "screening-006",
                "pair_name": "1gci_A__P29600",
            }
        ]
    )
    original_merge = pd.DataFrame.merge

    def expanding_merge(
        frame: pd.DataFrame,
        *args: object,
        **kwargs: object,
    ) -> pd.DataFrame:
        merged = original_merge(frame, *args, **kwargs)
        return pd.concat([merged, merged.iloc[[0]]], ignore_index=True)

    monkeypatch.setattr(pd.DataFrame, "merge", expanding_merge)

    with pytest.raises(ValueError, match="changed row count"):
        summary_module.enrich_pilot_mechanism_identity(
            mechanisms,
            pilot,
        )


@pytest.mark.parametrize(
    ("name", "evidence", "category"),
    [
        (
            "1AKE",
            complete_evidence(
                preflight_status="reference_quality_pass",
                ca_disagreement_p90=1.031,
                disagreement_segments=(
                    DisagreementSegment(42, 48, 7, 96.5, 2.69),
                ),
                segment_context_status="complete",
            ),
            "high_confidence_state_disagreement",
        ),
        ("Index 6", complete_evidence(), "easy_control"),
        (
            "Index 8",
            complete_evidence(
                ca_disagreement_p90=0.76,
                disagreement_segments=(
                    DisagreementSegment(10, 12, 3, 97.0, 1.2),
                ),
                segment_context_status="complete",
            ),
            "ordinary_or_unclassified",
        ),
        ("Index 24", complete_evidence(), "easy_control"),
        (
            "Index 36",
            complete_evidence(
                observed_ca_fraction=0.943,
                ca_disagreement_p90=9.79,
                longest_internal_below_80_length=7,
                pae_strata={
                    "48_95": PaeStratumStats(
                        9192, 2.0, 20.0, 5.6, 0.175, 0.145
                    ),
                    "96_plus": PaeStratumStats(
                        14028, 3.5, 26.0, 9.72, 0.367, 0.320
                    ),
                },
                disagreement_segments=(
                    DisagreementSegment(10, 11, 2, 98.0, 5.0),
                ),
                segment_context_status="complete",
            ),
            "high_pae_long_range",
        ),
        (
            "Index 35",
            complete_evidence(
                preflight_status="warn_construct_difference",
                pilot_role="construct_difference_negative_control",
                full_length_mapping_coverage=0.898,
            ),
            "construct_difference_control",
        ),
    ],
)
def test_reference_and_geometry_pilot_regressions(
    config: dict,
    name: str,
    evidence: A0Evidence,
    category: str,
) -> None:
    assert name
    assert classify_candidate(evidence, config).primary_category == category
