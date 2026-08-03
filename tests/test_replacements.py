from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from dual_uq.replacements import (
    build_replacement_audit,
    build_replacement_shortlist,
    evaluate_full_length_proxy,
    merge_replacement_preflight,
)

THRESHOLDS = {
    "min_pdb_to_uniprot_length_ratio": 0.90,
    "max_pdb_to_uniprot_length_ratio": 1.10,
    "min_full_length_mapping_coverage": 0.90,
    "min_entity_mapping_coverage": 0.90,
    "min_sequence_identity": 0.95,
    "min_observed_ca_fraction": 0.90,
}
CONFIDENCE_RANKING = {
    "enabled": True,
    "direction": "ascending",
    "hard_max": None,
}
ROOT = Path(__file__).resolve().parents[1]


def _quality_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "canonical_uniprot_length": 100,
        "pdb_entity_length": 100,
        "full_length_mapping_coverage": 0.95,
        "entity_mapping_coverage": 0.95,
        "sequence_identity": 0.98,
        "observed_ca_fraction_of_mapped": 0.95,
        "afdb_global_plddt": 80.0,
        "preflight_status": "pass_full_length",
    }
    row.update(overrides)
    return row


def _candidate(index: int, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "pdb_id": f"{index:04d}"[-4:],
        "chain_id": "A",
        "uniprot_id": f"P{index:05d}",
        "length": 100,
        "canonical_uniprot_length": 100,
        "afdb_global_plddt": 70.0 + index / 100,
        "resolution": 1.0 + index / 100,
        "organism": f"organism-{index % 8}",
        "sequence_cluster": f"cluster-{index}",
        "discovery_status": "eligible",
        "experimental_method": "X-RAY DIFFRACTION",
        "protein_chain_count": 1,
        "full_length_mapping_coverage": 0.95,
        "entity_mapping_coverage": 0.95,
        "sequence_identity": 0.98,
        "observed_ca_fraction_of_mapped": 0.95,
    }
    row.update(overrides)
    return row


@pytest.mark.parametrize("ratio", [0.90, 1.00, 1.10])
def test_length_ratio_boundaries_are_inclusive(ratio: float) -> None:
    result = evaluate_full_length_proxy(
        _quality_row(pdb_entity_length=ratio * 100),
        THRESHOLDS,
    )

    assert result.eligible
    assert result.pdb_to_uniprot_length_ratio == pytest.approx(ratio)


@pytest.mark.parametrize(
    ("ratio", "reason"),
    [(0.899, "low_length_ratio"), (1.101, "high_length_ratio")],
)
def test_length_ratio_outside_boundary_fails(ratio: float, reason: str) -> None:
    result = evaluate_full_length_proxy(
        _quality_row(pdb_entity_length=ratio * 100),
        THRESHOLDS,
    )

    assert not result.eligible
    assert reason in result.exclusion_reasons


@pytest.mark.parametrize(
    ("field", "boundary", "reason"),
    [
        (
            "full_length_mapping_coverage",
            0.90,
            "low_full_length_mapping_coverage",
        ),
        ("entity_mapping_coverage", 0.90, "low_entity_mapping_coverage"),
        ("sequence_identity", 0.95, "low_sequence_identity"),
        (
            "observed_ca_fraction_of_mapped",
            0.90,
            "low_observed_ca_fraction",
        ),
    ],
)
def test_quality_boundaries_are_inclusive(
    field: str,
    boundary: float,
    reason: str,
) -> None:
    at_boundary = evaluate_full_length_proxy(
        _quality_row(**{field: boundary}),
        THRESHOLDS,
    )
    below_boundary = evaluate_full_length_proxy(
        _quality_row(**{field: boundary - 0.001}),
        THRESHOLDS,
    )

    assert at_boundary.eligible
    assert not below_boundary.eligible
    assert reason in below_boundary.exclusion_reasons


@pytest.mark.parametrize(
    ("field", "reason"),
    [
        ("canonical_uniprot_length", "missing_canonical_uniprot_length"),
        ("pdb_entity_length", "missing_pdb_entity_length"),
        (
            "full_length_mapping_coverage",
            "missing_full_length_mapping_coverage",
        ),
        ("entity_mapping_coverage", "missing_entity_mapping_coverage"),
        ("sequence_identity", "missing_sequence_identity"),
        (
            "observed_ca_fraction_of_mapped",
            "missing_observed_ca",
        ),
        ("afdb_global_plddt", "missing_afdb_global_plddt"),
    ],
)
def test_missing_required_metric_fails_with_explicit_reason(
    field: str,
    reason: str,
) -> None:
    result = evaluate_full_length_proxy(
        _quality_row(**{field: float("nan")}),
        THRESHOLDS,
    )

    assert not result.eligible
    assert reason in result.exclusion_reasons


def test_all_exclusion_reasons_are_retained() -> None:
    result = evaluate_full_length_proxy(
        _quality_row(
            pdb_entity_length=80,
            full_length_mapping_coverage=0.80,
            observed_ca_fraction_of_mapped=float("nan"),
        ),
        THRESHOLDS,
    )

    assert result.exclusion_reasons == (
        "low_length_ratio",
        "low_full_length_mapping_coverage",
        "missing_observed_ca",
    )


def test_existing_pool_and_internal_uniprot_duplicates_are_excluded() -> None:
    discovered = pd.DataFrame(
        [
            _candidate(1, uniprot_id="EXISTING"),
            _candidate(
                2,
                uniprot_id="DUPLICATE",
                afdb_global_plddt=60.0,
                resolution=1.5,
            ),
            _candidate(
                3,
                uniprot_id="DUPLICATE",
                afdb_global_plddt=60.0,
                resolution=1.0,
            ),
            *[_candidate(i) for i in range(4, 25)],
        ]
    )
    current = pd.DataFrame(
        [
            {
                "pdb_id": "xxxx",
                "chain_id": "A",
                "uniprot_id": "EXISTING",
            }
        ]
    )

    result = build_replacement_shortlist(
        discovered,
        current,
        target_count=15,
        confidence_ranking=CONFIDENCE_RANKING,
        diversity_config={
            "unique_uniprot": True,
            "unique_sequence_cluster": True,
            "max_per_organism": 4,
        },
        seed=20260730,
    )

    assert "EXISTING" not in set(result["uniprot_id"])
    assert result["uniprot_id"].is_unique
    duplicate = result.loc[result["uniprot_id"] == "DUPLICATE"].iloc[0]
    assert duplicate["pdb_id"] == "0003"


def test_duplicate_priority_uses_mapping_and_observed_ca_before_resolution() -> None:
    discovered = pd.DataFrame(
        [
            _candidate(
                1,
                uniprot_id="DUP",
                afdb_global_plddt=60.0,
                full_length_mapping_coverage=0.91,
                observed_ca_fraction_of_mapped=0.99,
                resolution=0.5,
            ),
            _candidate(
                2,
                uniprot_id="DUP",
                afdb_global_plddt=60.0,
                full_length_mapping_coverage=0.99,
                observed_ca_fraction_of_mapped=0.91,
                resolution=2.0,
            ),
            *[_candidate(i) for i in range(3, 22)],
        ]
    )

    result = build_replacement_shortlist(
        discovered,
        pd.DataFrame(columns=["pdb_id", "chain_id", "uniprot_id"]),
        target_count=15,
        confidence_ranking=CONFIDENCE_RANKING,
        diversity_config={"unique_uniprot": True},
        seed=20260730,
    )

    duplicate = result.loc[result["uniprot_id"] == "DUP"].iloc[0]
    assert duplicate["pdb_id"] == "0002"


def test_global_plddt_is_a_ranking_prior_not_a_hard_qualification_gate() -> None:
    discovered = pd.DataFrame(
        [
            *[_candidate(i, afdb_global_plddt=80 + i / 100) for i in range(1, 6)],
            *[_candidate(i, afdb_global_plddt=90 + i / 100) for i in range(6, 21)],
        ]
    )

    result = build_replacement_shortlist(
        discovered,
        pd.DataFrame(columns=["pdb_id", "chain_id", "uniprot_id"]),
        target_count=15,
        confidence_ranking=CONFIDENCE_RANKING,
        diversity_config={"unique_uniprot": True},
        seed=20260730,
    )

    assert len(result) == 15
    assert result["afdb_global_plddt"].is_monotonic_increasing
    assert (result["afdb_global_plddt"] > 85).any()
    assert result.attrs["selection_audit"]["shortfall_count"] == 0
    assert "is_low_conf_local" not in result.columns
    assert "primary_category" not in result.columns


@pytest.mark.parametrize("target_count", [14, 21])
def test_target_count_must_be_between_15_and_20(target_count: int) -> None:
    with pytest.raises(ValueError, match="15.*20"):
        build_replacement_shortlist(
            pd.DataFrame([_candidate(i) for i in range(1, 22)]),
            pd.DataFrame(columns=["pdb_id", "chain_id", "uniprot_id"]),
            target_count=target_count,
            confidence_ranking=CONFIDENCE_RANKING,
            diversity_config={},
            seed=20260730,
        )


def test_shortfall_is_reported_without_duplication_or_threshold_relaxation() -> None:
    result = build_replacement_shortlist(
        pd.DataFrame([_candidate(i) for i in range(1, 4)]),
        pd.DataFrame(columns=["pdb_id", "chain_id", "uniprot_id"]),
        target_count=15,
        confidence_ranking=CONFIDENCE_RANKING,
        diversity_config={},
        seed=20260730,
    )

    assert len(result) == 3
    assert result.attrs["selection_audit"]["shortfall_count"] == 12
    assert result["uniprot_id"].is_unique


def test_diversity_constraints_and_unknown_values_are_stable() -> None:
    discovered = pd.DataFrame(
        [
            _candidate(
                i,
                organism=None if i <= 2 else f"organism-{i % 4}",
                sequence_cluster=None if i <= 2 else f"cluster-{i}",
            )
            for i in range(1, 25)
        ]
    )

    result = build_replacement_shortlist(
        discovered,
        pd.DataFrame(columns=["pdb_id", "chain_id", "uniprot_id"]),
        target_count=15,
        confidence_ranking=CONFIDENCE_RANKING,
        diversity_config={
            "unique_uniprot": True,
            "unique_sequence_cluster": True,
            "max_per_organism": 4,
        },
        seed=20260730,
    )

    assert len(result) == 15
    assert result["organism"].notna().all()
    assert result["sequence_cluster"].notna().all()
    assert result["sequence_cluster"].is_unique
    assert result["organism"].value_counts().max() <= 4


def test_unsupported_fragment_remains_a_distinct_status_after_exact_join() -> None:
    shortlist = pd.DataFrame(
        [
            {
                **_candidate(1),
                "screening_index": 101,
            }
        ]
    )
    preflight = pd.DataFrame(
        [
            {
                "screening_index": 101,
                "pdb_id": "0001",
                "chain_id": "A",
                "uniprot_id": "P00001",
                "preflight_status": "unsupported_afdb_fragment",
                "preflight_reason": "no_afdb_fragment_covers_mapped_interval",
            }
        ]
    )

    result = merge_replacement_preflight(shortlist, preflight)

    assert result.iloc[0]["preflight_status"] == "unsupported_afdb_fragment"
    assert result.iloc[0]["preflight_reason"] == (
        "no_afdb_fragment_covers_mapped_interval"
    )


def test_preflight_merge_requires_full_composite_key_match() -> None:
    shortlist = pd.DataFrame(
        [{**_candidate(1), "screening_index": 101}]
    )
    preflight = pd.DataFrame(
        [
            {
                "screening_index": 101,
                "pdb_id": "wrong",
                "chain_id": "A",
                "uniprot_id": "P00001",
                "preflight_status": "pass_full_length",
            }
        ]
    )

    with pytest.raises(ValueError, match="composite key"):
        merge_replacement_preflight(shortlist, preflight)


def test_preflight_merge_preserves_discovery_metrics_for_runtime_failure() -> None:
    shortlist = pd.DataFrame(
        [
            {
                **_candidate(1),
                "screening_index": 101,
                "pdb_to_uniprot_length_ratio": 1.0,
            },
            {
                **_candidate(2),
                "screening_index": 102,
                "pdb_to_uniprot_length_ratio": 0.98,
            },
        ]
    )
    preflight = pd.DataFrame(
        [
            {
                "screening_index": 101,
                "pdb_id": "0001",
                "chain_id": "A",
                "uniprot_id": "P00001",
                "pdb_to_uniprot_length_ratio": 1.0,
                "preflight_status": "pass_full_length",
            },
            {
                "screening_index": 102,
                "pdb_id": "0002",
                "chain_id": "A",
                "uniprot_id": "P00002",
                "pdb_to_uniprot_length_ratio": float("nan"),
                "preflight_status": "failed_runtime",
            },
        ]
    )

    result = merge_replacement_preflight(shortlist, preflight)

    assert result["pdb_to_uniprot_length_ratio"].tolist() == [1.0, 0.98]


def test_runtime_failure_does_not_inherit_discovery_afdb_selection() -> None:
    shortlist = pd.DataFrame(
        [
            {
                **_candidate(1),
                "screening_index": 101,
                "afdb_model_entity_id": "AF-P00001-F1",
            },
            {
                **_candidate(2),
                "screening_index": 102,
                "afdb_model_entity_id": "AF-P00002-F1",
            },
        ]
    )
    preflight = pd.DataFrame(
        [
            {
                "screening_index": 101,
                "pdb_id": "0001",
                "chain_id": "A",
                "uniprot_id": "P00001",
                "afdb_model_entity_id": "AF-P00001-F2",
                "afdb_fragment_start": 201,
                "afdb_fragment_end": 400,
                "preflight_status": "pass_full_length",
            },
            {
                "screening_index": 102,
                "pdb_id": "0002",
                "chain_id": "A",
                "uniprot_id": "P00002",
                "afdb_model_entity_id": None,
                "afdb_fragment_start": None,
                "afdb_fragment_end": None,
                "preflight_status": "failed_runtime",
            },
        ]
    )

    result = merge_replacement_preflight(shortlist, preflight)

    assert result.loc[0, "afdb_model_entity_id"] == "AF-P00001-F2"
    assert pd.isna(result.loc[1, "afdb_model_entity_id"])
    assert pd.isna(result.loc[1, "afdb_fragment_start"])
    assert pd.isna(result.loc[1, "afdb_fragment_end"])


def test_shortlist_order_is_reproducible_for_same_seed() -> None:
    discovered = pd.DataFrame([_candidate(i) for i in range(1, 25)])
    current = pd.DataFrame(columns=["pdb_id", "chain_id", "uniprot_id"])
    kwargs = {
        "target_count": 15,
        "confidence_ranking": CONFIDENCE_RANKING,
        "diversity_config": {"unique_uniprot": True},
        "seed": 20260730,
    }

    first = build_replacement_shortlist(discovered, current, **kwargs)
    second = build_replacement_shortlist(discovered, current, **kwargs)

    assert first["uniprot_id"].tolist() == second["uniprot_id"].tolist()


def test_confidence_ranking_rejects_nonascending_direction() -> None:
    with pytest.raises(ValueError, match="direction must be ascending"):
        build_replacement_shortlist(
            pd.DataFrame([_candidate(i) for i in range(1, 22)]),
            pd.DataFrame(columns=["pdb_id", "chain_id", "uniprot_id"]),
            target_count=15,
            confidence_ranking={
                **CONFIDENCE_RANKING,
                "direction": "descending",
            },
            diversity_config={"unique_uniprot": True},
            seed=20260730,
        )


def test_audit_reports_real_shortfall_and_structured_status_counts() -> None:
    candidates = pd.DataFrame(
        [
            {
                **_candidate(i),
                **_quality_row(),
                "screening_index": 100 + i,
                "uniprot_id": f"P{i:05d}",
                "preflight_status": status,
                "exclusion_reasons": (
                    "" if status == "pass_full_length" else "preflight_quality_failure"
                ),
            }
            for i, status in enumerate(
                [
                    "pass_full_length",
                    "pass_full_length",
                    "warn_construct_difference",
                    "fail_preflight",
                    "unsupported_afdb_fragment",
                    "failed_runtime",
                ],
                start=1,
            )
        ]
    )

    audit = build_replacement_audit(
        candidates,
        requested_count=15,
        thresholds=THRESHOLDS,
        seed=20260730,
        source_counts={
            "source_eligible_count": 20,
            "existing_pool_exclusions": 2,
            "length_proxy_eligible_count": 10,
            "confidence_ranked_count": 4,
            "diversity_eligible_count": 6,
        },
        ranking_config=CONFIDENCE_RANKING,
    )

    assert audit["selected_shortlist_count"] == 6
    assert audit["shortfall_count"] == 9
    assert audit["pass_full_length_count"] == 2
    assert audit["warn_construct_difference_count"] == 1
    assert audit["fail_preflight_count"] == 1
    assert audit["unsupported_afdb_fragment_count"] == 1
    assert audit["failed_runtime_count"] == 1
    assert audit["warn_count"] == 1
    assert audit["fail_count"] == 1
    assert audit["unsupported_count"] == 1
    assert audit["runtime_failure_count"] == 1
    assert audit["unsupported_indices"] == [105]
    assert not audit["audit_pass"]
    assert "selected_shortlist_below_15" in audit["blocked_reasons"]
    assert "fewer_than_5_full_length_passes" in audit["blocked_reasons"]


def test_audit_accepts_high_global_plddt_when_hard_quality_gates_pass() -> None:
    candidates = pd.DataFrame(
        [
            {
                **_candidate(i, afdb_global_plddt=90.0),
                **_quality_row(afdb_global_plddt=90.0),
                "screening_index": 100 + i,
                "uniprot_id": f"P{i:05d}",
                "preflight_status": "pass_full_length",
                "exclusion_reasons": "",
            }
            for i in range(1, 16)
        ]
    )

    audit = build_replacement_audit(
        candidates,
        requested_count=15,
        thresholds=THRESHOLDS,
        seed=20260730,
        ranking_config=CONFIDENCE_RANKING,
    )

    assert audit["audit_pass"]
    assert audit["global_plddt_min"] == 90.0
    assert audit["global_plddt_q25"] == 90.0
    assert audit["global_plddt_median"] == 90.0
    assert audit["global_plddt_q75"] == 90.0
    assert audit["global_plddt_max"] == 90.0
    assert audit["hard_thresholds"] == THRESHOLDS
    assert audit["ranking_features"][0] == "afdb_global_plddt:ascending"
    assert audit["blocked_reasons"] == []


def test_audit_rejects_nonascending_confidence_ranking() -> None:
    candidates = pd.DataFrame(
        [
            {
                **_candidate(i),
                **_quality_row(),
                "screening_index": 100 + i,
                "uniprot_id": f"P{i:05d}",
                "preflight_status": "pass_full_length",
                "exclusion_reasons": "",
            }
            for i in range(1, 16)
        ]
    )

    audit = build_replacement_audit(
        candidates,
        requested_count=15,
        thresholds=THRESHOLDS,
        seed=20260730,
        ranking_config={
            **CONFIDENCE_RANKING,
            "direction": "descending",
        },
    )

    assert not audit["audit_pass"]
    assert (
        "invalid_confidence_ranking_configuration"
        in audit["blocked_reasons"]
    )


def test_audit_rejects_output_that_is_not_confidence_ranked() -> None:
    candidates = pd.DataFrame(
        [
            {
                **_candidate(i),
                **_quality_row(afdb_global_plddt=100.0 - i),
                "screening_index": 100 + i,
                "uniprot_id": f"P{i:05d}",
                "preflight_status": "pass_full_length",
                "exclusion_reasons": "",
            }
            for i in range(1, 16)
        ]
    )

    audit = build_replacement_audit(
        candidates,
        requested_count=15,
        thresholds=THRESHOLDS,
        seed=20260730,
        ranking_config=CONFIDENCE_RANKING,
    )

    assert not audit["audit_pass"]
    assert "confidence_ranking_order_violation" in audit["blocked_reasons"]


def test_config_declares_global_plddt_as_nonbinding_ranking_prior() -> None:
    config = yaml.safe_load(
        (
            ROOT
            / "configs"
            / "legacy"
            / "a0_screening"
            / "lower_conf_replacement.yaml"
        ).read_text()
    )

    assert {
        name: config["selection"][name] for name in THRESHOLDS
    } == THRESHOLDS
    assert "afdb_global_plddt_max" not in config["selection"]
    assert config["confidence_ranking"]["enabled"] is True
    assert config["confidence_ranking"]["direction"] == "ascending"
    assert config["confidence_ranking"]["hard_max"] is None
    assert (
        config["confidence_ranking"]["historical_nonbinding_reference_max"]
        == 85.0
    )
