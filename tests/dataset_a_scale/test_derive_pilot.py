from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from dual_uq.dataset.derivation import account_stage_failure, verify_bound_hash
from dual_uq.dataset.fragments import (
    resolve_exact_fragment,
    validate_asset_model_binding,
    validate_bound_arrays,
)
from dual_uq.dataset.identity import extract_canonical_sequence
from dual_uq.dataset.mapping import (
    annotate_gap_semantics,
    are_peptide_adjacent,
    audit_sequence_discrepancies,
    compute_pair_quality,
    pair_qc_attrition,
    prepare_confidence_mapping,
)
from dual_uq.dataset.models import DerivationError
from dual_uq.dataset.observability import recompute_sampling_prior, state_segments
from dual_uq.dataset.reporting import write_report_mapping
from dual_uq.dataset_a_scale.pae import AFDBFragment, PAEMappingError

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "dataset_a"
    / "derivation"
    / "derive_pilot.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("derive_pilot", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    namespace = dict(vars(module))
    namespace.update(
        {
            "extract_canonical_sequence": extract_canonical_sequence,
            "verify_bound_hash": verify_bound_hash,
            "annotate_gap_semantics": annotate_gap_semantics,
            "are_peptide_adjacent": are_peptide_adjacent,
            "audit_sequence_discrepancies": audit_sequence_discrepancies,
            "resolve_exact_fragment": resolve_exact_fragment,
            "validate_bound_arrays": validate_bound_arrays,
            "account_stage_failure": account_stage_failure,
            "recompute_sampling_prior": recompute_sampling_prior,
            "_state_segments": state_segments,
            "validate_asset_model_binding": validate_asset_model_binding,
            "write_outputs": write_report_mapping,
            "prepare_confidence_mapping": prepare_confidence_mapping,
            "pair_qc_attrition": pair_qc_attrition,
            "_pair_quality": compute_pair_quality,
            "DerivePilotError": DerivationError,
            "AFDBFragment": AFDBFragment,
            "PAEMappingError": PAEMappingError,
        }
    )
    return SimpleNamespace(**namespace)


def _inventory() -> list[dict[str, object]]:
    rows = []
    values = {
        1: (147, 93.88),
        7: (754, 57.06),
        24: (None, 70.62),
        38: (131, 78.62),
        91: (205, 88.75),
        96: (125, 95.81),
        103: (539, 83.56),
        165: (1367, 40.31),
        208: (923, 79.12),
    }
    indices = sorted([1, 7, 24, 38, 91, 96, 103, 165, 208, *range(300, 339)])
    for offset, index in enumerate(indices):
        length, plddt = values.get(index, (300 + offset, 90.0))
        rows.append(
            {
                "candidate_index": index,
                "canonical_uniprot_length": length,
                "global_pLDDT_proxy": plddt,
            }
        )
    return rows


def _acquisition() -> list[dict[str, object]]:
    classes = {
        1: "needs_mapping_derivation_only",
        24: "needs_both_structure_sides",
        91: "needs_pdb_side_acquisition",
        96: "needs_pdb_side_acquisition",
    }
    indices = sorted([1, 7, 24, 38, 91, 96, 103, 165, 208, *range(300, 339)])
    ranks = {index: offset + 20 for offset, index in enumerate(indices)}
    ranks.update({1: 1, 91: 2, 24: 3, 165: 4, 208: 5, 96: 6})
    return [
        {
            "candidate_index": index,
            "batch1_rank": ranks[index],
            "pair_id": f"pair_{index}",
            "UniProt": f"P{index:05d}",
            "primary_acquisition_class": classes.get(
                index, "needs_afdb_side_acquisition"
            ),
        }
        for index in indices
    ]


def _resolution() -> list[dict[str, object]]:
    record_counts = {7: 2, 24: 3, 38: 3, 103: 2, 208: 3}
    return [
        {
            "candidate_index": index,
            "record_count": count,
            "payload_provenance": "acq3_immutable_rejected_evidence",
            "exact_accession_record_count": 1,
            "resolution_status": "exact_record_resolved",
        }
        for index, count in record_counts.items()
    ]


def _record(accession: str, model: str, start: int, end: int) -> dict[str, object]:
    return {
        "uniprotAccession": accession,
        "uniprotSequence": "A" * end,
        "modelEntityId": model,
        "entryId": model,
        "sequenceStart": start,
        "sequenceEnd": end,
    }


def test_pilot_selection_is_deterministic_and_uses_prederivation_metadata() -> None:
    module = _load_module()

    first = module.select_pilot_panel(_inventory(), _acquisition(), _resolution())
    second = module.select_pilot_panel(
        list(reversed(_inventory())),
        list(reversed(_acquisition())),
        list(reversed(_resolution())),
    )

    assert first == second
    assert [row["candidate_index"] for row in first] == [1, 7, 24, 38, 91, 96, 165, 208]
    assert all(row["selection_roles"] and row["selection_reason"] for row in first)


def test_pilot_selection_binds_acquisition_subset_of_full_inventory() -> None:
    module = _load_module()
    inventory = _inventory() + [
        {
            "candidate_index": 999,
            "canonical_uniprot_length": 1,
            "global_pLDDT_proxy": 1.0,
        }
    ]

    result = module.select_pilot_panel(inventory, _acquisition(), _resolution())

    assert 999 not in {row["candidate_index"] for row in result}
    assert [row["candidate_index"] for row in result] == [1, 7, 24, 38, 91, 96, 165, 208]


def test_exact_sibling_cannot_supply_canonical_sequence_or_model() -> None:
    module = _load_module()
    payload = json.dumps(
        [
            _record("P12345-2", "AF-P12345-2-F1", 1, 4),
            _record("P12345", "AF-P12345-F1", 1, 3),
        ]
    ).encode()

    result = module.extract_canonical_sequence(payload, "P12345")

    assert result["sequence"] == "AAA"
    assert result["model_entity_id"] == "AF-P12345-F1"
    assert result["nonselected_sibling_record_count"] == 1


def test_raw_hash_drift_is_rejected(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "raw.bin"
    path.write_bytes(b"changed")

    with pytest.raises(module.DerivePilotError, match="raw hash drift"):
        module.verify_bound_hash(path, hashlib.sha256(b"original").hexdigest())


def test_true_uniprot_gap_and_compressed_adjacency_remain_distinct() -> None:
    module = _load_module()
    mapping = pd.DataFrame(
        {
            "uniprot_residue_number": [20, 10, 21, 11],
            "auth_asym_id": ["A"] * 4,
            "auth_seq_id": [20, 10, 21, 11],
            "insertion_code": [""] * 4,
            "label_asym_id": ["B"] * 4,
            "label_seq_id": [2, 1, 3, 2],
        }
    )

    table, summary = module.annotate_gap_semantics(mapping)

    assert table["uniprot_position"].tolist() == [10, 11, 20, 21]
    assert table["output_position"].tolist() == [1, 2, 3, 4]
    assert table["segment_id"].tolist() == [1, 1, 2, 2]
    assert summary["segment_count"] == 2
    assert summary["gap_count"] == 1
    assert summary["largest_uniprot_gap"] == 8
    assert pd.isna(table.loc[0, "gap_before"])
    assert pd.isna(table.loc[3, "gap_after"])
    assert table["segment_count"].tolist() == [2, 2, 2, 2]
    assert summary["gap_boundaries"] == [
        {
            "left_uniprot_position": 11,
            "right_uniprot_position": 20,
            "left_output_position": 2,
            "right_output_position": 3,
            "missing_uniprot_start": 12,
            "missing_uniprot_end": 19,
            "gap_length": 8,
        }
    ]
    assert module.are_peptide_adjacent(table.iloc[1], table.iloc[2]) is False
    assert table.loc[1, "label_seq_id"] == 2


def test_mapping_pdb_afdb_mismatches_remain_separate() -> None:
    module = _load_module()
    mapping = pd.DataFrame(
        {"uniprot_position": [1, 2, 3], "mapping_aa": ["A", "D", "G"]}
    )
    pdb = {1: "A", 2: "N", 3: "G"}
    afdb = {1: "A", 2: "D", 3: "V"}

    result = module.audit_sequence_discrepancies(mapping, pdb, afdb)

    assert [row["uniprot_position"] for row in result["mapping_vs_pdb"]] == [2]
    assert [row["uniprot_position"] for row in result["mapping_vs_afdb"]] == [3]
    assert [row["uniprot_position"] for row in result["pdb_vs_afdb"]] == [2, 3]


def test_zero_covering_fragments_is_structured_failure() -> None:
    module = _load_module()
    records = [_record("P12345", "AF-P12345-F1", 1, 50)]

    result = module.resolve_exact_fragment(records, "P12345", (40, 80))

    assert result["fragment_resolution_status"] == "no_full_covering_fragment"
    assert result["full_cover_count"] == 0
    assert result["selected_model_entity_id"] is None


def test_multiple_covering_fragments_is_structured_ambiguity() -> None:
    module = _load_module()
    records = [
        _record("P12345", "AF-P12345-F1", 1, 100),
        _record("P12345", "AF-P12345-F2", 20, 120),
    ]

    result = module.resolve_exact_fragment(records, "P12345", (30, 80))

    assert result["fragment_resolution_status"] == "ambiguous_full_covering_fragments"
    assert result["full_cover_count"] == 2
    assert result["selected_model_entity_id"] is None


def test_pae_length_mismatch_rejected_without_padding_or_trimming() -> None:
    module = _load_module()
    fragment = module.AFDBFragment("AF-P12345-F1", 1, 3, 3)

    with pytest.raises(module.PAEMappingError) as exc:
        module.validate_bound_arrays(fragment, np.zeros((2, 2)), np.ones(3) * 90)

    assert exc.value.code == "pae_fragment_length_mismatch"


def test_confidence_length_mismatch_rejected_without_offset_fallback() -> None:
    module = _load_module()
    fragment = module.AFDBFragment("AF-P12345-F1", 1, 3, 3)

    with pytest.raises(module.DerivePilotError, match="confidence/model length mismatch"):
        module.validate_bound_arrays(fragment, np.zeros((3, 3)), np.ones(2) * 90)


def test_upstream_failure_marks_downstream_unavailable_not_independent() -> None:
    module = _load_module()
    stages = {
        "raw_complete": True,
        "canonical_sequence_complete": True,
        "pair_qc_complete": False,
        "mapping_complete": False,
        "fragment_resolved": False,
        "pae_bound": False,
        "confidence_bound": False,
        "mechanism_observable": False,
    }

    result = module.account_stage_failure(
        stages,
        failure_stage="pair_qc",
        failure_code="no_sifts_mapping",
        attrition_class="mapping_issue",
    )

    assert result["primary_failure_stage"] == "pair_qc"
    assert result["primary_failure_code"] == "no_sifts_mapping"
    assert result["independent_failure_count"] == 1
    assert result["dependent_unavailable_stages"] == [
        "mapping",
        "fragment",
        "pae",
        "confidence",
        "mechanism",
    ]


def test_multilabel_mechanism_evidence_is_preserved_before_sampling_precedence() -> None:
    module = _load_module()
    evidence = {
        "easy_control": False,
        "low_confidence_local": True,
        "high_pae_long_range": True,
        "state_disagreement": False,
        "quality_pass": True,
        "available": {
            "easy_control": True,
            "low_confidence_local": True,
            "high_pae_long_range": True,
            "state_disagreement": True,
        },
    }

    result = module.recompute_sampling_prior(evidence)

    assert result["positive_evidence_available"] == [
        "low_confidence_local_prior",
        "high_pae_long_range_prior",
    ]
    assert result["sampling_stratum_prior_recomputed"] == "high_pae_long_range_prior"


def test_sampling_prior_requires_positive_observable_evidence() -> None:
    module = _load_module()
    evidence = {
        "easy_control": False,
        "low_confidence_local": False,
        "high_pae_long_range": False,
        "state_disagreement": False,
        "quality_pass": False,
        "available": {
            "easy_control": False,
            "low_confidence_local": False,
            "high_pae_long_range": False,
            "state_disagreement": False,
        },
    }

    result = module.recompute_sampling_prior(evidence)

    assert result["mechanism_observability_status"] == "still_unobservable"
    assert result["sampling_stratum_prior_recomputed"] == "uncertain_or_unclassified"
    assert result["positive_evidence_available"] == []


def test_positive_mechanism_evidence_without_quality_pass_is_not_a_sampling_prior() -> None:
    module = _load_module()
    evidence = {
        "easy_control": False,
        "low_confidence_local": True,
        "high_pae_long_range": True,
        "state_disagreement": False,
        "quality_pass": False,
        "available": {
            "easy_control": True,
            "low_confidence_local": True,
            "high_pae_long_range": True,
            "state_disagreement": True,
        },
    }

    result = module.recompute_sampling_prior(evidence)

    assert result["positive_evidence_available"] == [
        "low_confidence_local_prior",
        "high_pae_long_range_prior",
    ]
    assert result["supported_sampling_priors"] == []
    assert result["sampling_stratum_prior_recomputed"] == "uncertain_or_unclassified"


def test_state_segments_form_by_disagreement_before_median_confidence_gate() -> None:
    module = _load_module()
    table = pd.DataFrame(
        {
            "uniprot_position": [1, 2, 3],
            "segment_id": [1, 1, 1],
            "plddt": [100.0, 80.0, 100.0],
            "ca_disagreement": [1.2, 1.2, 1.2],
        }
    )
    config = {
        "thresholds": {
            "high_confidence_state_disagreement": {
                "minimum_segment_median_plddt": 90.0,
                "minimum_segment_disagreement": 1.0,
            }
        }
    }

    segments = module._state_segments(table, config)

    assert len(segments) == 1
    assert segments[0]["residue_count"] == 3
    assert segments[0]["median_plddt"] == 100.0


def test_asset_binding_rejects_same_length_wrong_model_identity() -> None:
    module = _load_module()

    with pytest.raises(module.DerivePilotError, match="asset/model identity"):
        module.validate_asset_model_binding(
            selected_model_id="AF-P12345-F1",
            asset_records={
                "afdb_structure": {"exact_record_model_identity": "AF-P12345-F1"},
                "afdb_pae": {"exact_record_model_identity": "AF-P99999-F1"},
                "afdb_confidence": {"exact_record_model_identity": "AF-P12345-F1"},
            },
        )


def test_report_writer_is_deterministic_and_clean(tmp_path: Path) -> None:
    module = _load_module()
    report = {
        "schema_version": "dataset-a.derive-pilot.v1",
        "summary": {"pilot_candidate_count": 1, "pipeline_verdict": "DERIVATION_PIPELINE_PASS"},
        "candidates": [
            {
                "candidate_index": 1,
                "pair_id": "1abc_A__P12345",
                "selection_roles": ["A"],
                "raw_complete": True,
                "canonical_sequence_complete": True,
                "pair_qc_complete": True,
                "mapping_complete": True,
                "fragment_resolved": True,
                "pae_bound": True,
                "confidence_bound": True,
                "mechanism_observable": True,
                "primary_failure_stage": None,
                "primary_failure_code": None,
                "sampling_stratum_prior_recomputed": "observed_no_positive_prior",
                "z_optional_failure_detail": None,
            }
        ],
    }

    module.write_outputs(report, tmp_path)
    first = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    module.write_outputs(report, tmp_path)
    second = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    assert first == second
    assert set(first) == {
        "derive_pilot_v1.json",
        "derive_pilot_v1.tsv",
        "derive_pilot_v1.md",
    }
    assert b"\r" not in first["derive_pilot_v1.tsv"]
    assert all(
        not line.endswith(b"\t")
        for line in first["derive_pilot_v1.tsv"].splitlines()
    )
    markdown = first["derive_pilot_v1.md"].decode()
    assert "## Stage outcomes" in markdown
    assert "## Candidate audit" in markdown
    assert "primary failure" in markdown


def test_preflight_requires_48_raw_complete_exact_resolved_candidates() -> None:
    module = _load_module()
    bindings = {"inventory_sha256": "a", "batch1_sha256": "b"}
    state = {
        "summary": {
            "candidate_count": 48,
            "candidate_raw_complete": 48,
            "record_level_identity_resolved": 48,
            "current_candidate_identity_failures": 0,
        },
        "source_bindings": bindings,
    }

    result = module.validate_preflight_contract(
        state,
        expected_bindings=bindings,
        acquisition_candidate_indices=list(range(1, 49)),
    )

    assert result["preflight_pass"] is True
    assert result["candidate_count"] == 48


def test_preflight_binding_drift_stops_derivation() -> None:
    module = _load_module()
    state = {
        "summary": {
            "candidate_count": 48,
            "candidate_raw_complete": 48,
            "record_level_identity_resolved": 48,
            "current_candidate_identity_failures": 0,
        },
        "source_bindings": {"inventory_sha256": "changed"},
    }

    with pytest.raises(module.DerivePilotError, match="binding drift"):
        module.validate_preflight_contract(
            state,
            expected_bindings={"inventory_sha256": "expected"},
            acquisition_candidate_indices=list(range(1, 49)),
        )


def test_protocol_binding_is_content_derived_and_order_independent() -> None:
    module = _load_module()
    first = module._protocol_binding(
        {"identity": 0.95, "coverage": 0.9},
        {"easy": {"minimum": 0.8}},
    )
    reordered = module._protocol_binding(
        {"coverage": 0.9, "identity": 0.95},
        {"easy": {"minimum": 0.8}},
    )
    changed = module._protocol_binding(
        {"identity": 0.95, "coverage": 0.9},
        {"easy": {"minimum": 0.7}},
    )

    assert first == reordered
    assert first != changed
    assert len(first) == 64


def test_batch1_rank_is_bound_from_frozen_plan_not_silently_defaulted() -> None:
    module = _load_module()
    acquisition = [
        {"candidate_index": 21, "pair_id": "x"},
        {"candidate_index": 91, "pair_id": "y"},
    ]
    plan = [
        {"candidate_index": 91, "batch1_rank": 2},
        {"candidate_index": 21, "batch1_rank": 20},
    ]

    result = module.attach_batch1_ranks(acquisition, plan)

    assert {row["candidate_index"]: row["batch1_rank"] for row in result} == {
        21: 20,
        91: 2,
    }


def test_confidence_builder_boundary_removes_only_derived_observed_column() -> None:
    module = _load_module()
    mapping = pd.DataFrame(
        {
            "uniprot_residue_number": [1],
            "auth_asym_id": ["A"],
            "auth_seq_id": [42],
            "insertion_code": ["A"],
            "label_asym_id": ["B"],
            "label_seq_id": [7],
            "observed_ca": [True],
        }
    )

    result = module.prepare_confidence_mapping(mapping)

    assert "observed_ca" not in result
    assert result.loc[0, "auth_seq_id"] == 42
    assert result.loc[0, "insertion_code"] == "A"
    assert result.loc[0, "label_seq_id"] == 7


def test_pair_qc_predicate_failure_is_scientific_not_stage_failure() -> None:
    module = _load_module()
    result = module.pair_qc_attrition("pair_qc_fail")

    assert result == {
        "pair_qc_eligible": False,
        "scientific_attrition_flags": ["pair_qc_threshold_not_met"],
        "scientific_attrition_classes": ["protocol_eligibility_issue"],
    }


def test_pair_quality_uses_full_frozen_preflight_metric_contract() -> None:
    module = _load_module()
    mapping = pd.DataFrame(
        {
            "pdb_residue_name": ["ALA", "CYS", "ASP", "GLU"],
            "uniprot_residue_name": ["A", "C", "D", "E"],
            "uniprot_residue_number": [1, 2, 3, 4],
            "auth_asym_id": ["A"] * 4,
            "auth_seq_id": [1, 2, 3, 4],
            "insertion_code": [""] * 4,
            "label_asym_id": ["B"] * 4,
            "label_seq_id": [1, 2, 3, 4],
        }
    )
    ca = mapping[
        [
            "auth_asym_id",
            "auth_seq_id",
            "insertion_code",
            "label_asym_id",
            "label_seq_id",
        ]
    ].assign(x=0.0, y=0.0, z=0.0)
    thresholds = {
        "min_full_length_mapping_coverage": 0.90,
        "min_entity_mapping_coverage": 0.90,
        "min_sequence_identity": 0.95,
        "min_observed_ca_fraction": 0.90,
        "warn_full_length_mapping_coverage": 0.70,
        "max_internal_unmapped_fraction": 0.05,
    }

    result = module._pair_quality(
        mapping,
        ca,
        canonical_length=4,
        pdb_entity_length=4,
        thresholds=thresholds,
    )

    assert result["pair_qc_status"] == "pair_qc_pass"
    assert result["preflight_status"] == "pass_full_length"
    assert result["full_length_mapping_coverage"] == 1.0
    assert result["entity_mapping_coverage"] == 1.0
    assert result["observed_ca_fraction_of_mapped"] == 1.0
