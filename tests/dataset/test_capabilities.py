from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from dual_uq.core.errors import PAEMappingError
from dual_uq.dataset.audits.observability import recompute_sampling_prior, state_segments
from dual_uq.dataset.models import AFDBFragment, DerivationError
from dual_uq.dataset.policies.fragments import (
    resolve_exact_fragment,
    validate_bound_arrays,
    validate_frozen_model_artifacts,
)
from dual_uq.dataset.policies.identity import (
    extract_canonical_sequence,
    extract_prediction_record_sequence,
)
from dual_uq.dataset.services.mapping import (
    annotate_gap_semantics,
    are_peptide_adjacent,
    audit_sequence_discrepancies,
)
from dual_uq.dataset.stages.resolution import P0ValidationError


def _record(accession: str, model: str, start: int, end: int) -> dict[str, object]:
    return {
        "uniprotAccession": accession,
        "uniprotSequence": "A" * end,
        "modelEntityId": model,
        "entryId": model,
        "sequenceStart": start,
        "sequenceEnd": end,
    }


def test_exact_sibling_never_supplies_sequence_or_model() -> None:
    payload = json.dumps(
        [
            _record("P12345-2", "AF-P12345-2-F1", 1, 4),
            _record("P12345", "AF-P12345-F1", 1, 3),
        ]
    ).encode()

    result = extract_canonical_sequence(payload, "P12345")

    assert result["sequence"] == "AAA"
    assert result["model_entity_id"] == "AF-P12345-F1"
    assert result["nonselected_sibling_record_count"] == 1


def test_fragment_sequence_is_not_promoted_to_canonical_sequence() -> None:
    payload = json.dumps(
        [_record("P0DTD1", "AF-0000000365840311", 1368, 1493) | {
            "uniprotSequence": "A" * 126,
        }]
    ).encode()

    prediction = extract_prediction_record_sequence(payload, "P0DTD1")

    assert prediction["prediction_sequence_length"] == 126
    assert prediction["prediction_interval"] == [1368, 1493]
    assert prediction["model_entity_id"] == "AF-0000000365840311"
    with pytest.raises(
        DerivationError, match="does not establish the canonical UniProt sequence"
    ) as error:
        extract_canonical_sequence(payload, "P0DTD1")
    assert error.value.code == "missing_canonical_sequence_provenance"


def test_true_gap_is_not_compressed_peptide_adjacency() -> None:
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

    table, summary = annotate_gap_semantics(mapping)

    assert table["uniprot_position"].tolist() == [10, 11, 20, 21]
    assert table["output_position"].tolist() == [1, 2, 3, 4]
    assert table["segment_id"].tolist() == [1, 1, 2, 2]
    assert summary["gap_count"] == 1
    assert summary["largest_uniprot_gap"] == 8
    assert are_peptide_adjacent(table.iloc[1], table.iloc[2]) is False


def test_three_sequence_discrepancy_sets_remain_separate() -> None:
    mapping = pd.DataFrame(
        {"uniprot_position": [1, 2, 3], "mapping_aa": ["A", "D", "G"]}
    )

    result = audit_sequence_discrepancies(
        mapping,
        pdb_sequence={1: "A", 2: "N", 3: "G"},
        afdb_sequence={1: "A", 2: "D", 3: "V"},
    )

    assert [row["uniprot_position"] for row in result["mapping_vs_pdb"]] == [2]
    assert [row["uniprot_position"] for row in result["mapping_vs_afdb"]] == [3]
    assert [row["uniprot_position"] for row in result["pdb_vs_afdb"]] == [2, 3]


def test_fragment_resolution_preserves_zero_and_multiple_cover_failures() -> None:
    zero = resolve_exact_fragment(
        [_record("P12345", "AF-P12345-F1", 1, 50)],
        "P12345",
        (40, 80),
    )
    multiple = resolve_exact_fragment(
        [
            _record("P12345", "AF-P12345-F1", 1, 100),
            _record("P12345", "AF-P12345-F2", 20, 120),
        ],
        "P12345",
        (30, 80),
    )

    assert zero["fragment_resolution_status"] == "no_full_covering_fragment"
    assert zero["full_cover_count"] == 0
    assert multiple["fragment_resolution_status"] == "ambiguous_full_covering_fragments"
    assert multiple["full_cover_count"] == 2


def test_bound_arrays_never_pad_trim_or_infer_offsets() -> None:
    fragment = AFDBFragment("AF-P12345-F1", 1, 3, 3)
    with pytest.raises(PAEMappingError) as pae_error:
        validate_bound_arrays(fragment, np.zeros((2, 2)), np.ones(3) * 90)
    with pytest.raises(DerivationError, match="confidence/model length mismatch"):
        validate_bound_arrays(fragment, np.zeros((3, 3)), np.ones(2) * 90)

    assert pae_error.value.code == "pae_fragment_length_mismatch"


def test_frozen_artifact_validation_preserves_structured_failure_code(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def rejected(*args, **kwargs):
        del args, kwargs
        raise P0ValidationError("afdb_model_identity_mismatch", "wrong model")

    monkeypatch.setattr(
        "dual_uq.dataset.policies.fragments._validate_artifact_identities", rejected
    )

    with pytest.raises(DerivationError) as error:
        validate_frozen_model_artifacts(
            {},
            model_id="AF-P12345-F1",
            version=6,
            model_path=tmp_path / "model.cif",
            expected_length=3,
        )

    assert error.value.code == "afdb_model_identity_mismatch"


def test_multilabel_evidence_is_preserved_but_quality_gates_supported_prior() -> None:
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

    result = recompute_sampling_prior(evidence)

    assert result["positive_evidence_available"] == [
        "low_confidence_local_prior",
        "high_pae_long_range_prior",
    ]
    assert result["supported_sampling_priors"] == []
    assert result["sampling_stratum_prior_recomputed"] == "uncertain_or_unclassified"


def test_state_segments_form_before_median_confidence_gate() -> None:
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

    segments = state_segments(table, config)

    assert len(segments) == 1
    assert segments[0]["residue_count"] == 3
    assert segments[0]["median_plddt"] == 100.0
