from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import yaml

from dual_uq.lifecycle import build_candidate_lifecycle, write_lifecycle_outputs

ROOT = Path(__file__).resolve().parents[1]


def _load_script(filename: str, module_name: str) -> ModuleType:
    path = ROOT / "scripts" / filename
    spec = spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def summary_script() -> ModuleType:
    return _load_script(
        "12_build_a0_candidate_summary.py",
        "a0_multi_lifecycle_summary_script",
    )


def _lifecycle_row(
    index: int,
    *,
    pdb_id: str,
    chain_id: str = "A",
    uniprot_id: str,
    source: str,
    status: str = "complete",
    segment_status: str = "successful_no_segments",
) -> dict[str, object]:
    return {
        "screening_index": index,
        "pdb_id": pdb_id,
        "chain_id": chain_id,
        "uniprot_id": uniprot_id,
        "pair_name": f"{pdb_id.lower()}_{chain_id}__{uniprot_id}",
        "source": source,
        "pair_status": status,
        "geometry_status": status,
        "robust_status": status,
        "segment_context_status": segment_status,
    }


def _write_lifecycle(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_lifecycle_custom_output_paths_do_not_overwrite_each_other(
    tmp_path: Path,
) -> None:
    original_csv = tmp_path / "original.csv"
    original_audit = tmp_path / "original.json"
    replacement_csv = tmp_path / "replacement.csv"
    replacement_audit = tmp_path / "replacement.json"

    write_lifecycle_outputs(
        pd.DataFrame([{"screening_index": 1}]),
        {"cohort": "original"},
        output_csv=original_csv,
        output_audit=original_audit,
    )
    write_lifecycle_outputs(
        pd.DataFrame([{"screening_index": 101}]),
        {"cohort": "replacement"},
        output_csv=replacement_csv,
        output_audit=replacement_audit,
    )

    assert pd.read_csv(original_csv)["screening_index"].tolist() == [1]
    assert pd.read_csv(replacement_csv)["screening_index"].tolist() == [101]
    assert '"original"' in original_audit.read_text(encoding="utf-8")
    assert '"replacement"' in replacement_audit.read_text(encoding="utf-8")
    assert not list(tmp_path.glob("*.tmp"))


def test_lifecycle_builder_accepts_pool_with_embedded_preflight_fields(
    tmp_path: Path,
) -> None:
    row = {
        "screening_index": 101,
        "pdb_id": "8pb5",
        "chain_id": "A",
        "uniprot_id": "P0DPA9",
        "preflight_status": "pass_full_length",
        "preflight_reason": "pass",
        "full_length_mapping_coverage": 1.0,
        "entity_mapping_coverage": 0.96,
        "sequence_identity": 0.99,
        "observed_ca_fraction_of_mapped": 1.0,
    }

    lifecycle, audit = build_candidate_lifecycle(
        pool=pd.DataFrame([row]),
        preflight=pd.DataFrame([row]),
        status=pd.DataFrame(),
        pair_root=tmp_path,
    )

    assert lifecycle.loc[0, "preflight_status"] == "pass_full_length"
    assert lifecycle.loc[0, "full_length_mapping_coverage"] == 1.0
    assert audit["preflight_status_counts"] == {"pass_full_length": 1}


def test_summary_merges_multiple_lifecycle_inputs_stably(
    summary_script: ModuleType,
    tmp_path: Path,
) -> None:
    original = tmp_path / "original.csv"
    replacement = tmp_path / "replacement.csv"
    _write_lifecycle(
        original,
        [
            _lifecycle_row(
                6,
                pdb_id="1gci",
                uniprot_id="P29600",
                source="screening_pool",
            )
        ],
    )
    _write_lifecycle(
        replacement,
        [
            _lifecycle_row(
                101,
                pdb_id="8pb5",
                uniprot_id="P0DPA9",
                source="replacement_pool",
            )
        ],
    )

    first = summary_script.load_lifecycle_tables(
        tmp_path,
        [str(original), str(replacement)],
    )
    second = summary_script.load_lifecycle_tables(
        tmp_path,
        [str(replacement), str(original)],
    )

    columns = [
        "screening_index",
        "pdb_id",
        "chain_id",
        "uniprot_id",
        "pair_status",
        "lifecycle_cohort",
        "lifecycle_source_path",
    ]
    pd.testing.assert_frame_equal(first[columns], second[columns])
    assert first["screening_index"].tolist() == [6, 101]
    assert first["pair_status"].tolist() == ["complete", "complete"]
    assert first["lifecycle_cohort"].tolist() == [
        "screening_pool",
        "replacement_pool",
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate_key", "duplicate_candidate_key"),
        ("index_identity", "screening_index_identity_conflict"),
        ("pair_identity", "pair_name_identity_conflict"),
    ],
)
def test_lifecycle_merge_rejects_cross_file_identity_conflicts(
    summary_script: ModuleType,
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    row = _lifecycle_row(
        101,
        pdb_id="8pb5",
        uniprot_id="P0DPA9",
        source="replacement_pool",
    )
    conflict = dict(row)
    if mutation == "index_identity":
        conflict["pdb_id"] = "9xyz"
        conflict["pair_name"] = "9xyz_A__P0DPA9"
    elif mutation == "pair_identity":
        conflict["screening_index"] = 102
        conflict["pdb_id"] = "9xyz"
    _write_lifecycle(first, [row])
    _write_lifecycle(second, [conflict])

    with pytest.raises(ValueError, match=message):
        summary_script.load_lifecycle_tables(
            tmp_path,
            [str(first), str(second)],
        )


def test_lifecycle_validation_rejects_internal_duplicates(
    summary_script: ModuleType,
    tmp_path: Path,
) -> None:
    path = tmp_path / "duplicate.csv"
    row = _lifecycle_row(
        101,
        pdb_id="8pb5",
        uniprot_id="P0DPA9",
        source="replacement_pool",
    )
    _write_lifecycle(path, [row, row])

    with pytest.raises(ValueError, match="duplicate_candidate_key"):
        summary_script.load_lifecycle_tables(tmp_path, [str(path)])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"robust_status": None}, "missing_required_lifecycle_columns"),
        ({"pair_status": "made_up"}, "invalid_lifecycle_status"),
        ({"screening_index": 101.5}, "invalid_screening_index"),
    ],
)
def test_lifecycle_validation_rejects_invalid_schema_and_values(
    summary_script: ModuleType,
    tmp_path: Path,
    mutation: dict[str, object],
    message: str,
) -> None:
    path = tmp_path / "invalid.csv"
    row = _lifecycle_row(
        101,
        pdb_id="8pb5",
        uniprot_id="P0DPA9",
        source="replacement_pool",
    )
    if mutation == {"robust_status": None}:
        row.pop("robust_status")
    else:
        row.update(mutation)
    _write_lifecycle(path, [row])

    with pytest.raises(ValueError, match=message):
        summary_script.load_lifecycle_tables(tmp_path, [str(path)])


def test_summary_lifecycle_default_is_backward_compatible(
    summary_script: ModuleType,
) -> None:
    args = summary_script.parse_args(["--project-root", str(ROOT)])

    assert args.lifecycle is None
    loaded = summary_script.load_lifecycle_tables(ROOT, args.lifecycle)
    expected = pd.read_csv(ROOT / "reports/candidate_lifecycle.csv")
    identity = [
        "screening_index",
        "pdb_id",
        "chain_id",
        "uniprot_id",
        "pair_name",
        "pair_status",
        "geometry_status",
        "robust_status",
        "segment_context_status",
    ]
    pd.testing.assert_frame_equal(
        loaded[identity].reset_index(drop=True),
        expected[identity].reset_index(drop=True),
        check_dtype=False,
    )
    assert len(loaded) == 36


def _replacement_inputs(
    index: int = 101,
) -> tuple[dict[str, object], dict[str, object]]:
    replacement_table = pd.read_csv(
        ROOT / "data/manifests/lower_conf_replacement_pool.tsv",
        sep="\t",
    )
    mapped_table = pd.read_csv(
        ROOT / "reports/replacement_mapped_confidence.csv",
    )
    replacement = replacement_table.loc[
        replacement_table["screening_index"] == index
    ].iloc[0].to_dict()
    mapped = mapped_table.loc[
        mapped_table["screening_index"] == index
    ].iloc[0].to_dict()
    candidate = {
        **replacement,
        "source": "replacement_pool",
        "pair_name": (
            f"{str(replacement['pdb_id']).lower()}_"
            f"{replacement['chain_id']}__{replacement['uniprot_id']}"
        ),
    }
    return candidate, mapped


@pytest.mark.parametrize("index", [101, 113])
def test_current_replacements_remain_partial_without_lifecycle(
    summary_script: ModuleType,
    index: int,
) -> None:
    candidate, mapped = _replacement_inputs(index)
    config = yaml.safe_load(
        (ROOT / "configs/legacy/a0_screening/a0_selection.yaml").read_text(encoding="utf-8")
    )

    row = summary_script._assemble_candidate(
        candidate,
        root=ROOT,
        config=config,
        preflight={},
        lifecycle={},
        replacement=candidate,
        pilot={},
        mechanism={},
        mapped_confidence=mapped,
    )

    assert row["is_low_conf_local"] is True
    assert row["classification_status"] == "partial_classification"
    assert row["selection_eligible"] is False
    assert row["lifecycle_record_available"] is False


@pytest.mark.parametrize(
    "segment_status",
    ["complete", "successful_no_segments"],
)
def test_complete_replacement_lifecycle_propagates_to_classification(
    summary_script: ModuleType,
    tmp_path: Path,
    segment_status: str,
) -> None:
    candidate, mapped = _replacement_inputs(101)
    config = yaml.safe_load(
        (ROOT / "configs/legacy/a0_screening/a0_selection.yaml").read_text(encoding="utf-8")
    )
    pair_dir = tmp_path / "data/processed/pairs" / candidate["pair_name"]
    pair_dir.mkdir(parents=True)
    positions = np.array([1, 50, 100], dtype=int)
    np.savez(
        pair_dir / "pairwise_geometry.npz",
        uniprot_positions=positions,
        symmetric_pae=np.array(
            [
                [0.0, 3.0, 3.0],
                [3.0, 0.0, 3.0],
                [3.0, 3.0, 0.0],
            ]
        ),
    )
    if segment_status == "complete":
        pd.DataFrame(
            [
                {
                    "start_position": 10,
                    "end_position": 11,
                    "residue_count": 2,
                    "median_plddt": 70.0,
                    "median_disagreement": 0.5,
                }
            ]
        ).to_csv(pair_dir / "disagreement_segments.csv", index=False)
    lifecycle = {
        **_lifecycle_row(
            101,
            pdb_id="8pb5",
            uniprot_id="P0DPA9",
            source="replacement_pool",
            segment_status=segment_status,
        ),
        "lifecycle_source_path": "/tmp/replacement.csv",
        "lifecycle_cohort": "replacement_pool",
    }
    mechanism = {
        "confidence_model_match": True,
        "selected_afdb_model_entity_id": "AF-P0DPA9-F1",
        "plddt_model_entity_id": "AF-P0DPA9-F1",
        "pae_model_entity_id": "AF-P0DPA9-F1",
        "geometry_model_entity_id": "AF-P0DPA9-F1",
        "selected_afdb_version": 6,
        "plddt_model_version": 6,
        "pae_model_version": 6,
        "geometry_model_version": 6,
        "ca_disagreement_p90": 0.5,
    }

    row = summary_script._assemble_candidate(
        candidate,
        root=tmp_path,
        config=config,
        preflight={},
        lifecycle=lifecycle,
        replacement=candidate,
        pilot={},
        mechanism=mechanism,
        mapped_confidence=mapped,
    )

    assert row["pair_status"] == "complete"
    assert row["segment_context_status"] == segment_status
    assert row["lifecycle_record_available"] is True
    assert row["full_diagnostic_complete"] is True
    assert row["classification_status"] == "complete_classification"
    assert row["selection_eligible"] is True
    assert row["is_low_conf_local"] is True
    assert row["primary_category"] == "low_confidence_local"


def test_explicit_not_started_lifecycle_is_not_promoted(
    summary_script: ModuleType,
) -> None:
    candidate, mapped = _replacement_inputs(101)
    config = yaml.safe_load(
        (ROOT / "configs/legacy/a0_screening/a0_selection.yaml").read_text(encoding="utf-8")
    )
    lifecycle = _lifecycle_row(
        101,
        pdb_id="8pb5",
        uniprot_id="P0DPA9",
        source="replacement_pool",
        status="not_started",
        segment_status="not_started",
    )

    row = summary_script._assemble_candidate(
        candidate,
        root=ROOT,
        config=config,
        preflight={},
        lifecycle=lifecycle,
        replacement=candidate,
        pilot={},
        mechanism={},
        mapped_confidence=mapped,
    )

    assert row["lifecycle_record_available"] is True
    assert row["full_diagnostic_complete"] is False
    assert row["classification_status"] == "partial_classification"
    assert row["selection_eligible"] is False
