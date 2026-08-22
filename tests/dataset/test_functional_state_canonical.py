from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import dual_uq.dataset.functional_state_canonical as canonical_module
from dual_uq.benchmark.schema_registry import SchemaRegistry
from dual_uq.dataset.functional_state_canonical import (
    FunctionalStateCanonicalConfig,
    FunctionalStateCanonicalError,
    _merge_existing_canonical_frames,
    _write_table,
    canonicalize_admission_tables,
    materialize_canonical_tables,
    repair_primary_residue_mapping_identity,
    validate_canonical_tables,
)

REPO_ROOT = Path(__file__).parents[2]


def _frames() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    candidate_pairs = pd.DataFrame(
        [
            {
                "pair_id": "P1__1aaa_1__2bbb_1",
                "uniprot_id": "P1",
                "state_family": "OPEN_CLOSED",
                "state_a": "OPEN",
                "state_b": "CLOSED",
                "entity_a": "1aaa_1",
                "entity_b": "2bbb_1",
                "pdb_a": "1aaa",
                "pdb_b": "2bbb",
                "chain_a": "A",
                "chain_b": "A",
                "sequence_exact": True,
                "state_contrast_category": "FUNCTIONAL_STATE_INDEPENDENT_OF_LIGAND_LABEL",
                "final_pair_state": "PRIMARY",
                "admission_status": "ADMITTED",
                "admission_reasons": None,
                "common_mapped_count": 2,
                "common_mapped_fraction": 1.0,
                "common_coordinate_visible_count": 2,
                "common_coordinate_visible_fraction": 1.0,
                "construct_overlap_fraction": 1.0,
                "assembly_comparable": True,
                "mapping_valid": True,
            },
            {
                "pair_id": "P2__3ccc_1__4ddd_1",
                "uniprot_id": "P2",
                "state_family": "PRE_POST",
                "state_a": "PRE_TRANSITION",
                "state_b": "POST_TRANSITION",
                "entity_a": "3ccc_1",
                "entity_b": "4ddd_1",
                "pdb_a": "3ccc",
                "pdb_b": "4ddd",
                "chain_a": "A",
                "chain_b": "A",
                "sequence_exact": False,
                "state_contrast_category": "UNRESOLVED_LIGAND_CONTEXT",
                "final_pair_state": "UNRESOLVED",
                "admission_status": "UNRESOLVED",
                "admission_reasons": "unresolved_state_evidence",
            },
        ]
    )
    structures = pd.DataFrame(
        [
            {
                "pdb_id": pdb,
                "polymer_entity_id": entity,
                "chain_id": "A",
                "uniprot_id": protein,
                "canonical_sequence": sequence,
                "canonical_sequence_status": "BOUND",
                "canonical_sequence_length": len(sequence),
                "pdb_asset_status": "AVAILABLE",
                "uniprot_asset_status": "AVAILABLE",
                "mapping_status": "MAPPING_VALID",
                "mapping_relative_path": f"data/raw/functional_states/sifts/{pdb}.xml.gz",
                "mmcif_relative_path": f"data/raw/functional_states/pdb/{pdb}.cif",
                "uniprot_relative_path": f"data/raw/functional_states/uniprot/{protein}.json",
                "assembly_ids": "1",
                "experimental_method": "X-RAY DIFFRACTION",
                "resolution": 2.0,
            }
            for pdb, entity, protein, sequence in (
                ("1aaa", "1aaa_1", "P1", "AC"),
                ("2bbb", "2bbb_1", "P1", "AC"),
                ("3ccc", "3ccc_1", "P2", "AC"),
                ("4ddd", "4ddd_1", "P2", "AC"),
            )
        ]
    )
    annotations = pd.DataFrame(
        [
            {"polymer_entity_id": "1aaa_1", "state_label": "OPEN", "state_family": "OPEN_CLOSED", "evidence_tier": "TIER_1", "evidence_source": "paper", "evidence_text": "open"},
            {"polymer_entity_id": "2bbb_1", "state_label": "CLOSED", "state_family": "OPEN_CLOSED", "evidence_tier": "TIER_1", "evidence_source": "paper", "evidence_text": "closed"},
        ]
    )
    ledger = pd.DataFrame(
        [
            {"asset_type": "pdb_mmcif", "pdb_id": pdb, "uniprot_id": None, "relative_path": f"data/raw/functional_states/pdb/{pdb}.cif", "sha256": f"sha-{pdb}", "retrieval_status": "AVAILABLE_LOCAL_REUSE"}
            for pdb in ("1aaa", "2bbb", "3ccc", "4ddd")
        ]
        + [
            {"asset_type": "sifts_xml", "pdb_id": pdb, "uniprot_id": None, "relative_path": f"data/raw/functional_states/sifts/{pdb}.xml.gz", "sha256": f"sha-sifts-{pdb}", "retrieval_status": "AVAILABLE_LOCAL_REUSE"}
            for pdb in ("1aaa", "2bbb", "3ccc", "4ddd")
        ]
        + [
            {"asset_type": "uniprot_canonical_json", "pdb_id": None, "uniprot_id": protein, "relative_path": f"data/raw/functional_states/uniprot/{protein}.json", "sha256": f"sha-{protein}", "retrieval_status": "AVAILABLE_LOCAL_REUSE"}
            for protein in ("P1", "P2")
        ]
    )
    admission = {
        "candidate_pairs": candidate_pairs,
        "primary_pairs": candidate_pairs.iloc[[0]].copy(),
        "alternative_pairs": candidate_pairs.iloc[0:0].copy(),
        "structures": structures,
        "annotations": annotations,
        "asset_ledger": ledger,
        "pair_structural_descriptors": pd.DataFrame(),
        "residue_structural_descriptors": pd.DataFrame(),
        "cluster_assignments": pd.DataFrame(),
    }
    discovery = {"candidate_pairs": candidate_pairs, "structures": structures, "annotations": annotations}
    return admission, discovery


def _persisted_mapping_with_identity() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pair_id": "P1__1aaa_1__2bbb_1",
                "state_family": "OPEN_CLOSED",
                "canonical_position": 1,
                "canonical_residue": "A",
                "coordinate_observable_a": True,
                "coordinate_observable_b": True,
                "condition_1_residue_id": "A:101",
                "condition_2_residue_id": "B:201",
                "condition_1_aa": "A",
                "condition_2_aa": "A",
            },
            {
                "pair_id": "P1__1aaa_1__2bbb_1",
                "state_family": "OPEN_CLOSED",
                "canonical_position": 2,
                "canonical_residue": "C",
                "coordinate_observable_a": True,
                "coordinate_observable_b": True,
                "condition_1_residue_id": "A:102",
                "condition_2_residue_id": "B:202",
                "condition_1_aa": "S",
                "condition_2_aa": "C",
            },
        ]
    )


def _persisted_mapping_without_identity() -> pd.DataFrame:
    return _persisted_mapping_with_identity().drop(
        columns=[
            "condition_1_residue_id",
            "condition_2_residue_id",
            "condition_1_aa",
            "condition_2_aa",
        ]
    )


def test_canonicalization_keeps_all_terminal_candidates_but_pairs_only_admitted() -> None:
    admission, discovery = _frames()
    admission["residue_mappings"] = _persisted_mapping_with_identity()
    tables = canonicalize_admission_tables(admission, discovery)
    assert len(tables["candidate_attrition"]) == 2
    assert set(tables["condition_pairs"]["pair_id"]) == {"P1__1aaa_1__2bbb_1"}
    assert set(tables["residue_mappings"]["canonical_position"]) == {1, 2}
    assert tables["residue_mappings"]["common_coordinate_visible"].all()
    assert tables["residue_mappings"].loc[1, "condition_1_residue_id"] == "A:102"
    assert tables["residue_mappings"].loc[1, "condition_1_aa"] == "S"
    assert tables["residue_mappings"].loc[1, "canonical_aa"] == "C"
    assert not any("proteinmpnn" in column.lower() or "esm_if1" in column.lower() for column in tables["benchmark_instances"].columns)
    assert set(tables["condition_pairs"]["arm"]) == {"functional_state"}
    assert set(tables["structures"]["arm"]) == {"functional_state"}
    assert set(tables["benchmark_instances"]["arm"]) == {"functional_state"}
    assert not any(
        column.lower() in {"arm_id", "arm_name"}
        for frame in tables.values()
        for column in frame.columns
    )
    validate_canonical_tables(tables)


def test_persisted_mapping_projection_does_not_invoke_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    admission, discovery = _frames()
    admission["residue_mappings"] = _persisted_mapping_with_identity().iloc[[0]].copy()

    def fail_if_called(**_: object) -> None:
        raise AssertionError("canonical materialization must not invoke map_condition_pair")

    monkeypatch.setattr(canonical_module, "map_condition_pair", fail_if_called, raising=False)

    tables = canonicalize_admission_tables(admission, discovery)

    assert len(tables["residue_mappings"]) == 1
    assert bool(tables["residue_mappings"].iloc[0]["common_mapped"])


def test_projection_recovers_identity_from_persisted_sifts_facts_without_canonical_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission, discovery = _frames()
    admission["residue_mappings"] = _persisted_mapping_without_identity()
    facts = {
        "1aaa_1": pd.DataFrame(
            [
                {"canonical_position": 1, "residue_id": "A:101", "aa": "A", "mapped": True},
                {"canonical_position": 2, "residue_id": "A:102", "aa": "S", "mapped": True},
            ]
        ),
        "2bbb_1": pd.DataFrame(
            [
                {"canonical_position": 1, "residue_id": "B:201", "aa": "A", "mapped": True},
                {"canonical_position": 2, "residue_id": "B:202", "aa": "C", "mapped": True},
            ]
        ),
    }
    monkeypatch.setattr(
        canonical_module,
        "_load_persisted_identity_facts",
        lambda *_args, **_kwargs: facts,
        raising=False,
    )

    tables = canonicalize_admission_tables(admission, discovery)

    mapping = tables["residue_mappings"]
    assert mapping.loc[1, "condition_1_residue_id"] == "A:102"
    assert mapping.loc[1, "condition_1_aa"] == "S"
    assert mapping.loc[1, "canonical_aa"] == "C"


def test_projection_recovers_nullable_identity_columns_from_persisted_sifts_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission, discovery = _frames()
    persisted = _persisted_mapping_with_identity()
    persisted.loc[1, [
        "condition_1_residue_id",
        "condition_2_residue_id",
        "condition_1_aa",
        "condition_2_aa",
    ]] = None
    admission["residue_mappings"] = persisted
    facts = {
        "1aaa_1": pd.DataFrame(
            [
                {"canonical_position": 1, "residue_id": "A:101", "aa": "A", "mapped": True},
                {"canonical_position": 2, "residue_id": "A:102", "aa": "S", "mapped": True},
            ]
        ),
        "2bbb_1": pd.DataFrame(
            [
                {"canonical_position": 1, "residue_id": "B:201", "aa": "A", "mapped": True},
                {"canonical_position": 2, "residue_id": "B:202", "aa": "C", "mapped": True},
            ]
        ),
    }
    monkeypatch.setattr(
        canonical_module,
        "_load_persisted_identity_facts",
        lambda *_args, **_kwargs: facts,
        raising=False,
    )

    tables = canonicalize_admission_tables(admission, discovery)

    mapping = tables["residue_mappings"]
    assert mapping.loc[0, "condition_1_residue_id"] == "A:101"
    assert mapping.loc[1, "condition_1_residue_id"] == "A:102"
    assert mapping.loc[1, "condition_1_aa"] == "S"
    assert mapping.loc[1, "canonical_aa"] == "C"


def test_projection_rejects_conflicting_direct_identity_and_persisted_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission, discovery = _frames()
    persisted = _persisted_mapping_with_identity()
    persisted.loc[1, "condition_1_residue_id"] = None
    persisted.loc[1, "condition_1_aa"] = "A"
    admission["residue_mappings"] = persisted
    facts = {
        "1aaa_1": pd.DataFrame(
            [
                {"canonical_position": 1, "residue_id": "A:101", "aa": "A", "mapped": True},
                {"canonical_position": 2, "residue_id": "A:999", "aa": "S", "mapped": True},
            ]
        ),
        "2bbb_1": pd.DataFrame(
            [
                {"canonical_position": 1, "residue_id": "B:201", "aa": "A", "mapped": True},
                {"canonical_position": 2, "residue_id": "B:202", "aa": "C", "mapped": True},
            ]
        ),
    }
    monkeypatch.setattr(
        canonical_module,
        "_load_persisted_identity_facts",
        lambda *_args, **_kwargs: facts,
        raising=False,
    )

    with pytest.raises(FunctionalStateCanonicalError, match="identity conflict"):
        canonicalize_admission_tables(admission, discovery)


def test_primary_identity_repair_preserves_non_identity_mapping_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission, discovery = _frames()
    admission["residue_mappings"] = _persisted_mapping_with_identity()
    tables = canonicalize_admission_tables(admission, discovery)
    current_mapping = tables["residue_mappings"].copy()
    current_mapping.loc[1, [
        "condition_1_residue_id",
        "condition_2_residue_id",
        "condition_1_aa",
        "condition_2_aa",
    ]] = None
    admission["residue_mappings"] = _persisted_mapping_without_identity()
    facts = {
        "1aaa_1": pd.DataFrame(
            [
                {"canonical_position": 1, "residue_id": "A:101", "aa": "A", "mapped": True},
                {"canonical_position": 2, "residue_id": "A:102", "aa": "S", "mapped": True},
            ]
        ),
        "2bbb_1": pd.DataFrame(
            [
                {"canonical_position": 1, "residue_id": "B:201", "aa": "A", "mapped": True},
                {"canonical_position": 2, "residue_id": "B:202", "aa": "C", "mapped": True},
            ]
        ),
    }
    monkeypatch.setattr(
        canonical_module,
        "_load_persisted_identity_facts",
        lambda *_args, **_kwargs: facts,
        raising=False,
    )

    repaired, summary = repair_primary_residue_mapping_identity(
        current_mapping,
        admission["primary_pairs"],
        admission["residue_mappings"],
        admission["structures"],
        admission["asset_ledger"],
        schema_root=REPO_ROOT / "schemas",
        project_root=REPO_ROOT,
    )

    assert summary["identity_cells_missing_before"] == 4
    assert summary["identity_cells_missing_after"] == 0
    pd.testing.assert_frame_equal(
        repaired,
        tables["residue_mappings"],
        check_dtype=False,
    )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("condition_1_residue_id", None, "mapped condition side requires residue_id"),
        ("condition_2_aa", None, "mapped condition side requires aa"),
        ("condition_1_mapped", False, "coordinate-visible condition side must be mapped"),
    ],
)
def test_validate_canonical_tables_enforces_mapping_semantic_invariants(
    column: str,
    value: object,
    message: str,
) -> None:
    admission, discovery = _frames()
    admission["residue_mappings"] = _persisted_mapping_with_identity()
    tables = canonicalize_admission_tables(admission, discovery)
    tables["residue_mappings"].loc[0, column] = value

    with pytest.raises(FunctionalStateCanonicalError, match=message):
        validate_canonical_tables(tables)


def test_validate_canonical_tables_rejects_common_mapping_without_both_sides() -> None:
    admission, discovery = _frames()
    admission["residue_mappings"] = _persisted_mapping_with_identity()
    tables = canonicalize_admission_tables(admission, discovery)
    tables["residue_mappings"].loc[0, "condition_2_mapped"] = False
    tables["residue_mappings"].loc[0, "condition_2_coordinate_visible"] = False

    with pytest.raises(
        FunctionalStateCanonicalError,
        match="common_mapped requires both condition sides mapped",
    ):
        validate_canonical_tables(tables)


def test_validate_canonical_tables_rejects_cross_structure_identity_conflict() -> None:
    admission, discovery = _frames()
    admission["residue_mappings"] = _persisted_mapping_with_identity()
    tables = canonicalize_admission_tables(admission, discovery)

    duplicate_pair = tables["condition_pairs"].iloc[[0]].copy()
    duplicate_pair["pair_id"] = "P1_DUPLICATE"
    tables["condition_pairs"] = pd.concat(
        [tables["condition_pairs"], duplicate_pair], ignore_index=True
    )
    duplicate_instance = tables["benchmark_instances"].iloc[[0]].copy()
    duplicate_instance["instance_id"] = "benchmark_instance:duplicate"
    duplicate_instance["pair_id"] = "P1_DUPLICATE"
    tables["benchmark_instances"] = pd.concat(
        [tables["benchmark_instances"], duplicate_instance], ignore_index=True
    )
    duplicate_mapping = tables["residue_mappings"].copy()
    duplicate_mapping["pair_id"] = "P1_DUPLICATE"
    duplicate_mapping.loc[duplicate_mapping["canonical_position"].eq(1), "condition_1_residue_id"] = "A:999"
    tables["residue_mappings"] = pd.concat(
        [tables["residue_mappings"], duplicate_mapping], ignore_index=True
    )

    with pytest.raises(FunctionalStateCanonicalError, match="structure identity mapping consistency"):
        validate_canonical_tables(tables)


def test_canonical_table_writer_ignores_non_persistent_dataframe_attrs(tmp_path: Path) -> None:
    frame = pd.DataFrame({"value": [1]})
    frame.attrs["source_frame"] = pd.DataFrame({"value": [1]})

    _write_table(frame, tmp_path / "table.parquet")

    assert pd.read_parquet(tmp_path / "table.parquet")["value"].tolist() == [1]


def test_candidate_attrition_normalizes_existing_terminal_state_columns() -> None:
    admission, discovery = _frames()
    candidate = admission["candidate_pairs"].copy()
    candidate["terminal_status"] = "STALE"
    candidate["terminal_reason"] = "stale"
    admission["candidate_pairs"] = candidate

    tables = canonicalize_admission_tables(admission, discovery)

    row = tables["candidate_attrition"].iloc[0]
    assert row["terminal_status"] == "PRIMARY"
    assert pd.isna(row["terminal_reason"])


def test_canonical_validation_rejects_unknown_arm_fields() -> None:
    admission, discovery = _frames()
    tables = canonicalize_admission_tables(admission, discovery)
    tables["proteins"]["arm_id"] = "forbidden"
    with pytest.raises(FunctionalStateCanonicalError, match="outside frozen schema"):
        validate_canonical_tables(tables)


def test_materialization_is_immutable_and_writes_minimal_completion_record(tmp_path) -> None:
    admission, discovery = _frames()
    tables = canonicalize_admission_tables(admission, discovery, project_root=REPO_ROOT)
    config = FunctionalStateCanonicalConfig(
        project_root=REPO_ROOT,
        admission_root=tmp_path / "admission",
        discovery_root=tmp_path / "discovery",
        output_root=tmp_path / "benchmark",
        audit_root=tmp_path / "functional_state" / "construction",
    )
    first = materialize_canonical_tables(tables, config, input_artifacts={"discovery/summary.json": "abc"})
    second = materialize_canonical_tables(tables, config, input_artifacts={"discovery/summary.json": "abc"})
    assert first == second
    completion = json.loads((config.audit_root / "construction_completion.json").read_text())
    assert completion["completion_status"] == "SUCCESS"
    assert completion["validation_status"] == "PASSED"
    assert completion["structural_condition_semantics"] == "functional_state"
    assert "artifact_hashes" not in completion
    assert "manifest_sha" not in json.dumps(completion).lower()
    assert (config.output_root / "core" / "condition_pairs.parquet").is_file()
    assert (config.output_root / "annotations" / "pair_structural_descriptors.parquet").is_file()
    assert not (config.output_root / "condition_pairs.parquet").exists()


def test_pure_ligand_state_is_preserved_as_excluded_not_primary() -> None:
    admission, discovery = _frames()
    admission["candidate_pairs"] = admission["candidate_pairs"].copy()
    admission["candidate_pairs"].loc[0, "state_contrast_category"] = "PURE_LIGAND_STATE_ONLY"
    admission["candidate_pairs"].loc[0, "final_pair_state"] = "PRIMARY"
    admission["primary_pairs"] = admission["primary_pairs"].copy()
    admission["primary_pairs"].loc[0, "state_contrast_category"] = "PURE_LIGAND_STATE_ONLY"
    tables = canonicalize_admission_tables(admission, discovery)
    assert tables["condition_pairs"].empty
    row = tables["candidate_attrition"].loc[tables["candidate_attrition"]["pair_id"].eq("P1__1aaa_1__2bbb_1")].iloc[0]
    assert row["terminal_status"] == "EXCLUDED"
    assert row["terminal_reason"] == "pure_ligand_state_only"


def test_canonical_merge_preserves_existing_structural_condition_category(tmp_path) -> None:
    admission, discovery = _frames()
    tables = canonicalize_admission_tables(admission, discovery, project_root=REPO_ROOT)
    existing = tables["condition_pairs"].copy()
    existing.loc[0, "pair_id"] = "ligand-pair"
    existing.loc[0, "arm"] = "ligand_state"
    existing.loc[0, "condition_1_structure_id"] = "ligand-apo-structure"
    existing.loc[0, "condition_2_structure_id"] = "ligand-holo-structure"
    existing.loc[0, "condition_1_label"] = "APO"
    existing.loc[0, "condition_2_label"] = "HOLO"
    destination = tmp_path / "condition_pairs.parquet"
    existing.to_parquet(destination, index=False)

    merged = _merge_existing_canonical_frames(
        {"condition_pairs": tables["condition_pairs"]},
        {"condition_pairs": destination},
        SchemaRegistry(REPO_ROOT / "schemas"),
    )["condition_pairs"]

    assert set(merged["pair_id"]) == {"ligand-pair", "P1__1aaa_1__2bbb_1"}
    assert set(merged["arm"]) == {"ligand_state", "functional_state"}
