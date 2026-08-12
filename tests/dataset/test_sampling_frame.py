from __future__ import annotations

import gzip
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from dual_uq.core.hashing import sha256_file
from dual_uq.core.paths import ProjectPaths
from dual_uq.dataset.sampling_frame import (
    NeutralityEvidence,
    SamplingFrameAuditConfig,
    SamplingFrameAuditError,
    audit_sampling_frame,
    classify_local_availability,
    materialize_sampling_frame_audit,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _paths(root: Path) -> ProjectPaths:
    (root / "src/dual_uq").mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    return ProjectPaths.discover(project_root=root)


def _mmcif(entry_id: str, *, chain: str = "A", entity: str = "1") -> str:
    return f"""data_{entry_id}
_entry.id {entry_id}
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
ATOM 1 C CA ALA {chain} {entity} 1 {chain} 1
"""


def _write_candidate_assets(
    root: Path,
    *,
    pdb_id: str,
    chain: str,
    accession: str,
    with_pdb: bool,
    with_afdb: bool,
    with_mapping: bool,
) -> None:
    if with_pdb:
        path = root / f"data/raw/pdb/{pdb_id}.cif"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_mmcif(pdb_id.upper(), chain=chain), encoding="utf-8")
    if with_afdb:
        afdb = root / f"data/raw/afdb/{accession}/AF-{accession}-F1"
        afdb.mkdir(parents=True, exist_ok=True)
        (afdb / "metadata.json").write_text(
            json.dumps(
                {
                    "uniprotAccession": accession,
                    "modelEntityId": f"AF-{accession}-F1",
                    "sequence": "A",
                    "sequenceStart": 1,
                    "sequenceEnd": 1,
                }
            ),
            encoding="utf-8",
        )
        (afdb / "model.cif").write_text(
            _mmcif(f"AF-{accession}-F1"), encoding="utf-8"
        )
    if with_mapping:
        path = root / f"data/raw/mappings/{pdb_id}.xml.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wb") as handle:
            handle.write(b"<?xml version='1.0'?><entry/>")


def _fixture_config(root: Path) -> SamplingFrameAuditConfig:
    candidates = [
        {
            "candidate_index": 1,
            "canonical_source_row": 0,
            "polymer_entity_id": "1AAA_1",
            "pair_id": "1aaa_A__P00001",
            "PDB": "1aaa",
            "chain": "A",
            "UniProt": "P00001",
        },
        {
            "candidate_index": 2,
            "canonical_source_row": 1,
            "polymer_entity_id": "2BBB_1",
            "pair_id": "2bbb_A__P00002",
            "PDB": "2bbb",
            "chain": "A",
            "UniProt": "P00002",
        },
        {
            "candidate_index": 3,
            "canonical_source_row": 2,
            "polymer_entity_id": "3CCC_1",
            "pair_id": "3ccc_A__P00003",
            "PDB": "3ccc",
            "chain": "A",
            "UniProt": "P00003",
        },
    ]
    discovery_path = root / "data/processed/discovery/discovered_candidates.parquet"
    discovery_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "polymer_entity_id": row["polymer_entity_id"],
                "pdb_id": row["PDB"],
                "chain_id": row["chain"],
                "uniprot_id": row["UniProt"],
                "discovery_status": "eligible",
            }
            for row in candidates
        ]
    ).to_parquet(discovery_path, index=False)
    inventory_path = root / "artifacts/dataset/reports/census/candidate_inventory_v1.json"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(
        json.dumps(
            {
                "schema_version": "fixture.inventory.v1",
                "inventory_status": "metadata_only_plan_not_admission",
                "candidate_index_semantics": {
                    "ordering": "eligible canonical source order",
                    "identity": "inventory convenience ID only",
                },
                "source_metadata": {
                    "canonical_source_path": (
                        "data/processed/discovery/discovered_candidates.parquet"
                    ),
                    "source_sha256": sha256_file(discovery_path),
                    "total_row_count": 3,
                    "eligible_row_count": 3,
                    "indexing_policy_version": "inventory-source-order-v1",
                },
                "candidates": candidates,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(candidates).to_csv(
        inventory_path.with_suffix(".tsv"), sep="\t", index=False
    )

    source_refs = (
        "scripts/13_discover_screening_pool.py",
        "src/dual_uq/rcsb_discovery.py",
        "configs/legacy/a0_screening/screening_pool.yaml",
        "reports/screening_pool_discovery.log",
        "reports/screening_pool_discovery_summary.json",
    )
    for logical in source_refs:
        path = root / logical
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture source: {logical}\n", encoding="utf-8")

    stage0 = root / "experiments/p2_design_baseline/stage0"
    stage0.mkdir(parents=True, exist_ok=True)
    panel = [{"candidate_index": 1, "protein_id": "1aaa_A__P00001"}]
    (stage0 / "stage0_intervention_panel_v1.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in panel), encoding="utf-8"
    )
    (stage0 / "stage0_intervention_admission_v1.jsonl").write_text(
        json.dumps(
            {
                **panel[0],
                "intervention_admission_status": "ADMITTED",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (stage0 / "protein_manifest.json").write_text(
        json.dumps({"proteins": [{**panel[0], "mapping_provenance": {}}]}),
        encoding="utf-8",
    )
    _write_candidate_assets(
        root,
        pdb_id="1aaa",
        chain="A",
        accession="P00001",
        with_pdb=True,
        with_afdb=True,
        with_mapping=True,
    )
    _write_candidate_assets(
        root,
        pdb_id="2bbb",
        chain="A",
        accession="P00002",
        with_pdb=True,
        with_afdb=False,
        with_mapping=False,
    )
    _write_candidate_assets(
        root,
        pdb_id="3ccc",
        chain="A",
        accession="P00003",
        with_pdb=False,
        with_afdb=True,
        with_mapping=False,
    )
    expected_hashes = tuple(
        (logical, sha256_file(root / logical)) for logical in source_refs
    )
    return SamplingFrameAuditConfig(
        expected_total_discovered=3,
        expected_candidates=3,
        expected_stage0_declared=1,
        expected_stage0_admitted=1,
        neutrality_source_hashes=expected_hashes,
        neutrality_evidence=NeutralityEvidence(
            candidate_selection_precedes_stage0_scoring=True,
            source_derivation_reproducible=True,
            outcome_fields_used_in_frame_selection=(),
            geometry_enrichment_used=False,
            uncertainty_enrichment_used=False,
            plddt_recorded_but_not_used_for_frame_inclusion=True,
            evidence_complete=True,
        ),
    )


def test_repository_sampling_frame_is_exact_unique_ordered_and_stage0_traceable() -> None:
    paths = ProjectPaths.discover(project_root=REPOSITORY_ROOT)
    result = audit_sampling_frame(paths)

    assert len(result.candidates) == 213
    assert result.candidates["sampling_frame_index"].tolist() == list(range(1, 214))
    assert result.candidates["candidate_id"].is_unique
    assert result.candidates["pair_id"].is_unique
    assert result.candidates["polymer_entity_id"].is_unique
    assert result.manifest["sampling_frame_neutrality"]["status"] == (
        "SAMPLING_FRAME_VALID"
    )
    trace = result.manifest["stage0_traceability"]
    assert trace["declared_count"] == 11
    assert trace["declared_traceable_count"] == 11
    assert trace["admitted_count"] == 8
    assert trace["admitted_traceable_count"] == 8
    assert trace["exceptions"] == []
    source_artifacts = result.manifest["source_candidate_artifacts"]
    assert [row["path"] for row in source_artifacts] == [
        "artifacts/dataset/reports/census/candidate_inventory_v1.json",
        "artifacts/dataset/reports/census/candidate_inventory_v1.tsv",
    ]
    assert all(len(row["sha256"]) == 64 for row in source_artifacts)


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({}, "SAMPLING_FRAME_VALID"),
        (
            {"outcome_fields_used_in_frame_selection": ("SDFI",)},
            "SAMPLING_FRAME_INVALID_OUTCOME_DEPENDENT",
        ),
        ({"evidence_complete": False}, "SAMPLING_FRAME_PROVENANCE_INSUFFICIENT"),
        ({"source_derivation_reproducible": False}, "SAMPLING_FRAME_AMBIGUOUS"),
    ],
)
def test_source_neutrality_conclusion_is_deterministic(
    tmp_path: Path, updates: dict[str, object], expected: str
) -> None:
    config = _fixture_config(tmp_path)
    config = replace(
        config,
        neutrality_evidence=replace(config.neutrality_evidence, **updates),
    )
    result = audit_sampling_frame(_paths(tmp_path), config=config)
    assert result.manifest["sampling_frame_neutrality"]["status"] == expected


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        (
            {"ambiguous_local_source": True, "unreadable_local_file": True},
            "MULTIPLE_LOCAL_SOURCE_AMBIGUITY",
        ),
        (
            {"unreadable_local_file": True, "missing_pdb": True},
            "UNREADABLE_LOCAL_FILE",
        ),
        ({"missing_pdb": True, "missing_afdb": True}, "MISSING_PDB_AND_AFDB"),
        ({"missing_pdb": True}, "MISSING_PDB"),
        ({"missing_afdb": True}, "MISSING_AFDB"),
        ({"missing_canonical_sequence": True}, "MISSING_CANONICAL_SEQUENCE"),
        ({"missing_identity_metadata": True}, "MISSING_IDENTITY_METADATA"),
        ({"missing_mapping_metadata": True}, "MISSING_MAPPING_METADATA"),
        ({"missing_provenance_metadata": True}, "MISSING_PROVENANCE_METADATA"),
        ({}, "LOCAL_READY_FOR_SCALE1A"),
    ],
)
def test_availability_status_precedence_is_frozen(
    flags: dict[str, bool], expected: str
) -> None:
    assert classify_local_availability(flags)[0] == expected


def test_local_audit_preserves_every_row_and_uses_only_portable_paths(
    tmp_path: Path,
) -> None:
    config = _fixture_config(tmp_path)
    result = audit_sampling_frame(_paths(tmp_path), config=config)

    assert len(result.candidates) == 3
    assert result.summary["n_sampling_frame"] == 3
    assert result.summary["n_local_ready"] == 1
    assert result.candidates["local_availability_status"].tolist() == [
        "LOCAL_READY_FOR_SCALE1A",
        "MISSING_AFDB",
        "MISSING_PDB",
    ]
    serialized = json.dumps(
        {
            "records": result.candidates.to_dict(orient="records"),
            "summary": result.summary,
            "manifest": result.manifest,
        },
        allow_nan=False,
    )
    assert str(tmp_path) not in serialized
    assert "/home/" not in serialized


def test_immutable_materialization_reuses_identical_and_rejects_conflict(
    tmp_path: Path,
) -> None:
    config = _fixture_config(tmp_path)
    paths = _paths(tmp_path)
    result = audit_sampling_frame(paths, config=config)

    first = materialize_sampling_frame_audit(result, paths, config=config)
    second = materialize_sampling_frame_audit(result, paths, config=config)
    assert set(first["write_status"].values()) == {"created"}
    assert set(second["write_status"].values()) == {"reused_identical"}

    changed = replace(
        result,
        candidates=result.candidates.assign(local_pair_complete=False),
    )
    with pytest.raises(SamplingFrameAuditError, match="immutable"):
        materialize_sampling_frame_audit(changed, paths, config=config)


def test_audit_is_offline_and_does_not_mutate_stage0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path)
    paths = _paths(tmp_path)
    stage0 = tmp_path / config.stage0_panel_ref
    before = sha256_file(stage0)

    def forbidden_network(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr("socket.create_connection", forbidden_network)
    result = audit_sampling_frame(paths, config=config)

    assert result.manifest["scope_confirmations"]["network_access_used"] is False
    assert sha256_file(stage0) == before
