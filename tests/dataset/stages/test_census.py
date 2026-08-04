from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dual_uq.dataset.stages.census import (
    CENSUS_CONFIG,
    AfdbMismatchSite,
    CensusSkip,
    IdentityRecord,
    IdentitySources,
    MismatchSite,
    analyze_sequence_mismatches,
    build_manifest_row,
    discover_local_pairs,
    evaluate_protein,
    load_identity_sources,
    run_round1_census,
    write_round1_report,
)
from dual_uq.dataset.stages.resolution import P0_STAGE_NAME, run_p0

# ---------------------------------------------------------------------------
# load_identity_sources
# ---------------------------------------------------------------------------


def _write_lifecycle_csv(path: Path, rows: list[dict[str, str]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_identity_sources_joins_primary_and_replacement_pools(tmp_path: Path) -> None:
    primary = tmp_path / "candidate_lifecycle.csv"
    replacement = tmp_path / "replacement_candidate_lifecycle.csv"
    _write_lifecycle_csv(
        primary,
        [{"pair_name": "1gci_A__P29600", "screening_index": "6", "primary_category": "easy_control"}],
    )
    _write_lifecycle_csv(
        replacement,
        [
            {
                "pair_name": "5avd_A__P00772",
                "screening_index": "103",
                "primary_category": "easy_control",
            }
        ],
    )

    sources = load_identity_sources(primary, replacement)

    assert sources.records["1gci_A__P29600"] == IdentityRecord(
        screening_index=6, mechanism_label_prior="easy_control", source="candidate_lifecycle"
    )
    assert sources.records["5avd_A__P00772"] == IdentityRecord(
        screening_index=103,
        mechanism_label_prior="easy_control",
        source="replacement_candidate_lifecycle",
    )
    assert sources.conflicts == {}


def test_load_identity_sources_agreeing_duplicate_is_not_a_conflict(tmp_path: Path) -> None:
    primary = tmp_path / "candidate_lifecycle.csv"
    replacement = tmp_path / "replacement_candidate_lifecycle.csv"
    row = {"pair_name": "3zoj_A__F2QVG4", "screening_index": "36", "primary_category": "high_pae_long_range"}
    _write_lifecycle_csv(primary, [row])
    _write_lifecycle_csv(replacement, [row])

    sources = load_identity_sources(primary, replacement)

    assert sources.records["3zoj_A__F2QVG4"].screening_index == 36
    assert sources.conflicts == {}


def test_load_identity_sources_flags_conflicting_screening_index(tmp_path: Path) -> None:
    primary = tmp_path / "candidate_lifecycle.csv"
    replacement = tmp_path / "replacement_candidate_lifecycle.csv"
    _write_lifecycle_csv(
        primary,
        [{"pair_name": "1abc_A__P1", "screening_index": "6", "primary_category": "easy_control"}],
    )
    _write_lifecycle_csv(
        replacement,
        [{"pair_name": "1abc_A__P1", "screening_index": "106", "primary_category": "easy_control"}],
    )

    sources = load_identity_sources(primary, replacement)

    assert "1abc_A__P1" not in sources.records
    assert len(sources.conflicts["1abc_A__P1"]) == 2


# ---------------------------------------------------------------------------
# discover_local_pairs
# ---------------------------------------------------------------------------


def test_discover_local_pairs_lists_only_directories_with_pair_qc(tmp_path: Path) -> None:
    pairs_root = tmp_path / "pairs"
    (pairs_root / "1abc_A__P1").mkdir(parents=True)
    (pairs_root / "1abc_A__P1" / "pair_qc.json").write_text("{}", encoding="utf-8")
    (pairs_root / "incomplete_pair").mkdir(parents=True)

    assert discover_local_pairs(pairs_root) == ["1abc_A__P1"]


# ---------------------------------------------------------------------------
# build_manifest_row
# ---------------------------------------------------------------------------


def _write_local_pair(
    project_root: Path,
    pair_id: str,
    *,
    pdb_id: str = "1abc",
    uniprot_id: str = "P12345",
    model_entity_id: str = "AF-P12345-F1",
) -> dict[str, Path]:
    pair_dir = project_root / "data/processed/pairs" / pair_id
    pair_dir.mkdir(parents=True)
    afdb_dir = project_root / "data/raw/afdb" / uniprot_id / model_entity_id
    afdb_dir.mkdir(parents=True)
    pdb_path = project_root / "data/raw/pdb" / f"{pdb_id}.cif"
    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    pdb_path.write_text("data_fixture\n#\n", encoding="utf-8")
    mapping_path = pair_dir / "residue_mapping.parquet"
    mapping_path.write_text("", encoding="utf-8")
    model_path = afdb_dir / "model.cif"
    plddt_path = afdb_dir / "plddt.json"
    pae_path = afdb_dir / "pae.json"
    metadata_path = afdb_dir / "metadata.json"
    for path in (model_path, plddt_path, pae_path, metadata_path):
        path.write_text("{}", encoding="utf-8")
    pair_qc_path = pair_dir / "pair_qc.json"
    pair_qc_path.write_text(
        json.dumps(
            {
                "pdb_id": pdb_id,
                "chain_id": "A",
                "uniprot_id": uniprot_id,
                "quality_flag": "pass",
                "uniprot_length": 500,
                "pdb_path": str(pdb_path),
                "mapping_path": str(mapping_path),
                "afdb_model_path": str(model_path),
                "plddt_path": str(plddt_path),
                "pae_path": str(pae_path),
            }
        ),
        encoding="utf-8",
    )
    return {
        "pair_dir": pair_dir,
        "pair_qc_path": pair_qc_path,
        "pdb_path": pdb_path,
        "mapping_path": mapping_path,
        "afdb_dir": afdb_dir,
        "metadata_path": metadata_path,
    }


def _write_original_preflight(
    project_root: Path,
    *,
    screening_index: int,
    uniprot_length: int | str,
    canonical_uniprot_length: int | str = "",
) -> None:
    path = project_root / "data/manifests/screening_pool_preflight.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "screening_index": screening_index,
                "uniprot_length": uniprot_length,
                "canonical_uniprot_length": canonical_uniprot_length,
            }
        ]
    ).to_csv(path, sep="\t", index=False)


def _write_replacement_preflight(
    project_root: Path, *, screening_index: int, canonical_uniprot_length: int
) -> None:
    path = project_root / "data/manifests/lower_conf_replacement_pool.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "screening_index": screening_index,
                "canonical_uniprot_length": canonical_uniprot_length,
            }
        ]
    ).to_csv(path, sep="\t", index=False)


def test_build_manifest_row_success(tmp_path: Path) -> None:
    project_root = tmp_path
    paths = _write_local_pair(project_root, "1abc_A__P12345")
    _write_original_preflight(project_root, screening_index=6, uniprot_length=500)
    sources = IdentitySources(
        records={
            "1abc_A__P12345": IdentityRecord(
                screening_index=6, mechanism_label_prior="easy_control", source="candidate_lifecycle"
            )
        },
        conflicts={},
    )

    result = build_manifest_row(
        "1abc_A__P12345",
        pairs_root=project_root / "data/processed/pairs",
        identity_sources=sources,
        project_root=project_root,
    )

    assert isinstance(result, dict)
    assert result["protein_id"] == "index6"
    assert result["screening_index"] == 6
    assert result["pair_id"] == "1abc_A__P12345"
    assert result["mechanism_label"] == "easy_control"
    assert result["tier"] == 1
    assert result["pair_qc_path"] == str(paths["pair_qc_path"])
    assert result["residue_mapping_path"] == str(paths["mapping_path"])
    assert result["afdb_metadata_path"] == str(paths["metadata_path"])
    assert result["uniprot_full_length"] == 500


def test_build_manifest_row_uses_canonical_length_not_selected_fragment_length(
    tmp_path: Path,
) -> None:
    project_root = tmp_path
    paths = _write_local_pair(project_root, "7kr0_A__P0DTD1")
    pair_qc = json.loads(paths["pair_qc_path"].read_text(encoding="utf-8"))
    pair_qc["uniprot_length"] = 126
    paths["pair_qc_path"].write_text(json.dumps(pair_qc), encoding="utf-8")
    paths["metadata_path"].write_text(
        json.dumps({"uniprotStart": 1368, "uniprotEnd": 1493}), encoding="utf-8"
    )
    _write_original_preflight(
        project_root,
        screening_index=9,
        uniprot_length=7095,
        canonical_uniprot_length=7095,
    )
    sources = IdentitySources(
        records={
            "7kr0_A__P0DTD1": IdentityRecord(
                screening_index=9,
                mechanism_label_prior="ordinary_or_unclassified",
                source="candidate_lifecycle",
            )
        },
        conflicts={},
    )

    result = build_manifest_row(
        "7kr0_A__P0DTD1",
        pairs_root=project_root / "data/processed/pairs",
        identity_sources=sources,
        project_root=project_root,
    )

    assert isinstance(result, dict)
    assert result["uniprot_full_length"] == 7095
    assert result["uniprot_full_length"] != 126
    assert result["uniprot_full_length_source"] == (
        "screening_pool_preflight.canonical_uniprot_length"
    )
    assert result["uniprot_full_length_corrected"] is True


def test_build_manifest_row_skips_when_canonical_length_source_is_missing(tmp_path: Path) -> None:
    project_root = tmp_path
    _write_local_pair(project_root, "1abc_A__P12345")
    sources = IdentitySources(
        records={
            "1abc_A__P12345": IdentityRecord(
                screening_index=6,
                mechanism_label_prior="easy_control",
                source="candidate_lifecycle",
            )
        },
        conflicts={},
    )

    result = build_manifest_row(
        "1abc_A__P12345",
        pairs_root=project_root / "data/processed/pairs",
        identity_sources=sources,
        project_root=project_root,
    )

    assert isinstance(result, CensusSkip)
    assert result.code == "missing_canonical_uniprot_full_length"


def test_build_manifest_row_uses_replacement_preflight_canonical_length(tmp_path: Path) -> None:
    project_root = tmp_path
    _write_local_pair(project_root, "8pb5_A__P0DPA9")
    _write_replacement_preflight(
        project_root, screening_index=101, canonical_uniprot_length=309
    )
    sources = IdentitySources(
        records={
            "8pb5_A__P0DPA9": IdentityRecord(
                screening_index=101,
                mechanism_label_prior="high_confidence_state_disagreement",
                source="replacement_candidate_lifecycle",
            )
        },
        conflicts={},
    )

    result = build_manifest_row(
        "8pb5_A__P0DPA9",
        pairs_root=project_root / "data/processed/pairs",
        identity_sources=sources,
        project_root=project_root,
    )

    assert isinstance(result, dict)
    assert result["uniprot_full_length"] == 309
    assert result["uniprot_full_length_source"] == (
        "lower_conf_replacement_pool.canonical_uniprot_length"
    )


def test_build_manifest_row_skips_on_conflicting_explicit_canonical_lengths(
    tmp_path: Path,
) -> None:
    project_root = tmp_path
    _write_local_pair(project_root, "7kr0_A__P0DTD1")
    _write_original_preflight(
        project_root,
        screening_index=9,
        uniprot_length=7000,
        canonical_uniprot_length=7095,
    )
    sources = IdentitySources(
        records={
            "7kr0_A__P0DTD1": IdentityRecord(
                screening_index=9,
                mechanism_label_prior="ordinary_or_unclassified",
                source="candidate_lifecycle",
            )
        },
        conflicts={},
    )

    result = build_manifest_row(
        "7kr0_A__P0DTD1",
        pairs_root=project_root / "data/processed/pairs",
        identity_sources=sources,
        project_root=project_root,
    )

    assert isinstance(result, CensusSkip)
    assert result.code == "conflicting_canonical_uniprot_full_length"
    assert result.details["values"] == [7000, 7095]


def test_build_manifest_row_skips_when_no_identity_source(tmp_path: Path) -> None:
    project_root = tmp_path
    _write_local_pair(project_root, "1abc_A__P12345")
    sources = IdentitySources(records={}, conflicts={})

    result = build_manifest_row(
        "1abc_A__P12345",
        pairs_root=project_root / "data/processed/pairs",
        identity_sources=sources,
        project_root=project_root,
    )

    assert result == CensusSkip(pair_id="1abc_A__P12345", code="no_identity_source", details={})


def test_build_manifest_row_skips_on_identity_conflict(tmp_path: Path) -> None:
    project_root = tmp_path
    _write_local_pair(project_root, "1abc_A__P12345")
    conflicting = [
        IdentityRecord(screening_index=6, mechanism_label_prior="easy_control", source="candidate_lifecycle"),
        IdentityRecord(
            screening_index=106,
            mechanism_label_prior="easy_control",
            source="replacement_candidate_lifecycle",
        ),
    ]
    sources = IdentitySources(records={}, conflicts={"1abc_A__P12345": conflicting})

    result = build_manifest_row(
        "1abc_A__P12345",
        pairs_root=project_root / "data/processed/pairs",
        identity_sources=sources,
        project_root=project_root,
    )

    assert isinstance(result, CensusSkip)
    assert result.code == "identity_source_conflict"


def test_build_manifest_row_skips_when_declared_file_missing(tmp_path: Path) -> None:
    project_root = tmp_path
    paths = _write_local_pair(project_root, "1abc_A__P12345")
    _write_original_preflight(project_root, screening_index=6, uniprot_length=500)
    paths["pdb_path"].unlink()
    sources = IdentitySources(
        records={
            "1abc_A__P12345": IdentityRecord(
                screening_index=6, mechanism_label_prior="easy_control", source="candidate_lifecycle"
            )
        },
        conflicts={},
    )

    result = build_manifest_row(
        "1abc_A__P12345",
        pairs_root=project_root / "data/processed/pairs",
        identity_sources=sources,
        project_root=project_root,
    )

    assert isinstance(result, CensusSkip)
    assert result.code == "declared_file_missing"
    assert result.details["field"] == "pdb_structure_path"


# ---------------------------------------------------------------------------
# Real P0/P1 fixture shared by count_all_sequence_mismatches / evaluate_protein tests
# ---------------------------------------------------------------------------

_ATOM_COLUMNS = (
    "group_PDB",
    "type_symbol",
    "label_atom_id",
    "label_alt_id",
    "label_comp_id",
    "label_asym_id",
    "label_seq_id",
    "auth_asym_id",
    "auth_seq_id",
    "pdbx_PDB_ins_code",
    "Cartn_x",
    "Cartn_y",
    "Cartn_z",
    "occupancy",
    "pdbx_PDB_model_num",
)

_ONE_TO_THREE = {"M": "MET", "A": "ALA", "G": "GLY", "S": "SER"}


def _write_mmcif(path: Path, entry_id: str, residues: list[tuple[int, str]], *, offset: float) -> None:
    values = []
    serial = 0
    for auth_seq_id, resname in residues:
        for atom_index, atom_name in enumerate(("N", "CA", "C", "O")):
            serial += 1
            values.append(
                " ".join(
                    str(value)
                    for value in (
                        "ATOM",
                        atom_name[0],
                        atom_name,
                        ".",
                        resname,
                        "A",
                        auth_seq_id,
                        "A",
                        auth_seq_id,
                        "?",
                        offset + serial,
                        offset + atom_index + 0.25,
                        offset + auth_seq_id / 10.0,
                        1.0,
                        1,
                    )
                )
            )
    path.write_text(
        f"data_{entry_id}\n_entry.id {entry_id}\nloop_\n"
        + "\n".join(f"_atom_site.{column}" for column in _ATOM_COLUMNS)
        + "\n"
        + "\n".join(values)
        + "\n#\n",
        encoding="utf-8",
    )


def _build_real_fixture(
    root: Path,
    *,
    pdb_sequence: str,
    mapping_sequence: str = "MAG",
    afdb_sequence: str | None = None,
    omit_pdb_output_positions: frozenset[int] = frozenset(),
    uniprot_full_length: int = 500,
) -> dict[str, object]:
    """Build one real, valid-except-for-pdb_sequence P0-ready pair fixture.

    mapping canonical sequence defaults to 'MAG'; pdb_sequence controls what
    residue identities are actually written into the PDB mmCIF, so callers can
    induce zero, one, or several isolated amino-acid mismatches. Both must be
    the same length; pass a longer `mapping_sequence` to test shape statistics
    (max_consecutive_run / min_pairwise_spacing) that need more than 3 sites.
    `afdb_sequence` defaults to `mapping_sequence`; pass a different value to
    induce an AFDB-side identity divergence independent of the PDB side.
    `omit_pdb_output_positions` drops the named 1-indexed output positions from
    the PDB mmCIF entirely, simulating a real missing-density coverage gap
    (P1's `residue_count_mismatch`) distinct from an amino-acid mismatch.
    """
    afdb_sequence = mapping_sequence if afdb_sequence is None else afdb_sequence
    assert len(pdb_sequence) == len(mapping_sequence)
    assert len(afdb_sequence) == len(mapping_sequence)
    n = len(mapping_sequence)
    pair_id = "1abc_A__P12345"
    model_id = "AF-P12345-F1"

    pair_dir = root / "data/processed/pairs" / pair_id
    afdb_dir = root / "data/raw/afdb/P12345" / model_id
    pdb_path = root / "data/raw/pdb/1abc.cif"
    pair_dir.mkdir(parents=True)
    afdb_dir.mkdir(parents=True)
    pdb_path.parent.mkdir(parents=True)

    auth_positions = list(range(42, 42 + n))
    uniprot_positions = list(range(100, 100 + n))
    uniprot_end = 99 + n
    mapping = pd.DataFrame(
        {
            "output_position": list(range(1, n + 1)),
            "uniprot_id": ["P12345"] * n,
            "uniprot_residue_number": uniprot_positions,
            "uniprot_residue_name": list(mapping_sequence),
            "pdb_residue_name": [_ONE_TO_THREE[aa] for aa in mapping_sequence],
            "auth_asym_id": ["A"] * n,
            "auth_seq_id": auth_positions,
            "insertion_code": [""] * n,
            "label_asym_id": ["A"] * n,
            "label_seq_id": auth_positions,
        }
    )
    mapping_path = pair_dir / "residue_mapping.tsv"
    mapping.to_csv(mapping_path, sep="\t", index=False, lineterminator="\n")

    pdb_residues = [
        (auth_seq, _ONE_TO_THREE[aa])
        for output_position, (auth_seq, aa) in enumerate(
            zip(auth_positions, pdb_sequence, strict=True), start=1
        )
        if output_position not in omit_pdb_output_positions
    ]
    afdb_residues = [(index + 1, _ONE_TO_THREE[aa]) for index, aa in enumerate(afdb_sequence)]
    _write_mmcif(pdb_path, "1ABC", pdb_residues, offset=0.0)
    model_path = afdb_dir / "model.cif"
    _write_mmcif(model_path, model_id, afdb_residues, offset=100.0)

    metadata_path = afdb_dir / "metadata.json"
    plddt_path = afdb_dir / "plddt.json"
    pae_path = afdb_dir / "pae.json"
    metadata_path.write_text(
        json.dumps(
            {
                "modelEntityId": model_id,
                "entryId": model_id,
                "latestVersion": 6,
                "uniprotStart": 100,
                "uniprotEnd": uniprot_end,
                "sequenceStart": 100,
                "sequenceEnd": uniprot_end,
                "cifUrl": f"https://example.test/{model_id}-model_v6.cif",
                "plddtDocUrl": f"https://example.test/{model_id}-confidence_v6.json",
                "paeDocUrl": f"https://example.test/{model_id}-predicted_aligned_error_v6.json",
            }
        ),
        encoding="utf-8",
    )
    plddt_path.write_text(
        json.dumps(
            {"residueNumber": list(range(1, n + 1)), "confidenceScore": [95.0] * n}
        ),
        encoding="utf-8",
    )
    pae_path.write_text(
        json.dumps(
            [{"predicted_aligned_error": [[0.0] * n for _ in range(n)], "max_predicted_aligned_error": 31.75}]
        ),
        encoding="utf-8",
    )
    pair_qc_path = pair_dir / "pair_qc.json"
    pair_qc_path.write_text(
        json.dumps(
            {
                "pdb_id": "1abc",
                "chain_id": "A",
                "uniprot_id": "P12345",
                "quality_flag": "pass",
                "uniprot_length": uniprot_full_length,
                "mapped_residue_count": n,
                "mapped_uniprot_start": 100,
                "mapped_uniprot_end": uniprot_end,
                "pdb_path": str(pdb_path),
                "mapping_path": str(mapping_path),
                "afdb_model_path": str(model_path),
                "plddt_path": str(plddt_path),
                "pae_path": str(pae_path),
                "afdb_model_entity_id": model_id,
                "afdb_version": 6,
                "afdb_fragment_start": 100,
                "afdb_fragment_end": uniprot_end,
                "afdb_fragment_length": n,
            }
        ),
        encoding="utf-8",
    )
    row = {
        "protein_id": "index6",
        "screening_index": 6,
        "pair_id": pair_id,
        "mechanism_label": "easy_control",
        "tier": 1,
        "pair_qc_path": str(pair_qc_path),
        "residue_mapping_path": str(mapping_path),
        "pdb_structure_path": str(pdb_path),
        "afdb_metadata_path": str(metadata_path),
        "afdb_model_path": str(model_path),
        "afdb_plddt_path": str(plddt_path),
        "afdb_pae_path": str(pae_path),
        "uniprot_full_length": uniprot_full_length,
    }
    return {
        "row": row,
        "pair_id": pair_id,
        "pdb_path": pdb_path,
        "afdb_path": model_path,
        "fragment_start": 100,
        "model_length": n,
    }


# ---------------------------------------------------------------------------
# analyze_sequence_mismatches
# ---------------------------------------------------------------------------


def _run_p0_for_fixture(tmp_path: Path, fixture: dict[str, object]) -> Path:
    stage_dir = tmp_path / "census" / "index6" / P0_STAGE_NAME
    result = run_p0(
        manifest_row=fixture["row"],
        project_root=tmp_path,
        stage_dir=stage_dir,
        config=CENSUS_CONFIG,
        pipeline_version="dataset-a.v1",
        run_id="census-test",
    )
    assert result.validation.validation_pass is True
    return stage_dir


def test_analyze_sequence_mismatches_counts_every_position_not_just_first(tmp_path: Path) -> None:
    fixture = _build_real_fixture(tmp_path, pdb_sequence="GAS")  # mismatches at position 1 and 3
    stage_dir = _run_p0_for_fixture(tmp_path, fixture)

    analysis = analyze_sequence_mismatches(
        pdb_path=fixture["pdb_path"],
        afdb_path=fixture["afdb_path"],
        p0_stage_dir=stage_dir,
        pair_id=fixture["pair_id"],
        fragment_start=fixture["fragment_start"],
        model_length=fixture["model_length"],
    )

    assert analysis.mismatch_count == 2
    assert analysis.missing_pdb_count == 0
    assert analysis.missing_afdb_count == 0
    assert analysis.mapped_length == 3


def test_analyze_sequence_mismatches_is_zero_for_matching_sequence(tmp_path: Path) -> None:
    fixture = _build_real_fixture(tmp_path, pdb_sequence="MAG")
    stage_dir = _run_p0_for_fixture(tmp_path, fixture)

    analysis = analyze_sequence_mismatches(
        pdb_path=fixture["pdb_path"],
        afdb_path=fixture["afdb_path"],
        p0_stage_dir=stage_dir,
        pair_id=fixture["pair_id"],
        fragment_start=fixture["fragment_start"],
        model_length=fixture["model_length"],
    )

    assert analysis.mismatch_count == 0
    assert analysis.sequence_identity_mapped == 1.0
    assert analysis.sequence_identity_paired == 1.0
    assert analysis.max_consecutive_run == 0
    assert analysis.min_pairwise_spacing is None
    assert analysis.mismatches == ()


def test_analyze_sequence_mismatches_excludes_missing_pdb_coverage_gap(tmp_path: Path) -> None:
    """A mapped position with zero PDB atom coverage is P1's `residue_count_mismatch`,
    not an amino-acid identity mismatch, and must never be folded into the count.

    Mirrors a real round-1 case (1i1w_A__P23360, screening_index 111): one mapped
    position has no PDB atoms at all *and* the protein separately has genuine AA
    mismatches elsewhere. Manually cross-checking that real protein's frozen mapping
    against its PDB structure confirmed the mismatch count reports 6, not 7 -- i.e.
    it already excludes the coverage gap. This test locks that behavior in with a
    synthetic, controlled fixture instead of relying on ad hoc real-data checks.
    """
    fixture = _build_real_fixture(
        tmp_path, pdb_sequence="GAG", omit_pdb_output_positions=frozenset({3})
    )
    stage_dir = _run_p0_for_fixture(tmp_path, fixture)

    analysis = analyze_sequence_mismatches(
        pdb_path=fixture["pdb_path"],
        afdb_path=fixture["afdb_path"],
        p0_stage_dir=stage_dir,
        pair_id=fixture["pair_id"],
        fragment_start=fixture["fragment_start"],
        model_length=fixture["model_length"],
    )

    assert analysis.mismatch_count == 1  # only the genuine mismatch at position 1
    assert analysis.missing_pdb_count == 1  # position 3 is a coverage gap, not a mismatch
    assert analysis.mapped_length == 3
    assert analysis.sequence_identity_mapped == pytest.approx(1 - 1 / 3)
    assert analysis.sequence_identity_paired == pytest.approx(1 - 1 / 2)


def test_analyze_sequence_mismatches_reports_mismatch_site_details(tmp_path: Path) -> None:
    fixture = _build_real_fixture(tmp_path, pdb_sequence="GAS")  # mismatches at position 1 and 3
    stage_dir = _run_p0_for_fixture(tmp_path, fixture)

    analysis = analyze_sequence_mismatches(
        pdb_path=fixture["pdb_path"],
        afdb_path=fixture["afdb_path"],
        p0_stage_dir=stage_dir,
        pair_id=fixture["pair_id"],
        fragment_start=fixture["fragment_start"],
        model_length=fixture["model_length"],
    )

    assert analysis.mismatches == (
        MismatchSite(output_position=1, uniprot_position=100, mapping_aa="M", pdb_aa="G"),
        MismatchSite(output_position=3, uniprot_position=102, mapping_aa="G", pdb_aa="S"),
    )


def test_analyze_sequence_mismatches_computes_shape_statistics(tmp_path: Path) -> None:
    # mismatches at output_position 1, 2 (consecutive) and 7 (isolated)
    fixture = _build_real_fixture(
        tmp_path, mapping_sequence="MAGSMAGSM", pdb_sequence="AGGSMAMSM"
    )
    stage_dir = _run_p0_for_fixture(tmp_path, fixture)

    analysis = analyze_sequence_mismatches(
        pdb_path=fixture["pdb_path"],
        afdb_path=fixture["afdb_path"],
        p0_stage_dir=stage_dir,
        pair_id=fixture["pair_id"],
        fragment_start=fixture["fragment_start"],
        model_length=fixture["model_length"],
    )

    assert analysis.mismatch_count == 3
    assert [site.output_position for site in analysis.mismatches] == [1, 2, 7]
    assert analysis.max_consecutive_run == 2  # positions 1,2
    assert analysis.min_pairwise_spacing == 1  # positions 1,2 are 1 apart


def test_analyze_sequence_mismatches_min_pairwise_spacing_is_null_not_sentinel_for_single_mismatch(
    tmp_path: Path,
) -> None:
    fixture = _build_real_fixture(tmp_path, pdb_sequence="GAG")  # single isolated mismatch
    stage_dir = _run_p0_for_fixture(tmp_path, fixture)

    analysis = analyze_sequence_mismatches(
        pdb_path=fixture["pdb_path"],
        afdb_path=fixture["afdb_path"],
        p0_stage_dir=stage_dir,
        pair_id=fixture["pair_id"],
        fragment_start=fixture["fragment_start"],
        model_length=fixture["model_length"],
    )

    assert analysis.mismatch_count == 1
    assert analysis.min_pairwise_spacing is None
    assert analysis.max_consecutive_run == 1


def test_analyze_sequence_mismatches_flags_missing_afdb_coverage(tmp_path: Path) -> None:
    """missing_afdb_count is expected to always be 0 in practice (AFDB models are

    contiguous by construction), but this must still be verified rather than
    assumed. Writes P0's frozen mapping output directly rather than going
    through run_p0, since P0's own AFDB completeness checks are out of scope
    for this diagnostic.
    """
    pair_id = "1abc_A__P12345"
    p0_stage_dir = tmp_path / "stage"
    (p0_stage_dir / "outputs").mkdir(parents=True)
    mapping = pd.DataFrame(
        {
            "output_position": [1, 2, 3],
            "uniprot_id": ["P12345"] * 3,
            "uniprot_residue_number": [100, 101, 102],
            "uniprot_residue_name": list("MAG"),
            "pdb_residue_name": ["MET", "ALA", "GLY"],
            "auth_asym_id": ["A"] * 3,
            "auth_seq_id": [42, 43, 44],
            "insertion_code": [""] * 3,
            "label_asym_id": ["A"] * 3,
            "label_seq_id": [42, 43, 44],
        }
    )
    mapping.to_csv(
        p0_stage_dir / "outputs" / "residue_mapping.tsv", sep="\t", index=False, lineterminator="\n"
    )
    pdb_path = tmp_path / "pdb.cif"
    _write_mmcif(pdb_path, "1ABC", [(42, "MET"), (43, "ALA"), (44, "GLY")], offset=0.0)
    afdb_path = tmp_path / "afdb.cif"
    _write_mmcif(afdb_path, "MODEL", [(1, "MET"), (3, "GLY")], offset=100.0)  # position 2 omitted

    analysis = analyze_sequence_mismatches(
        pdb_path=pdb_path,
        afdb_path=afdb_path,
        p0_stage_dir=p0_stage_dir,
        pair_id=pair_id,
        fragment_start=100,
        model_length=3,
    )

    assert analysis.missing_afdb_count == 1
    assert analysis.mismatch_count == 0


def test_analyze_sequence_mismatches_flags_afdb_amino_acid_mismatch(tmp_path: Path) -> None:
    """AFDB-side identity divergence (distinct from missing coverage) must be

    counted separately from the PDB-side mismatch_count, never conflated with
    it. Confirmed relevant by a real round-1 case (8pb5_A__P0DPA9, screening
    index 101): PDB and AFDB agree with each other (both K) while `mapping`
    alone says R -- i.e. the defect is in the mapping data, not a real PDB-
    side variant. AFDB is never exempted by the allow-list (PDR-01 §1.5).
    """
    pair_id = "1abc_A__P12345"
    p0_stage_dir = tmp_path / "stage"
    (p0_stage_dir / "outputs").mkdir(parents=True)
    mapping = pd.DataFrame(
        {
            "output_position": [1, 2, 3],
            "uniprot_id": ["P12345"] * 3,
            "uniprot_residue_number": [100, 101, 102],
            "uniprot_residue_name": list("MAG"),
            "pdb_residue_name": ["MET", "ALA", "GLY"],
            "auth_asym_id": ["A"] * 3,
            "auth_seq_id": [42, 43, 44],
            "insertion_code": [""] * 3,
            "label_asym_id": ["A"] * 3,
            "label_seq_id": [42, 43, 44],
        }
    )
    mapping.to_csv(
        p0_stage_dir / "outputs" / "residue_mapping.tsv", sep="\t", index=False, lineterminator="\n"
    )
    pdb_path = tmp_path / "pdb.cif"
    _write_mmcif(pdb_path, "1ABC", [(42, "MET"), (43, "ALA"), (44, "GLY")], offset=0.0)
    afdb_path = tmp_path / "afdb.cif"
    # position 2 (model-local auth_seq_id=2) diverges from mapping: SER instead of ALA
    _write_mmcif(afdb_path, "MODEL", [(1, "MET"), (2, "SER"), (3, "GLY")], offset=100.0)

    analysis = analyze_sequence_mismatches(
        pdb_path=pdb_path,
        afdb_path=afdb_path,
        p0_stage_dir=p0_stage_dir,
        pair_id=pair_id,
        fragment_start=100,
        model_length=3,
    )

    assert analysis.mismatch_count == 0  # PDB side is clean
    assert analysis.missing_afdb_count == 0  # residue is present, just wrong identity
    assert analysis.afdb_mismatch_count == 1
    assert analysis.afdb_mismatches == (
        AfdbMismatchSite(output_position=2, uniprot_position=101, mapping_aa="A", afdb_aa="S"),
    )


# ---------------------------------------------------------------------------
# evaluate_protein / run_round1_census
# ---------------------------------------------------------------------------


def test_evaluate_protein_reports_ok_outcome_for_valid_pair(tmp_path: Path) -> None:
    fixture = _build_real_fixture(tmp_path, pdb_sequence="MAG")

    record = evaluate_protein(
        fixture["row"],
        project_root=tmp_path,
        census_stage_root=tmp_path / "census",
        config=CENSUS_CONFIG,
    )

    assert record.outcome == "p1_pairing_ok"
    assert record.failure_code is None
    assert record.stage_reached == "p1_resolved"
    assert record.mapped_length == 3
    assert record.uniprot_full_length == 500


def test_evaluate_protein_reports_p1_failure_with_mismatch_count(tmp_path: Path) -> None:
    fixture = _build_real_fixture(tmp_path, pdb_sequence="GAG")  # single isolated mismatch at position 1

    record = evaluate_protein(
        fixture["row"],
        project_root=tmp_path,
        census_stage_root=tmp_path / "census",
        config=CENSUS_CONFIG,
    )

    assert record.outcome == "p1_failure"
    assert record.failure_code == "pdb_amino_acid_mismatch"
    assert record.stage_reached == "p0_frozen"
    assert record.mismatch_count == 1
    assert record.missing_pdb_count == 0
    assert record.missing_afdb_count == 0
    assert record.sequence_identity_paired == pytest.approx(1 - 1 / 3)
    assert record.max_consecutive_run == 1
    assert record.min_pairwise_spacing is None
    assert record.mismatches == (
        MismatchSite(output_position=1, uniprot_position=100, mapping_aa="M", pdb_aa="G"),
    )
    assert record.afdb_mismatch_count == 0
    assert record.afdb_mismatches == ()


def test_evaluate_protein_reports_afdb_mismatch_alongside_pdb_mismatch(tmp_path: Path) -> None:
    """A protein can hit pdb_amino_acid_mismatch (fail-fast, position 1) while

    *also* carrying an independent AFDB-side divergence elsewhere (position 2).
    Both must be visible in the same record -- this is exactly the real
    8pb5_A__P0DPA9 shape (PDR-01 finding): AFDB disagreement at the known
    mismatch position itself, surfaced here as afdb_mismatch data on the same
    ProteinCensusRecord as the PDB-side pdb_amino_acid_mismatch failure.
    """
    fixture = _build_real_fixture(tmp_path, pdb_sequence="GAG", afdb_sequence="MSG")

    record = evaluate_protein(
        fixture["row"],
        project_root=tmp_path,
        census_stage_root=tmp_path / "census",
        config=CENSUS_CONFIG,
    )

    assert record.outcome == "p1_failure"
    assert record.failure_code == "pdb_amino_acid_mismatch"
    assert record.mismatch_count == 1
    assert record.afdb_mismatch_count == 1
    assert record.afdb_mismatches == (
        AfdbMismatchSite(output_position=2, uniprot_position=101, mapping_aa="A", afdb_aa="S"),
    )


def test_evaluate_protein_reports_p0_failure_without_writing_stage(tmp_path: Path) -> None:
    fixture = _build_real_fixture(tmp_path, pdb_sequence="MAG")
    bad_row = {**fixture["row"], "pair_qc_path": str(tmp_path / "does_not_exist.json")}

    record = evaluate_protein(
        bad_row,
        project_root=tmp_path,
        census_stage_root=tmp_path / "census",
        config=CENSUS_CONFIG,
    )

    assert record.outcome == "p0_failure"
    assert record.failure_code == "missing_required_file"
    assert record.stage_reached == "manifest_built"
    assert record.mapped_length is None
    assert not (tmp_path / "census").exists()


def test_run_round1_census_end_to_end_produces_report_with_summary(tmp_path: Path) -> None:
    project_root = tmp_path
    ok_fixture = _build_real_fixture(project_root, pdb_sequence="MAG")
    (project_root / "reports").mkdir()
    _write_lifecycle_csv(
        project_root / "reports/candidate_lifecycle.csv",
        [{"pair_name": ok_fixture["pair_id"], "screening_index": "6", "primary_category": "easy_control"}],
    )
    _write_lifecycle_csv(project_root / "reports/replacement_candidate_lifecycle.csv", [])
    _write_original_preflight(project_root, screening_index=6, uniprot_length=500)

    report = run_round1_census(
        pairs_root=project_root / "data/processed/pairs",
        candidate_lifecycle_path=project_root / "reports/candidate_lifecycle.csv",
        replacement_lifecycle_path=project_root / "reports/replacement_candidate_lifecycle.csv",
        project_root=project_root,
        census_stage_root=project_root / "census",
        config=CENSUS_CONFIG,
        run_id="dataset_a_census_round1_test",
    )

    assert len(report.records) == 1
    record = report.records[0]
    assert record.outcome == "p1_pairing_ok"
    assert record.stage_reached == "p1_resolved"
    assert record.mapped_length == 3
    assert record.uniprot_full_length == 500
    assert record.mapped_length != record.uniprot_full_length
    assert report.summary["total_local_pairs"] == 1
    assert report.summary["outcome_counts"] == {"p1_pairing_ok": 1}
    assert report.summary["p0_failure_code_counts"] == {}
    assert report.summary["p1_failure_code_counts"] == {}
    assert report.summary["stage_reached_counts"] == {"p1_resolved": 1}
    assert report.summary["full_length_provenance_audit"] == {
        "matched_count": 1,
        "corrected_count": 0,
        "missing_count": 0,
        "conflict_count": 0,
        "corrected_proteins": [],
    }
    assert "1i1w" in report.summary["censoring_note"]

    out_path = project_root / "artifacts/dataset/reports/census/round1_report.json"
    write_round1_report(report, out_path)
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["summary"]["outcome_counts"] == {"p1_pairing_ok": 1}
    assert payload["records"][0]["pair_id"] == ok_fixture["pair_id"]
    assert payload["records"][0]["stage_reached"] == "p1_resolved"
