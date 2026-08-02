"""Dataset A round-1 eligibility census.

Read-only-in-spirit measurement harness described in
docs/Dual-UQ_Dataset-A_实验设计方案_v0.1.md §5.0: it walks the proteins that
already have real inputs on local disk, resolves each through P0/P1, and
tallies the failure_code distribution needed to freeze the D1/D2 protocol
decisions. It does not decide D1-D5, does not download anything, and does not
alter any A1-A8 stage contract.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from dual_uq.dataset_a_scale.stages.p0 import P0_INPUT_PATH_FIELDS, P0_STAGE_NAME, run_p0
from dual_uq.dataset_a_scale.stages.p1 import (
    P1ValidationError,
    _index_records,
    _load_atom_records,
    _load_mapping,
    resolve_p1_inputs,
)
from dual_uq.dataset_a_scale.structures import ResidueKey, select_backbone_atoms

CENSUS_RUN_ID = "dataset_a_census_round1"
CENSUS_CONFIG = {
    "purpose": "dataset_a_census_round1",
    "mapping_policy": "explicit_auth_label",
    "fragment_policy": "full_coverage",
}

_PAIR_QC_KEYS = {
    "pdb_structure_path": "pdb_path",
    "residue_mapping_path": "mapping_path",
    "afdb_model_path": "afdb_model_path",
    "afdb_plddt_path": "plddt_path",
    "afdb_pae_path": "pae_path",
}


@dataclass(frozen=True)
class IdentityRecord:
    screening_index: int
    mechanism_label_prior: str
    source: str


@dataclass(frozen=True)
class IdentitySources:
    records: dict[str, IdentityRecord]
    conflicts: dict[str, list[IdentityRecord]]


@dataclass(frozen=True)
class CensusSkip:
    pair_id: str
    code: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProteinCensusRecord:
    pair_id: str
    protein_id: str | None
    screening_index: int | None
    mechanism_label_prior: str | None
    outcome: str
    failure_code: str | None = None
    failure_details: dict[str, Any] = field(default_factory=dict)
    mismatch_count: int | None = None


@dataclass(frozen=True)
class Round1Report:
    records: tuple[ProteinCensusRecord, ...]
    summary: dict[str, Any]


def _read_lifecycle_table(path: Path, source_label: str) -> dict[str, IdentityRecord]:
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return {}
    records: dict[str, IdentityRecord] = {}
    for _, row in frame.iterrows():
        pair_id = row["pair_name"]
        if not pair_id:
            continue
        records[pair_id] = IdentityRecord(
            screening_index=int(row["screening_index"]),
            mechanism_label_prior=row["primary_category"],
            source=source_label,
        )
    return records


def load_identity_sources(
    candidate_lifecycle_path: Path, replacement_lifecycle_path: Path
) -> IdentitySources:
    """Join the primary (36) and replacement (20) candidate pool tables by pair_name.

    A pair_id present in both tables with the same screening_index is not a
    conflict. A pair_id present in both with a *different* screening_index is
    a data-integrity problem and is excluded from `records`, reported in
    `conflicts` instead of silently picking one source.
    """
    primary = _read_lifecycle_table(candidate_lifecycle_path, "candidate_lifecycle")
    replacement = _read_lifecycle_table(replacement_lifecycle_path, "replacement_candidate_lifecycle")

    records: dict[str, IdentityRecord] = {}
    conflicts: dict[str, list[IdentityRecord]] = {}
    for pair_id in sorted(set(primary) | set(replacement)):
        candidates = [table[pair_id] for table in (primary, replacement) if pair_id in table]
        if len({candidate.screening_index for candidate in candidates}) > 1:
            conflicts[pair_id] = candidates
            continue
        records[pair_id] = candidates[0]
    return IdentitySources(records=records, conflicts=conflicts)


def discover_local_pairs(pairs_root: Path) -> list[str]:
    """List pair_ids that have a real local pair_qc.json (no download required)."""
    return sorted(
        entry.name
        for entry in pairs_root.iterdir()
        if entry.is_dir() and (entry / "pair_qc.json").is_file()
    )


def build_manifest_row(
    pair_id: str,
    *,
    pairs_root: Path,
    identity_sources: IdentitySources,
    project_root: Path,
) -> dict[str, Any] | CensusSkip:
    """Assemble one Dataset A scale manifest row from real on-disk artifacts.

    Never resolves a fallback path or invents an identity: any missing
    identity or declared file is returned as an explicit CensusSkip so round-1
    attrition stays visible instead of silently shrinking the pool.
    """
    if pair_id in identity_sources.conflicts:
        return CensusSkip(
            pair_id=pair_id,
            code="identity_source_conflict",
            details={"sources": [record.source for record in identity_sources.conflicts[pair_id]]},
        )
    identity = identity_sources.records.get(pair_id)
    if identity is None:
        return CensusSkip(pair_id=pair_id, code="no_identity_source", details={})

    pair_qc_path = pairs_root / pair_id / "pair_qc.json"
    pair_qc = json.loads(pair_qc_path.read_text(encoding="utf-8"))

    row: dict[str, Any] = {
        "protein_id": f"index{identity.screening_index}",
        "screening_index": identity.screening_index,
        "pair_id": pair_id,
        "mechanism_label": identity.mechanism_label_prior,
        "tier": 1,
        "pair_qc_path": str(pair_qc_path),
    }
    for manifest_field, pair_qc_key in _PAIR_QC_KEYS.items():
        row[manifest_field] = str(pair_qc[pair_qc_key])
    row["afdb_metadata_path"] = str(Path(row["afdb_model_path"]).with_name("metadata.json"))

    for manifest_field in P0_INPUT_PATH_FIELDS:
        candidate = Path(row[manifest_field])
        resolved = candidate if candidate.is_absolute() else project_root / candidate
        if not resolved.exists():
            return CensusSkip(
                pair_id=pair_id,
                code="declared_file_missing",
                details={"field": manifest_field, "path": str(candidate)},
            )
    return row


def count_all_sequence_mismatches(*, pdb_path: Path, p0_stage_dir: Path, pair_id: str) -> int:
    """Count every mapped position where the PDB residue disagrees with the mapping AA.

    P1's own `pdb_amino_acid_mismatch` check (stages/p1.py) is fail-fast by
    frozen contract: it raises on the first mismatch and never trims or
    continues. That contract is correct for P1 and must not change. D1
    (docs/Dual-UQ_Dataset-A_实验设计方案_v0.1.md §3) needs the *total* mismatch
    count per protein to tell an isolated variant from a broken mapping, so
    this walks the same frozen P0 output mapping and the same PDB parsing/
    selection primitives P1 uses, without stopping at the first hit.
    """
    mapping = _load_mapping(p0_stage_dir / "outputs" / "residue_mapping.tsv")
    pdb_index = _index_records(_load_atom_records(pdb_path, pair_id))
    mismatches = 0
    for row in mapping.itertuples(index=False):
        key = ResidueKey(pair_id, str(row.auth_asym_id), int(row.auth_seq_id), str(row.insertion_code))
        atoms = pdb_index.get(key)
        if atoms is None:
            continue
        try:
            selection = select_backbone_atoms(atoms)
        except ValueError:
            continue
        if selection.residue.canonical_aa != str(row.canonical_aa):
            mismatches += 1
    return mismatches


def evaluate_protein(
    row: dict[str, Any],
    *,
    project_root: Path,
    census_stage_root: Path,
    config: dict[str, Any],
    pipeline_version: str = "dataset-a.v1",
) -> ProteinCensusRecord:
    """Resolve one manifest row through P0 (write, to a census scratch dir) then P1 (read-only)."""
    pair_id = row["pair_id"]
    identity_fields = {
        "pair_id": pair_id,
        "protein_id": row["protein_id"],
        "screening_index": row["screening_index"],
        "mechanism_label_prior": row["mechanism_label"],
    }

    p0_stage_dir = census_stage_root / row["protein_id"] / P0_STAGE_NAME
    p0_result = run_p0(
        manifest_row=row,
        project_root=project_root,
        stage_dir=p0_stage_dir,
        config=config,
        pipeline_version=pipeline_version,
        run_id=CENSUS_RUN_ID,
    )
    if not p0_result.validation.validation_pass:
        return ProteinCensusRecord(
            **identity_fields,
            outcome="p0_failure",
            failure_code=p0_result.failure_code,
            failure_details=dict(p0_result.validation.details),
        )

    try:
        resolve_p1_inputs(
            project_root=project_root,
            p0_stage_dir=p0_stage_dir,
            config=config,
            pipeline_version=pipeline_version,
        )
    except P1ValidationError as exc:
        mismatch_count = None
        if exc.code == "pdb_amino_acid_mismatch":
            mismatch_count = count_all_sequence_mismatches(
                pdb_path=Path(row["pdb_structure_path"]), p0_stage_dir=p0_stage_dir, pair_id=pair_id
            )
        return ProteinCensusRecord(
            **identity_fields,
            outcome="p1_failure",
            failure_code=exc.code,
            failure_details={"message": str(exc), **exc.details},
            mismatch_count=mismatch_count,
        )

    return ProteinCensusRecord(**identity_fields, outcome="p1_pairing_ok")


def _summarize(records: list[ProteinCensusRecord], *, total_local_pairs: int) -> dict[str, Any]:
    mismatch_hits = [record for record in records if record.failure_code == "pdb_amino_acid_mismatch"]
    return {
        "total_local_pairs": total_local_pairs,
        "outcome_counts": dict(Counter(record.outcome for record in records)),
        "failure_code_counts": dict(
            Counter(record.failure_code for record in records if record.failure_code)
        ),
        "pdb_amino_acid_mismatch_hit_count": len(mismatch_hits),
        "pdb_amino_acid_mismatch_counts": [record.mismatch_count for record in mismatch_hits],
    }


def run_round1_census(
    *,
    pairs_root: Path,
    candidate_lifecycle_path: Path,
    replacement_lifecycle_path: Path,
    project_root: Path,
    census_stage_root: Path,
    config: dict[str, Any] = CENSUS_CONFIG,
    pipeline_version: str = "dataset-a.v1",
) -> Round1Report:
    identity_sources = load_identity_sources(candidate_lifecycle_path, replacement_lifecycle_path)
    pair_ids = discover_local_pairs(pairs_root)

    records: list[ProteinCensusRecord] = []
    for pair_id in pair_ids:
        row_or_skip = build_manifest_row(
            pair_id, pairs_root=pairs_root, identity_sources=identity_sources, project_root=project_root
        )
        if isinstance(row_or_skip, CensusSkip):
            identity = identity_sources.records.get(pair_id)
            records.append(
                ProteinCensusRecord(
                    pair_id=pair_id,
                    protein_id=f"index{identity.screening_index}" if identity else None,
                    screening_index=identity.screening_index if identity else None,
                    mechanism_label_prior=identity.mechanism_label_prior if identity else None,
                    outcome="skipped",
                    failure_code=row_or_skip.code,
                    failure_details=row_or_skip.details,
                )
            )
            continue
        records.append(
            evaluate_protein(
                row_or_skip,
                project_root=project_root,
                census_stage_root=census_stage_root,
                config=config,
                pipeline_version=pipeline_version,
            )
        )

    return Round1Report(
        records=tuple(records), summary=_summarize(records, total_local_pairs=len(pair_ids))
    )


def write_round1_report(report: Round1Report, out_path: Path) -> None:
    payload = {
        "summary": report.summary,
        "records": [asdict(record) for record in report.records],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
