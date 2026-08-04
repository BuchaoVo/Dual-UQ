"""Dataset A round-1 eligibility census.

Read-only-in-spirit measurement harness described in
docs/design/Dual-UQ_Dataset-A_实验设计方案_v0.1.md §5.0: it walks the proteins that
already have real inputs on local disk, resolves each through P0/P1, and
tallies the failure_code distribution needed to freeze the D1/D2 protocol
decisions. It does not decide D1-D5, does not download anything, and does not
alter any A1-A8 stage contract.

TASK-A extension (docs/handoff/Dataset-A_后续任务规格包_v0.1.md): adds per-protein
stage_reached / mapped_length / uniprot_full_length, and for the
pdb_amino_acid_mismatch cohort, a full non-fail-fast replay
(analyze_sequence_mismatches) producing the site table and shape statistics
PDR-01 §1.2 needs. This is a read-only diagnostic layered on top of the
frozen P0/P1 stages; it does not modify them.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import pandas as pd

from dual_uq.core.atomic_io import atomic_write_text

from ..models import ResidueKey, group_residue_records, select_backbone_atoms
from .derivation import (
    P1ValidationError,
    _index_records,
    _load_atom_records,
    _load_mapping,
    resolve_p1_inputs,
)
from .resolution import (
    P0_INPUT_PATH_FIELDS,
    P0_STAGE_NAME,
    P0ValidationError,
    resolve_p0_inputs,
    run_p0,
)

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

_STAGE_ORDER = ("manifest_built", "p0_resolved", "p0_frozen", "p1_resolved")

_CENSORING_NOTE = (
    "Each failure_code count is a first-failure lower bound, not a prevalence: the "
    "pipeline is staged and checks are fail-fast, so a protein with multiple "
    "independent problems is reported under whichever check triggers first. "
    "Confirmed case: 1i1w_A__P23360 (screening_index 111) has both a genuine "
    "pdb_amino_acid_mismatch and a separate missing_pdb coverage gap; the coverage "
    "gap's own failure code (residue_count_mismatch, raised only after P1's full "
    "per-row loop completes -- see p1.py:729-734) is masked because the AA mismatch "
    "is hit first within the same loop. stage_not_reached_counts is derived from "
    "stage_reached and is the same censoring at stage granularity, not per check."
)


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
class CanonicalLengthRecord:
    value: int
    source: str


@dataclass(frozen=True)
class CanonicalLengthSources:
    records: dict[int, CanonicalLengthRecord]
    conflicts: dict[int, tuple[CanonicalLengthRecord, ...]]


@dataclass(frozen=True)
class CensusSkip:
    pair_id: str
    code: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MismatchSite:
    output_position: int
    uniprot_position: int
    mapping_aa: str
    pdb_aa: str


@dataclass(frozen=True)
class AfdbMismatchSite:
    output_position: int
    uniprot_position: int
    mapping_aa: str
    afdb_aa: str


@dataclass(frozen=True)
class SequenceMismatchAnalysis:
    mismatch_count: int
    missing_pdb_count: int
    missing_afdb_count: int
    mapped_length: int
    sequence_identity_mapped: float
    sequence_identity_paired: float | None
    max_consecutive_run: int
    min_pairwise_spacing: int | None
    mismatches: tuple[MismatchSite, ...]
    afdb_mismatch_count: int
    afdb_mismatches: tuple[AfdbMismatchSite, ...]


@dataclass(frozen=True)
class ProteinCensusRecord:
    pair_id: str
    protein_id: str | None
    screening_index: int | None
    mechanism_label_prior: str | None
    outcome: str
    stage_reached: str | None = None
    failure_code: str | None = None
    failure_details: dict[str, Any] = field(default_factory=dict)
    mapped_length: int | None = None
    uniprot_full_length: int | None = None
    uniprot_full_length_source: str | None = None
    pair_qc_uniprot_length: int | None = None
    uniprot_full_length_corrected: bool | None = None
    mismatch_count: int | None = None
    missing_pdb_count: int | None = None
    missing_afdb_count: int | None = None
    sequence_identity_mapped: float | None = None
    sequence_identity_paired: float | None = None
    max_consecutive_run: int | None = None
    min_pairwise_spacing: int | None = None
    mismatches: tuple[MismatchSite, ...] = ()
    afdb_mismatch_count: int | None = None
    afdb_mismatches: tuple[AfdbMismatchSite, ...] = ()


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


def _positive_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    integer = int(numeric)
    return integer if numeric == integer and integer > 0 else None


def load_canonical_length_sources(project_root: Path) -> CanonicalLengthSources:
    """Load explicit canonical UniProt lengths from existing preflight evidence.

    ``screening_pool_preflight.uniprot_length`` predates the explicitly named
    ``canonical_uniprot_length`` migration column but has canonical full-length
    semantics.  Both are checked when populated.  Replacement preflight evidence
    is frozen in ``lower_conf_replacement_pool.tsv``.  Selected AFDB model,
    mapped-region, and paired-backbone lengths are deliberately not candidates.
    """
    specifications = (
        (
            project_root / "data/manifests/screening_pool_preflight.tsv",
            (
                "canonical_uniprot_length",
                "uniprot_length",
            ),
            "screening_pool_preflight",
        ),
        (
            project_root / "data/manifests/lower_conf_replacement_pool.tsv",
            ("canonical_uniprot_length",),
            "lower_conf_replacement_pool",
        ),
    )
    candidates: dict[int, list[CanonicalLengthRecord]] = {}
    for path, columns, source_name in specifications:
        if not path.is_file():
            continue
        try:
            frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        except pd.errors.EmptyDataError:
            continue
        if "screening_index" not in frame.columns:
            continue
        for row in frame.to_dict(orient="records"):
            screening_index = _positive_int(row.get("screening_index"))
            if screening_index is None:
                continue
            for column in columns:
                value = _positive_int(row.get(column))
                if value is not None:
                    candidates.setdefault(screening_index, []).append(
                        CanonicalLengthRecord(
                            value=value,
                            source=f"{source_name}.{column}",
                        )
                    )

    records: dict[int, CanonicalLengthRecord] = {}
    conflicts: dict[int, tuple[CanonicalLengthRecord, ...]] = {}
    for screening_index, values in candidates.items():
        distinct_values = {item.value for item in values}
        if len(distinct_values) > 1:
            conflicts[screening_index] = tuple(values)
        else:
            records[screening_index] = values[0]
    return CanonicalLengthSources(records=records, conflicts=conflicts)


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
    canonical_length_sources: CanonicalLengthSources | None = None,
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

    canonical_length_sources = canonical_length_sources or load_canonical_length_sources(
        project_root
    )
    if identity.screening_index in canonical_length_sources.conflicts:
        conflicting = canonical_length_sources.conflicts[identity.screening_index]
        return CensusSkip(
            pair_id=pair_id,
            code="conflicting_canonical_uniprot_full_length",
            details={
                "screening_index": identity.screening_index,
                "values": sorted({item.value for item in conflicting}),
                "sources": [item.source for item in conflicting],
            },
        )
    canonical_length = canonical_length_sources.records.get(identity.screening_index)
    if canonical_length is None:
        return CensusSkip(
            pair_id=pair_id,
            code="missing_canonical_uniprot_full_length",
            details={"screening_index": identity.screening_index},
        )

    pair_qc_path = pairs_root / pair_id / "pair_qc.json"
    pair_qc = json.loads(pair_qc_path.read_text(encoding="utf-8"))
    pair_qc_length = _positive_int(pair_qc.get("uniprot_length"))

    row: dict[str, Any] = {
        "protein_id": f"index{identity.screening_index}",
        "screening_index": identity.screening_index,
        "pair_id": pair_id,
        "mechanism_label": identity.mechanism_label_prior,
        "tier": 1,
        "pair_qc_path": str(pair_qc_path),
        "uniprot_full_length": canonical_length.value,
        "uniprot_full_length_source": canonical_length.source,
        "pair_qc_uniprot_length": pair_qc_length,
        "uniprot_full_length_corrected": pair_qc_length != canonical_length.value,
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


def _max_consecutive_run(positions: list[int]) -> int:
    if not positions:
        return 0
    ordered = sorted(positions)
    longest = current = 1
    for previous, current_position in pairwise(ordered):
        if current_position == previous + 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


def _min_pairwise_spacing(positions: list[int]) -> int | None:
    if len(positions) < 2:
        return None
    ordered = sorted(positions)
    return min(b - a for a, b in pairwise(ordered))


def analyze_sequence_mismatches(
    *,
    pdb_path: Path,
    afdb_path: Path,
    p0_stage_dir: Path,
    pair_id: str,
    fragment_start: int,
    model_length: int,
) -> SequenceMismatchAnalysis:
    """Full, non-fail-fast replay of P1's AA-identity and coverage checks.

    P1's own checks (stages/p1.py) are fail-fast by frozen contract and must
    stay that way. PDR-01 D1 needs the complete per-protein picture -- total
    mismatch count, site list, and shape statistics -- to classify a protein
    as variant-like vs mapping-error-like, so this reuses the same frozen-
    mapping/atom-parsing/backbone-selection primitives P1 uses
    (`p1._load_mapping`, `p1._load_atom_records`, `p1._index_records`,
    `structures.select_backbone_atoms`, `structures.group_residue_records`)
    and walks every mapped position instead of stopping at the first failure.
    A missing PDB residue or missing AFDB residue at a position is a coverage
    problem (P1's `residue_count_mismatch`), not an amino-acid mismatch, and
    is counted separately, never folded into mismatch_count. Symmetrically,
    an AFDB residue that is present but whose canonical AA disagrees with the
    mapping is counted in afdb_mismatch_count, never folded into
    mismatch_count either -- AFDB is predicted directly from the declared
    UniProt canonical sequence, so an AFDB-side divergence points at a
    mapping data defect, not a PDB-side sequence variant (PDR-01 §1.5/§1.10;
    confirmed real by 8pb5_A__P0DPA9, where PDB and AFDB agree with each
    other and only `mapping` is the outlier).
    """
    mapping = _load_mapping(p0_stage_dir / "outputs" / "residue_mapping.tsv")
    pdb_index = _index_records(_load_atom_records(pdb_path, pair_id))
    afdb_local_positions: dict[int, tuple] = {}
    for group in group_residue_records(_load_atom_records(afdb_path, pair_id)):
        key = group.residue.key
        if key.insertion_code or key.auth_seq_id in afdb_local_positions:
            continue
        afdb_local_positions[key.auth_seq_id] = group.atoms

    missing_pdb_count = 0
    missing_afdb_count = 0
    mismatches: list[MismatchSite] = []
    afdb_mismatches: list[AfdbMismatchSite] = []
    for row in mapping.itertuples(index=False):
        output_position = int(row.output_position)
        uniprot_position = int(row.uniprot_residue_number)
        mapping_aa = str(row.canonical_aa)

        key = ResidueKey(pair_id, str(row.auth_asym_id), int(row.auth_seq_id), str(row.insertion_code))
        pdb_atoms = pdb_index.get(key)
        if pdb_atoms is None:
            missing_pdb_count += 1
        else:
            try:
                selection = select_backbone_atoms(pdb_atoms)
            except ValueError:
                selection = None
            if selection is not None and selection.residue.canonical_aa != mapping_aa:
                mismatches.append(
                    MismatchSite(
                        output_position=output_position,
                        uniprot_position=uniprot_position,
                        mapping_aa=mapping_aa,
                        pdb_aa=selection.residue.canonical_aa,
                    )
                )

        model_position = uniprot_position - fragment_start + 1
        if not (1 <= model_position <= model_length):
            missing_afdb_count += 1
        else:
            afdb_atoms = afdb_local_positions.get(model_position)
            if afdb_atoms is None:
                missing_afdb_count += 1
            else:
                try:
                    afdb_selection = select_backbone_atoms(afdb_atoms)
                except ValueError:
                    afdb_selection = None
                if afdb_selection is not None and afdb_selection.residue.canonical_aa != mapping_aa:
                    afdb_mismatches.append(
                        AfdbMismatchSite(
                            output_position=output_position,
                            uniprot_position=uniprot_position,
                            mapping_aa=mapping_aa,
                            afdb_aa=afdb_selection.residue.canonical_aa,
                        )
                    )

    mapped_length = len(mapping)
    mismatch_count = len(mismatches)
    sequence_identity_mapped = 1.0 - mismatch_count / mapped_length
    paired_length = mapped_length - missing_pdb_count
    sequence_identity_paired = (
        1.0 - mismatch_count / paired_length if paired_length > 0 else None
    )
    mismatch_positions = [site.output_position for site in mismatches]

    return SequenceMismatchAnalysis(
        mismatch_count=mismatch_count,
        missing_pdb_count=missing_pdb_count,
        missing_afdb_count=missing_afdb_count,
        afdb_mismatch_count=len(afdb_mismatches),
        afdb_mismatches=tuple(afdb_mismatches),
        mapped_length=mapped_length,
        sequence_identity_mapped=sequence_identity_mapped,
        sequence_identity_paired=sequence_identity_paired,
        max_consecutive_run=_max_consecutive_run(mismatch_positions),
        min_pairwise_spacing=_min_pairwise_spacing(mismatch_positions),
        mismatches=tuple(mismatches),
    )


def evaluate_protein(
    row: dict[str, Any],
    *,
    project_root: Path,
    census_stage_root: Path,
    config: dict[str, Any],
    pipeline_version: str = "dataset-a.v1",
    run_id: str = CENSUS_RUN_ID,
) -> ProteinCensusRecord:
    """Resolve one manifest row through P0 (write, to a census scratch dir) then P1 (read-only).

    Calls `resolve_p0_inputs` (pure in-memory) before `run_p0` (write) purely
    to distinguish `p0_resolved` (scientific validation passed) from
    `p0_frozen` (outputs actually written) in `stage_reached` -- both stages
    are only reachable together in practice, but the distinction matters if a
    write-time failure ever diverges from a resolve-time one.
    """
    pair_id = row["pair_id"]
    identity_fields = {
        "pair_id": pair_id,
        "protein_id": row["protein_id"],
        "screening_index": row["screening_index"],
        "mechanism_label_prior": row["mechanism_label"],
        "uniprot_full_length": row.get("uniprot_full_length"),
        "uniprot_full_length_source": row.get("uniprot_full_length_source"),
        "pair_qc_uniprot_length": row.get("pair_qc_uniprot_length"),
        "uniprot_full_length_corrected": row.get("uniprot_full_length_corrected"),
    }

    try:
        resolution = resolve_p0_inputs(
            row, project_root=project_root, config=config, pipeline_version=pipeline_version
        )
    except P0ValidationError as exc:
        return ProteinCensusRecord(
            **identity_fields,
            outcome="p0_failure",
            stage_reached="manifest_built",
            failure_code=exc.code,
            failure_details={"message": str(exc), **exc.details},
        )
    mapped_length = len(resolution.mapping)

    p0_stage_dir = census_stage_root / row["protein_id"] / P0_STAGE_NAME
    p0_result = run_p0(
        manifest_row=row,
        project_root=project_root,
        stage_dir=p0_stage_dir,
        config=config,
        pipeline_version=pipeline_version,
        run_id=run_id,
    )
    if not p0_result.validation.validation_pass:
        return ProteinCensusRecord(
            **identity_fields,
            outcome="p0_failure",
            stage_reached="p0_resolved",
            mapped_length=mapped_length,
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
        analysis_fields: dict[str, Any] = {}
        if exc.code == "pdb_amino_acid_mismatch":
            analysis = analyze_sequence_mismatches(
                pdb_path=Path(row["pdb_structure_path"]),
                afdb_path=Path(row["afdb_model_path"]),
                p0_stage_dir=p0_stage_dir,
                pair_id=pair_id,
                fragment_start=resolution.fragment.uniprot_start,
                model_length=resolution.fragment.model_residue_count,
            )
            analysis_fields = {
                "mismatch_count": analysis.mismatch_count,
                "missing_pdb_count": analysis.missing_pdb_count,
                "missing_afdb_count": analysis.missing_afdb_count,
                "sequence_identity_mapped": analysis.sequence_identity_mapped,
                "sequence_identity_paired": analysis.sequence_identity_paired,
                "max_consecutive_run": analysis.max_consecutive_run,
                "min_pairwise_spacing": analysis.min_pairwise_spacing,
                "mismatches": analysis.mismatches,
                "afdb_mismatch_count": analysis.afdb_mismatch_count,
                "afdb_mismatches": analysis.afdb_mismatches,
            }
        return ProteinCensusRecord(
            **identity_fields,
            outcome="p1_failure",
            stage_reached="p0_frozen",
            mapped_length=mapped_length,
            failure_code=exc.code,
            failure_details={"message": str(exc), **exc.details},
            **analysis_fields,
        )

    return ProteinCensusRecord(
        **identity_fields,
        outcome="p1_pairing_ok",
        stage_reached="p1_resolved",
        mapped_length=mapped_length,
    )


def _summarize(records: list[ProteinCensusRecord], *, total_local_pairs: int) -> dict[str, Any]:
    evaluated = [record for record in records if record.outcome != "skipped"]
    p0_failures = [record for record in records if record.outcome == "p0_failure"]
    p1_failures = [record for record in records if record.outcome == "p1_failure"]
    mismatch_hits = [record for record in records if record.failure_code == "pdb_amino_acid_mismatch"]
    afdb_mismatch_hits = [record for record in mismatch_hits if (record.afdb_mismatch_count or 0) > 0]
    stage_reached_counts = dict(
        Counter(record.stage_reached for record in evaluated if record.stage_reached)
    )
    stage_not_reached_counts = {
        stage: sum(
            1
            for record in evaluated
            if record.stage_reached is not None
            and _STAGE_ORDER.index(record.stage_reached) < _STAGE_ORDER.index(stage)
        )
        for stage in _STAGE_ORDER
    }
    corrected = [record for record in records if record.uniprot_full_length_corrected is True]
    matched = [record for record in records if record.uniprot_full_length_corrected is False]
    full_length_provenance_audit = {
        "matched_count": len(matched),
        "corrected_count": len(corrected),
        "missing_count": sum(
            record.failure_code == "missing_canonical_uniprot_full_length" for record in records
        ),
        "conflict_count": sum(
            record.failure_code == "conflicting_canonical_uniprot_full_length"
            for record in records
        ),
        "corrected_proteins": [
            {
                "screening_index": record.screening_index,
                "pair_id": record.pair_id,
                "previous_pair_qc_length": record.pair_qc_uniprot_length,
                "canonical_uniprot_full_length": record.uniprot_full_length,
            }
            for record in corrected
        ],
    }
    return {
        "total_local_pairs": total_local_pairs,
        "outcome_counts": dict(Counter(record.outcome for record in records)),
        "failure_code_counts": dict(
            Counter(record.failure_code for record in records if record.failure_code)
        ),
        "p0_failure_code_counts": dict(
            Counter(record.failure_code for record in p0_failures if record.failure_code)
        ),
        "p1_failure_code_counts": dict(
            Counter(record.failure_code for record in p1_failures if record.failure_code)
        ),
        "stage_reached_counts": stage_reached_counts,
        "stage_not_reached_counts": stage_not_reached_counts,
        "pdb_amino_acid_mismatch_hit_count": len(mismatch_hits),
        "pdb_amino_acid_mismatch_counts": [record.mismatch_count for record in mismatch_hits],
        "afdb_amino_acid_mismatch_hit_count": len(afdb_mismatch_hits),
        "afdb_amino_acid_mismatch_hit_pair_ids": [record.pair_id for record in afdb_mismatch_hits],
        "full_length_provenance_audit": full_length_provenance_audit,
        "censoring_note": _CENSORING_NOTE,
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
    run_id: str = CENSUS_RUN_ID,
) -> Round1Report:
    identity_sources = load_identity_sources(candidate_lifecycle_path, replacement_lifecycle_path)
    canonical_length_sources = load_canonical_length_sources(project_root)
    pair_ids = discover_local_pairs(pairs_root)

    records: list[ProteinCensusRecord] = []
    for pair_id in pair_ids:
        row_or_skip = build_manifest_row(
            pair_id,
            pairs_root=pairs_root,
            identity_sources=identity_sources,
            project_root=project_root,
            canonical_length_sources=canonical_length_sources,
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
                run_id=run_id,
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
    atomic_write_text(
        out_path,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
