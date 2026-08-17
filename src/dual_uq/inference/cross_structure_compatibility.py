"""ProteinMPNN cross-structure scoring for generated sequences."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from dual_uq.core.atomic_io import atomic_write_new_bytes
from dual_uq.core.hashing import sha256_bytes
from dual_uq.inference.generative_propagation import (
    load_generation_inputs,
    load_generation_records,
)
from dual_uq.models.proteinmpnn import (
    ProteinMPNNScore,
    ProteinMPNNStructureInput,
    make_decoding_realization,
)
from dual_uq.models.proteinmpnn_generation import GeneratedSequenceRecord
from dual_uq.models.scoring import ScorerBinding

CROSS_SCORE_PROTOCOL = "stage0_fixed_sequence_autoregressive_mask_logp_v1"
_SHARD_SCHEMA = "cross_structure_compatibility_score_shard_v1"


class CrossStructureScoringError(ValueError):
    """Structured cross-score input, execution, or artifact failure."""


class CrossScoreAdapter(Protocol):
    def score_sequences(
        self,
        structure: ProteinMPNNStructureInput,
        sequences: tuple[str, ...],
        realization: Any,
        *,
        batch_size: int,
    ) -> tuple[ProteinMPNNScore, ...]: ...


@dataclass(frozen=True, slots=True)
class CrossStructureInputs:
    """Validated generated records and their exact common-mask projections."""

    records: tuple[GeneratedSequenceRecord, ...]
    projections: Mapping[tuple[str, str], ProteinMPNNStructureInput]
    expected_protein_count: int = 68

    def __post_init__(self) -> None:
        records = tuple(self.records)
        if not records:
            raise CrossStructureScoringError("generated records are required")
        proteins = {record.request.protein_id for record in records}
        if len(proteins) != self.expected_protein_count:
            raise CrossStructureScoringError("cross-score cohort cardinality differs")
        key_set = {
            (record.request.protein_id, record.request.backbone_condition, record.request.sample_index)
            for record in records
        }
        expected = {
            (protein, condition, sample_index)
            for protein in proteins
            for condition in ("PDB", "AFDB")
            for sample_index in range(256)
        }
        if key_set != expected:
            raise CrossStructureScoringError("generated record grid is incomplete")
        projections = dict(self.projections)
        expected_projection_keys = {
            (protein, condition) for protein in proteins for condition in ("PDB", "AFDB")
        }
        if set(projections) != expected_projection_keys:
            raise CrossStructureScoringError("cross-score projection grid is incomplete")
        for key, projection in projections.items():
            if not isinstance(projection, ProteinMPNNStructureInput):
                raise CrossStructureScoringError("cross-score projection is invalid")
            if (projection.protein_id, projection.backbone_condition) != key:
                raise CrossStructureScoringError("cross-score projection identity differs")
        for record in records:
            projection = projections[(record.request.protein_id, record.request.backbone_condition)]
            if (
                record.request.structure_sha256 != projection.structure_sha256
                or record.request.canonical_positions != projection.uniprot_positions
                or record.request.wt_sequence_projection != projection.wt_sequence_projection
            ):
                raise CrossStructureScoringError("generated record is not bound to its projection")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "projections", projections)


@dataclass(frozen=True, slots=True)
class CrossStructureScoreRow:
    """One generated sequence scored under one evaluated structure."""

    protein_id: str
    generated_condition: str
    evaluated_condition: str
    structure_sha256: str
    generated_structure_sha256: str
    sample_index: int
    seed: int
    sample_class: str
    sequence_hash: str
    sequence: str
    score_sum_logp_mask: float
    score_mean_logp_mask: float
    scored_residue_count: int
    scoring_realization_id: str
    scorer_id: str
    implementation_id: str
    checkpoint_id: str | None
    score_contract_id: str


@dataclass(frozen=True, slots=True)
class CrossStructureScoreRunResult:
    """Rows and execution accounting for one deterministic scoring run."""

    rows: tuple[CrossStructureScoreRow, ...]
    executed_proteins: int
    reused_proteins: int


def build_cross_structure_inputs(project_root: Path, generation_root: Path) -> CrossStructureInputs:
    """Resolve generated records and the same frozen projections used for generation."""
    records = load_generation_records(generation_root)
    generation_inputs = load_generation_inputs(project_root)
    projections = {
        (condition.protein_id, condition.backbone_condition): condition.structure.projection
        for condition in generation_inputs.conditions
    }
    return CrossStructureInputs(
        records=records,
        projections=projections,
        expected_protein_count=generation_inputs.expected_protein_count,
    )


def _binding(adapter: object) -> ScorerBinding:
    observed = getattr(adapter, "binding", None)
    implementation_id = getattr(observed, "implementation_id", None)
    checkpoint_id = getattr(observed, "checkpoint_id", None)
    if implementation_id is None:
        implementation_id = getattr(adapter, "implementation_id", None)
    if checkpoint_id is None:
        checkpoint_id = getattr(adapter, "checkpoint_id", None)
    if not isinstance(implementation_id, str) or not isinstance(checkpoint_id, str):
        raise CrossStructureScoringError("scorer identity is unavailable")
    return ScorerBinding(
        scorer_id="ProteinMPNN",
        implementation_id=implementation_id,
        checkpoint_id=checkpoint_id,
        score_contract_id=CROSS_SCORE_PROTOCOL,
    )


def _row_payload(row: CrossStructureScoreRow) -> dict[str, Any]:
    return {
        "protein_id": row.protein_id,
        "generated_condition": row.generated_condition,
        "evaluated_condition": row.evaluated_condition,
        "structure_sha256": row.structure_sha256,
        "generated_structure_sha256": row.generated_structure_sha256,
        "sample_index": row.sample_index,
        "seed": row.seed,
        "sample_class": row.sample_class,
        "sequence_hash": row.sequence_hash,
        "sequence": row.sequence,
        "score_sum_logp_mask": row.score_sum_logp_mask,
        "score_mean_logp_mask": row.score_mean_logp_mask,
        "scored_residue_count": row.scored_residue_count,
        "scoring_realization_id": row.scoring_realization_id,
        "scorer_id": row.scorer_id,
        "implementation_id": row.implementation_id,
        "checkpoint_id": row.checkpoint_id,
        "score_contract_id": row.score_contract_id,
    }


def _row_from_payload(value: Mapping[str, Any]) -> CrossStructureScoreRow:
    try:
        return CrossStructureScoreRow(**dict(value))
    except (TypeError, ValueError) as exc:
        raise CrossStructureScoringError("malformed cross-score row") from exc


def _protein_shard_path(output_root: Path, protein_id: str) -> Path:
    digest = sha256_bytes(protein_id.encode("utf-8"))
    return output_root / "shards" / digest[:2] / f"{digest}.json"


def _render_shard(
    protein_id: str,
    rows: tuple[CrossStructureScoreRow, ...],
    binding: ScorerBinding,
) -> bytes:
    if len(rows) != 1024:
        raise CrossStructureScoringError("cross-score shard row count differs")
    payload = {
        "schema_version": _SHARD_SCHEMA,
        "protein_id": protein_id,
        "scorer_binding": {
            "scorer_id": binding.scorer_id,
            "implementation_id": binding.implementation_id,
            "checkpoint_id": binding.checkpoint_id,
            "score_contract_id": binding.score_contract_id,
        },
        "rows": [_row_payload(row) for row in rows],
    }
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _read_shard(path: Path, protein_id: str, binding: ScorerBinding) -> tuple[CrossStructureScoreRow, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        observed = ScorerBinding(**payload["scorer_binding"])
        rows = tuple(_row_from_payload(value) for value in payload["rows"])
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise CrossStructureScoringError(f"malformed cross-score shard: {path}") from exc
    if payload.get("schema_version") != _SHARD_SCHEMA or payload.get("protein_id") != protein_id:
        raise CrossStructureScoringError("cross-score shard identity differs")
    if observed != binding or len(rows) != 1024:
        raise CrossStructureScoringError("cross-score shard binding differs")
    keys = {
        (row.generated_condition, row.evaluated_condition, row.sample_index, row.sequence_hash)
        for row in rows
    }
    expected = {
        (generated, evaluated, index)
        for generated in ("PDB", "AFDB")
        for evaluated in ("PDB", "AFDB")
        for index in range(256)
    }
    observed = {(row.generated_condition, row.evaluated_condition, row.sample_index) for row in rows}
    if len(keys) != 1024 or observed != expected or any(row.protein_id != protein_id for row in rows):
        raise CrossStructureScoringError("cross-score shard keys are invalid")
    if not np.isfinite([row.score_mean_logp_mask for row in rows]).all():
        raise CrossStructureScoringError("cross-score shard contains non-finite values")
    return tuple(sorted(rows, key=lambda row: (row.generated_condition, row.evaluated_condition, row.sample_index)))


def _score_protein(
    protein_id: str,
    records: tuple[GeneratedSequenceRecord, ...],
    projections: Mapping[tuple[str, str], ProteinMPNNStructureInput],
    adapter: CrossScoreAdapter,
    binding: ScorerBinding,
    *,
    batch_size: int,
) -> tuple[CrossStructureScoreRow, ...]:
    by_condition = {
        condition: tuple(
            sorted(
                (record for record in records if record.request.backbone_condition == condition),
                key=lambda record: record.request.sample_index,
            )
        )
        for condition in ("PDB", "AFDB")
    }
    rows: list[CrossStructureScoreRow] = []
    for generated_condition in ("PDB", "AFDB"):
        generated = by_condition[generated_condition]
        if len(generated) != 256:
            raise CrossStructureScoringError("cross-score generated condition is incomplete")
        projection_for_order = projections[(protein_id, generated_condition)]
        realization = make_decoding_realization(
            protein_id=protein_id,
            mask_length=projection_for_order.residue_count,
            repeat_index=0,
            seed=0,
            protocol_version=CROSS_SCORE_PROTOCOL,
        )
        for evaluated_condition in ("PDB", "AFDB"):
            projection = projections[(protein_id, evaluated_condition)]
            scores = adapter.score_sequences(
                projection,
                tuple(record.sequence for record in generated),
                realization,
                batch_size=batch_size,
            )
            if len(scores) != 256:
                raise CrossStructureScoringError("cross-score result count differs")
            for record, score in zip(generated, scores, strict=True):
                rows.append(
                    CrossStructureScoreRow(
                        protein_id=protein_id,
                        generated_condition=generated_condition,
                        evaluated_condition=evaluated_condition,
                        structure_sha256=str(projection.structure_sha256),
                        generated_structure_sha256=record.request.structure_sha256,
                        sample_index=record.request.sample_index,
                        seed=record.request.seed,
                        sample_class=record.request.sample_class,
                        sequence_hash=record.sequence_hash,
                        sequence=record.sequence,
                        score_sum_logp_mask=float(score.score_sum_logp_mask),
                        score_mean_logp_mask=float(score.score_mean_logp_mask),
                        scored_residue_count=projection.residue_count,
                        scoring_realization_id=realization.fingerprint,
                        scorer_id=binding.scorer_id,
                        implementation_id=binding.implementation_id,
                        checkpoint_id=binding.checkpoint_id,
                        score_contract_id=binding.score_contract_id,
                    )
                )
    return tuple(sorted(rows, key=lambda row: (row.generated_condition, row.evaluated_condition, row.sample_index)))


def score_cross_structure(
    inputs: CrossStructureInputs,
    adapter: CrossScoreAdapter,
    *,
    output_root: Path,
    batch_size: int = 32,
    resume: bool,
    worker_index: int = 0,
    worker_count: int = 1,
) -> CrossStructureScoreRunResult:
    """Score every generated sequence under both paired structures."""
    if not isinstance(inputs, CrossStructureInputs):
        raise CrossStructureScoringError("cross-score inputs are required")
    if batch_size <= 0 or worker_count <= 0 or not 0 <= worker_index < worker_count:
        raise CrossStructureScoringError("invalid cross-score execution partition")
    binding = _binding(adapter)
    root = Path(output_root).expanduser().resolve()
    proteins = sorted({record.request.protein_id for record in inputs.records})
    selected = tuple(protein for index, protein in enumerate(proteins) if index % worker_count == worker_index)
    rows: list[CrossStructureScoreRow] = []
    executed = reused = 0
    for protein_id in selected:
        path = _protein_shard_path(root, protein_id)
        if path.exists():
            if not resume:
                raise CrossStructureScoringError("cross-score shard already exists; use --resume")
            existing = _read_shard(path, protein_id, binding)
            rows.extend(existing)
            reused += 1
            continue
        protein_records = tuple(record for record in inputs.records if record.request.protein_id == protein_id)
        generated = _score_protein(
            protein_id,
            protein_records,
            inputs.projections,
            adapter,
            binding,
            batch_size=batch_size,
        )
        rendered = _render_shard(protein_id, generated, binding)
        try:
            atomic_write_new_bytes(path, rendered)
        except FileExistsError:
            existing = _read_shard(path, protein_id, binding)
            if existing != generated:
                raise CrossStructureScoringError("immutable cross-score shard conflict") from None
            rows.extend(existing)
            reused += 1
            continue
        rows.extend(_read_shard(path, protein_id, binding))
        executed += 1
    return CrossStructureScoreRunResult(
        rows=tuple(sorted(rows, key=lambda row: (row.protein_id, row.generated_condition, row.evaluated_condition, row.sample_index))),
        executed_proteins=executed,
        reused_proteins=reused,
    )


def load_cross_score_rows(output_root: Path) -> tuple[CrossStructureScoreRow, ...]:
    """Load and validate all immutable per-protein cross-score shards."""
    root = Path(output_root).expanduser().resolve()
    paths = tuple(sorted((root / "shards").glob("*/*.json")))
    if not paths:
        raise CrossStructureScoringError("no cross-score shards were found")
    rows: list[CrossStructureScoreRow] = []
    canonical_binding: ScorerBinding | None = None
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            binding = ScorerBinding(**payload["scorer_binding"])
            protein_id = str(payload["protein_id"])
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise CrossStructureScoringError(f"malformed cross-score shard: {path}") from exc
        if canonical_binding is None:
            canonical_binding = binding
        elif binding != canonical_binding:
            raise CrossStructureScoringError("cross-score shards have inconsistent scorer binding")
        rows.extend(_read_shard(path, protein_id, binding))
    return tuple(sorted(rows, key=lambda row: (row.protein_id, row.generated_condition, row.evaluated_condition, row.sample_index)))


def cross_score_rows_frame(rows: tuple[CrossStructureScoreRow, ...] | list[CrossStructureScoreRow]) -> pd.DataFrame:
    """Convert validated score rows to the canonical analysis input table."""
    return pd.DataFrame([_row_payload(row) for row in rows])
