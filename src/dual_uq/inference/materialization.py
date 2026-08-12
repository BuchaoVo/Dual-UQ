"""Validated immutable materialization for normalized formal score results."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any

from dual_uq.core.atomic_io import atomic_write_new_bytes
from dual_uq.core.hashing import sha256_file
from dual_uq.models.scoring import (
    ScoreDispatchError,
    ScorerBinding,
    ScoreRecord,
    ScoreRequest,
    VariantKind,
    execute_score_request,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_FORMAL_SHARD_SCHEMA = "stage0_fixed_probe_scoring_shard_v1"
_CONSOLIDATION_SCHEMA = "formal_measurement_consolidation_v1"

if TYPE_CHECKING:
    from dual_uq.inference.formal import FormalRequestDefinition




def _parquet_modules() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise FormalMaterializationError(
            "parquet_runtime_unavailable",
            "Bounded-memory consolidation requires the data environment",
        ) from exc
    return pa, pq


def _consolidated_schemas() -> tuple[Any, Any]:
    pa, _pq = _parquet_modules()
    wt = pa.schema(
        [
            ("protein_id", pa.string()),
            ("backbone_condition", pa.string()),
            ("backbone_sha256", pa.string()),
            ("repeat_index", pa.int64()),
            ("seed", pa.int64()),
            ("decoding_realization_sha256", pa.string()),
            ("score_sum_logp_mask", pa.float64()),
            ("score_mean_logp_mask", pa.float64()),
            ("scored_residue_count", pa.int64()),
            ("model_checkpoint_sha256", pa.string()),
            ("scoring_protocol", pa.string()),
        ]
    )
    probe = pa.schema(
        [
            ("protein_id", pa.string()),
            ("sequence_hash", pa.string()),
            ("position", pa.int64()),
            ("wt_aa", pa.string()),
            ("mut_aa", pa.string()),
            ("backbone_condition", pa.string()),
            ("backbone_sha256", pa.string()),
            ("repeat_index", pa.int64()),
            ("seed", pa.int64()),
            ("decoding_realization_sha256", pa.string()),
            ("score_sum_logp_mask", pa.float64()),
            ("score_mean_logp_mask", pa.float64()),
            ("delta_score_vs_wt", pa.float64()),
            ("scored_residue_count", pa.int64()),
            ("model_checkpoint_sha256", pa.string()),
            ("scoring_protocol", pa.string()),
        ]
    )
    return wt, probe


class FormalMaterializationError(ValueError):
    """Structured formal artifact validation or materialization failure."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class ArtifactInspectionState(str, Enum):
    """State observed directly from the authoritative artifact and validator."""

    MISSING = "MISSING"
    VALID = "VALID"
    INVALID = "INVALID"
    CONFLICT = "CONFLICT"


@dataclass(frozen=True, slots=True)
class FormalArtifactBinding:
    """Operational binding from one orchestration identity to one shard path."""

    orchestration_id: str
    relative_path: Path

    def __post_init__(self) -> None:
        _require_sha256(self.orchestration_id, "orchestration_id")
        relative = Path(self.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("formal artifact path must be portable and relative")
        object.__setattr__(self, "relative_path", relative)

    def resolve(self, root: Path) -> Path:
        """Resolve the portable binding beneath a caller-owned artifact root."""
        return Path(root).resolve() / self.relative_path


@dataclass(frozen=True, slots=True)
class ArtifactInspection:
    """One immutable validator observation used to derive inventory state."""

    state: ArtifactInspectionState
    artifact_sha256: str | None = None
    error_code: str | None = None
    detail: str | None = None


def _require_sha256(value: object, field_name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def bind_formal_artifact(orchestration_id: str) -> FormalArtifactBinding:
    """Bind one stable orchestration identity to a portable sharded JSON path."""
    identity = _require_sha256(orchestration_id, "orchestration_id")
    return FormalArtifactBinding(
        orchestration_id=identity,
        relative_path=Path("shards") / identity[:2] / f"{identity}.json",
    )


def formal_shard_binding_record(
    *,
    request: ScoreRequest,
    scorer_binding: ScorerBinding,
    scientific_fingerprint: str,
    orchestration_id: str,
) -> dict[str, Any]:
    if scorer_binding.score_contract_id != request.score_contract_id:
        raise FormalMaterializationError(
            "score_contract_mismatch",
            "Scorer binding differs from the formal request score contract",
        )
    return {
        "orchestration_id": _require_sha256(
            orchestration_id, "orchestration_id"
        ),
        "scientific_fingerprint": _require_sha256(
            scientific_fingerprint, "scientific_fingerprint"
        ),
        "protein_id": request.protein_id,
        "backbone_condition": request.condition.condition_id,
        "backbone_sha256": request.condition.structure_sha256,
        "common_mask_binding": request.scoring_domain_id,
        "repeat_index": request.repeat_index,
        "seed": request.seed,
        "decoding_realization_sha256": request.realization_id,
        "decoding_realization_algorithm": request.realization_algorithm,
        "candidate_identity_sha256": request.candidate_collection.collection_id,
        "candidate_count": len(request.candidate_collection.probes),
        "mask_length": len(request.canonical_positions),
        "scorer_id": scorer_binding.scorer_id,
        "implementation_commit": scorer_binding.implementation_id,
        "checkpoint_sha256": scorer_binding.checkpoint_id,
        "scoring_protocol": scorer_binding.score_contract_id,
    }


class _PersistedRecordScorer:
    def __init__(
        self, binding: ScorerBinding, records: tuple[ScoreRecord, ...]
    ) -> None:
        self.binding = binding
        self._records = records

    def score(self, request: ScoreRequest) -> tuple[ScoreRecord, ...]:
        del request
        return self._records


def _validate_records(
    *,
    request: ScoreRequest,
    scorer_binding: ScorerBinding,
    records: tuple[ScoreRecord, ...],
) -> tuple[ScoreRecord, ...]:
    try:
        return execute_score_request(
            _PersistedRecordScorer(scorer_binding, records), request
        )
    except ScoreDispatchError as exc:
        raise FormalMaterializationError(exc.code, str(exc)) from exc


def build_formal_shard_payload(
    *,
    request: ScoreRequest,
    scorer_binding: ScorerBinding,
    scientific_fingerprint: str,
    orchestration_id: str,
    records: tuple[ScoreRecord, ...],
    execution_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Adapt normalized records to the existing formal JSON shard schema."""
    validated = _validate_records(
        request=request, scorer_binding=scorer_binding, records=tuple(records)
    )
    if not execution_provenance:
        raise FormalMaterializationError(
            "missing_execution_provenance", "Execution provenance is required"
        )
    wt, *probes = validated

    def score_fields(record: ScoreRecord) -> dict[str, Any]:
        return {
            "score_sum_logp_mask": record.score_sum_logp_mask,
            "score_mean_logp_mask": record.score_mean_logp_mask,
            "scored_residue_count": record.scored_residue_count,
        }

    return {
        "schema_version": _FORMAL_SHARD_SCHEMA,
        "binding": formal_shard_binding_record(
            request=request,
            scorer_binding=scorer_binding,
            scientific_fingerprint=scientific_fingerprint,
            orchestration_id=orchestration_id,
        ),
        "wt_score": score_fields(wt),
        "candidate_scores": [
            {"sequence_hash": record.sequence_hash, **score_fields(record)}
            for record in probes
        ],
        "execution_environment": dict(execution_provenance),
    }


def _record_from_payload(
    *,
    request: ScoreRequest,
    scorer_binding: ScorerBinding,
    variant_index: int,
    payload: Mapping[str, Any],
) -> ScoreRecord:
    variant = request.candidate_collection.variants[variant_index]
    if variant.variant_kind is VariantKind.PROBE and payload.get(
        "sequence_hash"
    ) != variant.sequence_hash:
        raise FormalMaterializationError(
            "formal_shard_candidate_identity_mismatch",
            "Persisted probe identity or ordering differs",
        )
    try:
        return ScoreRecord(
            protein_id=request.protein_id,
            condition_id=request.condition.condition_id,
            structure_sha256=request.condition.structure_sha256,
            variant_kind=variant.variant_kind,
            variant_id=variant.variant_id,
            sequence_hash=(
                None
                if variant.variant_kind is VariantKind.WT
                else variant.sequence_hash
            ),
            position=variant.position,
            wt_aa=variant.wt_aa,
            mut_aa=variant.mut_aa,
            repeat_index=request.repeat_index,
            seed=request.seed,
            realization_id=request.realization_id,
            scorer_id=scorer_binding.scorer_id,
            implementation_id=scorer_binding.implementation_id,
            checkpoint_id=scorer_binding.checkpoint_id,
            score_contract_id=scorer_binding.score_contract_id,
            score_sum_logp_mask=float(payload["score_sum_logp_mask"]),
            score_mean_logp_mask=float(payload["score_mean_logp_mask"]),
            scored_residue_count=int(payload["scored_residue_count"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FormalMaterializationError(
            "invalid_formal_shard", "Persisted score record is invalid"
        ) from exc


def validate_formal_shard_payload(
    payload: Any,
    *,
    request: ScoreRequest,
    scorer_binding: ScorerBinding,
    scientific_fingerprint: str,
    orchestration_id: str,
) -> dict[str, Any]:
    """Validate schema, complete binding, membership, and score arithmetic."""
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        _FORMAL_SHARD_SCHEMA
    ):
        raise FormalMaterializationError(
            "invalid_formal_shard", "Formal shard schema differs"
        )
    expected_binding = formal_shard_binding_record(
        request=request,
        scorer_binding=scorer_binding,
        scientific_fingerprint=scientific_fingerprint,
        orchestration_id=orchestration_id,
    )
    if payload.get("binding") != expected_binding:
        raise FormalMaterializationError(
            "formal_shard_binding_mismatch", "Formal shard binding differs"
        )
    candidates = payload.get("candidate_scores")
    if not isinstance(candidates, list) or len(candidates) != len(
        request.candidate_collection.probes
    ):
        raise FormalMaterializationError(
            "formal_shard_candidate_count_mismatch",
            "Formal shard candidate count differs",
        )
    if not isinstance(payload.get("execution_environment"), dict):
        raise FormalMaterializationError(
            "missing_execution_provenance", "Formal shard provenance is absent"
        )
    wt_payload = payload.get("wt_score")
    if not isinstance(wt_payload, dict) or any(
        not isinstance(row, dict) for row in candidates
    ):
        raise FormalMaterializationError(
            "invalid_formal_shard", "Formal shard score records are invalid"
        )
    records = (
        _record_from_payload(
            request=request,
            scorer_binding=scorer_binding,
            variant_index=0,
            payload=wt_payload,
        ),
        *(
            _record_from_payload(
                request=request,
                scorer_binding=scorer_binding,
                variant_index=index,
                payload=row,
            )
            for index, row in enumerate(candidates, start=1)
        ),
    )
    _validate_records(
        request=request, scorer_binding=scorer_binding, records=tuple(records)
    )
    return payload


def _render_payload(payload: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FormalMaterializationError(
            "invalid_formal_shard", "Formal shard cannot be serialized"
        ) from exc


def inspect_formal_artifact(
    *,
    root: Path,
    artifact: FormalArtifactBinding,
    request: ScoreRequest,
    scorer_binding: ScorerBinding,
    scientific_fingerprint: str,
) -> ArtifactInspection:
    """Inspect one canonical artifact; file existence alone never completes it."""
    path = artifact.resolve(root)
    if not path.exists():
        return ArtifactInspection(ArtifactInspectionState.MISSING)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return ArtifactInspection(
            ArtifactInspectionState.INVALID,
            error_code="invalid_formal_shard",
            detail=str(exc),
        )
    try:
        validate_formal_shard_payload(
            payload,
            request=request,
            scorer_binding=scorer_binding,
            scientific_fingerprint=scientific_fingerprint,
            orchestration_id=artifact.orchestration_id,
        )
    except FormalMaterializationError as exc:
        state = (
            ArtifactInspectionState.CONFLICT
            if exc.code == "formal_shard_binding_mismatch"
            else ArtifactInspectionState.INVALID
        )
        return ArtifactInspection(state, error_code=exc.code, detail=str(exc))
    return ArtifactInspection(
        ArtifactInspectionState.VALID, artifact_sha256=sha256_file(path)
    )


def load_validated_formal_artifact(
    *,
    root: Path,
    artifact: FormalArtifactBinding,
    request: ScoreRequest,
    scorer_binding: ScorerBinding,
    scientific_fingerprint: str,
) -> dict[str, Any]:
    """Load one artifact only after the authoritative content validator passes."""
    path = artifact.resolve(root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FormalMaterializationError(
            "invalid_formal_shard", f"Unable to read formal shard: {artifact.relative_path}"
        ) from exc
    return validate_formal_shard_payload(
        payload,
        request=request,
        scorer_binding=scorer_binding,
        scientific_fingerprint=scientific_fingerprint,
        orchestration_id=artifact.orchestration_id,
    )


def materialize_formal_artifact(
    *,
    root: Path,
    artifact: FormalArtifactBinding,
    payload: dict[str, Any],
    request: ScoreRequest,
    scorer_binding: ScorerBinding,
    scientific_fingerprint: str,
) -> str:
    """Atomically create one validated shard or recognize identical bytes."""
    validate_formal_shard_payload(
        payload,
        request=request,
        scorer_binding=scorer_binding,
        scientific_fingerprint=scientific_fingerprint,
        orchestration_id=artifact.orchestration_id,
    )
    rendered = _render_payload(payload)
    path = artifact.resolve(root)
    if path.exists():
        if path.read_bytes() == rendered:
            return "reused_identical"
        raise FormalMaterializationError(
            "immutable_formal_shard_conflict",
            f"Existing immutable artifact differs: {artifact.relative_path}",
        )
    try:
        atomic_write_new_bytes(path, rendered)
    except FileExistsError as exc:
        if path.read_bytes() == rendered:
            return "reused_identical"
        raise FormalMaterializationError(
            "immutable_formal_shard_conflict",
            f"Concurrent immutable artifact differs: {artifact.relative_path}",
        ) from exc
    return "created"


def _validated_existing_consolidation(
    *,
    manifest_path: Path,
    wt_path: Path,
    probe_path: Path,
    expected_wt_rows: int,
    expected_probe_rows: int,
    provenance: Mapping[str, Any],
) -> dict[str, Any] | None:
    _pa, pq = _parquet_modules()
    wt_schema, probe_schema = _consolidated_schemas()
    if not manifest_path.exists():
        # A manifest-last interruption may leave one or both immutable Parquet
        # outputs in place. Rebuild the bounded temporary outputs and let the
        # immutable promotion compare their content before completing manifest.
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FormalMaterializationError(
            "invalid_consolidation_manifest",
            "Consolidation manifest is unreadable",
        ) from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != _CONSOLIDATION_SCHEMA
        or manifest.get("provenance") != dict(provenance)
        or manifest.get("counts")
        != {
            "canonical_shards": expected_wt_rows,
            "probe_rows": expected_probe_rows,
            "wt_rows": expected_wt_rows,
        }
    ):
        raise FormalMaterializationError(
            "immutable_consolidation_conflict",
            "Existing consolidation manifest differs from the requested universe",
        )
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise FormalMaterializationError(
            "invalid_consolidation_manifest", "Consolidation outputs are absent"
        )
    for key, path, rows, schema in (
        ("wt", wt_path, expected_wt_rows, wt_schema),
        ("probe", probe_path, expected_probe_rows, probe_schema),
    ):
        record = outputs.get(key)
        if (
            not isinstance(record, dict)
            or not path.is_file()
            or record.get("sha256") != sha256_file(path)
            or record.get("rows") != rows
        ):
            raise FormalMaterializationError(
                "invalid_consolidated_output",
                f"Existing consolidated {key} artifact does not validate",
            )
        metadata = pq.ParquetFile(path).metadata
        observed_schema = pq.read_schema(path)
        if metadata.num_rows != rows or not observed_schema.equals(schema):
            raise FormalMaterializationError(
                "invalid_consolidated_output",
                f"Existing consolidated {key} schema/cardinality differs",
            )
    return {
        "wt_path": wt_path,
        "probe_path": probe_path,
        "manifest_path": manifest_path,
        "write_status": {
            "manifest": "reused_identical",
            "probe": "reused_identical",
            "wt": "reused_identical",
        },
        "manifest": manifest,
    }


def _promote_immutable_parquet(temporary: Path, destination: Path) -> str:
    if destination.exists():
        if sha256_file(destination) == sha256_file(temporary):
            return "reused_identical"
        raise FormalMaterializationError(
            "immutable_consolidation_conflict",
            f"Existing consolidated artifact differs: {destination.name}",
        )
    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        if sha256_file(destination) == sha256_file(temporary):
            return "reused_identical"
        raise FormalMaterializationError(
            "immutable_consolidation_conflict",
            f"Consolidated artifact appeared concurrently: {destination.name}",
        ) from exc
    return "created"


def consolidate_formal_measurements(
    definitions: tuple[FormalRequestDefinition, ...],
    *,
    artifact_root: Path,
    output_root: Path,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Stream validated expected shards into canonical WT/probe Parquet outputs."""
    pa, pq = _parquet_modules()
    wt_schema, probe_schema = _consolidated_schemas()
    definitions = tuple(definitions)
    if not definitions or not provenance:
        raise FormalMaterializationError(
            "invalid_consolidation_request",
            "Expected definitions and consolidation provenance are required",
        )
    identities = [definition.orchestration_id for definition in definitions]
    if len(identities) != len(set(identities)):
        raise FormalMaterializationError(
            "duplicate_consolidation_request",
            "Consolidation request identities are not unique",
        )
    expected_wt_rows = len(definitions)
    expected_probe_rows = sum(
        len(definition.request.candidate_collection.probes)
        for definition in definitions
    )
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    wt_path = output_root / "fixed_probe_wt_scores.parquet"
    probe_path = output_root / "fixed_probe_scores.parquet"
    manifest_path = output_root / "fixed_probe_scoring_manifest.json"
    existing = _validated_existing_consolidation(
        manifest_path=manifest_path,
        wt_path=wt_path,
        probe_path=probe_path,
        expected_wt_rows=expected_wt_rows,
        expected_probe_rows=expected_probe_rows,
        provenance=provenance,
    )
    if existing is not None:
        return existing

    wt_temporary: Path | None = None
    probe_temporary: Path | None = None
    wt_writer: Any | None = None
    probe_writer: Any | None = None
    try:
        with NamedTemporaryFile(
            dir=output_root, prefix=".fixed_probe_wt_scores.", suffix=".tmp", delete=False
        ) as handle:
            wt_temporary = Path(handle.name)
        with NamedTemporaryFile(
            dir=output_root, prefix=".fixed_probe_scores.", suffix=".tmp", delete=False
        ) as handle:
            probe_temporary = Path(handle.name)
        wt_writer = pq.ParquetWriter(wt_temporary, wt_schema, compression="zstd")
        probe_writer = pq.ParquetWriter(
            probe_temporary, probe_schema, compression="zstd"
        )
        for definition in definitions:
            artifact = definition.artifact
            if artifact is None:
                raise FormalMaterializationError(
                    "missing_artifact_binding",
                    "Consolidation definition lacks an artifact binding",
                )
            payload = load_validated_formal_artifact(
                root=artifact_root,
                artifact=artifact,
                request=definition.request,
                scorer_binding=definition.scorer_binding,
                scientific_fingerprint=definition.scientific_fingerprint,
            )
            request = definition.request
            binding = payload["binding"]
            wt = payload["wt_score"]
            wt_writer.write_table(
                pa.Table.from_pydict(
                    {
                        "protein_id": [request.protein_id],
                        "backbone_condition": [request.condition.condition_id],
                        "backbone_sha256": [request.condition.structure_sha256],
                        "repeat_index": [request.repeat_index],
                        "seed": [request.seed],
                        "decoding_realization_sha256": [request.realization_id],
                        "score_sum_logp_mask": [float(wt["score_sum_logp_mask"])],
                        "score_mean_logp_mask": [float(wt["score_mean_logp_mask"])],
                        "scored_residue_count": [int(wt["scored_residue_count"])],
                        "model_checkpoint_sha256": [binding["checkpoint_sha256"]],
                        "scoring_protocol": [binding["scoring_protocol"]],
                    },
                    schema=wt_schema,
                )
            )
            probes = request.candidate_collection.probes
            scores = payload["candidate_scores"]
            count = len(probes)
            wt_mean = float(wt["score_mean_logp_mask"])
            probe_writer.write_table(
                pa.Table.from_pydict(
                    {
                        "protein_id": [request.protein_id] * count,
                        "sequence_hash": [row.sequence_hash for row in probes],
                        "position": [row.position for row in probes],
                        "wt_aa": [row.wt_aa for row in probes],
                        "mut_aa": [row.mut_aa for row in probes],
                        "backbone_condition": [request.condition.condition_id] * count,
                        "backbone_sha256": [request.condition.structure_sha256] * count,
                        "repeat_index": [request.repeat_index] * count,
                        "seed": [request.seed] * count,
                        "decoding_realization_sha256": [request.realization_id] * count,
                        "score_sum_logp_mask": [
                            float(row["score_sum_logp_mask"]) for row in scores
                        ],
                        "score_mean_logp_mask": [
                            float(row["score_mean_logp_mask"]) for row in scores
                        ],
                        "delta_score_vs_wt": [
                            float(row["score_mean_logp_mask"]) - wt_mean
                            for row in scores
                        ],
                        "scored_residue_count": [
                            int(row["scored_residue_count"]) for row in scores
                        ],
                        "model_checkpoint_sha256": [binding["checkpoint_sha256"]]
                        * count,
                        "scoring_protocol": [binding["scoring_protocol"]] * count,
                    },
                    schema=probe_schema,
                )
            )
        wt_writer.close()
        wt_writer = None
        probe_writer.close()
        probe_writer = None
        for path, rows, schema in (
            (wt_temporary, expected_wt_rows, wt_schema),
            (probe_temporary, expected_probe_rows, probe_schema),
        ):
            if pq.ParquetFile(path).metadata.num_rows != rows or not pq.read_schema(
                path
            ).equals(schema):
                raise FormalMaterializationError(
                    "invalid_consolidated_output",
                    "Temporary consolidated output failed schema/cardinality validation",
                )
        wt_status = _promote_immutable_parquet(wt_temporary, wt_path)
        probe_status = _promote_immutable_parquet(probe_temporary, probe_path)
        manifest = {
            "schema_version": _CONSOLIDATION_SCHEMA,
            "status": "COMPLETE",
            "counts": {
                "canonical_shards": expected_wt_rows,
                "probe_rows": expected_probe_rows,
                "wt_rows": expected_wt_rows,
            },
            "outputs": {
                "wt": {
                    "path": wt_path.name,
                    "rows": expected_wt_rows,
                    "sha256": sha256_file(wt_path),
                },
                "probe": {
                    "path": probe_path.name,
                    "rows": expected_probe_rows,
                    "sha256": sha256_file(probe_path),
                },
            },
            "provenance": dict(provenance),
        }
        rendered = (
            json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        atomic_write_new_bytes(manifest_path, rendered)
        return {
            "wt_path": wt_path,
            "probe_path": probe_path,
            "manifest_path": manifest_path,
            "write_status": {
                "manifest": "created",
                "probe": probe_status,
                "wt": wt_status,
            },
            "manifest": manifest,
        }
    finally:
        if wt_writer is not None:
            wt_writer.close()
        if probe_writer is not None:
            probe_writer.close()
        if wt_temporary is not None:
            wt_temporary.unlink(missing_ok=True)
        if probe_temporary is not None:
            probe_temporary.unlink(missing_ok=True)
