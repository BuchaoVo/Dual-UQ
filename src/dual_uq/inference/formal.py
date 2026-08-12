"""Derived formal request inventory and scorer-dispatch composition."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pandas as pd

from dual_uq.core.atomic_io import atomic_write_new_bytes
from dual_uq.core.hashing import sha256_bytes, sha256_canonical
from dual_uq.inference.materialization import (
    ArtifactInspectionState,
    FormalArtifactBinding,
    bind_formal_artifact,
    build_formal_shard_payload,
    inspect_formal_artifact,
    load_validated_formal_artifact,
    materialize_formal_artifact,
)
from dual_uq.models.scoring import (
    ScorerBinding,
    ScoreRecord,
    ScoreRequest,
    SequenceScorer,
    execute_score_request,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")


class FormalInventoryError(ValueError):
    """Structured formal-inventory invariant failure."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class MaterializationState(str, Enum):
    """Request state derived from canonical artifacts and reuse evidence."""

    VALID_CANONICAL_COMPLETE = "VALID_CANONICAL_COMPLETE"
    HISTORICAL_REUSE_ACCEPTED = "HISTORICAL_REUSE_ACCEPTED"
    FRESH_EXECUTION_REQUIRED = "FRESH_EXECUTION_REQUIRED"
    INVALID_EXISTING = "INVALID_EXISTING"
    CONFLICT = "CONFLICT"


class HistoricalReuseStatus(str, Enum):
    """Per-request result after the protocol authorization gate."""

    NOT_APPLICABLE = "NOT_APPLICABLE"
    CANDIDATE = "HISTORICAL_REUSE_CANDIDATE"
    ACCEPTED = "HISTORICAL_REUSE_ACCEPTED"
    REJECTED = "HISTORICAL_REUSE_REJECTED"


def _sha256(value: object, field_name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def derive_orchestration_id(scientific_fingerprint: str) -> str:
    """Derive an operational key while retaining role separation in the API."""
    return _sha256(scientific_fingerprint, "scientific_fingerprint")


@dataclass(frozen=True, slots=True)
class HistoricalReuseDecision:
    """Explicit authorization-compatible evidence for one target request."""

    status: HistoricalReuseStatus = HistoricalReuseStatus.NOT_APPLICABLE
    reuse_contract_identity: str | None = None
    source_execution_identity: str | None = None
    source_artifact_reference: str | None = None
    source_artifact_sha256: str | None = None
    compatibility_validation_identity: str | None = None
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        status = HistoricalReuseStatus(self.status)
        object.__setattr__(self, "status", status)
        if status is HistoricalReuseStatus.NOT_APPLICABLE:
            return
        required = (
            self.reuse_contract_identity,
            self.source_execution_identity,
            self.source_artifact_reference,
            self.source_artifact_sha256,
            self.compatibility_validation_identity,
        )
        if any(type(value) is not str or not value for value in required):
            raise ValueError("historical reuse evidence is incomplete")
        _sha256(self.reuse_contract_identity, "reuse_contract_identity")
        _sha256(self.source_artifact_sha256, "source_artifact_sha256")
        _sha256(
            self.compatibility_validation_identity,
            "compatibility_validation_identity",
        )
        if status is HistoricalReuseStatus.REJECTED and not self.rejection_reason:
            raise ValueError("rejected historical reuse requires a reason")
        if status is HistoricalReuseStatus.ACCEPTED and self.rejection_reason is not None:
            raise ValueError("accepted historical reuse cannot have a rejection reason")


@dataclass(frozen=True, slots=True)
class FormalRequestDefinition:
    """Scientific request plus external binding used by the control plane."""

    scientific_fingerprint: str
    orchestration_id: str
    workflow_request_id: str
    request: ScoreRequest
    scorer_binding: ScorerBinding
    workflow_metadata: Mapping[str, Any]
    historical_reuse: HistoricalReuseDecision = field(
        default_factory=HistoricalReuseDecision
    )
    artifact: FormalArtifactBinding | None = None

    def __post_init__(self) -> None:
        fingerprint = _sha256(
            self.scientific_fingerprint, "scientific_fingerprint"
        )
        orchestration = _sha256(self.orchestration_id, "orchestration_id")
        if orchestration != derive_orchestration_id(fingerprint):
            raise ValueError("orchestration ID must derive from scientific fingerprint")
        if type(self.workflow_request_id) is not str or not self.workflow_request_id:
            raise ValueError("workflow_request_id is required")
        if not isinstance(self.request, ScoreRequest) or not isinstance(
            self.scorer_binding, ScorerBinding
        ):
            raise TypeError("formal definition requires ScoreRequest and ScorerBinding")
        object.__setattr__(
            self, "workflow_metadata", MappingProxyType(dict(self.workflow_metadata))
        )
        if not isinstance(self.historical_reuse, HistoricalReuseDecision):
            raise TypeError("historical_reuse must be a HistoricalReuseDecision")
        if self.artifact is None:
            object.__setattr__(self, "artifact", bind_formal_artifact(orchestration))


@dataclass(frozen=True, slots=True)
class FormalInventoryEntry:
    """One derived immutable request-state observation."""

    orchestration_id: str
    scientific_fingerprint: str
    workflow_request_id: str
    artifact_reference: str
    state: MaterializationState
    expected_wt_rows: int
    expected_probe_rows: int
    historical_reuse_status: HistoricalReuseStatus
    validation_artifact_sha256: str | None = None
    validation_error_code: str | None = None
    validation_detail: str | None = None
    reuse_source_artifact_reference: str | None = None
    reuse_source_artifact_sha256: str | None = None
    reuse_compatibility_validation_identity: str | None = None


@dataclass(frozen=True, slots=True)
class FormalInventorySnapshot:
    """Derived point-in-time view; never an independently mutable state store."""

    entries: tuple[FormalInventoryEntry, ...]

    @property
    def total_requests(self) -> int:
        return len(self.entries)

    @property
    def expected_wt_rows(self) -> int:
        return sum(entry.expected_wt_rows for entry in self.entries)

    @property
    def expected_probe_rows(self) -> int:
        return sum(entry.expected_probe_rows for entry in self.entries)

    @property
    def state_counts(self) -> dict[MaterializationState, int]:
        return dict(Counter(entry.state for entry in self.entries))


def _inventory_rows(snapshot: FormalInventorySnapshot) -> list[dict[str, Any]]:
    return [
        {
            "orchestration_id": entry.orchestration_id,
            "scientific_fingerprint": entry.scientific_fingerprint,
            "workflow_request_id": entry.workflow_request_id,
            "artifact_reference": entry.artifact_reference,
            "materialization_state": entry.state.value,
            "expected_wt_rows": entry.expected_wt_rows,
            "expected_probe_rows": entry.expected_probe_rows,
            "historical_reuse_status": entry.historical_reuse_status.value,
            "validation_artifact_sha256": entry.validation_artifact_sha256,
            "validation_error_code": entry.validation_error_code,
            "validation_detail": entry.validation_detail,
            "reuse_source_artifact_reference": (
                entry.reuse_source_artifact_reference
            ),
            "reuse_source_artifact_sha256": entry.reuse_source_artifact_sha256,
            "reuse_compatibility_validation_identity": (
                entry.reuse_compatibility_validation_identity
            ),
        }
        for entry in snapshot.entries
    ]


def _write_inventory_asset(path: Path, payload: bytes) -> str:
    if path.is_file():
        if path.read_bytes() == payload:
            return "reused_identical"
        raise FormalInventoryError(
            "immutable_inventory_conflict",
            f"Immutable inventory asset differs: {path.name}",
        )
    try:
        atomic_write_new_bytes(path, payload)
    except FileExistsError as exc:
        raise FormalInventoryError(
            "immutable_inventory_conflict",
            f"Inventory asset appeared concurrently: {path.name}",
        ) from exc
    return "created"


def materialize_inventory_snapshot(
    snapshot: FormalInventorySnapshot,
    *,
    output_root: Path,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize one immutable content-addressed derived inventory snapshot."""
    rows = _inventory_rows(snapshot)
    provenance_record = dict(provenance)
    try:
        snapshot_id = sha256_canonical(
            {"inventory_rows": rows, "provenance": provenance_record}
        )
    except (TypeError, ValueError) as exc:
        raise FormalInventoryError(
            "invalid_inventory_provenance",
            "Inventory provenance is not canonical JSON",
        ) from exc
    frame = pd.DataFrame(rows)
    buffer = BytesIO()
    frame.to_parquet(buffer, index=False)
    inventory_bytes = buffer.getvalue()
    directory = Path(output_root).resolve() / snapshot_id
    inventory_path = directory / "execution_inventory.parquet"
    manifest_path = directory / "execution_inventory_manifest.json"
    counts = {
        "total_requests": snapshot.total_requests,
        "expected_wt_rows": snapshot.expected_wt_rows,
        "expected_probe_rows": snapshot.expected_probe_rows,
        "states": {
            state.value: count
            for state, count in sorted(
                snapshot.state_counts.items(), key=lambda item: item[0].value
            )
        },
        "inventory_gaps": 0,
        "binding_collisions": 0,
        "unresolved_conflicts": snapshot.state_counts.get(
            MaterializationState.CONFLICT, 0
        ),
    }
    manifest = {
        "schema_version": "formal_execution_inventory_snapshot_v1",
        "status": "FORMAL_INVENTORY_SNAPSHOT_COMPLETE",
        "snapshot_id": snapshot_id,
        "inventory": {
            "path": "execution_inventory.parquet",
            "sha256": sha256_bytes(inventory_bytes),
            "row_count": len(frame),
        },
        "counts": counts,
        "provenance": provenance_record,
    }
    manifest_bytes = (
        json.dumps(
            manifest,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    inventory_status = _write_inventory_asset(inventory_path, inventory_bytes)
    manifest_status = _write_inventory_asset(manifest_path, manifest_bytes)
    return {
        "status": "FORMAL_INVENTORY_SNAPSHOT_COMPLETE",
        "snapshot_id": snapshot_id,
        "inventory_path": inventory_path,
        "manifest_path": manifest_path,
        "write_status": {
            "inventory": inventory_status,
            "manifest": manifest_status,
        },
        "counts": counts,
    }


def derive_and_materialize_inventory(
    definitions: tuple[FormalRequestDefinition, ...],
    *,
    artifact_root: Path,
    output_root: Path,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Compose authoritative validation with one immutable inventory snapshot."""
    snapshot = derive_execution_inventory(
        definitions, artifact_root=artifact_root
    )
    return materialize_inventory_snapshot(
        snapshot, output_root=output_root, provenance=provenance
    )


def _validate_definition_set(
    definitions: tuple[FormalRequestDefinition, ...],
) -> None:
    identities = [definition.orchestration_id for definition in definitions]
    if len(identities) != len(set(identities)):
        raise FormalInventoryError(
            "duplicate_orchestration_identity", "Found duplicate orchestration identity"
        )
    paths = [str(definition.artifact.relative_path) for definition in definitions]
    if len(paths) != len(set(paths)):
        raise FormalInventoryError(
            "artifact_binding_collision", "Found artifact-binding collision"
        )


def derive_execution_inventory(
    definitions: tuple[FormalRequestDefinition, ...], *, artifact_root: Path
) -> FormalInventorySnapshot:
    """Recompute request states from authoritative artifacts and reuse evidence."""
    definitions = tuple(definitions)
    _validate_definition_set(definitions)
    rows = []
    for definition in definitions:
        artifact = definition.artifact
        if artifact is None:
            raise FormalInventoryError(
                "missing_artifact_binding", "Formal request lacks artifact binding"
            )
        observed = inspect_formal_artifact(
            root=artifact_root,
            artifact=artifact,
            request=definition.request,
            scorer_binding=definition.scorer_binding,
            scientific_fingerprint=definition.scientific_fingerprint,
        )
        if observed.state is ArtifactInspectionState.VALID:
            state = MaterializationState.VALID_CANONICAL_COMPLETE
        elif observed.state is ArtifactInspectionState.INVALID:
            state = MaterializationState.INVALID_EXISTING
        elif observed.state is ArtifactInspectionState.CONFLICT:
            state = MaterializationState.CONFLICT
        elif definition.historical_reuse.status is HistoricalReuseStatus.ACCEPTED:
            state = MaterializationState.HISTORICAL_REUSE_ACCEPTED
        else:
            state = MaterializationState.FRESH_EXECUTION_REQUIRED
        reuse = definition.historical_reuse
        rows.append(
            FormalInventoryEntry(
                orchestration_id=definition.orchestration_id,
                scientific_fingerprint=definition.scientific_fingerprint,
                workflow_request_id=definition.workflow_request_id,
                artifact_reference=artifact.relative_path.as_posix(),
                state=state,
                expected_wt_rows=1,
                expected_probe_rows=len(
                    definition.request.candidate_collection.probes
                ),
                historical_reuse_status=reuse.status,
                validation_artifact_sha256=observed.artifact_sha256,
                validation_error_code=observed.error_code,
                validation_detail=observed.detail,
                reuse_source_artifact_reference=reuse.source_artifact_reference,
                reuse_source_artifact_sha256=reuse.source_artifact_sha256,
                reuse_compatibility_validation_identity=(
                    reuse.compatibility_validation_identity
                ),
            )
        )
    return FormalInventorySnapshot(tuple(rows))


def select_fresh_work(
    snapshot: FormalInventorySnapshot,
    definitions: tuple[FormalRequestDefinition, ...],
) -> tuple[FormalRequestDefinition, ...]:
    """Select only missing non-reused work; invalid/conflicting work stays blocked."""
    fresh = {
        entry.orchestration_id
        for entry in snapshot.entries
        if entry.state is MaterializationState.FRESH_EXECUTION_REQUIRED
    }
    by_id = {definition.orchestration_id: definition for definition in definitions}
    if set(by_id) != {entry.orchestration_id for entry in snapshot.entries}:
        raise FormalInventoryError(
            "inventory_definition_mismatch", "Snapshot and definitions differ"
        )
    return tuple(by_id[entry.orchestration_id] for entry in snapshot.entries if entry.orchestration_id in fresh)


def select_deterministic_preflight_pair(
    snapshot: FormalInventorySnapshot,
    definitions: tuple[FormalRequestDefinition, ...],
) -> tuple[FormalRequestDefinition, ...]:
    """Select the smallest fully pending PDB/AFDB realization pair."""
    pending = {
        entry.orchestration_id
        for entry in snapshot.entries
        if entry.state is MaterializationState.FRESH_EXECUTION_REQUIRED
    }
    by_id = {definition.orchestration_id: definition for definition in definitions}
    if set(by_id) != {entry.orchestration_id for entry in snapshot.entries}:
        raise FormalInventoryError(
            "inventory_definition_mismatch", "Snapshot and definitions differ"
        )
    groups: dict[
        tuple[str, int, int, str], dict[str, FormalRequestDefinition]
    ] = {}
    for orchestration_id in pending:
        definition = by_id[orchestration_id]
        request = definition.request
        key = (
            request.protein_id,
            request.repeat_index,
            request.seed,
            request.realization_id,
        )
        groups.setdefault(key, {})[request.condition.condition_id] = definition
    candidates = [
        (tuple(sorted(row[condition].orchestration_id for condition in ("PDB", "AFDB"))), row)
        for row in groups.values()
        if set(row) == {"PDB", "AFDB"}
    ]
    if not candidates:
        ordered = sorted(
            (by_id[identity] for identity in pending),
            key=lambda row: row.orchestration_id,
        )
        return tuple(ordered[:1])
    _identity, selected = min(candidates, key=lambda row: row[0])
    return selected["PDB"], selected["AFDB"]


def partition_formal_work(
    definitions: tuple[FormalRequestDefinition, ...],
    *,
    worker_count: int,
    worker_index: int,
) -> tuple[FormalRequestDefinition, ...]:
    """Return one static disjoint partition in orchestration-ID order."""
    if (
        isinstance(worker_count, bool)
        or not isinstance(worker_count, int)
        or worker_count <= 0
        or isinstance(worker_index, bool)
        or not isinstance(worker_index, int)
        or not 0 <= worker_index < worker_count
    ):
        raise FormalInventoryError(
            "invalid_worker_partition", "Worker index/count are invalid"
        )
    ordered = tuple(sorted(definitions, key=lambda row: row.orchestration_id))
    return ordered[worker_index::worker_count]


def execute_formal_partition(
    definitions: tuple[FormalRequestDefinition, ...],
    *,
    scorer: SequenceScorer,
    artifact_root: Path,
    execution_provenance: Mapping[str, Any],
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, int]:
    """Execute a static partition, validating canonical state after each request."""
    counts = {"selected": len(definitions), "executed": 0, "reused_valid": 0}
    for definition in definitions:
        before = derive_execution_inventory(
            (definition,), artifact_root=artifact_root
        ).entries[0]
        if before.state is MaterializationState.VALID_CANONICAL_COMPLETE:
            counts["reused_valid"] += 1
            continue
        if before.state is not MaterializationState.FRESH_EXECUTION_REQUIRED:
            raise FormalInventoryError(
                "formal_execution_state_blocked",
                f"Request cannot execute from state {before.state.value}",
            )
        materialize_fresh_request(
            definition,
            scorer=scorer,
            artifact_root=artifact_root,
            execution_provenance=execution_provenance,
        )
        after = derive_execution_inventory(
            (definition,), artifact_root=artifact_root
        ).entries[0]
        if after.state is not MaterializationState.VALID_CANONICAL_COMPLETE:
            raise FormalInventoryError(
                "formal_post_execution_validation_failed",
                "Canonical shard did not validate after execution",
            )
        counts["executed"] += 1
        if progress_callback is not None:
            progress_callback(
                {
                    "orchestration_id": definition.orchestration_id,
                    "workflow_request_id": definition.workflow_request_id,
                    "executed": counts["executed"],
                    "selected": counts["selected"],
                }
            )
    return counts


def materialize_fresh_request(
    definition: FormalRequestDefinition,
    *,
    scorer: SequenceScorer,
    artifact_root: Path,
    execution_provenance: Mapping[str, Any],
) -> str:
    """Compose normalized dispatch with the authoritative artifact materializer."""
    if scorer.binding != definition.scorer_binding:
        raise FormalInventoryError(
            "scorer_binding_mismatch", "Concrete scorer differs from formal definition"
        )
    records = execute_score_request(scorer, definition.request)
    artifact = definition.artifact
    if artifact is None:
        raise FormalInventoryError(
            "missing_artifact_binding", "Formal request lacks artifact binding"
        )
    payload = build_formal_shard_payload(
        request=definition.request,
        scorer_binding=definition.scorer_binding,
        scientific_fingerprint=definition.scientific_fingerprint,
        orchestration_id=definition.orchestration_id,
        records=records,
        execution_provenance=execution_provenance,
    )
    return materialize_formal_artifact(
        root=artifact_root,
        artifact=artifact,
        payload=payload,
        request=definition.request,
        scorer_binding=definition.scorer_binding,
        scientific_fingerprint=definition.scientific_fingerprint,
    )


def materialize_historical_reuse(
    definition: FormalRequestDefinition,
    *,
    records: tuple[ScoreRecord, ...],
    artifact_root: Path,
) -> str:
    """Canonicalize an accepted exact historical result without altering its source."""
    reuse = definition.historical_reuse
    if reuse.status is not HistoricalReuseStatus.ACCEPTED:
        raise FormalInventoryError(
            "historical_reuse_not_accepted",
            "Historical records require an accepted exact compatibility decision",
        )
    artifact = definition.artifact
    if artifact is None:
        raise FormalInventoryError(
            "missing_artifact_binding", "Formal request lacks artifact binding"
        )
    provenance = {
        "materialization_source": "HISTORICAL_REUSE",
        "historical_reuse": {
            "target_scientific_fingerprint": definition.scientific_fingerprint,
            "source_execution_identity": reuse.source_execution_identity,
            "source_artifact_reference": reuse.source_artifact_reference,
            "source_artifact_sha256": reuse.source_artifact_sha256,
            "reuse_contract_identity": reuse.reuse_contract_identity,
            "compatibility_validation_identity": (
                reuse.compatibility_validation_identity
            ),
        },
    }
    payload = build_formal_shard_payload(
        request=definition.request,
        scorer_binding=definition.scorer_binding,
        scientific_fingerprint=definition.scientific_fingerprint,
        orchestration_id=definition.orchestration_id,
        records=records,
        execution_provenance=provenance,
    )
    return materialize_formal_artifact(
        root=artifact_root,
        artifact=artifact,
        payload=payload,
        request=definition.request,
        scorer_binding=definition.scorer_binding,
        scientific_fingerprint=definition.scientific_fingerprint,
    )


def materialize_historical_reuse_set(
    items: Iterable[
        tuple[FormalRequestDefinition, tuple[ScoreRecord, ...]]
    ],
    *,
    artifact_root: Path,
) -> dict[str, int]:
    """Materialize a streamed set of validated historical result bindings."""
    counts = {"accepted": 0, "created": 0, "reused_identical": 0}
    seen: set[str] = set()
    for definition, records in items:
        if definition.orchestration_id in seen:
            raise FormalInventoryError(
                "duplicate_historical_reuse_item",
                "Historical reuse stream repeats an orchestration identity",
            )
        seen.add(definition.orchestration_id)
        status = materialize_historical_reuse(
            definition, records=records, artifact_root=artifact_root
        )
        counts["accepted"] += 1
        counts[status] += 1
    return counts


def discover_valid_materializations(
    definitions: tuple[FormalRequestDefinition, ...], *, artifact_root: Path
) -> dict[str, dict[str, Any]]:
    """Discover only artifacts that pass the same canonical content validator."""
    discovered = {}
    for definition in definitions:
        artifact = definition.artifact
        if artifact is None:
            raise FormalInventoryError(
                "missing_artifact_binding", "Formal request lacks artifact binding"
            )
        inspected = inspect_formal_artifact(
            root=artifact_root,
            artifact=artifact,
            request=definition.request,
            scorer_binding=definition.scorer_binding,
            scientific_fingerprint=definition.scientific_fingerprint,
        )
        if inspected.state is ArtifactInspectionState.VALID:
            discovered[definition.orchestration_id] = load_validated_formal_artifact(
                root=artifact_root,
                artifact=artifact,
                request=definition.request,
                scorer_binding=definition.scorer_binding,
                scientific_fingerprint=definition.scientific_fingerprint,
            )
    return discovered
