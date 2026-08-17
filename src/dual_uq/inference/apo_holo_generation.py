"""Deterministic APO/HOLO sequence-generation execution and shard storage.

This module is deliberately separate from the frozen PDB/AFDB generation
protocol.  It reuses the verified model adapters, but keeps experimental state
labels, seed domains, and artifact identity specific to the apo/holo cohort.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from dual_uq.core.atomic_io import atomic_write_new_bytes
from dual_uq.core.hashing import sha256_bytes, sha256_canonical
from dual_uq.models.proteinmpnn import sequence_sha256, validate_protein_sequence
from dual_uq.models.proteinmpnn_generation import (
    GenerationRequest,
    ProteinMPNNGenerationAdapter,
    independent_seed,
)

GenerationState = Literal["APO", "HOLO"]
GENERATION_TEMPERATURE = 0.1
GENERATION_SAMPLE_COUNT = 64
GENERATION_SCHEMA = "apo_holo_generation_shard_v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ApoHoloGenerationError(ValueError):
    """Raised when a generation input, record, or shard is invalid."""


def generation_seed(
    protein_id: str,
    state: GenerationState,
    sample_index: int,
    *,
    model_name: str = "fixture",
) -> int:
    """Derive a deterministic, state-disjoint seed from scientific identity."""
    if type(sample_index) is not int or not 0 <= sample_index < GENERATION_SAMPLE_COUNT:
        raise ApoHoloGenerationError("sample_index must be in 0..63")
    if state not in ("APO", "HOLO"):
        raise ApoHoloGenerationError("state must be APO or HOLO")
    if model_name == "ProteinMPNN":
        # Reuse the frozen generation adapter's independent seed domains.
        # APO and HOLO intentionally never share a stochastic realization.
        return (1_000_000 if state == "APO" else 2_000_000) + sample_index
    payload = f"apo_holo_generation_v1|{model_name}|{protein_id}|{state}|{sample_index}".encode()
    value = int(hashlib.sha256(payload).hexdigest()[:16], 16) % (2**32 - 1)
    return max(1, value)


def _digest(value: object, field: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ApoHoloGenerationError(f"{field} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class ApoHoloGenerationCondition:
    """One frozen apo or holo condition bound to the common-mask projection."""

    protein_id: str
    pair_id: str
    state: GenerationState
    structure_sha256: str
    canonical_positions: tuple[int, ...]
    wt_sequence_projection: str
    coordinates: np.ndarray | None
    structure_path: Path | None
    chain_id: str | None
    proteinmpnn_structure: Any = None

    def __post_init__(self) -> None:
        if type(self.protein_id) is not str or not self.protein_id:
            raise ApoHoloGenerationError("protein_id is required")
        if type(self.pair_id) is not str or not self.pair_id:
            raise ApoHoloGenerationError("pair_id is required")
        if self.state not in ("APO", "HOLO"):
            raise ApoHoloGenerationError("state must be APO or HOLO")
        _digest(self.structure_sha256, "structure_sha256")
        positions = tuple(self.canonical_positions)
        if (
            not positions
            or any(type(value) is not int or value <= 0 for value in positions)
            or any(right <= left for left, right in pairwise(positions))
        ):
            raise ApoHoloGenerationError("canonical_positions must be strictly increasing")
        sequence = validate_protein_sequence(self.wt_sequence_projection)
        if len(sequence) != len(positions):
            raise ApoHoloGenerationError("projection sequence length differs from positions")
        object.__setattr__(self, "canonical_positions", positions)
        object.__setattr__(self, "wt_sequence_projection", sequence)
        if self.coordinates is not None:
            coordinates = np.asarray(self.coordinates, dtype=np.float32)
            if coordinates.ndim != 3 or coordinates.shape[0] != len(positions):
                raise ApoHoloGenerationError("coordinates must have one row per projection position")
            if coordinates.shape[1:] not in ((3, 3), (4, 3)):
                raise ApoHoloGenerationError("coordinates must have shape (L,3,3) or (L,4,3)")
            if np.isinf(coordinates).any():
                raise ApoHoloGenerationError("coordinates contain infinity")
            coordinates = coordinates.copy()
            coordinates.setflags(write=False)
            object.__setattr__(self, "coordinates", coordinates)

    @property
    def identity(self) -> str:
        coordinates_sha = None
        if self.coordinates is not None:
            coordinates_sha = sha256_bytes(
                np.asarray(self.coordinates, dtype="<f4").tobytes()
            )
        return sha256_canonical(
            {
                "protein_id": self.protein_id,
                "pair_id": self.pair_id,
                "state": self.state,
                "structure_sha256": self.structure_sha256,
                "canonical_positions": list(self.canonical_positions),
                "wt_sequence_projection": self.wt_sequence_projection,
                "coordinates_sha256": coordinates_sha,
            }
        )


@dataclass(frozen=True, slots=True)
class ApoHoloGenerationRecord:
    """One generated projected sequence with explicit provenance."""

    condition: ApoHoloGenerationCondition
    sample_index: int
    seed: int
    temperature: float
    sequence: str
    sequence_hash: str
    model_name: str
    implementation_id: str
    checkpoint_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.condition, ApoHoloGenerationCondition):
            raise ApoHoloGenerationError("condition is invalid")
        if type(self.sample_index) is not int or not 0 <= self.sample_index < GENERATION_SAMPLE_COUNT:
            raise ApoHoloGenerationError("sample_index must be in 0..63")
        expected_seed = generation_seed(
            self.condition.protein_id,
            self.condition.state,
            self.sample_index,
            model_name=self.model_name,
        )
        if type(self.seed) is not int or self.seed != expected_seed:
            raise ApoHoloGenerationError("seed does not match deterministic generation plan")
        if float(self.temperature) != GENERATION_TEMPERATURE:
            raise ApoHoloGenerationError("generation temperature must be exactly 0.1")
        sequence = validate_protein_sequence(self.sequence)
        if len(sequence) != len(self.condition.canonical_positions):
            raise ApoHoloGenerationError("generated sequence length differs from projection")
        if sequence_sha256(sequence) != _digest(self.sequence_hash, "sequence_hash"):
            raise ApoHoloGenerationError("sequence_hash does not match sequence")
        if type(self.model_name) is not str or not self.model_name:
            raise ApoHoloGenerationError("model_name is required")
        _digest(self.checkpoint_id, "checkpoint_id")
        if type(self.implementation_id) is not str or not self.implementation_id:
            raise ApoHoloGenerationError("implementation_id is required")
        object.__setattr__(self, "sequence", sequence)


class ApoHoloGenerationAdapter(Protocol):
    """Minimal model-specific sampling boundary."""

    model_name: str
    implementation_id: str
    checkpoint_id: str

    def generate(
        self,
        condition: ApoHoloGenerationCondition,
        *,
        n_samples: int,
    ) -> tuple[str, ...]: ...


@dataclass(frozen=True, slots=True)
class ProteinMPNNApoHoloGenerationAdapter:
    """Adapt the verified ProteinMPNN generator to the APO/HOLO contract.

    The existing generator's public multi-sample path starts with paired
    seeds.  This adapter deliberately invokes its validated single-sample
    path with the frozen independent seed domain, so APO and HOLO ensembles
    remain independent while all model semantics stay in the canonical
    ProteinMPNN adapter.
    """

    adapter: ProteinMPNNGenerationAdapter

    @property
    def model_name(self) -> str:
        return "ProteinMPNN"

    @property
    def implementation_id(self) -> str:
        return self.adapter.adapter.implementation_id

    @property
    def checkpoint_id(self) -> str:
        return self.adapter.adapter.checkpoint_id

    def generate(
        self,
        condition: ApoHoloGenerationCondition,
        *,
        n_samples: int,
    ) -> tuple[str, ...]:
        if n_samples != GENERATION_SAMPLE_COUNT:
            raise ApoHoloGenerationError("the frozen first-pass sample count is exactly 64")
        structure = condition.proteinmpnn_structure
        if structure is None:
            raise ApoHoloGenerationError("ProteinMPNN condition lacks a full-chain structure")
        backbone_condition = "PDB" if condition.state == "APO" else "AFDB"
        requests = tuple(
            GenerationRequest(
                protein_id=condition.protein_id,
                backbone_condition=backbone_condition,
                structure_sha256=condition.structure_sha256,
                canonical_positions=condition.canonical_positions,
                wt_sequence_projection=condition.wt_sequence_projection,
                temperature=GENERATION_TEMPERATURE,
                sample_index=GENERATION_SAMPLE_COUNT * 2 + index,
                seed=independent_seed(backbone_condition, index),
                sample_class="independent",
                decoding_realization="0" * 64,
            )
            for index in range(GENERATION_SAMPLE_COUNT)
        )
        sequences: list[str] = []
        for start in range(0, GENERATION_SAMPLE_COUNT, self.adapter.batch_size):
            generated = self.adapter.generate_requests_batched(
                requests[start : start + self.adapter.batch_size], structure
            )
            sequences.extend(record.sequence for record in generated)
        if len(sequences) != GENERATION_SAMPLE_COUNT:
            raise ApoHoloGenerationError("ProteinMPNN returned the wrong sample count")
        return tuple(sequences)


@dataclass(frozen=True, slots=True)
class ESMIF1ApoHoloGenerationAdapter:
    """Adapt the official ESM-IF1 sampler to the same immutable record API."""

    adapter: Any
    batch_size: int = 8

    def __post_init__(self) -> None:
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool) or self.batch_size <= 0:
            raise ApoHoloGenerationError("batch_size must be a positive integer")
        binding = self.adapter.binding()
        if not isinstance(binding, Mapping):
            raise ApoHoloGenerationError("ESM-IF1 adapter lacks a binding")
        for field in ("implementation_revision", "checkpoint_sha256"):
            if not isinstance(binding.get(field), str) or not binding[field]:
                raise ApoHoloGenerationError(f"ESM-IF1 binding lacks {field}")

    @property
    def model_name(self) -> str:
        binding = self.adapter.binding()
        return str(binding.get("model_name") or "ESM-IF1")

    @property
    def implementation_id(self) -> str:
        return str(self.adapter.binding()["implementation_revision"])

    @property
    def checkpoint_id(self) -> str:
        return str(self.adapter.binding()["checkpoint_sha256"])

    def generate(
        self,
        condition: ApoHoloGenerationCondition,
        *,
        n_samples: int,
    ) -> tuple[str, ...]:
        if n_samples != GENERATION_SAMPLE_COUNT:
            raise ApoHoloGenerationError("the frozen first-pass sample count is exactly 64")
        if condition.coordinates is None:
            raise ApoHoloGenerationError("ESM-IF1 condition lacks coordinates")
        coordinates = np.asarray(condition.coordinates, dtype=np.float32)
        if coordinates.shape[1:] != (3, 3):
            raise ApoHoloGenerationError("ESM-IF1 coordinates must have shape (L,3,3)")
        seeds = tuple(
            generation_seed(condition.protein_id, condition.state, index, model_name=self.model_name)
            for index in range(GENERATION_SAMPLE_COUNT)
        )
        allow_missing = bool(np.isnan(coordinates).any())
        values: list[str] = []
        for start in range(0, GENERATION_SAMPLE_COUNT, self.batch_size):
            batch_seeds = seeds[start : start + self.batch_size]
            batch_coordinates = np.repeat(coordinates[None, ...], len(batch_seeds), axis=0)
            values.extend(
                self.adapter.sample_batch(
                    batch_coordinates,
                    tuple(GENERATION_TEMPERATURE for _ in batch_seeds),
                    batch_seeds,
                    allow_missing_coordinates=allow_missing,
                )
            )
        if len(values) != GENERATION_SAMPLE_COUNT:
            raise ApoHoloGenerationError("ESM-IF1 returned the wrong sample count")
        return tuple(values)


def validate_generation_records(
    records: Iterable[ApoHoloGenerationRecord],
    *,
    expected_protein_count: int,
) -> tuple[ApoHoloGenerationRecord, ...]:
    """Validate the complete two-state × 64-sample grid."""
    values = tuple(records)
    if expected_protein_count <= 0:
        raise ApoHoloGenerationError("expected_protein_count must be positive")
    if not values:
        raise ApoHoloGenerationError("generation records are required")
    keys = [(r.condition.protein_id, r.condition.state, r.sample_index) for r in values]
    if len(set(keys)) != len(keys):
        raise ApoHoloGenerationError("generation sample keys are duplicated")
    proteins = {r.condition.protein_id for r in values}
    if len(proteins) != expected_protein_count:
        raise ApoHoloGenerationError("generation protein count differs")
    expected = {
        (protein, state, index)
        for protein in proteins
        for state in ("APO", "HOLO")
        for index in range(GENERATION_SAMPLE_COUNT)
    }
    if set(keys) != expected:
        raise ApoHoloGenerationError("generation state/sample grid is incomplete")
    for protein in sorted(proteins):
        group = [r for r in values if r.condition.protein_id == protein]
        identities = {(r.condition.pair_id, r.condition.canonical_positions, r.condition.wt_sequence_projection) for r in group}
        if len(identities) != 1:
            raise ApoHoloGenerationError(f"condition projection differs within protein: {protein}")
    return tuple(sorted(values, key=lambda r: (r.condition.protein_id, r.condition.state, r.sample_index)))


def _record_payload(record: ApoHoloGenerationRecord) -> dict[str, Any]:
    condition = record.condition
    return {
        "condition": _condition_binding(condition),
        "sample_index": record.sample_index,
        "seed": record.seed,
        "temperature": record.temperature,
        "sequence": record.sequence,
        "sequence_hash": record.sequence_hash,
        "model_name": record.model_name,
        "implementation_id": record.implementation_id,
        "checkpoint_id": record.checkpoint_id,
    }


def _condition_binding(condition: ApoHoloGenerationCondition) -> dict[str, Any]:
    """Return the serialized scientific binding independent of runtime payloads."""
    return {
        "protein_id": condition.protein_id,
        "pair_id": condition.pair_id,
        "state": condition.state,
        "structure_sha256": condition.structure_sha256,
        "canonical_positions": list(condition.canonical_positions),
        "wt_sequence_projection": condition.wt_sequence_projection,
    }


def _record_from_payload(value: Mapping[str, Any]) -> ApoHoloGenerationRecord:
    condition_value = value.get("condition")
    if not isinstance(condition_value, Mapping):
        raise ApoHoloGenerationError("malformed generation condition")
    condition = ApoHoloGenerationCondition(
        protein_id=str(condition_value["protein_id"]),
        pair_id=str(condition_value["pair_id"]),
        state=str(condition_value["state"]),  # type: ignore[arg-type]
        structure_sha256=str(condition_value["structure_sha256"]),
        canonical_positions=tuple(int(x) for x in condition_value["canonical_positions"]),
        wt_sequence_projection=str(condition_value["wt_sequence_projection"]),
        coordinates=None,
        structure_path=None,
        chain_id=None,
    )
    try:
        return ApoHoloGenerationRecord(
            condition=condition,
            sample_index=int(value["sample_index"]),
            seed=int(value["seed"]),
            temperature=float(value["temperature"]),
            sequence=str(value["sequence"]),
            sequence_hash=str(value["sequence_hash"]),
            model_name=str(value["model_name"]),
            implementation_id=str(value["implementation_id"]),
            checkpoint_id=str(value["checkpoint_id"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ApoHoloGenerationError("malformed generation record") from exc


def _shard_path(root: Path, condition: ApoHoloGenerationCondition) -> Path:
    return root / "shards" / condition.identity[:2] / f"{condition.identity}.json"


def generate_apo_holo_conditions(
    conditions: Iterable[ApoHoloGenerationCondition],
    adapter: ApoHoloGenerationAdapter,
    *,
    output_root: Path,
    resume: bool,
    n_samples: int = GENERATION_SAMPLE_COUNT,
) -> tuple[tuple[ApoHoloGenerationRecord, ...], int, int]:
    """Generate one immutable shard per state and return records and counts."""
    if type(n_samples) is not int or n_samples != GENERATION_SAMPLE_COUNT:
        raise ApoHoloGenerationError("the frozen first-pass sample count is exactly 64")
    ordered_conditions = tuple(sorted(conditions, key=lambda c: (c.protein_id, c.state)))
    if not ordered_conditions:
        raise ApoHoloGenerationError("generation conditions are required")
    if len({(c.protein_id, c.state) for c in ordered_conditions}) != len(ordered_conditions):
        raise ApoHoloGenerationError("generation condition keys are duplicated")
    root = Path(output_root).expanduser().resolve()
    records: list[ApoHoloGenerationRecord] = []
    executed = reused = 0
    for condition in ordered_conditions:
        path = _shard_path(root, condition)
        if path.exists():
            if not resume:
                raise ApoHoloGenerationError("generation shard already exists; use --resume")
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("schema_version") != GENERATION_SCHEMA:
                    raise ApoHoloGenerationError("malformed generation shard schema")
                shard_condition = payload.get("condition")
                shard_records = tuple(_record_from_payload(item) for item in payload["records"])
                if len(shard_records) != n_samples or shard_condition != _condition_binding(condition):
                    raise ApoHoloGenerationError("generation shard binding differs")
                if any(
                    _condition_binding(item.condition) != _condition_binding(condition)
                    for item in shard_records
                ):
                    raise ApoHoloGenerationError("generation shard condition identity differs")
            except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ApoHoloGenerationError("malformed generation shard") from exc
            records.extend(shard_records)
            reused += 1
            continue
        sequences = tuple(adapter.generate(condition, n_samples=n_samples))
        if len(sequences) != n_samples:
            raise ApoHoloGenerationError("generation adapter returned the wrong sample count")
        generated = tuple(
            ApoHoloGenerationRecord(
                condition=condition,
                sample_index=index,
                seed=generation_seed(
                    condition.protein_id,
                    condition.state,
                    index,
                    model_name=adapter.model_name,
                ),
                temperature=GENERATION_TEMPERATURE,
                sequence=sequence,
                sequence_hash=sequence_sha256(sequence),
                model_name=adapter.model_name,
                implementation_id=adapter.implementation_id,
                checkpoint_id=adapter.checkpoint_id,
            )
            for index, sequence in enumerate(sequences)
        )
        payload = {
            "schema_version": GENERATION_SCHEMA,
            "condition": _condition_binding(condition),
            "binding": {
                "model_name": adapter.model_name,
                "implementation_id": adapter.implementation_id,
                "checkpoint_id": adapter.checkpoint_id,
            },
            "records": [_record_payload(record) for record in generated],
        }
        rendered = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        try:
            atomic_write_new_bytes(path, rendered)
        except FileExistsError:
            existing = path.read_bytes()
            if existing != rendered:
                raise ApoHoloGenerationError("immutable generation shard conflict") from None
        records.extend(generated)
        executed += 1
    return tuple(sorted(records, key=lambda r: (r.condition.protein_id, r.condition.state, r.sample_index))), executed, reused


def load_apo_holo_generation_records(output_root: Path) -> tuple[ApoHoloGenerationRecord, ...]:
    """Load and validate all immutable state shards in stable order."""
    root = Path(output_root).expanduser().resolve()
    paths = tuple(sorted((root / "shards").glob("*/*.json")))
    if not paths:
        raise ApoHoloGenerationError("no apo/holo generation shards were found")
    records: list[ApoHoloGenerationRecord] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != GENERATION_SCHEMA:
                raise ApoHoloGenerationError("malformed generation shard schema")
            records.extend(_record_from_payload(item) for item in payload["records"])
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ApoHoloGenerationError(f"malformed generation shard: {path}") from exc
    return validate_generation_records(records, expected_protein_count=len({r.condition.protein_id for r in records}))
