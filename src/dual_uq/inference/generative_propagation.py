"""Validated execution of the frozen clean-cohort ProteinMPNN generation protocol."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from dual_uq.core.atomic_io import atomic_write_new_bytes
from dual_uq.core.hashing import sha256_bytes, sha256_canonical
from dual_uq.dataset.models import BACKBONE_ATOM_NAMES
from dual_uq.models.proteinmpnn import ProteinMPNNStructureInput, sequence_sha256
from dual_uq.models.proteinmpnn_generation import (
    GeneratedSequenceRecord,
    GenerationContractError,
    GenerationRequest,
    ProteinMPNNGenerationStructure,
)
from dual_uq.models.scoring import ScorerBinding
from dual_uq.structure_io import (
    load_atom_site_table,
    load_chain_ca_table,
    residue_name_to_one_letter,
)

_SHARD_SCHEMA = "generative_propagation_sequence_shard_v1"
_SAMPLES_PER_CONDITION = 256
_CLEAN_PROTEIN_COUNT = 68


class GenerationExecutionError(ValueError):
    """A frozen generation input or immutable shard is invalid."""


def _load_generation_chain(
    structure_path: Path,
    chain_id: str,
    required_keys: tuple[tuple[int, str], ...],
) -> tuple[tuple[tuple[int, str], ...], str, np.ndarray, tuple[int, ...]]:
    """Load the same first-model backbone representation used by apo/holo cases.

    The released apo/holo case builder deliberately selects the first coordinate
    model and retains chemically modified amino-acid names that have an explicit
    standard one-letter mapping (for example MLY -> K).  Generation must bind to
    that representation rather than the stricter single-model/P1 parser used by
    unrelated admission stages.
    """
    atoms = load_atom_site_table(structure_path)
    atoms = atoms.loc[atoms["model_number"].eq(int(atoms["model_number"].min()))].copy()
    atoms = atoms.loc[atoms["auth_asym_id"].astype(str).eq(str(chain_id))].copy()
    if atoms.empty:
        raise GenerationExecutionError(f"generation structure chain is absent: {chain_id}")
    atoms = atoms.loc[atoms["atom_name"].astype(str).str.upper().isin(BACKBONE_ATOM_NAMES)].copy()
    atoms["_alt_rank"] = atoms["alt_id"].map(
        lambda value: 0 if str(value).strip().upper() in {"", ".", "?", "A"} else 1
    )
    atoms = atoms.sort_values(
        ["_alt_rank", "occupancy"], ascending=[True, False], kind="mergesort"
    )
    selected: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    residue_names: dict[tuple[int, str], str] = {}
    label_order: dict[tuple[int, str], tuple[int, int]] = {}
    for row in atoms.itertuples(index=False):
        if pd.isna(row.auth_seq_id):
            continue
        key = (int(row.auth_seq_id), str(row.insertion_code or "").strip().upper())
        label_seq = int(row.label_seq_id) if not pd.isna(row.label_seq_id) else 10**9
        label_order.setdefault(key, (label_seq, len(label_order)))
        residue_names.setdefault(key, str(row.residue_name).strip().upper())
        atom_name = str(row.atom_name).strip().upper()
        residue = selected.setdefault(key, {})
        if atom_name in residue:
            continue
        coordinate = np.asarray([row.x, row.y, row.z], dtype=np.float32)
        if np.isfinite(coordinate).all():
            residue[atom_name] = coordinate

    ca = load_chain_ca_table(structure_path, chain_id)
    ca_residue_names: dict[tuple[int, str], str] = {}
    for row in ca.itertuples(index=False):
        key = (int(row.auth_seq_id), str(row.insertion_code or "").strip().upper())
        ca_residue_names[key] = str(row.residue_one_letter).strip().upper()
        label_seq = int(row.label_seq_id) if not pd.isna(row.label_seq_id) else 10**9
        label_order.setdefault(key, (label_seq, len(label_order)))
    keys = sorted(
        set(ca_residue_names).union(
            key for key in required_keys if key in selected
        ),
        key=lambda key: label_order.get(key, (10**9, key[0])),
    )
    if not keys:
        raise GenerationExecutionError("generation structure chain has no residues")
    sequence: list[str] = []
    coordinate_rows: list[np.ndarray] = []
    for key in keys:
        sequence.append(ca_residue_names.get(key) or residue_name_to_one_letter(residue_names[key]))
        residue = selected.get(key, {})
        coordinate_rows.append(
            np.asarray(
                [residue.get(atom, np.full(3, np.nan, dtype=np.float32)) for atom in BACKBONE_ATOM_NAMES],
                dtype=np.float32,
            )
        )
    auth_positions = [key[0] for key in keys]
    segment_ends = tuple(
        [index + 1 for index, (left, right) in enumerate(pairwise(auth_positions)) if right != left + 1]
        + [len(keys)]
    )
    return tuple(keys), "".join(sequence), np.asarray(coordinate_rows, dtype=np.float32), segment_ends


class GenerationAdapter(Protocol):
    binding: ScorerBinding

    def generate(
        self,
        request: GenerationRequest,
        structure: ProteinMPNNGenerationStructure,
        *,
        n_samples: int,
    ) -> tuple[GeneratedSequenceRecord, ...]: ...


@dataclass(frozen=True, slots=True)
class GenerationCondition:
    """One exact PDB or AFDB structure binding for a clean-cohort protein."""

    structure: ProteinMPNNGenerationStructure

    @classmethod
    def from_structure(cls, structure: ProteinMPNNGenerationStructure) -> GenerationCondition:
        if not isinstance(structure, ProteinMPNNGenerationStructure):
            raise GenerationExecutionError("generation structure must be validated")
        return cls(structure)

    @property
    def protein_id(self) -> str:
        return self.structure.projection.protein_id

    @property
    def backbone_condition(self) -> str:
        return self.structure.projection.backbone_condition

    @property
    def identity(self) -> str:
        projection = self.structure.projection
        return sha256_canonical(
            {
                "protein_id": projection.protein_id,
                "backbone_condition": projection.backbone_condition,
                "structure_sha256": projection.structure_sha256,
                "canonical_positions": list(projection.uniprot_positions),
                "wt_sequence_projection": projection.wt_sequence_projection,
                "chain_id": self.structure.chain_id,
                "chain_sequence_sha256": sequence_sha256(self.structure.chain_sequence),
                "chain_coordinates_sha256": sha256_bytes(
                    np.asarray(self.structure.chain_coordinates, dtype="<f4").tobytes()
                ),
                "chain_positions": list(self.structure.chain_positions),
                "chain_segment_ends": list(self.structure.chain_segment_ends),
                "temperature": 0.1,
                "generation_protocol": _SHARD_SCHEMA,
            }
        )

    def initial_request(self) -> GenerationRequest:
        projection = self.structure.projection
        if projection.structure_sha256 is None:
            raise GenerationExecutionError("generation structure lacks SHA-256 identity")
        return GenerationRequest(
            protein_id=projection.protein_id,
            backbone_condition=projection.backbone_condition,
            structure_sha256=projection.structure_sha256,
            canonical_positions=projection.uniprot_positions,
            wt_sequence_projection=projection.wt_sequence_projection,
            temperature=0.1,
            sample_index=0,
            seed=0,
            sample_class="paired",
            decoding_realization="0" * 64,
        )


@dataclass(frozen=True, slots=True)
class GenerationInputs:
    """Clean generation condition grid, with explicit expected cohort cardinality."""

    conditions: tuple[GenerationCondition, ...]
    expected_protein_count: int = _CLEAN_PROTEIN_COUNT
    input_provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.expected_protein_count <= 0:
            raise GenerationExecutionError("expected protein count must be positive")
        conditions = tuple(self.conditions)
        if not conditions:
            raise GenerationExecutionError("generation conditions are required")
        keys = {(row.protein_id, row.backbone_condition) for row in conditions}
        if len(keys) != len(conditions):
            raise GenerationExecutionError("generation condition identity is duplicated")
        proteins = {row.protein_id for row in conditions}
        if len(proteins) != self.expected_protein_count:
            raise GenerationExecutionError("generation cohort cardinality differs")
        expected = {(protein, condition) for protein in proteins for condition in ("PDB", "AFDB")}
        if keys != expected:
            raise GenerationExecutionError("each clean protein requires an exact PDB/AFDB pair")
        order = {"PDB": 0, "AFDB": 1}
        object.__setattr__(
            self,
            "conditions",
            tuple(sorted(conditions, key=lambda row: (row.protein_id, order[row.backbone_condition]))),
        )
        object.__setattr__(self, "input_provenance", dict(self.input_provenance or {}))


@dataclass(frozen=True, slots=True)
class GenerationRunResult:
    records: tuple[GeneratedSequenceRecord, ...]
    executed_conditions: int
    reused_conditions: int


def _full_chain_generation_structure(
    projection: ProteinMPNNStructureInput,
    *,
    structure_path: Path,
    source_id: str,
    chain_id: str | None,
    canonicalize_projected_sequence: bool = False,
    projection_chain_auth_keys: tuple[tuple[int, str], ...] | None = None,
    rebind_projection_coordinates: bool = False,
) -> ProteinMPNNGenerationStructure:
    """Bind a projection to the complete validated chain used by ProteinMPNN.

    Projection coordinates are matched against explicit author residue keys when
    supplied, or by exact bytes as a legacy fallback.  No sequence alignment,
    offset inference, or missing-residue compression is used.  A chain with a
    true numbering gap is represented as an explicit ProteinMPNN chain break;
    no compressed numbering or inferred offset is used.
    """
    try:
        if chain_id is None:
            atoms = load_atom_site_table(structure_path)
            chains = set(atoms["auth_asym_id"].dropna().astype(str))
            if len(chains) != 1:
                raise GenerationExecutionError(
                    "generation structure chain identity is ambiguous"
                )
            chain_id = next(iter(chains))
        chain_keys, sequence, coordinates, segment_ends = _load_generation_chain(
            structure_path,
            chain_id,
            projection_chain_auth_keys or (),
        )
        if not sequence or len(sequence) != len(chain_keys):
            raise GenerationExecutionError("generation structure chain sequence is invalid")
        if projection_chain_auth_keys is not None:
            if len(projection_chain_auth_keys) != projection.residue_count:
                raise GenerationExecutionError(
                    "generation projection auth-key mapping length differs"
                )
            auth_positions = {key: index + 1 for index, key in enumerate(chain_keys)}
            try:
                chain_positions = [
                    auth_positions[(int(seq_id), str(insertion).strip().upper())]
                    for seq_id, insertion in projection_chain_auth_keys
                ]
            except KeyError as exc:
                raise GenerationExecutionError(
                    "generation projection auth-key is absent from full chain"
                ) from exc
        else:
            coordinate_keys: dict[bytes, int] = {}
            for index, row in enumerate(coordinates):
                key = row.astype("<f4", copy=False).tobytes()
                if key in coordinate_keys:
                    raise GenerationExecutionError(
                        "generation projection coordinate binding is ambiguous"
                    )
                coordinate_keys[key] = index + 1
            chain_positions = []
            for row in np.asarray(projection.coordinates, dtype=np.float32):
                key = row.astype("<f4", copy=False).tobytes()
                position = coordinate_keys.get(key)
                if position is None:
                    raise GenerationExecutionError(
                        "generation projection coordinates are absent from full chain"
                    )
                chain_positions.append(position)
        if chain_positions != sorted(chain_positions) or len(set(chain_positions)) != len(chain_positions):
            raise GenerationExecutionError(
                "generation projection does not preserve full-chain order"
            )
        if canonicalize_projected_sequence:
            # Apo/holo generation is defined on the frozen canonical WT
            # sequence. Coordinates remain exactly those observed in the
            # released structure; only the projected sequence labels are
            # canonicalized when an explicit structure-level conflict exists.
            canonicalized = list(sequence)
            for chain_position, residue in zip(
                chain_positions, projection.wt_sequence_projection
            ):
                canonicalized[chain_position - 1] = residue
            sequence = "".join(canonicalized)
        projection_for_structure = projection
        if rebind_projection_coordinates:
            projection_for_structure = ProteinMPNNStructureInput(
                protein_id=projection.protein_id,
                backbone_condition=projection.backbone_condition,
                uniprot_positions=projection.uniprot_positions,
                wt_sequence_projection=projection.wt_sequence_projection,
                coordinates=np.asarray(
                    [coordinates[position - 1] for position in chain_positions],
                    dtype=np.float32,
                ),
                structure_sha256=projection.structure_sha256,
            )
        return ProteinMPNNGenerationStructure(
            projection=projection_for_structure,
            chain_id=chain_id,
            chain_sequence=sequence,
            chain_coordinates=coordinates,
            chain_positions=tuple(chain_positions),
            chain_segment_ends=segment_ends,
        )
    except GenerationExecutionError:
        raise
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise GenerationExecutionError(
            "generation full-chain structure validation failed"
        ) from exc


def load_generation_inputs(project_root: Path) -> GenerationInputs:
    """Resolve exactly the existing clean cohort and its authoritative structures."""
    from dual_uq.core.hashing import sha256_file
    from dual_uq.evaluation.operational_pairs import (
        frozen_operational_paths,
        load_operational_conditions,
    )

    root = Path(project_root).expanduser().resolve()
    frozen = load_operational_conditions(root)
    conditions = tuple(
        GenerationCondition.from_structure(
            _full_chain_generation_structure(
                ProteinMPNNStructureInput(
                    protein_id=item.protein_id,
                    backbone_condition=item.condition,
                    uniprot_positions=item.canonical_positions,
                    wt_sequence_projection=item.wt_sequence_projection,
                    coordinates=item.coordinates,
                    structure_sha256=item.source_sha256,
                ),
                structure_path=item.source_path,
                source_id=item.source_id,
                chain_id=item.source_chain_id,
                projection_chain_auth_keys=item.auth_keys,
            )
        )
        for item in frozen
    )
    paths = frozen_operational_paths(root)
    return GenerationInputs(
        conditions,
        input_provenance={
            "owner": "dual_uq.evaluation.operational_pairs",
            "clean_protein_count": len({item.protein_id for item in frozen}),
            "cohort_sha256": sha256_file(paths["cohort"]),
            "mask_sha256": sha256_file(paths["masks"]),
            "validity_sha256": sha256_file(paths["validity"]),
        },
    )


def _binding(condition: GenerationCondition) -> dict[str, Any]:
    request = condition.initial_request()
    return {
        "condition_identity": condition.identity,
        "protein_id": request.protein_id,
        "backbone_condition": request.backbone_condition,
        "structure_sha256": request.structure_sha256,
        "canonical_positions": list(request.canonical_positions),
        "wt_sequence_projection": request.wt_sequence_projection,
        "chain_id": condition.structure.chain_id,
        "chain_sequence_sha256": sequence_sha256(condition.structure.chain_sequence),
        "chain_coordinates_sha256": sha256_bytes(
            np.asarray(condition.structure.chain_coordinates, dtype="<f4").tobytes()
        ),
        "chain_positions": list(condition.structure.chain_positions),
        "chain_segment_ends": list(condition.structure.chain_segment_ends),
        "temperature": request.temperature,
        "sample_count": _SAMPLES_PER_CONDITION,
    }


def _record_payload(record: GeneratedSequenceRecord) -> dict[str, Any]:
    request = record.request
    return {
        "request": {
            "protein_id": request.protein_id,
            "backbone_condition": request.backbone_condition,
            "structure_sha256": request.structure_sha256,
            "canonical_positions": list(request.canonical_positions),
            "wt_sequence_projection": request.wt_sequence_projection,
            "temperature": request.temperature,
            "sample_index": request.sample_index,
            "seed": request.seed,
            "sample_class": request.sample_class,
            "decoding_realization": request.decoding_realization,
        },
        "sequence": record.sequence,
        "sequence_hash": record.sequence_hash,
        "scorer_binding": {
            "scorer_id": record.scorer_binding.scorer_id,
            "implementation_id": record.scorer_binding.implementation_id,
            "checkpoint_id": record.scorer_binding.checkpoint_id,
            "score_contract_id": record.scorer_binding.score_contract_id,
        },
    }


def _record_from_payload(value: Any) -> GeneratedSequenceRecord:
    if not isinstance(value, dict):
        raise GenerationExecutionError("malformed generation shard record")
    try:
        request = GenerationRequest(**value["request"])
        binding = ScorerBinding(**value["scorer_binding"])
        return GeneratedSequenceRecord(
            request=request,
            sequence=value["sequence"],
            sequence_hash=value["sequence_hash"],
            scorer_binding=binding,
        )
    except (GenerationContractError, KeyError, TypeError, ValueError) as exc:
        raise GenerationExecutionError("malformed generation shard record") from exc


def load_generation_records(generation_root: Path) -> tuple[GeneratedSequenceRecord, ...]:
    """Load records from the immutable generation shards in stable order.

    Shard parsing belongs to the generation artifact owner; downstream
    analysis consumes the returned records and does not reimplement the
    on-disk schema.  Completeness and scientific sample-grid checks remain in
    the canonical analysis validator.
    """
    root = Path(generation_root).expanduser().resolve()
    shard_paths = tuple(sorted((root / "shards").glob("*/*.json")))
    if not shard_paths:
        raise GenerationExecutionError("no generation shards were found")
    records: list[GeneratedSequenceRecord] = []
    seen: set[tuple[str, str, int]] = set()
    for path in shard_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise GenerationExecutionError(f"malformed generation shard: {path}") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != _SHARD_SCHEMA:
            raise GenerationExecutionError(f"malformed generation shard: {path}")
        values = payload.get("records")
        if not isinstance(values, list):
            raise GenerationExecutionError(f"malformed generation shard records: {path}")
        for value in values:
            record = _record_from_payload(value)
            key = (
                record.request.protein_id,
                record.request.backbone_condition,
                record.request.sample_index,
            )
            if key in seen:
                raise GenerationExecutionError("generation shard records are duplicated")
            seen.add(key)
            records.append(record)
    condition_order = {"PDB": 0, "AFDB": 1}
    return tuple(
        sorted(
            records,
            key=lambda item: (
                item.request.protein_id,
                condition_order[item.request.backbone_condition],
                item.request.sample_index,
            ),
        )
    )


def _validate_shard(
    value: Any,
    condition: GenerationCondition,
    expected_binding: ScorerBinding,
) -> tuple[GeneratedSequenceRecord, ...]:
    if not isinstance(value, dict) or value.get("schema_version") != _SHARD_SCHEMA:
        raise GenerationExecutionError("malformed generation shard")
    if value.get("binding") != _binding(condition):
        raise GenerationExecutionError("malformed generation shard binding")
    values = value.get("records")
    if not isinstance(values, list) or len(values) != _SAMPLES_PER_CONDITION:
        raise GenerationExecutionError("malformed generation shard sample count")
    records = tuple(_record_from_payload(item) for item in values)
    request = condition.initial_request()
    if {item.request.sample_index for item in records} != set(range(_SAMPLES_PER_CONDITION)):
        raise GenerationExecutionError("malformed generation shard sample grid")
    scorer_bindings = {item.scorer_binding for item in records}
    if len(scorer_bindings) != 1 or next(iter(scorer_bindings)) != expected_binding:
        raise GenerationExecutionError("malformed generation shard scorer identity")
    for item in records:
        observed = item.request
        if (
            observed.protein_id != request.protein_id
            or observed.backbone_condition != request.backbone_condition
            or observed.structure_sha256 != request.structure_sha256
            or observed.canonical_positions != request.canonical_positions
            or observed.wt_sequence_projection != request.wt_sequence_projection
            or observed.temperature != request.temperature
            or sequence_sha256(item.sequence) != item.sequence_hash
        ):
            raise GenerationExecutionError("malformed generation shard binding")
    return tuple(sorted(records, key=lambda item: item.request.sample_index))


def _shard_path(root: Path, condition: GenerationCondition) -> Path:
    return root / "shards" / condition.identity[:2] / f"{condition.identity}.json"


def _read_shard(
    path: Path, condition: GenerationCondition, expected_binding: ScorerBinding
) -> tuple[GeneratedSequenceRecord, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GenerationExecutionError("malformed generation shard") from exc
    return _validate_shard(payload, condition, expected_binding)


def _render_shard(
    condition: GenerationCondition,
    records: tuple[GeneratedSequenceRecord, ...],
    expected_binding: ScorerBinding,
) -> bytes:
    _validate_shard(
        {
            "schema_version": _SHARD_SCHEMA,
            "binding": _binding(condition),
            "records": [_record_payload(record) for record in records],
        },
        condition,
        expected_binding,
    )
    return (
        json.dumps(
            {
                "schema_version": _SHARD_SCHEMA,
                "binding": _binding(condition),
                "records": [_record_payload(record) for record in records],
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def generate_clean_cohort(
    inputs: GenerationInputs,
    adapter: GenerationAdapter,
    *,
    output_root: Path,
    resume: bool,
) -> GenerationRunResult:
    """Generate one immutable shard per condition and validate every reuse."""
    if not isinstance(inputs, GenerationInputs):
        raise GenerationExecutionError("generation inputs are required")
    expected_binding = getattr(adapter, "binding", None)
    if not isinstance(expected_binding, ScorerBinding):
        raise GenerationExecutionError("generation adapter lacks scorer binding")
    root = Path(output_root).expanduser().resolve()
    records: list[GeneratedSequenceRecord] = []
    executed = 0
    reused = 0
    for condition in inputs.conditions:
        path = _shard_path(root, condition)
        if path.exists():
            existing = _read_shard(path, condition, expected_binding)
            if not resume:
                raise GenerationExecutionError("generation shard already exists; use --resume")
            records.extend(existing)
            reused += 1
            continue
        generated = tuple(adapter.generate(condition.initial_request(), condition.structure, n_samples=256))
        if any(record.scorer_binding != expected_binding for record in generated):
            raise GenerationExecutionError("generated record scorer identity differs")
        rendered = _render_shard(condition, generated, expected_binding)
        try:
            atomic_write_new_bytes(path, rendered)
        except FileExistsError:
            existing = _read_shard(path, condition, expected_binding)
            if existing != tuple(sorted(generated, key=lambda item: item.request.sample_index)):
                raise GenerationExecutionError("immutable generation shard conflict") from None
            records.extend(existing)
            reused += 1
            continue
        records.extend(_read_shard(path, condition, expected_binding))
        executed += 1
    condition_order = {"PDB": 0, "AFDB": 1}
    ordered = tuple(
        sorted(
            records,
            key=lambda item: (
                item.request.protein_id,
                condition_order[item.request.backbone_condition],
                item.request.sample_index,
            ),
        )
    )
    expected_rows = inputs.expected_protein_count * 2 * _SAMPLES_PER_CONDITION
    if len(ordered) != expected_rows:
        raise GenerationExecutionError("generation run record cardinality differs")
    return GenerationRunResult(ordered, executed, reused)
