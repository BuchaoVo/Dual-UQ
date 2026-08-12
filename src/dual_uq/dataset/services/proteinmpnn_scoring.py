"""ProteinMPNN-compatible projections and scoring primitives.

This module keeps model imports lazy. Dataset validation and projection tests do
not require Torch, while runtime scoring uses the one authorized submodule.
"""

from __future__ import annotations

import json
import platform
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from dual_uq.core.atomic_io import atomic_write_new_bytes
from dual_uq.core.hashing import sha256_bytes, sha256_canonical, sha256_file
from dual_uq.models.proteinmpnn import (
    AUTHORIZED_CHECKPOINT_SHA256,
    AUTHORIZED_IMPLEMENTATION_COMMIT,
    PROTEINMPNN_SCORER_ID,
    DecodingRealization,
    ProteinMPNNAdapter,
    ProteinMPNNScorer,
    ProteinMPNNScoringError,
    ProteinMPNNStructureInput,
    implementation_worktree_is_clean,
    load_authorized_proteinmpnn_adapter,
    sequence_sha256,
    validate_protein_sequence,
    validate_structure_input,
)
from dual_uq.models.proteinmpnn import (
    DECODING_REALIZATION_ALGORITHM as _DECODING_REALIZATION_ALGORITHM,
)
from dual_uq.models.proteinmpnn import ProteinMPNNScore as _ProteinMPNNScore
from dual_uq.models.proteinmpnn import (
    make_decoding_realization as _make_decoding_realization,
)
from dual_uq.models.proteinmpnn import (
    score_target_log_probs as _score_target_log_probs,
)
from dual_uq.models.proteinmpnn import (
    tile_decoding_order as _tile_decoding_order,
)
from dual_uq.models.scoring import (
    CandidateCollection,
    ScoreDispatchError,
    ScoreRecord,
    ScoreRequest,
    ScoringVariant,
    VariantKind,
    execute_score_request,
)
from dual_uq.structure import StructureCondition

BACKBONE_ATOM_NAMES = ("N", "CA", "C", "O")
DECODING_REALIZATION_ALGORITHM = _DECODING_REALIZATION_ALGORITHM
FixedProbeScoringError = ProteinMPNNScoringError
ProteinMPNNRuntime = ProteinMPNNAdapter
ProteinMPNNScore = _ProteinMPNNScore
ScoringBackboneProjection = ProteinMPNNStructureInput
load_authorized_proteinmpnn_runtime = load_authorized_proteinmpnn_adapter
make_decoding_realization = _make_decoding_realization
score_target_log_probs = _score_target_log_probs
tile_decoding_order = _tile_decoding_order


@dataclass(frozen=True)
class PairedScoringProjection:
    pdb: ScoringBackboneProjection
    afdb: ScoringBackboneProjection


def _normalized_score_fields(record: dict[str, Any]) -> dict[str, Any]:
    """Adapt fields shared by authoritative WT and probe persisted rows."""
    try:
        return {
            "protein_id": str(record["protein_id"]),
            "condition_id": str(record["backbone_condition"]),
            "structure_sha256": str(record["backbone_sha256"]),
            "repeat_index": int(record["repeat_index"]),
            "seed": int(record["seed"]),
            "realization_id": str(record["decoding_realization_sha256"]),
            "scorer_id": PROTEINMPNN_SCORER_ID,
            "implementation_id": AUTHORIZED_IMPLEMENTATION_COMMIT,
            "checkpoint_id": str(record["model_checkpoint_sha256"]),
            "score_contract_id": str(record["scoring_protocol"]),
            "score_sum_logp_mask": float(record["score_sum_logp_mask"]),
            "score_mean_logp_mask": float(record["score_mean_logp_mask"]),
            "scored_residue_count": int(record["scored_residue_count"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ProteinMPNNScoringError(
            "invalid_normalized_score_record",
            "Authoritative persisted score fields are incomplete",
        ) from exc


def normalized_wt_score_record(record: dict[str, Any]) -> ScoreRecord:
    """Map one authoritative WT persisted row to the normalized envelope."""
    try:
        return ScoreRecord(
            **_normalized_score_fields(record),
            variant_kind=VariantKind.WT,
            variant_id=None,
            sequence_hash=None,
            position=None,
            wt_aa=None,
            mut_aa=None,
        )
    except ValueError as exc:
        raise ProteinMPNNScoringError(
            "invalid_normalized_score_record", str(exc)
        ) from exc


def normalized_probe_score_record(record: dict[str, Any]) -> ScoreRecord:
    """Map one authoritative fixed-probe persisted row without deriving delta."""
    try:
        sequence_hash = str(record["sequence_hash"])
        return ScoreRecord(
            **_normalized_score_fields(record),
            variant_kind=VariantKind.PROBE,
            variant_id=sequence_hash,
            sequence_hash=sequence_hash,
            position=int(record["position"]),
            wt_aa=str(record["wt_aa"]),
            mut_aa=str(record["mut_aa"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProteinMPNNScoringError(
            "invalid_normalized_score_record", str(exc)
        ) from exc


def _validate_projection(projection: ScoringBackboneProjection) -> None:
    if projection.backbone_condition not in {"PDB", "AFDB"}:
        raise FixedProbeScoringError(
            "invalid_backbone_condition", "Backbone condition must be PDB or AFDB"
        )
    validate_structure_input(projection)


def validate_paired_projection(
    pdb: ScoringBackboneProjection,
    afdb: ScoringBackboneProjection,
    *,
    expected_positions: tuple[int, ...],
) -> PairedScoringProjection:
    """Require identical manifest-authoritative PDB and AFDB domains."""
    _validate_projection(pdb)
    _validate_projection(afdb)
    if (
        pdb.protein_id != afdb.protein_id
        or pdb.backbone_condition != "PDB"
        or afdb.backbone_condition != "AFDB"
        or pdb.uniprot_positions != expected_positions
        or afdb.uniprot_positions != expected_positions
        or pdb.wt_sequence_projection != afdb.wt_sequence_projection
    ):
        raise FixedProbeScoringError(
            "scoring_projection_domain_mismatch",
            "PDB and AFDB scoring projections must equal the frozen manifest mask",
        )
    return PairedScoringProjection(pdb=pdb, afdb=afdb)


def project_candidate_sequence(
    *,
    full_sequence: str,
    sequence_hash: str,
    uniprot_positions: tuple[int, ...],
    canonical_wt_sequence: str,
    mutation_position: int,
    wt_aa: str,
    mut_aa: str,
) -> str:
    """Validate one declared single mutant and project it onto the frozen mask."""
    candidate = validate_protein_sequence(full_sequence)
    canonical = validate_protein_sequence(canonical_wt_sequence)
    if sequence_sha256(candidate) != sequence_hash:
        raise FixedProbeScoringError(
            "candidate_sequence_hash_mismatch", "Candidate sequence SHA differs"
        )
    if len(candidate) != len(canonical):
        raise FixedProbeScoringError(
            "candidate_mutation_mismatch", "Candidate and WT sequence lengths differ"
        )
    differences = tuple(
        index
        for index, (reference, observed) in enumerate(
            zip(canonical, candidate, strict=True), start=1
        )
        if reference != observed
    )
    if (
        differences != (mutation_position,)
        or mutation_position not in uniprot_positions
        or canonical[mutation_position - 1] != wt_aa
        or candidate[mutation_position - 1] != mut_aa
        or wt_aa == mut_aa
    ):
        raise FixedProbeScoringError(
            "candidate_mutation_mismatch",
            "Candidate sequence does not match its declared single substitution",
        )
    try:
        return "".join(candidate[position - 1] for position in uniprot_positions)
    except IndexError as exc:
        raise FixedProbeScoringError(
            "candidate_projection_out_of_range",
            "Frozen mask exceeds candidate sequence length",
        ) from exc


def _bound_path(
    record: dict[str, Any],
    *,
    path_field: str,
    hash_field: str,
    project_root: Path,
) -> Path:
    logical = record.get(path_field)
    expected = record.get(hash_field)
    if not isinstance(logical, str) or not isinstance(expected, str):
        raise FixedProbeScoringError(
            "missing_projection_binding", f"Missing {path_field}/{hash_field}"
        )
    path = (project_root / logical).resolve()
    try:
        path.relative_to(project_root.resolve())
    except ValueError as exc:
        raise FixedProbeScoringError(
            "nonportable_scoring_path", f"{path_field} escapes the repository"
        ) from exc
    if not path.is_file() or sha256_file(path) != expected:
        raise FixedProbeScoringError(
            "projection_asset_hash_mismatch", f"Frozen asset differs: {path_field}"
        )
    return path


def _coordinates(selection: Any) -> np.ndarray:
    atoms = {atom.atom_name: atom for atom in selection.selected_atoms}
    try:
        return np.asarray(
            [atoms[name].coordinates for name in BACKBONE_ATOM_NAMES],
            dtype=np.float32,
        )
    except KeyError as exc:
        raise FixedProbeScoringError(
            "invalid_backbone_coordinates", "A scoring residue lacks N/CA/C/O"
        ) from exc


def build_paired_scoring_projection(
    protein: dict[str, Any], project_root: Path
) -> PairedScoringProjection:
    """Reuse frozen mapping/fragment/backbone semantics for one manifest record."""
    from dual_uq.dataset.policies.fragments import resolve_exact_fragment
    from dual_uq.dataset.policies.identity import metadata_records
    from dual_uq.dataset.services.mapping import (
        mapping_with_provenance,
        parse_sifts_mapping_with_explicit_labels,
    )
    from dual_uq.dataset.stages.derivation import (
        P1ValidationError,
        _load_atom_records,
        _pair_residues,
    )
    from dual_uq.structure_io import (
        load_chain_ca_table,
        residue_name_to_one_letter,
    )

    root = project_root.resolve()
    protein_id = str(protein.get("protein_id", ""))
    accession = str(protein.get("uniprot_accession", ""))
    canonical = validate_protein_sequence(str(protein.get("canonical_wt_sequence", "")))
    if sequence_sha256(canonical) != protein.get("canonical_sequence_sha256"):
        raise FixedProbeScoringError(
            "canonical_sequence_hash_mismatch", f"Canonical sequence drift: {protein_id}"
        )
    expected_positions = tuple(protein.get("mask_positions", ()))
    if len(expected_positions) != protein.get("mask_length"):
        raise FixedProbeScoringError(
            "scoring_projection_domain_mismatch", "Manifest mask length differs"
        )
    pdb_path = _bound_path(
        protein,
        path_field="pdb_backbone_path",
        hash_field="pdb_backbone_sha256",
        project_root=root,
    )
    afdb_path = _bound_path(
        protein,
        path_field="afdb_backbone_path",
        hash_field="afdb_backbone_sha256",
        project_root=root,
    )
    mapping_provenance = protein.get("mapping_provenance")
    sequence_provenance = protein.get("canonical_sequence_provenance")
    if not isinstance(mapping_provenance, dict) or not isinstance(sequence_provenance, dict):
        raise FixedProbeScoringError(
            "missing_projection_binding", "Manifest mapping/sequence provenance is absent"
        )
    binding_record = {
        "sifts_path": mapping_provenance.get("sifts_path"),
        "sifts_sha256": mapping_provenance.get("sifts_sha256"),
        "metadata_path": sequence_provenance.get("source_path"),
        "metadata_sha256": sequence_provenance.get("source_sha256"),
    }
    sifts_path = _bound_path(
        binding_record,
        path_field="sifts_path",
        hash_field="sifts_sha256",
        project_root=root,
    )
    metadata_path = _bound_path(
        binding_record,
        path_field="metadata_path",
        hash_field="metadata_sha256",
        project_root=root,
    )
    try:
        mapping = parse_sifts_mapping_with_explicit_labels(
            sifts_path,
            pdb_path,
            chain_id=str(protein.get("pdb_chain", "")),
            uniprot_id=accession,
        )
        pdb_ca = load_chain_ca_table(pdb_path, str(protein.get("pdb_chain", "")))
        mapping, _gaps, _diagnostics = mapping_with_provenance(mapping, pdb_ca)
        mapped_interval = (
            int(mapping["uniprot_position"].min()),
            int(mapping["uniprot_position"].max()),
        )
        fragment_result = resolve_exact_fragment(
            metadata_records(metadata_path.read_bytes()), accession, mapped_interval
        )
        fragment = fragment_result["selected_fragment"]
        if fragment is None or fragment.model_entity_id != protein.get("afdb_model_id"):
            raise FixedProbeScoringError(
                "projection_fragment_mismatch", "Frozen AFDB fragment cannot be resolved"
            )
        selected = mapping.loc[
            mapping["uniprot_position"].isin(expected_positions)
        ].copy()
        selected = selected.sort_values("uniprot_position", kind="mergesort").reset_index(
            drop=True
        )
        selected_positions = tuple(selected["uniprot_position"].astype(int))
        if selected_positions != expected_positions:
            raise FixedProbeScoringError(
                "scoring_projection_domain_mismatch",
                "SIFTS projection differs from the frozen manifest mask",
            )
        selected["uniprot_residue_number"] = selected["uniprot_position"].astype(int)
        selected["output_position"] = np.arange(1, len(selected) + 1)
        selected["canonical_aa"] = [
            canonical[position - 1] for position in selected_positions
        ]
        mapping_amino_acids = tuple(
            residue_name_to_one_letter(value)
            for value in selected["uniprot_residue_name"]
        )
        if mapping_amino_acids != tuple(selected["canonical_aa"]):
            raise FixedProbeScoringError(
                "mapping_amino_acid_mismatch",
                "SIFTS mapping conflicts with the canonical sequence",
            )
        residues = _pair_residues(
            mapping=selected,
            pdb_records=_load_atom_records(pdb_path, protein_id),
            afdb_records=_load_atom_records(afdb_path, fragment.model_entity_id),
            pair_id=protein_id,
            model_entity_id=fragment.model_entity_id,
            fragment_start=fragment.uniprot_start,
            fragment_end=fragment.uniprot_end,
        )
    except FixedProbeScoringError:
        raise
    except P1ValidationError as exc:
        raise FixedProbeScoringError(exc.code, str(exc)) from exc
    positions = tuple(residue.uniprot_position for residue in residues)
    sequence_projection = "".join(residue.canonical_aa for residue in residues)
    pdb_coordinates = np.stack(
        [_coordinates(residue.pdb_backbone) for residue in residues]
    )
    afdb_coordinates = np.stack(
        [_coordinates(residue.afdb_backbone) for residue in residues]
    )
    return validate_paired_projection(
        ScoringBackboneProjection(
            protein_id=protein_id,
            backbone_condition="PDB",
            uniprot_positions=positions,
            wt_sequence_projection=sequence_projection,
            coordinates=pdb_coordinates,
        ),
        ScoringBackboneProjection(
            protein_id=protein_id,
            backbone_condition="AFDB",
            uniprot_positions=positions,
            wt_sequence_projection=sequence_projection,
            coordinates=afdb_coordinates,
        ),
        expected_positions=expected_positions,
    )


def _runtime_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProteinMPNNScoringError(
            "invalid_runtime_request", f"Unable to read runtime JSON: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ProteinMPNNScoringError(
            "invalid_runtime_request", "Runtime request must be a JSON object"
        )
    return payload


def _runtime_projection(
    protein: dict[str, Any], backbone_condition: str
) -> ScoringBackboneProjection:
    return ScoringBackboneProjection(
        protein_id=str(protein["protein_id"]),
        backbone_condition=backbone_condition,
        uniprot_positions=tuple(int(value) for value in protein["uniprot_positions"]),
        wt_sequence_projection=str(protein["sequences"][0]),
        coordinates=np.asarray(
            protein[f"{backbone_condition.lower()}_coordinates"], dtype=np.float32
        ),
    )


def execute_g2_runtime_request(
    *,
    request_path: Path,
    response_path: Path,
    project_root: Path,
    device_name: str,
) -> dict[str, Any]:
    """Execute a data-prepared G2 request in the Torch-only model environment."""
    request = _runtime_json(request_path)
    if request.get("schema_version") != "stage0_g2_runtime_request_v1":
        raise ProteinMPNNScoringError(
            "invalid_runtime_request", "Unsupported G2 runtime request schema"
        )
    model = request.get("model_identity")
    proteins = request.get("proteins")
    if not isinstance(model, dict) or not isinstance(proteins, list) or len(proteins) != 2:
        raise ProteinMPNNScoringError(
            "invalid_runtime_request", "G2 runtime request identities are incomplete"
        )
    if (
        model.get("implementation_commit") != AUTHORIZED_IMPLEMENTATION_COMMIT
        or model.get("checkpoint_sha256") != AUTHORIZED_CHECKPOINT_SHA256
    ):
        raise ProteinMPNNScoringError(
            "unauthorized_scoring_model", "Runtime request model identity differs"
        )
    implementation_path = (project_root / str(model["implementation_path"])).resolve()
    checkpoint_path = (project_root / str(model["checkpoint_path"])).resolve()
    if sha256_file(checkpoint_path) != AUTHORIZED_CHECKPOINT_SHA256:
        raise ProteinMPNNScoringError(
            "checkpoint_hash_mismatch", "Runtime checkpoint SHA differs"
        )
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(implementation_path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProteinMPNNScoringError(
            "implementation_commit_unavailable", "Cannot resolve runtime commit"
        ) from exc
    if commit != AUTHORIZED_IMPLEMENTATION_COMMIT:
        raise ProteinMPNNScoringError(
            "implementation_commit_mismatch", "Runtime implementation commit differs"
        )
    if not implementation_worktree_is_clean(implementation_path):
        raise ProteinMPNNScoringError(
            "implementation_worktree_dirty",
            "Runtime ProteinMPNN implementation has tracked-file drift",
        )
    runtime = load_authorized_proteinmpnn_runtime(
        implementation_path=implementation_path,
        checkpoint_path=checkpoint_path,
        device_name=device_name,
        backbone_noise=float(request["scoring_protocol"]["backbone_noise"]),
    )
    calls: list[dict[str, Any]] = []

    def score_vector(seed: int, batch_size: int) -> None:
        for protein in proteins:
            sequences = tuple(str(value) for value in protein["sequences"])
            realization_record = next(
                item for item in protein["realizations"] if int(item["seed"]) == seed
            )
            order = tuple(int(value) for value in realization_record["order"])
            fingerprint = sha256_bytes(np.asarray(order, dtype="<i8").tobytes())
            if fingerprint != realization_record["fingerprint"]:
                raise ProteinMPNNScoringError(
                    "decoding_realization_mismatch", "Runtime order fingerprint differs"
                )
            realization = DecodingRealization(
                protein_id=str(protein["protein_id"]),
                repeat_index=int(realization_record["repeat_index"]),
                seed=seed,
                protocol_version=str(request["scoring_protocol"]["protocol_version"]),
                algorithm=str(realization_record["algorithm"]),
                order=order,
                fingerprint=fingerprint,
            )
            for backbone in ("PDB", "AFDB"):
                projection = _runtime_projection(protein, backbone)
                scores = runtime.score_sequences(
                    projection,
                    sequences,
                    realization,
                    batch_size=batch_size,
                )
                calls.append(
                    {
                        "protein_id": projection.protein_id,
                        "backbone_condition": backbone,
                        "seed": seed,
                        "repeat_index": realization.repeat_index,
                        "realization_fingerprint": realization.fingerprint,
                        "batch_size": batch_size,
                        "sequence_count": len(sequences),
                        "scores": [
                            {
                                "score_sum_logp_mask": score.score_sum_logp_mask,
                                "score_mean_logp_mask": score.score_mean_logp_mask,
                            }
                            for score in scores
                        ],
                    }
                )

    for _ in range(3):
        score_vector(0, 8)
    score_vector(0, 1)
    score_vector(0, 8)
    for seed in (0, 1, 2, 3):
        score_vector(seed, 8)
    torch = runtime.torch
    device_display_name = None
    if str(runtime.device).startswith("cuda") and torch.cuda.is_available():
        device_display_name = str(torch.cuda.get_device_name(runtime.device))
    response = {
        "schema_version": "stage0_g2_runtime_response_v1",
        "request_sha256": sha256_file(request_path),
        "calls": calls,
        "runtime_model": {
            "checkpoint_num_edges": runtime.checkpoint_num_edges,
            "checkpoint_noise_level": runtime.checkpoint_noise_level,
        },
        "execution_environment": {
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "cuda_runtime_version": (
                None if torch.version.cuda is None else str(torch.version.cuda)
            ),
            "cuda_available": bool(torch.cuda.is_available()),
            "device": str(runtime.device),
            "device_name": device_display_name,
        },
    }
    rendered = (
        json.dumps(response, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    if response_path.exists():
        raise ProteinMPNNScoringError(
            "runtime_response_exists", "Runtime response path already exists"
        )
    atomic_write_new_bytes(response_path, rendered)
    return response


def _render_formal_shard(payload: dict[str, Any]) -> bytes:
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
        raise ProteinMPNNScoringError(
            "invalid_formal_runtime_request", "Formal shard is not JSON-serializable"
        ) from exc


def _write_runtime_shard(path: Path, payload: dict[str, Any]) -> str:
    rendered = _render_formal_shard(payload)
    if path.exists():
        if path.read_bytes() != rendered:
            raise ProteinMPNNScoringError(
                "immutable_formal_shard_conflict",
                f"Existing runtime shard differs: {path}",
            )
        return "reused_identical"
    atomic_write_new_bytes(path, rendered)
    return "created"


@dataclass(frozen=True)
class FormalRuntimeScoreRequestView:
    """Normalized scientific request plus legacy execution-only bindings."""

    request: ScoreRequest
    projection: ScoringBackboneProjection
    binding: Mapping[str, Any]
    output_filename: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "binding", MappingProxyType(dict(self.binding)))


def _formal_candidate_collection(
    request: Mapping[str, Any],
    *,
    protein_id: str,
    positions: tuple[int, ...],
) -> CandidateCollection:
    canonical = validate_protein_sequence(
        str(request.get("canonical_wt_sequence", ""))
    )
    canonical_hash = str(request.get("canonical_sequence_sha256", ""))
    if sequence_sha256(canonical) != canonical_hash:
        raise ProteinMPNNScoringError(
            "formal_candidate_identity_mismatch",
            "Formal canonical sequence identity differs",
        )
    expected_domain = sha256_canonical(
        {
            "protein_id": protein_id,
            "canonical_positions": list(positions),
            "canonical_sequence_sha256": canonical_hash,
        }
    )
    if request.get("common_mask_binding") != expected_domain:
        raise ProteinMPNNScoringError(
            "formal_scoring_domain_mismatch",
            "Formal common-mask scientific binding differs",
        )

    records = request.get("candidate_records")
    candidate_hashes = request.get("candidate_sequence_hashes")
    if (
        not isinstance(records, list)
        or not records
        or not isinstance(candidate_hashes, list)
        or len(records) != len(candidate_hashes)
    ):
        raise ProteinMPNNScoringError(
            "invalid_formal_runtime_request",
            "Formal full-sequence candidate records are incomplete",
        )
    variants = []
    for index, (record, expected_hash) in enumerate(
        zip(records, candidate_hashes, strict=True)
    ):
        if not isinstance(record, dict) or record.get("sequence_hash") != expected_hash:
            raise ProteinMPNNScoringError(
                "formal_candidate_identity_mismatch",
                f"Formal candidate record {index} identity differs",
            )
        try:
            variants.append(
                ScoringVariant(
                    variant_kind=VariantKind.PROBE,
                    variant_id=str(record["sequence_hash"]),
                    sequence_hash=str(record["sequence_hash"]),
                    sequence=str(record["full_sequence"]),
                    position=int(record["position"]),
                    wt_aa=str(record["wt_aa"]),
                    mut_aa=str(record["mut_aa"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProteinMPNNScoringError(
                "formal_candidate_identity_mismatch",
                f"Formal candidate record {index} is invalid",
            ) from exc
    collection_id = sha256_canonical(
        {"sequence_hashes": [str(value) for value in candidate_hashes]}
    )
    try:
        return CandidateCollection(
            collection_id=collection_id,
            wt=ScoringVariant(
                variant_kind=VariantKind.WT,
                variant_id=None,
                sequence_hash=canonical_hash,
                sequence=canonical,
                position=None,
                wt_aa=None,
                mut_aa=None,
            ),
            probes=tuple(variants),
        )
    except (TypeError, ValueError) as exc:
        raise ProteinMPNNScoringError(
            "formal_candidate_identity_mismatch",
            "Formal candidate collection violates the normalized request contract",
        ) from exc


def adapt_formal_runtime_requests(
    request: Mapping[str, Any],
) -> tuple[FormalRuntimeScoreRequestView, ...]:
    """Adapt one historical formal worker request into normalized score requests."""
    schema_version = request.get("schema_version")
    if schema_version == "stage0_formal_protein_request_v1":
        raise ProteinMPNNScoringError(
            "legacy_formal_request_requires_enrichment",
            "Legacy v1 request lacks canonical full-sequence probe identity; "
            "rebuild missing work from its frozen plan instead of inferring it",
        )
    if schema_version != "stage0_formal_protein_request_v2":
        raise ProteinMPNNScoringError(
            "invalid_formal_runtime_request", "Formal request schema differs"
        )
    model_identity = request.get("model_identity")
    score_contract = request.get("scoring_protocol")
    if (
        not isinstance(model_identity, dict)
        or model_identity.get("implementation_commit")
        != AUTHORIZED_IMPLEMENTATION_COMMIT
        or model_identity.get("checkpoint_sha256") != AUTHORIZED_CHECKPOINT_SHA256
        or score_contract != "stage0_fixed_sequence_autoregressive_mask_logp_v1"
    ):
        raise ProteinMPNNScoringError(
            "unauthorized_scoring_model", "Formal request model/protocol differs"
        )
    protein_id = request.get("protein_id")
    positions_raw = request.get("uniprot_positions")
    tasks = request.get("tasks")
    if (
        not isinstance(protein_id, str)
        or not isinstance(positions_raw, list)
        or not isinstance(tasks, list)
        or not tasks
    ):
        raise ProteinMPNNScoringError(
            "invalid_formal_runtime_request", "Formal request fields are incomplete"
        )
    try:
        positions = tuple(int(value) for value in positions_raw)
        collection = _formal_candidate_collection(
            request, protein_id=protein_id, positions=positions
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ProteinMPNNScoringError):
            raise
        raise ProteinMPNNScoringError(
            "invalid_formal_runtime_request", "Formal scoring domain is invalid"
        ) from exc

    projected_wt = "".join(collection.wt.sequence[position - 1] for position in positions)
    projected_candidates = tuple(
        "".join(probe.sequence[position - 1] for position in positions)
        for probe in collection.probes
    )
    declared_sequences = request.get("candidate_sequences")
    declared_projection_hashes = request.get("candidate_projection_sha256")
    if (
        request.get("wt_sequence") != projected_wt
        or declared_sequences != list(projected_candidates)
        or declared_projection_hashes
        != [sequence_sha256(sequence) for sequence in projected_candidates]
    ):
        raise ProteinMPNNScoringError(
            "formal_candidate_identity_mismatch",
            "Formal projected candidate identity differs",
        )

    coordinates = {
        "PDB": np.asarray(request.get("pdb_coordinates"), dtype=np.float32),
        "AFDB": np.asarray(request.get("afdb_coordinates"), dtype=np.float32),
    }
    fingerprints: dict[int, set[str]] = {}
    canonical_realization_mismatches = []
    views = []
    for task in tasks:
        if not isinstance(task, dict) or not isinstance(task.get("binding"), dict):
            raise ProteinMPNNScoringError(
                "invalid_formal_runtime_request", "Formal task binding is absent"
            )
        binding = task["binding"]
        backbone = binding.get("backbone_condition")
        repeat_index = binding.get("repeat_index")
        seed = binding.get("seed")
        order_raw = task.get("order")
        output_filename = task.get("output_filename")
        if (
            binding.get("protein_id") != protein_id
            or backbone not in {"PDB", "AFDB"}
            or not isinstance(repeat_index, int)
            or not isinstance(seed, int)
            or not isinstance(order_raw, list)
            or sorted(order_raw) != list(range(len(positions)))
            or not isinstance(output_filename, str)
            or Path(output_filename).name != output_filename
            or binding.get("candidate_count") != len(collection.probes)
            or binding.get("candidate_identity_sha256") != collection.collection_id
            or binding.get("checkpoint_sha256") != AUTHORIZED_CHECKPOINT_SHA256
            or binding.get("implementation_commit")
            != AUTHORIZED_IMPLEMENTATION_COMMIT
            or binding.get("scoring_protocol") != score_contract
            or binding.get("mask_length") != len(positions)
            or binding.get("backbone_sha256") is None
        ):
            raise ProteinMPNNScoringError(
                "invalid_formal_runtime_request", "Formal task binding differs"
            )
        expected_realization = make_decoding_realization(
            protein_id=protein_id,
            mask_length=len(positions),
            repeat_index=repeat_index,
            seed=seed,
            protocol_version=str(score_contract),
        )
        order = tuple(int(value) for value in order_raw)
        observed_fingerprint = sha256_bytes(
            np.asarray(order, dtype="<i8").tobytes()
        )
        if (
            binding.get("decoding_realization_sha256") != observed_fingerprint
            or binding.get("decoding_realization_algorithm")
            != expected_realization.algorithm
        ):
            raise ProteinMPNNScoringError(
                "decoding_realization_mismatch",
                "Formal decoding realization differs",
            )
        fingerprints.setdefault(repeat_index, set()).add(observed_fingerprint)
        canonical_realization_mismatches.append(
            order != expected_realization.order
            or observed_fingerprint != expected_realization.fingerprint
        )
        structure_sha = str(binding["backbone_sha256"])
        condition = StructureCondition(
            protein_id=protein_id,
            condition_id=str(backbone),
            source="PDB" if backbone == "PDB" else "AlphaFoldDB",
            structure_sha256=structure_sha,
        )
        projection = ScoringBackboneProjection(
            protein_id=protein_id,
            backbone_condition=str(backbone),
            uniprot_positions=positions,
            wt_sequence_projection=projected_wt,
            coordinates=coordinates[str(backbone)],
            structure_sha256=structure_sha,
        )
        _validate_projection(projection)
        normalized = ScoreRequest(
            condition=condition,
            scoring_domain_id=str(request["common_mask_binding"]),
            canonical_positions=positions,
            candidate_collection=collection,
            repeat_index=repeat_index,
            seed=seed,
            realization_id=expected_realization.fingerprint,
            realization_algorithm=expected_realization.algorithm,
            score_contract_id=str(score_contract),
        )
        views.append(
            FormalRuntimeScoreRequestView(
                request=normalized,
                projection=projection,
                binding=binding,
                output_filename=output_filename,
            )
        )
    if any(len(values) != 1 for values in fingerprints.values()):
        raise ProteinMPNNScoringError(
            "formal_realization_pairing_mismatch",
            "PDB and AFDB task realizations differ",
        )
    if any(canonical_realization_mismatches):
        raise ProteinMPNNScoringError(
            "decoding_realization_mismatch",
            "Formal decoding realization differs from the frozen algorithm",
        )
    return tuple(views)


def formal_shard_payload_from_score_records(
    *,
    view: FormalRuntimeScoreRequestView,
    records: tuple[ScoreRecord, ...],
    execution_environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Translate normalized measurements to the authoritative legacy shard."""
    expected = view.request.candidate_collection.variants
    if len(records) != len(expected):
        raise ProteinMPNNScoringError(
            "formal_shard_candidate_count_mismatch",
            "Normalized record count differs from the formal candidate collection",
        )
    binding = view.binding
    for index, (record, variant) in enumerate(zip(records, expected, strict=True)):
        expected_hash = (
            None if variant.variant_kind is VariantKind.WT else variant.sequence_hash
        )
        if (
            record.protein_id != view.request.protein_id
            or record.condition_id != view.request.condition.condition_id
            or record.structure_sha256 != view.request.condition.structure_sha256
            or record.repeat_index != view.request.repeat_index
            or record.seed != view.request.seed
            or record.realization_id != view.request.realization_id
            or record.scorer_id != PROTEINMPNN_SCORER_ID
            or record.score_contract_id != view.request.score_contract_id
            or record.implementation_id != binding.get("implementation_commit")
            or record.checkpoint_id != binding.get("checkpoint_sha256")
            or record.variant_kind is not variant.variant_kind
            or record.variant_id != variant.variant_id
            or record.sequence_hash != expected_hash
            or record.position != variant.position
            or record.wt_aa != variant.wt_aa
            or record.mut_aa != variant.mut_aa
            or record.scored_residue_count != binding.get("mask_length")
        ):
            raise ProteinMPNNScoringError(
                "formal_score_record_binding_mismatch",
                f"Normalized record {index} differs from the formal shard binding",
            )
    wt = records[0]
    probes = records[1:]
    if wt.variant_kind is not VariantKind.WT or any(
        record.variant_kind is not VariantKind.PROBE for record in probes
    ):
        raise ProteinMPNNScoringError(
            "formal_candidate_identity_mismatch",
            "Normalized WT/probe association differs",
        )

    def score_fields(record: ScoreRecord) -> dict[str, Any]:
        return {
            "score_sum_logp_mask": record.score_sum_logp_mask,
            "score_mean_logp_mask": record.score_mean_logp_mask,
            "scored_residue_count": record.scored_residue_count,
        }

    return {
        "schema_version": "stage0_fixed_probe_scoring_shard_v1",
        "binding": dict(view.binding),
        "wt_score": score_fields(wt),
        "candidate_scores": [
            {"sequence_hash": record.sequence_hash, **score_fields(record)}
            for record in probes
        ],
        "execution_environment": dict(execution_environment),
    }


def run_formal_runtime_request(
    *,
    request: dict[str, Any],
    runtime: Any,
    shard_directory: Path,
    batch_size: int,
    execution_environment: dict[str, Any],
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Score all requested logical cells for one protein with one loaded model."""
    views = adapt_formal_runtime_requests(request)
    model_identity = request["model_identity"]
    if (
        getattr(runtime, "implementation_id", None)
        != model_identity["implementation_commit"]
        or getattr(runtime, "checkpoint_id", None)
        != model_identity["checkpoint_sha256"]
    ):
        raise ProteinMPNNScoringError(
            "unauthorized_scoring_model",
            "Loaded runtime identity differs from the formal request",
        )
    projections = {
        (view.request.protein_id, view.request.condition.condition_id): view.projection
        for view in views
    }
    scorer = ProteinMPNNScorer(
        adapter=runtime,
        projection_resolver=lambda normalized: projections[
            (normalized.protein_id, normalized.condition.condition_id)
        ],
        score_contract_id=str(request["scoring_protocol"]),
        batch_size=batch_size,
    )
    shard_directory.mkdir(parents=True, exist_ok=True)
    statuses = {"created": 0, "reused_identical": 0}
    for ordinal, view in enumerate(views, start=1):
        try:
            records = execute_score_request(scorer, view.request)
        except ScoreDispatchError as exc:
            raise ProteinMPNNScoringError(exc.code, str(exc)) from exc
        payload = formal_shard_payload_from_score_records(
            view=view,
            records=records,
            execution_environment=execution_environment,
        )
        status = _write_runtime_shard(
            shard_directory / view.output_filename, payload
        )
        statuses[status] += 1
        if progress_callback is not None:
            progress_callback(
                {
                    "protein_id": view.request.protein_id,
                    "backbone_condition": view.request.condition.condition_id,
                    "repeat_index": view.request.repeat_index,
                    "completed": ordinal,
                    "total": len(views),
                    "write_status": status,
                }
            )
    return {
        "status": "complete",
        "protein_id": views[0].request.protein_id,
        "candidate_count": len(views[0].request.candidate_collection.probes),
        "completed_shards": len(views),
        "created_shards": statuses["created"],
        "reused_shards": statuses["reused_identical"],
        "execution_environment": execution_environment,
    }


def execute_formal_runtime_request(
    *,
    request_path: Path,
    response_path: Path,
    shard_directory: Path,
    project_root: Path,
    device_name: str,
    batch_size: int,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Load the authorized model once and execute formal shards for one protein."""
    request = _runtime_json(request_path)
    model = request.get("model_identity")
    if not isinstance(model, dict):
        raise ProteinMPNNScoringError(
            "invalid_formal_runtime_request", "Formal model identity is absent"
        )
    implementation_path = (project_root / str(model.get("implementation_path"))).resolve()
    checkpoint_path = (project_root / str(model.get("checkpoint_path"))).resolve()
    if (
        model.get("implementation_commit") != AUTHORIZED_IMPLEMENTATION_COMMIT
        or model.get("checkpoint_sha256") != AUTHORIZED_CHECKPOINT_SHA256
        or sha256_file(checkpoint_path) != AUTHORIZED_CHECKPOINT_SHA256
    ):
        raise ProteinMPNNScoringError(
            "unauthorized_scoring_model", "Formal runtime model identity differs"
        )
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(implementation_path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProteinMPNNScoringError(
            "implementation_commit_unavailable", "Cannot resolve runtime commit"
        ) from exc
    if commit != AUTHORIZED_IMPLEMENTATION_COMMIT:
        raise ProteinMPNNScoringError(
            "implementation_commit_mismatch", "Runtime implementation commit differs"
        )
    if not implementation_worktree_is_clean(implementation_path):
        raise ProteinMPNNScoringError(
            "implementation_worktree_dirty",
            "Runtime ProteinMPNN implementation has tracked-file drift",
        )
    runtime = load_authorized_proteinmpnn_runtime(
        implementation_path=implementation_path,
        checkpoint_path=checkpoint_path,
        device_name=device_name,
        backbone_noise=0.0,
    )
    torch = runtime.torch
    device_display_name = None
    if str(runtime.device).startswith("cuda") and torch.cuda.is_available():
        device_display_name = str(torch.cuda.get_device_name(runtime.device))
    environment = {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": (
            None if torch.version.cuda is None else str(torch.version.cuda)
        ),
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(runtime.device),
        "device_name": device_display_name,
        "batch_size": batch_size,
        "process_boundary": "data_environment_to_model_environment_json_bridge_v1",
    }
    result = run_formal_runtime_request(
        request=request,
        runtime=runtime,
        shard_directory=shard_directory,
        batch_size=batch_size,
        execution_environment=environment,
        progress_callback=progress_callback,
    )
    response = {
        "schema_version": "stage0_formal_runtime_response_v1",
        "request_sha256": sha256_file(request_path),
        **result,
    }
    rendered = (
        json.dumps(response, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    if response_path.exists():
        response_path.unlink()
    atomic_write_new_bytes(response_path, rendered)
    return response
