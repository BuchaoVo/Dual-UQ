"""ProteinMPNN-compatible projections and scoring primitives.

This module keeps model imports lazy. Dataset validation and projection tests do
not require Torch, while runtime scoring uses the one authorized submodule.
"""

from __future__ import annotations

import importlib.util
import json
import platform
import subprocess
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from dual_uq.core.atomic_io import atomic_write_new_bytes
from dual_uq.core.hashing import sha256_bytes, sha256_canonical, sha256_file
from dual_uq.dataset.storage.proteinmpnn import (
    sequence_sha256,
    validate_protein_sequence,
)

DECODING_REALIZATION_ALGORITHM = "sha256_ranked_permutation_v1"
PROTEINMPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
BACKBONE_ATOM_NAMES = ("N", "CA", "C", "O")
AUTHORIZED_IMPLEMENTATION_COMMIT = "8907e6671bfbfc92303b5f79c4b5e6ce47cdef57"
AUTHORIZED_CHECKPOINT_SHA256 = (
    "c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd"
)


class ProteinMPNNScoringError(ValueError):
    """Structured projection, runtime, or numerical scoring failure."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


FixedProbeScoringError = ProteinMPNNScoringError


def implementation_worktree_is_clean(path: Path) -> bool:
    """Return whether the authorized implementation has no tracked-file drift."""
    try:
        status = subprocess.check_output(
            [
                "git",
                "-C",
                str(path),
                "status",
                "--porcelain=v1",
                "--untracked-files=no",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProteinMPNNScoringError(
            "implementation_worktree_unavailable",
            "Cannot verify ProteinMPNN tracked-file state",
        ) from exc
    return not status.strip()


@dataclass(frozen=True)
class ScoringBackboneProjection:
    protein_id: str
    backbone_condition: str
    uniprot_positions: tuple[int, ...]
    wt_sequence_projection: str
    coordinates: np.ndarray

    @property
    def residue_count(self) -> int:
        return len(self.uniprot_positions)


@dataclass(frozen=True)
class PairedScoringProjection:
    pdb: ScoringBackboneProjection
    afdb: ScoringBackboneProjection


@dataclass(frozen=True)
class DecodingRealization:
    protein_id: str
    repeat_index: int
    seed: int
    protocol_version: str
    algorithm: str
    order: tuple[int, ...]
    fingerprint: str


@dataclass(frozen=True)
class ProteinMPNNScore:
    score_sum_logp_mask: float
    score_mean_logp_mask: float


@dataclass(frozen=True)
class ProteinMPNNRuntime:
    """Authorized loaded model plus frozen runtime identity."""

    model: Any
    torch: Any
    device: Any
    checkpoint_num_edges: int
    checkpoint_noise_level: float

    def score_sequences(
        self,
        projection: ScoringBackboneProjection,
        sequences: tuple[str, ...],
        realization: DecodingRealization,
        *,
        batch_size: int,
    ) -> tuple[ProteinMPNNScore, ...]:
        """Score projected sequences without generation or fresh randomness."""
        _validate_projection(projection)
        if batch_size <= 0:
            raise FixedProbeScoringError(
                "invalid_batch_size", "ProteinMPNN batch size must be positive"
            )
        if (
            realization.protein_id != projection.protein_id
            or len(realization.order) != projection.residue_count
        ):
            raise FixedProbeScoringError(
                "decoding_realization_mismatch",
                "Decoding realization does not match the scoring projection",
            )
        checked_sequences = tuple(validate_protein_sequence(value) for value in sequences)
        if not checked_sequences or any(
            len(value) != projection.residue_count for value in checked_sequences
        ):
            raise FixedProbeScoringError(
                "scoring_sequence_length_mismatch",
                "Projected sequence length differs from the scoring domain",
            )
        alphabet_index = {amino_acid: index for index, amino_acid in enumerate(PROTEINMPNN_ALPHABET)}
        results: list[ProteinMPNNScore] = []
        torch = self.torch
        coordinates = np.asarray(projection.coordinates, dtype=np.float32)
        positions = np.asarray(projection.uniprot_positions, dtype=np.int64)
        for start in range(0, len(checked_sequences), batch_size):
            chunk = checked_sequences[start : start + batch_size]
            size = len(chunk)
            target_indices = np.asarray(
                [[alphabet_index[amino_acid] for amino_acid in sequence] for sequence in chunk],
                dtype=np.int64,
            )
            x = torch.as_tensor(
                np.broadcast_to(coordinates, (size, *coordinates.shape)).copy(),
                dtype=torch.float32,
                device=self.device,
            )
            s = torch.as_tensor(target_indices, dtype=torch.long, device=self.device)
            mask = torch.ones((size, projection.residue_count), dtype=torch.float32, device=self.device)
            chain_m = torch.ones_like(mask)
            residue_idx = torch.as_tensor(
                np.broadcast_to(positions, (size, len(positions))).copy(),
                dtype=torch.long,
                device=self.device,
            )
            chain_encoding = torch.ones_like(residue_idx)
            randn = torch.zeros_like(mask)
            decoding_order = torch.as_tensor(
                tile_decoding_order(realization, batch_size=size),
                dtype=torch.long,
                device=self.device,
            )
            with torch.inference_mode():
                log_probs = self.model(
                    x,
                    s,
                    mask,
                    chain_m,
                    residue_idx,
                    chain_encoding,
                    randn,
                    use_input_decoding_order=True,
                    decoding_order=decoding_order,
                )
            results.extend(
                score_target_log_probs(
                    log_probs.detach().to("cpu").numpy(), target_indices
                )
            )
        return tuple(results)


def make_decoding_realization(
    *,
    protein_id: str,
    mask_length: int,
    repeat_index: int,
    seed: int,
    protocol_version: str,
) -> DecodingRealization:
    """Create a portable explicit autoregressive order without runtime RNG state."""
    if (
        not protein_id
        or not protocol_version
        or isinstance(mask_length, bool)
        or not isinstance(mask_length, int)
        or mask_length <= 0
        or isinstance(repeat_index, bool)
        or not isinstance(repeat_index, int)
        or repeat_index < 0
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise FixedProbeScoringError(
            "invalid_decoding_realization", "Decoding realization inputs are invalid"
        )
    ranked = sorted(
        (
            sha256_canonical(
                {
                    "algorithm": DECODING_REALIZATION_ALGORITHM,
                    "protocol_version": protocol_version,
                    "protein_id": protein_id,
                    "repeat_index": repeat_index,
                    "seed": seed,
                    "position_index": position,
                }
            ),
            position,
        )
        for position in range(mask_length)
    )
    order = tuple(position for _digest, position in ranked)
    canonical_order_bytes = np.asarray(order, dtype="<i8").tobytes()
    fingerprint = sha256_bytes(canonical_order_bytes)
    return DecodingRealization(
        protein_id=protein_id,
        repeat_index=repeat_index,
        seed=seed,
        protocol_version=protocol_version,
        algorithm=DECODING_REALIZATION_ALGORITHM,
        order=order,
        fingerprint=fingerprint,
    )


def tile_decoding_order(
    realization: DecodingRealization, *, batch_size: int
) -> np.ndarray:
    """Tile one exact scientific realization without drawing new randomness."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise FixedProbeScoringError(
            "invalid_batch_size", "ProteinMPNN batch size must be positive"
        )
    return np.broadcast_to(
        np.asarray(realization.order, dtype=np.int64),
        (batch_size, len(realization.order)),
    ).copy()


def score_target_log_probs(
    log_probs: np.ndarray, targets: np.ndarray
) -> tuple[ProteinMPNNScore, ...]:
    """Gather target-AA log probabilities directly; higher remains better."""
    probabilities = np.asarray(log_probs)
    target_indices = np.asarray(targets)
    if probabilities.ndim != 3:
        raise FixedProbeScoringError(
            "invalid_log_probs", "ProteinMPNN log_probs must have shape [B,L,A]"
        )
    if not np.isfinite(probabilities).all():
        raise FixedProbeScoringError(
            "nonfinite_score", "ProteinMPNN log_probs contain non-finite values"
        )
    if target_indices.ndim != 2 or target_indices.shape != probabilities.shape[:2]:
        raise FixedProbeScoringError(
            "score_shape_mismatch", "Target indices must match log_probs [B,L]"
        )
    if not np.issubdtype(target_indices.dtype, np.integer):
        raise FixedProbeScoringError(
            "invalid_target_indices", "Target amino-acid indices must be integers"
        )
    if target_indices.size == 0 or target_indices.min() < 0 or target_indices.max() >= probabilities.shape[2]:
        raise FixedProbeScoringError(
            "target_index_out_of_range", "Target amino-acid index is out of range"
        )
    gathered = np.take_along_axis(
        probabilities, target_indices[..., np.newaxis], axis=-1
    ).squeeze(-1)
    if not np.isfinite(gathered).all():
        raise FixedProbeScoringError(
            "nonfinite_score", "Target log-probability score is not finite"
        )
    return tuple(
        ProteinMPNNScore(
            score_sum_logp_mask=float(row.sum(dtype=np.float64)),
            score_mean_logp_mask=float(row.mean(dtype=np.float64)),
        )
        for row in gathered
    )


def load_authorized_proteinmpnn_runtime(
    *,
    implementation_path: Path,
    checkpoint_path: Path,
    device_name: str,
    backbone_noise: float,
) -> ProteinMPNNRuntime:
    """Lazy-load the hash-verified ProteinMPNN implementation and checkpoint."""
    try:
        import torch
    except ImportError as exc:
        raise FixedProbeScoringError(
            "torch_unavailable", "ProteinMPNN runtime requires the model environment"
        ) from exc
    utility_path = implementation_path / "protein_mpnn_utils.py"
    spec = importlib.util.spec_from_file_location(
        "_dual_uq_authorized_protein_mpnn_utils", utility_path
    )
    if spec is None or spec.loader is None:
        raise FixedProbeScoringError(
            "implementation_import_failure", "Cannot load authorized ProteinMPNN utilities"
        )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        checkpoint = torch.load(checkpoint_path, map_location=device_name)
        model = module.ProteinMPNN(
            ca_only=False,
            num_letters=21,
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            augment_eps=backbone_noise,
            k_neighbors=int(checkpoint["num_edges"]),
        )
        device = torch.device(device_name)
        model.to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
    except (OSError, KeyError, RuntimeError, ValueError) as exc:
        raise FixedProbeScoringError(
            "model_load_failure", "Authorized ProteinMPNN checkpoint failed to load"
        ) from exc
    return ProteinMPNNRuntime(
        model=model,
        torch=torch,
        device=device,
        checkpoint_num_edges=int(checkpoint["num_edges"]),
        checkpoint_noise_level=float(checkpoint["noise_level"]),
    )


def _validate_projection(projection: ScoringBackboneProjection) -> None:
    if projection.backbone_condition not in {"PDB", "AFDB"}:
        raise FixedProbeScoringError(
            "invalid_backbone_condition", "Backbone condition must be PDB or AFDB"
        )
    positions = projection.uniprot_positions
    if (
        not positions
        or any(isinstance(value, bool) or not isinstance(value, int) for value in positions)
        or any(right <= left for left, right in pairwise(positions))
    ):
        raise FixedProbeScoringError(
            "scoring_projection_domain_mismatch",
            "Scoring positions must be strictly increasing UniProt coordinates",
        )
    coordinates = np.asarray(projection.coordinates)
    if coordinates.shape != (len(positions), 4, 3) or not np.isfinite(coordinates).all():
        raise FixedProbeScoringError(
            "invalid_backbone_coordinates",
            "Scoring projection requires finite N/CA/C/O coordinates",
        )
    if len(projection.wt_sequence_projection) != len(positions):
        raise FixedProbeScoringError(
            "scoring_projection_sequence_mismatch",
            "Projected WT sequence and residue domain lengths differ",
        )
    validate_protein_sequence(projection.wt_sequence_projection)


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
    if request.get("schema_version") != "stage0_formal_protein_request_v1":
        raise ProteinMPNNScoringError(
            "invalid_formal_runtime_request", "Formal request schema differs"
        )
    model_identity = request.get("model_identity")
    if (
        not isinstance(model_identity, dict)
        or model_identity.get("implementation_commit")
        != AUTHORIZED_IMPLEMENTATION_COMMIT
        or model_identity.get("checkpoint_sha256") != AUTHORIZED_CHECKPOINT_SHA256
        or request.get("scoring_protocol")
        != "stage0_fixed_sequence_autoregressive_mask_logp_v1"
    ):
        raise ProteinMPNNScoringError(
            "unauthorized_scoring_model", "Formal request model/protocol differs"
        )
    protein_id = request.get("protein_id")
    positions_raw = request.get("uniprot_positions")
    candidate_hashes_raw = request.get("candidate_sequence_hashes")
    projection_hashes_raw = request.get("candidate_projection_sha256")
    candidate_sequences_raw = request.get("candidate_sequences")
    tasks = request.get("tasks")
    if (
        not isinstance(protein_id, str)
        or not isinstance(positions_raw, list)
        or not isinstance(candidate_hashes_raw, list)
        or not isinstance(projection_hashes_raw, list)
        or not isinstance(candidate_sequences_raw, list)
        or not isinstance(tasks, list)
        or not tasks
        or len(candidate_hashes_raw) != len(candidate_sequences_raw)
        or len(projection_hashes_raw) != len(candidate_sequences_raw)
    ):
        raise ProteinMPNNScoringError(
            "invalid_formal_runtime_request", "Formal request fields are incomplete"
        )
    positions = tuple(int(value) for value in positions_raw)
    wt_sequence = validate_protein_sequence(str(request.get("wt_sequence", "")))
    candidate_sequences = tuple(
        validate_protein_sequence(str(value)) for value in candidate_sequences_raw
    )
    candidate_hashes = tuple(str(value) for value in candidate_hashes_raw)
    projection_hashes = tuple(str(value) for value in projection_hashes_raw)
    if (
        len(wt_sequence) != len(positions)
        or any(len(value) != len(positions) for value in candidate_sequences)
        or tuple(sequence_sha256(value) for value in candidate_sequences)
        != projection_hashes
    ):
        raise ProteinMPNNScoringError(
            "formal_candidate_identity_mismatch",
            "Formal projected candidate identity differs",
        )
    candidate_identity = sha256_canonical(
        {"sequence_hashes": list(candidate_hashes)}
    )
    coordinates = {
        "PDB": np.asarray(request.get("pdb_coordinates"), dtype=np.float32),
        "AFDB": np.asarray(request.get("afdb_coordinates"), dtype=np.float32),
    }
    projections = {
        backbone: ScoringBackboneProjection(
            protein_id=protein_id,
            backbone_condition=backbone,
            uniprot_positions=positions,
            wt_sequence_projection=wt_sequence,
            coordinates=value,
        )
        for backbone, value in coordinates.items()
    }
    for projection in projections.values():
        _validate_projection(projection)

    fingerprints: dict[int, set[str]] = {}
    checked_tasks: list[
        tuple[dict[str, Any], DecodingRealization, str]
    ] = []
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
            or binding.get("candidate_count") != len(candidate_hashes)
            or binding.get("candidate_identity_sha256") != candidate_identity
            or binding.get("checkpoint_sha256") != AUTHORIZED_CHECKPOINT_SHA256
            or binding.get("implementation_commit") != AUTHORIZED_IMPLEMENTATION_COMMIT
            or binding.get("scoring_protocol")
            != "stage0_fixed_sequence_autoregressive_mask_logp_v1"
            or binding.get("mask_length") != len(positions)
        ):
            raise ProteinMPNNScoringError(
                "invalid_formal_runtime_request", "Formal task binding differs"
            )
        order = tuple(int(value) for value in order_raw)
        fingerprint = sha256_bytes(np.asarray(order, dtype="<i8").tobytes())
        if binding.get("decoding_realization_sha256") != fingerprint:
            raise ProteinMPNNScoringError(
                "decoding_realization_mismatch", "Formal decoding order SHA differs"
            )
        fingerprints.setdefault(repeat_index, set()).add(fingerprint)
        realization = DecodingRealization(
            protein_id=protein_id,
            repeat_index=repeat_index,
            seed=seed,
            protocol_version=str(binding["scoring_protocol"]),
            algorithm=str(binding["decoding_realization_algorithm"]),
            order=order,
            fingerprint=fingerprint,
        )
        checked_tasks.append((binding, realization, output_filename))
    if any(len(values) != 1 for values in fingerprints.values()):
        raise ProteinMPNNScoringError(
            "formal_realization_pairing_mismatch",
            "PDB and AFDB task realizations differ",
        )

    shard_directory.mkdir(parents=True, exist_ok=True)
    statuses = {"created": 0, "reused_identical": 0}
    sequences = (wt_sequence, *candidate_sequences)
    for ordinal, (binding, realization, output_filename) in enumerate(
        checked_tasks, start=1
    ):
        backbone = str(binding["backbone_condition"])
        scores = runtime.score_sequences(
            projections[backbone],
            sequences,
            realization,
            batch_size=batch_size,
        )
        if len(scores) != len(sequences):
            raise ProteinMPNNScoringError(
                "formal_shard_candidate_count_mismatch",
                "Formal runtime returned an incomplete score vector",
            )
        mask_length = len(positions)

        def score_record(
            score: ProteinMPNNScore, residue_count: int = mask_length
        ) -> dict[str, Any]:
            if not np.isfinite(
                (score.score_sum_logp_mask, score.score_mean_logp_mask)
            ).all():
                raise ProteinMPNNScoringError(
                    "nonfinite_score", "Formal runtime returned a non-finite score"
                )
            return {
                "score_sum_logp_mask": float(score.score_sum_logp_mask),
                "score_mean_logp_mask": float(score.score_mean_logp_mask),
                "scored_residue_count": residue_count,
            }

        payload = {
            "schema_version": "stage0_fixed_probe_scoring_shard_v1",
            "binding": binding,
            "wt_score": score_record(scores[0]),
            "candidate_scores": [
                {
                    "sequence_hash": sequence_hash,
                    **score_record(score),
                }
                for sequence_hash, score in zip(
                    candidate_hashes, scores[1:], strict=True
                )
            ],
            "execution_environment": execution_environment,
        }
        status = _write_runtime_shard(
            shard_directory / output_filename, payload
        )
        statuses[status] += 1
        if progress_callback is not None:
            progress_callback(
                {
                    "protein_id": protein_id,
                    "backbone_condition": backbone,
                    "repeat_index": realization.repeat_index,
                    "completed": ordinal,
                    "total": len(checked_tasks),
                    "write_status": status,
                }
            )
    return {
        "status": "complete",
        "protein_id": protein_id,
        "candidate_count": len(candidate_hashes),
        "completed_shards": len(checked_tasks),
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
