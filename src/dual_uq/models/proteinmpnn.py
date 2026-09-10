"""Canonical ProteinMPNN adapter and frozen non-forward score semantics."""

from __future__ import annotations

import importlib.util
import re
import subprocess
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from dual_uq.core.hashing import sha256_bytes, sha256_canonical, sha256_file
from dual_uq.models.scoring import (
    ScorerBinding,
    ScoreRecord,
    ScoreRequest,
    VariantKind,
)

STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
PROTEINMPNN_ALPHABET = f"{STANDARD_AMINO_ACIDS}X"
DECODING_REALIZATION_ALGORITHM = "sha256_ranked_permutation_v1"
PROTEINMPNN_SCORER_ID = "ProteinMPNN"
AUTHORIZED_IMPLEMENTATION_COMMIT = "8907e6671bfbfc92303b5f79c4b5e6ce47cdef57"
AUTHORIZED_CHECKPOINT_SHA256 = (
    "c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd"
)
AUTHORIZED_VANILLA_CHECKPOINTS = {
    "v_48_002.pt": (
        "925f2ca1007bf9b02e0e7f420ff00eb91f50fcc2722f64b42e644ae95adaa131",
        0.02,
    ),
    "v_48_010.pt": (
        "db866fae956a28661f926053d630610c55e9fc4bc03922f2aeeb98a37435ccce",
        0.10,
    ),
    "v_48_020.pt": (AUTHORIZED_CHECKPOINT_SHA256, 0.20),
    "v_48_030.pt": (
        "c34b7bfb38418ea30989fda3314f4781ac4e3920f9825731cf555f1fed44ac66",
        0.30,
    ),
}
_AUTHORIZED_PROTEINMPNN_ADAPTERS: weakref.WeakValueDictionary[
    int, ProteinMPNNAdapter
] = weakref.WeakValueDictionary()
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ProteinMPNNScoringError(ValueError):
    """Structured ProteinMPNN adapter or score-contract failure."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def validate_protein_sequence(sequence: object) -> str:
    """Validate the frozen ProteinMPNN candidate alphabet used by this project."""
    if type(sequence) is not str:
        raise TypeError("sequence must be a string.")
    if not sequence:
        raise ValueError("sequence must not be empty.")
    invalid = sorted(set(sequence).difference(STANDARD_AMINO_ACIDS))
    if invalid:
        raise ValueError(
            "sequence contains characters outside the standard uppercase 20-AA "
            f"alphabet: {invalid}"
        )
    return sequence


def sequence_sha256(sequence: str) -> str:
    """Hash only the validated ASCII sequence, independent of FASTA metadata."""
    return sha256_bytes(validate_protein_sequence(sequence).encode("ascii"))


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
class ProteinMPNNStructureInput:
    """ProteinMPNN computational structure input with an opaque condition label."""

    protein_id: str
    backbone_condition: str
    uniprot_positions: tuple[int, ...]
    wt_sequence_projection: str
    coordinates: np.ndarray
    structure_sha256: str | None = None

    @property
    def residue_count(self) -> int:
        return len(self.uniprot_positions)


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


def validate_structure_input(structure: ProteinMPNNStructureInput) -> None:
    """Validate only ProteinMPNN computational requirements, not pair policy."""
    positions = structure.uniprot_positions
    if (
        not structure.protein_id
        or not structure.backbone_condition
        or not positions
        or any(isinstance(value, bool) or not isinstance(value, int) for value in positions)
        or any(right <= left for left, right in pairwise(positions))
    ):
        raise ProteinMPNNScoringError(
            "scoring_projection_domain_mismatch",
            "Scoring positions must be strictly increasing coordinates",
        )
    coordinates = np.asarray(structure.coordinates)
    if coordinates.shape != (len(positions), 4, 3) or not np.isfinite(coordinates).all():
        raise ProteinMPNNScoringError(
            "invalid_backbone_coordinates",
            "Scoring input requires finite N/CA/C/O coordinates",
        )
    if len(structure.wt_sequence_projection) != len(positions):
        raise ProteinMPNNScoringError(
            "scoring_projection_sequence_mismatch",
            "Projected WT sequence and residue domain lengths differ",
        )
    validate_protein_sequence(structure.wt_sequence_projection)
    if (
        structure.structure_sha256 is not None
        and _SHA256.fullmatch(structure.structure_sha256) is None
    ):
        raise ProteinMPNNScoringError(
            "invalid_structure_identity",
            "Scoring structure SHA must be a lowercase SHA-256 digest",
        )


@dataclass(frozen=True)
class ProteinMPNNAdapter:
    """Authorized loaded model plus frozen ProteinMPNN-native scoring behavior."""

    model: Any
    torch: Any
    device: Any
    checkpoint_num_edges: int
    checkpoint_noise_level: float
    implementation_id: str
    checkpoint_id: str
    _tied_featurize: Callable[..., Any] | None = None

    def score_sequences(
        self,
        structure: ProteinMPNNStructureInput,
        sequences: tuple[str, ...],
        realization: DecodingRealization,
        *,
        batch_size: int,
    ) -> tuple[ProteinMPNNScore, ...]:
        """Score projected sequences without generation or fresh randomness."""
        results: list[ProteinMPNNScore] = []
        for log_probs, target_indices in self._forward_log_probs(
            structure, sequences, realization, batch_size=batch_size
        ):
            results.extend(score_target_log_probs(log_probs, target_indices))
        return tuple(results)

    def probability_distributions(
        self,
        structure: ProteinMPNNStructureInput,
        sequences: tuple[str, ...],
        realization: DecodingRealization,
        *,
        batch_size: int,
    ) -> tuple[np.ndarray, ...]:
        """Return model-native standard-20-AA distributions for each sequence.

        ProteinMPNN exposes a 21-token alphabet including ``X``.  This local
        response analysis uses the normalized standard-20-AA marginal only;
        the frozen target-score path remains unchanged.
        """
        distributions: list[np.ndarray] = []
        for log_probs, _target_indices in self._forward_log_probs(
            structure, sequences, realization, batch_size=batch_size
        ):
            values = np.asarray(log_probs, dtype=np.float64)
            if values.ndim != 3 or values.shape[2] < len(STANDARD_AMINO_ACIDS):
                raise ProteinMPNNScoringError(
                    "invalid_log_probs", "ProteinMPNN output lacks standard 20-AA logits"
                )
            standard = values[..., : len(STANDARD_AMINO_ACIDS)]
            maximum = np.max(standard, axis=-1, keepdims=True)
            weights = np.exp(standard - maximum)
            normalizer = weights.sum(axis=-1, keepdims=True)
            if not np.isfinite(weights).all() or not np.isfinite(normalizer).all() or (normalizer <= 0).any():
                raise ProteinMPNNScoringError(
                    "nonfinite_distribution", "ProteinMPNN standard-AA distribution is invalid"
                )
            distributions.extend(weights / normalizer)
        return tuple(distributions)

    def _forward_log_probs(
        self,
        structure: ProteinMPNNStructureInput,
        sequences: tuple[str, ...],
        realization: DecodingRealization,
        *,
        batch_size: int,
    ) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        """Run the exact frozen forward path and retain logits plus target indices."""
        validate_structure_input(structure)
        if batch_size <= 0:
            raise ProteinMPNNScoringError(
                "invalid_batch_size", "ProteinMPNN batch size must be positive"
            )
        if (
            realization.protein_id != structure.protein_id
            or len(realization.order) != structure.residue_count
        ):
            raise ProteinMPNNScoringError(
                "decoding_realization_mismatch",
                "Decoding realization does not match the scoring input",
            )
        checked_sequences = tuple(validate_protein_sequence(value) for value in sequences)
        if not checked_sequences or any(
            len(value) != structure.residue_count for value in checked_sequences
        ):
            raise ProteinMPNNScoringError(
                "scoring_sequence_length_mismatch",
                "Projected sequence length differs from the scoring domain",
            )
        alphabet_index = {
            amino_acid: index
            for index, amino_acid in enumerate(PROTEINMPNN_ALPHABET)
        }
        results: list[tuple[np.ndarray, np.ndarray]] = []
        torch = self.torch
        coordinates = np.asarray(structure.coordinates, dtype=np.float32)
        positions = np.asarray(structure.uniprot_positions, dtype=np.int64)
        for start in range(0, len(checked_sequences), batch_size):
            chunk = checked_sequences[start : start + batch_size]
            size = len(chunk)
            target_indices = np.asarray(
                [
                    [alphabet_index[amino_acid] for amino_acid in sequence]
                    for sequence in chunk
                ],
                dtype=np.int64,
            )
            x = torch.as_tensor(
                np.broadcast_to(coordinates, (size, *coordinates.shape)).copy(),
                dtype=torch.float32,
                device=self.device,
            )
            s = torch.as_tensor(target_indices, dtype=torch.long, device=self.device)
            mask = torch.ones(
                (size, structure.residue_count),
                dtype=torch.float32,
                device=self.device,
            )
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
            values = np.asarray(log_probs.detach().to("cpu").numpy(), dtype=np.float64)
            if values.ndim != 3 or values.shape[:2] != target_indices.shape or not np.isfinite(values).all():
                raise ProteinMPNNScoringError(
                    "invalid_log_probs", "ProteinMPNN log_probs have invalid shape or values"
                )
            results.append((values, target_indices))
        return tuple(results)


def is_authorized_proteinmpnn_adapter(adapter: object) -> bool:
    """Return whether this exact adapter instance came from the verified loader."""
    return (
        isinstance(adapter, ProteinMPNNAdapter)
        and _AUTHORIZED_PROTEINMPNN_ADAPTERS.get(id(adapter)) is adapter
    )


@dataclass(frozen=True)
class ProteinMPNNScorer:
    """SequenceScorer implementation over a model-independent request.

    The resolver materializes the already-bound comparable structural domain.
    ProteinMPNN-native decoding order, projected sequences, tensors, batching,
    and model calls remain entirely behind this adapter.
    """

    adapter: ProteinMPNNAdapter
    projection_resolver: Callable[[ScoreRequest], ProteinMPNNStructureInput]
    score_contract_id: str
    batch_size: int

    def __post_init__(self) -> None:
        if not self.score_contract_id:
            raise ProteinMPNNScoringError(
                "invalid_score_contract", "ProteinMPNN score contract is required"
            )
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size <= 0
        ):
            raise ProteinMPNNScoringError(
                "invalid_batch_size", "ProteinMPNN batch size must be positive"
            )

    @property
    def binding(self) -> ScorerBinding:
        """Return identity carried by the concrete loaded/scaffolded adapter."""
        try:
            return ScorerBinding(
                scorer_id=PROTEINMPNN_SCORER_ID,
                implementation_id=self.adapter.implementation_id,
                checkpoint_id=self.adapter.checkpoint_id,
                score_contract_id=self.score_contract_id,
            )
        except (AttributeError, ValueError) as exc:
            raise ProteinMPNNScoringError(
                "invalid_scorer_binding",
                "ProteinMPNN adapter lacks a valid implementation binding",
            ) from exc

    @staticmethod
    def _project_sequence(sequence: str, positions: tuple[int, ...]) -> str:
        try:
            return "".join(sequence[position - 1] for position in positions)
        except IndexError as exc:
            raise ProteinMPNNScoringError(
                "candidate_projection_out_of_range",
                "Candidate sequence does not cover the scoring domain",
            ) from exc

    def score(self, request: ScoreRequest) -> tuple[ScoreRecord, ...]:
        """Resolve one request and return one WT plus all probe measurements."""
        if request.score_contract_id != self.binding.score_contract_id:
            raise ProteinMPNNScoringError(
                "score_contract_mismatch",
                "Request score contract does not match the ProteinMPNN scorer",
            )
        projection = self.projection_resolver(request)
        validate_structure_input(projection)
        if (
            projection.protein_id != request.protein_id
            or projection.backbone_condition != request.condition.condition_id
            or projection.uniprot_positions != request.canonical_positions
        ):
            raise ProteinMPNNScoringError(
                "scoring_projection_domain_mismatch",
                "Resolved ProteinMPNN projection does not match the request",
            )
        if projection.structure_sha256 != request.condition.structure_sha256:
            raise ProteinMPNNScoringError(
                "scoring_projection_structure_mismatch",
                "Resolved coordinates are not bound to the requested structure SHA",
            )

        variants = request.candidate_collection.variants
        projected_sequences = tuple(
            self._project_sequence(variant.sequence, request.canonical_positions)
            for variant in variants
        )
        if projected_sequences[0] != projection.wt_sequence_projection:
            raise ProteinMPNNScoringError(
                "scoring_projection_sequence_mismatch",
                "Resolved WT projection does not match the request collection",
            )
        realization = make_decoding_realization(
            protein_id=request.protein_id,
            mask_length=len(request.canonical_positions),
            repeat_index=request.repeat_index,
            seed=request.seed,
            protocol_version=request.score_contract_id,
        )
        if (
            realization.algorithm != request.realization_algorithm
            or realization.fingerprint != request.realization_id
        ):
            raise ProteinMPNNScoringError(
                "decoding_realization_mismatch",
                "Request realization does not match ProteinMPNN interpretation",
            )
        scores = self.adapter.score_sequences(
            projection,
            projected_sequences,
            realization,
            batch_size=self.batch_size,
        )
        if len(scores) != request.result_count:
            raise ProteinMPNNScoringError(
                "scoring_result_count_mismatch",
                "ProteinMPNN did not return one score per collection member",
            )
        binding = self.binding
        return tuple(
            ScoreRecord(
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
                scorer_id=binding.scorer_id,
                implementation_id=binding.implementation_id,
                checkpoint_id=binding.checkpoint_id,
                score_contract_id=request.score_contract_id,
                score_sum_logp_mask=score.score_sum_logp_mask,
                score_mean_logp_mask=score.score_mean_logp_mask,
                scored_residue_count=len(request.canonical_positions),
            )
            for variant, score in zip(variants, scores, strict=True)
        )


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
        raise ProteinMPNNScoringError(
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
    fingerprint = sha256_bytes(np.asarray(order, dtype="<i8").tobytes())
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
        raise ProteinMPNNScoringError(
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
        raise ProteinMPNNScoringError(
            "invalid_log_probs", "ProteinMPNN log_probs must have shape [B,L,A]"
        )
    if not np.isfinite(probabilities).all():
        raise ProteinMPNNScoringError(
            "nonfinite_score", "ProteinMPNN log_probs contain non-finite values"
        )
    if target_indices.ndim != 2 or target_indices.shape != probabilities.shape[:2]:
        raise ProteinMPNNScoringError(
            "score_shape_mismatch", "Target indices must match log_probs [B,L]"
        )
    if not np.issubdtype(target_indices.dtype, np.integer):
        raise ProteinMPNNScoringError(
            "invalid_target_indices", "Target amino-acid indices must be integers"
        )
    if (
        target_indices.size == 0
        or target_indices.min() < 0
        or target_indices.max() >= probabilities.shape[2]
    ):
        raise ProteinMPNNScoringError(
            "target_index_out_of_range", "Target amino-acid index is out of range"
        )
    gathered = np.take_along_axis(
        probabilities, target_indices[..., np.newaxis], axis=-1
    ).squeeze(-1)
    if not np.isfinite(gathered).all():
        raise ProteinMPNNScoringError(
            "nonfinite_score", "Target log-probability score is not finite"
        )
    return tuple(
        ProteinMPNNScore(
            score_sum_logp_mask=float(row.sum(dtype=np.float64)),
            score_mean_logp_mask=float(row.mean(dtype=np.float64)),
        )
        for row in gathered
    )


def load_authorized_proteinmpnn_adapter(
    *,
    implementation_path: Path,
    checkpoint_path: Path,
    device_name: str,
    backbone_noise: float,
    allow_noncode_worktree_drift: bool = False,
) -> ProteinMPNNAdapter:
    """Lazy-load the hash-verified ProteinMPNN implementation and checkpoint."""
    try:
        implementation_commit = subprocess.check_output(
            ["git", "-C", str(implementation_path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProteinMPNNScoringError(
            "implementation_worktree_unavailable",
            "Cannot verify ProteinMPNN implementation commit",
        ) from exc
    if implementation_commit != AUTHORIZED_IMPLEMENTATION_COMMIT:
        raise ProteinMPNNScoringError(
            "implementation_commit_mismatch",
            "ProteinMPNN implementation commit differs from the authorization",
        )
    if not implementation_worktree_is_clean(implementation_path):
        code_is_clean = subprocess.run(
            ["git", "-C", str(implementation_path), "diff", "--quiet", "HEAD", "--", "*.py"],
            check=False,
        ).returncode == 0
        if not allow_noncode_worktree_drift or not code_is_clean:
            raise ProteinMPNNScoringError(
                "implementation_worktree_dirty",
                "ProteinMPNN implementation contains tracked-file drift",
            )
    try:
        checkpoint_id = sha256_file(checkpoint_path)
    except OSError as exc:
        raise ProteinMPNNScoringError(
            "checkpoint_unavailable", "Authorized ProteinMPNN checkpoint is unavailable"
        ) from exc
    authorized = next(
        (
            (filename, training_noise)
            for filename, (digest, training_noise) in AUTHORIZED_VANILLA_CHECKPOINTS.items()
            if digest == checkpoint_id
        ),
        None,
    )
    if authorized is None:
        raise ProteinMPNNScoringError(
            "checkpoint_hash_mismatch",
            "ProteinMPNN checkpoint differs from the authorization",
        )
    authorized_filename, authorized_training_noise = authorized
    try:
        import torch
    except ImportError as exc:
        raise ProteinMPNNScoringError(
            "torch_unavailable", "ProteinMPNN runtime requires the model environment"
        ) from exc
    utility_path = implementation_path / "protein_mpnn_utils.py"
    spec = importlib.util.spec_from_file_location(
        "_dual_uq_authorized_protein_mpnn_utils", utility_path
    )
    if spec is None or spec.loader is None:
        raise ProteinMPNNScoringError(
            "implementation_import_failure",
            "Cannot load authorized ProteinMPNN utilities",
        )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        tied_featurize = module.tied_featurize
        checkpoint = torch.load(checkpoint_path, map_location=device_name)
        if not np.isclose(
            float(checkpoint["noise_level"]),
            authorized_training_noise,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ProteinMPNNScoringError(
                "checkpoint_metadata_mismatch",
                f"ProteinMPNN {authorized_filename} training-noise metadata is invalid",
            )
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
    except (AttributeError, OSError, KeyError, RuntimeError, ValueError) as exc:
        raise ProteinMPNNScoringError(
            "model_load_failure", "Authorized ProteinMPNN checkpoint failed to load"
        ) from exc
    adapter = ProteinMPNNAdapter(
        model=model,
        torch=torch,
        device=device,
        checkpoint_num_edges=int(checkpoint["num_edges"]),
        checkpoint_noise_level=float(checkpoint["noise_level"]),
        implementation_id=implementation_commit,
        checkpoint_id=checkpoint_id,
        _tied_featurize=tied_featurize,
    )
    _AUTHORIZED_PROTEINMPNN_ADAPTERS[id(adapter)] = adapter
    return adapter
