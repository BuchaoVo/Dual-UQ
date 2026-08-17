"""Validated ProteinMPNN generation records and deterministic seed domains."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal

import numpy as np

from dual_uq.core.hashing import sha256_bytes
from dual_uq.models.proteinmpnn import (
    AUTHORIZED_CHECKPOINT_SHA256,
    AUTHORIZED_IMPLEMENTATION_COMMIT,
    PROTEINMPNN_ALPHABET,
    PROTEINMPNN_SCORER_ID,
    ProteinMPNNAdapter,
    ProteinMPNNStructureInput,
    is_authorized_proteinmpnn_adapter,
    sequence_sha256,
    validate_protein_sequence,
    validate_structure_input,
)
from dual_uq.models.scoring import ScorerBinding

GenerationSampleClass = Literal["paired", "independent"]

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAMPLES_PER_CLASS = 128
GENERATION_SCORE_CONTRACT_ID = "proteinmpnn_generation_temperature_0.1_v1"


class GenerationContractError(ValueError):
    """Raised when a generated-sequence contract is malformed or inconsistent."""


def _required_text(value: object, field: str) -> str:
    if type(value) is not str or not value:
        raise GenerationContractError(f"{field} must be a non-empty string")
    return value


def _sha256(value: object, field: str) -> str:
    value = _required_text(value, field)
    if _SHA256.fullmatch(value) is None:
        raise GenerationContractError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _seed_index(sample_index: object) -> int:
    if (
        isinstance(sample_index, bool)
        or not isinstance(sample_index, int)
        or not 0 <= sample_index < _SAMPLES_PER_CLASS
    ):
        raise GenerationContractError("sample_index must be an integer in 0..127")
    return sample_index


def paired_seed(sample_index: int) -> int:
    """Return the paired PDB/AFDB seed for a paired-sample local index."""
    return _seed_index(sample_index)


def independent_seed(condition: str, sample_index: int) -> int:
    """Return a condition-disjoint seed for an independent-sample local index."""
    sample_index = _seed_index(sample_index)
    if condition == "PDB":
        return 1_000_000 + sample_index
    if condition == "AFDB":
        return 2_000_000 + sample_index
    raise GenerationContractError("condition must be PDB or AFDB")


@dataclass(frozen=True, slots=True)
class GenerationSeedSpec:
    """One global sample index and its condition-specific deterministic seeds."""

    sample_index: int
    sample_class: GenerationSampleClass
    pdb_seed: int
    afdb_seed: int


def generation_seed_plan() -> tuple[GenerationSeedSpec, ...]:
    """Return the fixed 128 paired then 128 independent generation samples."""
    paired = tuple(
        GenerationSeedSpec(index, "paired", paired_seed(index), paired_seed(index))
        for index in range(_SAMPLES_PER_CLASS)
    )
    independent = tuple(
        GenerationSeedSpec(
            _SAMPLES_PER_CLASS + index,
            "independent",
            independent_seed("PDB", index),
            independent_seed("AFDB", index),
        )
        for index in range(_SAMPLES_PER_CLASS)
    )
    return (*paired, *independent)


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """One fully identified request to generate a projected ProteinMPNN sequence."""

    protein_id: str
    backbone_condition: str
    structure_sha256: str
    canonical_positions: tuple[int, ...]
    wt_sequence_projection: str
    temperature: float
    sample_index: int
    seed: int
    sample_class: GenerationSampleClass
    decoding_realization: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "protein_id", _required_text(self.protein_id, "protein_id"))
        condition = _required_text(self.backbone_condition, "backbone_condition")
        if condition not in ("PDB", "AFDB"):
            raise GenerationContractError("backbone_condition must be PDB or AFDB")
        object.__setattr__(self, "backbone_condition", condition)
        object.__setattr__(self, "structure_sha256", _sha256(self.structure_sha256, "structure_sha256"))
        object.__setattr__(
            self, "decoding_realization", _sha256(self.decoding_realization, "decoding_realization")
        )
        positions = tuple(self.canonical_positions)
        if (
            not positions
            or any(isinstance(value, bool) or not isinstance(value, int) for value in positions)
            or any(value <= 0 for value in positions)
            or any(right <= left for left, right in pairwise(positions))
        ):
            raise GenerationContractError(
                "canonical_positions must be positive and strictly increasing"
            )
        object.__setattr__(self, "canonical_positions", positions)
        sequence = validate_protein_sequence(self.wt_sequence_projection)
        if len(sequence) != len(positions):
            raise GenerationContractError(
                "sequence-domain lengths must equal canonical_positions length"
            )
        object.__setattr__(self, "wt_sequence_projection", sequence)
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(self.temperature)
            or self.temperature != 0.1
        ):
            raise GenerationContractError("temperature must equal 0.1")
        if self.sample_class not in ("paired", "independent"):
            raise GenerationContractError("sample_class must be paired or independent")
        if (
            isinstance(self.sample_index, bool)
            or not isinstance(self.sample_index, int)
            or not 0 <= self.sample_index < 2 * _SAMPLES_PER_CLASS
        ):
            raise GenerationContractError("sample_index must be an integer in 0..255")
        expected_class: GenerationSampleClass = (
            "paired" if self.sample_index < _SAMPLES_PER_CLASS else "independent"
        )
        if self.sample_class != expected_class:
            raise GenerationContractError("sample_index does not belong to sample_class")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise GenerationContractError("seed must be a non-negative integer")
        local_index = self.sample_index % _SAMPLES_PER_CLASS
        expected_seed = (
            paired_seed(local_index)
            if self.sample_class == "paired"
            else independent_seed(condition, local_index)
        )
        if self.seed != expected_seed:
            raise GenerationContractError("seed does not match the deterministic sample plan")


@dataclass(frozen=True, slots=True)
class GeneratedSequenceRecord:
    """A generated projected sequence with immutable request and scorer identity."""

    request: GenerationRequest
    sequence: str
    sequence_hash: str
    scorer_binding: ScorerBinding

    def __post_init__(self) -> None:
        if not isinstance(self.request, GenerationRequest):
            raise GenerationContractError("request must be a GenerationRequest")
        sequence = validate_protein_sequence(self.sequence)
        if len(sequence) != len(self.request.canonical_positions):
            raise GenerationContractError("sequence-domain lengths must match the request")
        digest = _sha256(self.sequence_hash, "sequence_hash")
        if sequence_sha256(sequence) != digest:
            raise GenerationContractError("sequence_hash does not match sequence content")
        if not isinstance(self.scorer_binding, ScorerBinding):
            raise GenerationContractError("scorer_binding must be a ScorerBinding")
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "sequence_hash", digest)


@dataclass(frozen=True, slots=True)
class ProteinMPNNGenerationStructure:
    """Full-chain generation input bound exactly to a scoring projection."""

    projection: ProteinMPNNStructureInput
    chain_id: str
    chain_sequence: str
    chain_coordinates: np.ndarray
    chain_positions: tuple[int, ...]
    chain_segment_ends: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.projection, ProteinMPNNStructureInput):
            raise GenerationContractError("projection must be a ProteinMPNNStructureInput")
        try:
            validate_structure_input(self.projection)
        except (TypeError, ValueError) as exc:
            raise GenerationContractError("projection is not a valid ProteinMPNN structure") from exc
        object.__setattr__(self, "chain_id", _required_text(self.chain_id, "chain_id"))
        sequence = validate_protein_sequence(self.chain_sequence)
        coordinates = np.asarray(self.chain_coordinates)
        if coordinates.shape != (len(sequence), 4, 3) or np.isinf(coordinates).any():
            raise GenerationContractError(
                "chain_coordinates must contain finite values or explicit NaN missingness"
            )
        positions = tuple(self.chain_positions)
        if (
            len(positions) != self.projection.residue_count
            or any(isinstance(value, bool) or not isinstance(value, int) for value in positions)
            or any(value <= 0 or value > len(sequence) for value in positions)
            or any(right <= left for left, right in pairwise(positions))
        ):
            raise GenerationContractError(
                "chain_positions must be strictly increasing chain indices for the projection"
            )
        segment_ends = (
            (len(sequence),)
            if self.chain_segment_ends is None
            else tuple(self.chain_segment_ends)
        )
        if (
            not segment_ends
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in segment_ends
            )
            or any(right <= left for left, right in pairwise(segment_ends))
            or segment_ends[-1] != len(sequence)
        ):
            raise GenerationContractError(
                "chain_segment_ends must partition the complete chain sequence"
            )
        projection_indices = np.asarray(positions, dtype=np.int64) - 1
        if "".join(sequence[index] for index in projection_indices) != self.projection.wt_sequence_projection:
            raise GenerationContractError("chain_sequence does not match the projected WT sequence")
        if not np.array_equal(
            coordinates[projection_indices], np.asarray(self.projection.coordinates), equal_nan=True
        ):
            raise GenerationContractError("chain_positions do not bind the projection coordinates")
        projection_coordinates = np.asarray(self.projection.coordinates).copy()
        projection_coordinates.setflags(write=False)
        object.__setattr__(
            self,
            "projection",
            ProteinMPNNStructureInput(
                protein_id=self.projection.protein_id,
                backbone_condition=self.projection.backbone_condition,
                uniprot_positions=self.projection.uniprot_positions,
                wt_sequence_projection=self.projection.wt_sequence_projection,
                coordinates=projection_coordinates,
                structure_sha256=self.projection.structure_sha256,
            ),
        )
        coordinates = coordinates.copy()
        coordinates.setflags(write=False)
        object.__setattr__(self, "chain_sequence", sequence)
        object.__setattr__(self, "chain_coordinates", coordinates)
        object.__setattr__(self, "chain_positions", positions)
        object.__setattr__(self, "chain_segment_ends", segment_ends)


@dataclass(frozen=True, slots=True)
class ProteinMPNNGenerationAdapter:
    """Sample a verified ProteinMPNN checkpoint on one fixed common-mask domain."""

    adapter: ProteinMPNNAdapter
    batch_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.adapter, ProteinMPNNAdapter):
            raise GenerationContractError("adapter must be a canonical ProteinMPNNAdapter")
        if (
            self.adapter.implementation_id != AUTHORIZED_IMPLEMENTATION_COMMIT
            or self.adapter.checkpoint_id != AUTHORIZED_CHECKPOINT_SHA256
        ):
            raise GenerationContractError("adapter is not the authorized ProteinMPNN implementation")
        if not is_authorized_proteinmpnn_adapter(self.adapter):
            raise GenerationContractError("adapter lacks authorized loader provenance")
        if not callable(self.adapter._tied_featurize):
            raise GenerationContractError("adapter lacks verified tied_featurize")
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size <= 0
        ):
            raise GenerationContractError("batch_size must be a positive integer")

    @property
    def binding(self) -> ScorerBinding:
        """Return the exact scorer identity persisted in generation shards."""
        return ScorerBinding(
            scorer_id=PROTEINMPNN_SCORER_ID,
            implementation_id=self.adapter.implementation_id,
            checkpoint_id=self.adapter.checkpoint_id,
            score_contract_id=GENERATION_SCORE_CONTRACT_ID,
        )

    @staticmethod
    def _validate_binding(
        request: GenerationRequest, structure: ProteinMPNNGenerationStructure
    ) -> None:
        projection = structure.projection
        if projection.protein_id != request.protein_id:
            raise GenerationContractError("structure protein_id does not match the request")
        if projection.backbone_condition != request.backbone_condition:
            raise GenerationContractError("structure backbone_condition does not match the request")
        if projection.structure_sha256 != request.structure_sha256:
            raise GenerationContractError("structure_sha256 does not match the request")
        if projection.uniprot_positions != request.canonical_positions:
            raise GenerationContractError("structure canonical positions do not match the request")
        if projection.wt_sequence_projection != request.wt_sequence_projection:
            raise GenerationContractError("structure projected WT sequence does not match the request")

    @staticmethod
    def _batch(structure: ProteinMPNNGenerationStructure) -> dict[str, Any]:
        coordinates = structure.chain_coordinates
        batch: dict[str, Any] = {
            "name": structure.projection.protein_id,
            "seq": structure.chain_sequence,
        }
        start = 0
        for segment_index, end in enumerate(structure.chain_segment_ends):
            chain_id = (
                structure.chain_id
                if segment_index == 0
                else f"{structure.chain_id}_{segment_index}"
            )
            segment = coordinates[start:end]
            sequence = structure.chain_sequence[start:end]
            batch[f"seq_chain_{chain_id}"] = sequence
            batch[f"coords_chain_{chain_id}"] = {
                f"N_chain_{chain_id}": segment[:, 0, :].tolist(),
                f"CA_chain_{chain_id}": segment[:, 1, :].tolist(),
                f"C_chain_{chain_id}": segment[:, 2, :].tolist(),
                f"O_chain_{chain_id}": segment[:, 3, :].tolist(),
            }
            start = end
        return batch

    @staticmethod
    def _chain_layout(
        structure: ProteinMPNNGenerationStructure,
    ) -> tuple[tuple[str, int, int], ...]:
        start = 0
        layout = []
        for segment_index, end in enumerate(structure.chain_segment_ends):
            chain_id = (
                structure.chain_id
                if segment_index == 0
                else f"{structure.chain_id}_{segment_index}"
            )
            layout.append((chain_id, start, end))
            start = end
        return tuple(layout)

    def generate(
        self,
        request: GenerationRequest,
        structure: ProteinMPNNGenerationStructure,
        *,
        n_samples: int = 256,
    ) -> tuple[GeneratedSequenceRecord, ...]:
        """Generate projected sequences with the request's explicit RNG seed."""
        if not isinstance(request, GenerationRequest):
            raise GenerationContractError("request must be a GenerationRequest")
        if not isinstance(structure, ProteinMPNNGenerationStructure):
            raise GenerationContractError(
                "structure must be a ProteinMPNNGenerationStructure"
            )
        if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples <= 0:
            raise GenerationContractError("n_samples must be a positive integer")
        if n_samples > 2 * _SAMPLES_PER_CLASS:
            raise GenerationContractError("n_samples must not exceed 256")
        self._validate_binding(request, structure)
        tied_featurize = self.adapter._tied_featurize
        protein_name = request.protein_id
        layout = self._chain_layout(structure)
        projected_positions = set(structure.chain_positions)
        masked_chains = [chain_id for chain_id, _start, _end in layout]
        fixed_positions = {
            chain_id: [
                position - start
                for position in range(start + 1, end + 1)
                if position not in projected_positions
            ]
            for chain_id, start, end in layout
        }
        x, s, mask, _lengths, chain_m, chain_encoding, *_rest = tied_featurize(
            [self._batch(structure)],
            self.adapter.device,
            {protein_name: (masked_chains, [])},
            {protein_name: fixed_positions},
        )
        (
            _chain_list,
            _visible_list,
            _masked_list,
            _masked_chain_lengths,
            chain_m_pos,
            omit_aa_mask,
            residue_idx,
            _dihedral_mask,
            _tied_positions,
            pssm_coef,
            pssm_bias,
            _pssm_log_odds,
            bias_by_res,
            _tied_beta,
        ) = _rest
        results: list[GeneratedSequenceRecord] = []
        alphabet = np.asarray(tuple(PROTEINMPNN_ALPHABET))
        torch = self.adapter.torch
        sample_requests = (
            (request,)
            if n_samples == 1
            else tuple(
                GenerationRequest(
                    protein_id=request.protein_id,
                    backbone_condition=request.backbone_condition,
                    structure_sha256=request.structure_sha256,
                    canonical_positions=request.canonical_positions,
                    wt_sequence_projection=request.wt_sequence_projection,
                    temperature=request.temperature,
                    sample_index=seed_spec.sample_index,
                    seed=(
                        seed_spec.pdb_seed
                        if request.backbone_condition == "PDB"
                        else seed_spec.afdb_seed
                    ),
                    sample_class=seed_spec.sample_class,
                    decoding_realization=request.decoding_realization,
                )
                for seed_spec in generation_seed_plan()[:n_samples]
            )
        )
        for sample_request in sample_requests:
            torch.manual_seed(sample_request.seed)
            random_order = torch.randn(chain_m.shape, device=x.device)
            with torch.inference_mode():
                sampled = self.adapter.model.sample(
                    X=x,
                    randn=random_order,
                    S_true=s,
                    chain_mask=chain_m,
                    chain_encoding_all=chain_encoding,
                    residue_idx=residue_idx,
                    mask=mask,
                    temperature=request.temperature,
                    omit_AAs_np=np.zeros(len(alphabet), dtype=np.float32),
                    bias_AAs_np=np.zeros(len(alphabet), dtype=np.float32),
                    chain_M_pos=chain_m_pos,
                    omit_AA_mask=omit_aa_mask,
                    pssm_coef=pssm_coef,
                    pssm_bias=pssm_bias,
                    pssm_multi=0.0,
                    pssm_log_odds_flag=False,
                    pssm_log_odds_mask=None,
                    pssm_bias_flag=False,
                    bias_by_res=bias_by_res,
                )
            sample_indices = np.asarray(sampled["S"].detach().to("cpu").numpy(), dtype=np.int64)
            order = np.asarray(
                sampled["decoding_order"].detach().to("cpu").numpy(), dtype=np.int64
            )
            if sample_indices.shape != (1, len(structure.chain_sequence)):
                raise GenerationContractError("ProteinMPNN sample shape does not match the structure chain")
            if order.shape != sample_indices.shape or not np.array_equal(
                np.sort(order[0]), np.arange(len(structure.chain_sequence), dtype=np.int64)
            ):
                raise GenerationContractError("ProteinMPNN returned an invalid decoding order")
            if sample_indices.min() < 0 or sample_indices.max() >= len(alphabet):
                raise GenerationContractError("ProteinMPNN returned an invalid amino-acid index")
            full_sequence = "".join(alphabet[sample_indices[0]].tolist())
            sequence = "".join(full_sequence[position - 1] for position in structure.chain_positions)
            decoding_realization = sha256_bytes(order[0].astype("<i8", copy=False).tobytes())
            realized_request = GenerationRequest(
                protein_id=request.protein_id,
                backbone_condition=request.backbone_condition,
                structure_sha256=request.structure_sha256,
                canonical_positions=request.canonical_positions,
                wt_sequence_projection=request.wt_sequence_projection,
                temperature=request.temperature,
                sample_index=sample_request.sample_index,
                seed=sample_request.seed,
                sample_class=sample_request.sample_class,
                decoding_realization=decoding_realization,
            )
            results.append(
                GeneratedSequenceRecord(
                    request=realized_request,
                    sequence=sequence,
                    sequence_hash=sequence_sha256(sequence),
                    scorer_binding=ScorerBinding(
                        scorer_id=PROTEINMPNN_SCORER_ID,
                        implementation_id=self.adapter.implementation_id,
                        checkpoint_id=self.adapter.checkpoint_id,
                        score_contract_id=GENERATION_SCORE_CONTRACT_ID,
                    ),
                )
            )
        return tuple(results)

    def generate_requests_batched(
        self,
        requests: tuple[GenerationRequest, ...],
        structure: ProteinMPNNGenerationStructure,
    ) -> tuple[GeneratedSequenceRecord, ...]:
        """Generate an explicit request batch without changing seed semantics.

        This is an execution optimization for protocols that already provide
        independent request seeds.  Each request gets the same featurized
        structure and its own deterministic random-order tensor; the model is
        invoked once for the batch.  The existing ``generate`` path remains
        the owner of the frozen PDB/AFDB sample-plan behavior.
        """
        if not requests:
            raise GenerationContractError("requests must not be empty")
        if not isinstance(structure, ProteinMPNNGenerationStructure):
            raise GenerationContractError("structure must be a ProteinMPNNGenerationStructure")
        for request in requests:
            if not isinstance(request, GenerationRequest):
                raise GenerationContractError("requests must contain GenerationRequest values")
            self._validate_binding(request, structure)
        tied_featurize = self.adapter._tied_featurize
        protein_name = requests[0].protein_id
        layout = self._chain_layout(structure)
        projected_positions = set(structure.chain_positions)
        masked_chains = [chain_id for chain_id, _start, _end in layout]
        fixed_positions = {
            chain_id: [
                position - start
                for position in range(start + 1, end + 1)
                if position not in projected_positions
            ]
            for chain_id, start, end in layout
        }
        batch = [self._batch(structure) for _ in requests]
        x, s, mask, _lengths, chain_m, chain_encoding, *_rest = tied_featurize(
            batch,
            self.adapter.device,
            {protein_name: (masked_chains, [])},
            {protein_name: fixed_positions},
        )
        (
            _chain_list,
            _visible_list,
            _masked_list,
            _masked_chain_lengths,
            chain_m_pos,
            omit_aa_mask,
            residue_idx,
            _dihedral_mask,
            _tied_positions,
            pssm_coef,
            pssm_bias,
            _pssm_log_odds,
            bias_by_res,
            _tied_beta,
        ) = _rest
        alphabet = np.asarray(tuple(PROTEINMPNN_ALPHABET))
        torch = self.adapter.torch
        random_orders = []
        for request in requests:
            torch.manual_seed(request.seed)
            random_orders.append(torch.randn(chain_m.shape[1:], device=x.device))
        randn = torch.stack(random_orders, dim=0)
        with torch.inference_mode():
            sampled = self.adapter.model.sample(
                X=x,
                randn=randn,
                S_true=s,
                chain_mask=chain_m,
                chain_encoding_all=chain_encoding,
                residue_idx=residue_idx,
                mask=mask,
                temperature=requests[0].temperature,
                omit_AAs_np=np.zeros(len(alphabet), dtype=np.float32),
                bias_AAs_np=np.zeros(len(alphabet), dtype=np.float32),
                chain_M_pos=chain_m_pos,
                omit_AA_mask=omit_aa_mask,
                pssm_coef=pssm_coef,
                pssm_bias=pssm_bias,
                pssm_multi=0.0,
                pssm_log_odds_flag=False,
                pssm_log_odds_mask=None,
                pssm_bias_flag=False,
                bias_by_res=bias_by_res,
            )
        sample_indices = np.asarray(sampled["S"].detach().to("cpu").numpy(), dtype=np.int64)
        orders = np.asarray(sampled["decoding_order"].detach().to("cpu").numpy(), dtype=np.int64)
        expected_shape = (len(requests), len(structure.chain_sequence))
        if sample_indices.shape != expected_shape or orders.shape != expected_shape:
            raise GenerationContractError("ProteinMPNN batched sample shape is invalid")
        results: list[GeneratedSequenceRecord] = []
        for row, request in enumerate(requests):
            order = orders[row]
            if not np.array_equal(np.sort(order), np.arange(len(structure.chain_sequence), dtype=np.int64)):
                raise GenerationContractError("ProteinMPNN returned an invalid decoding order")
            indices = sample_indices[row]
            if indices.min() < 0 or indices.max() >= len(alphabet):
                raise GenerationContractError("ProteinMPNN returned an invalid amino-acid index")
            full_sequence = "".join(alphabet[indices].tolist())
            sequence = "".join(full_sequence[position - 1] for position in structure.chain_positions)
            realized_request = GenerationRequest(
                protein_id=request.protein_id,
                backbone_condition=request.backbone_condition,
                structure_sha256=request.structure_sha256,
                canonical_positions=request.canonical_positions,
                wt_sequence_projection=request.wt_sequence_projection,
                temperature=request.temperature,
                sample_index=request.sample_index,
                seed=request.seed,
                sample_class=request.sample_class,
                decoding_realization=sha256_bytes(order.astype("<i8", copy=False).tobytes()),
            )
            results.append(
                GeneratedSequenceRecord(
                    request=realized_request,
                    sequence=sequence,
                    sequence_hash=sequence_sha256(sequence),
                    scorer_binding=ScorerBinding(
                        scorer_id=PROTEINMPNN_SCORER_ID,
                        implementation_id=self.adapter.implementation_id,
                        checkpoint_id=self.adapter.checkpoint_id,
                        score_contract_id=GENERATION_SCORE_CONTRACT_ID,
                    ),
                )
            )
        return tuple(results)
