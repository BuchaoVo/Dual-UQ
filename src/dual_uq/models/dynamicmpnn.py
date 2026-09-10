"""Project-facing adapter for the official DynamicMPNN two-state model.

The adapter deliberately keeps the official implementation under
``third_party/DynamicMPNN``.  This module owns only the frozen input contract,
source/checkpoint identity checks, and conversion of the canonical two-state
input into the official single-chain-k2 featurizer.
"""

from __future__ import annotations

import hashlib
import importlib
import random
import re
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dual_uq.core.hashing import sha256_file

STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
OFFICIAL_SOURCE_COMMIT = "1f3e326c0f4d275ee8b3918e4726e19d3eef6c3f"
DEFAULT_CHECKPOINT_RELATIVE_PATH = "third_party/DynamicMPNN/checkpoints/single_chain_k2.ckpt"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def build_natural_decoding_order(length: int) -> tuple[int, ...]:
    """Return the explicit residue-index order used by pilot scoring."""

    if isinstance(length, bool) or length <= 0:
        raise ValueError("decoding order length must be positive")
    return tuple(range(int(length)))


def teacher_forcing_edge_context(
    sequence: np.ndarray,
    edge_index: np.ndarray,
) -> np.ndarray:
    """Return native prefix token IDs for the upstream causal edge convention.

    ``src < dst`` edges may carry the source token.  Backward/future edges are
    represented by ``-1`` and are replaced by the encoder embedding in the
    DynamicMPNN decoder.  This small array-level helper makes the causal
    contract testable without importing the optional PyTorch stack.
    """

    values = np.asarray(sequence)
    edges = np.asarray(edge_index)
    if values.ndim != 1:
        raise ValueError("sequence must be one-dimensional")
    if edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if np.any(edges < 0) or np.any(edges >= len(values)):
        raise ValueError("edge_index contains an out-of-range residue")
    context = np.full(edges.shape[1], -1, dtype=values.dtype)
    prefix = edges[0] < edges[1]
    context[prefix] = values[edges[0, prefix]]
    return context


class DynamicMPNNInputError(ValueError):
    """A scientific input cannot be represented by the two-state model."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


@dataclass(frozen=True)
class DynamicMPNNInputCase:
    """One aligned APO/HOLO case on the full canonical residue axis.

    ``coordinates`` is ``[residue, state, atom, xyz]`` with N/CA/C atoms.
    Missing coordinates remain non-finite in this project-level object and are
    represented as virtual nodes only at the official featurizer boundary.
    """

    protein_id: str
    pair_id: str
    canonical_positions: tuple[int, ...]
    sequences: tuple[str, str]
    coordinates: np.ndarray
    coordinate_present: np.ndarray
    chain_ids: tuple[str, str]
    structure_sha256: tuple[str, str]

    @property
    def residue_count(self) -> int:
        return len(self.canonical_positions)


@dataclass(frozen=True)
class DynamicMPNNInputValidation:
    evaluable: bool
    reason: str | None
    residue_count: int
    missing_coordinate_count: int


@dataclass(frozen=True)
class DynamicMPNNSampleResult:
    protein_id: str
    pair_id: str
    seed: int
    sequences: tuple[str, ...]
    residue_count: int
    checkpoint_sha256: str
    source_commit: str


@dataclass(frozen=True)
class DynamicMPNNScoringResult:
    """Logits and explicit order returned by one scoring mode."""

    logits: Any
    valid_mask: Any
    decoding_order: tuple[tuple[int, ...], ...]


def _raise(reason: str, message: str) -> None:
    raise DynamicMPNNInputError(reason, message)


def _check_digest(value: str, field: str) -> None:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def validate_dynamic_input(case: DynamicMPNNInputCase) -> DynamicMPNNInputValidation:
    """Validate strict shared-axis and two-conformation semantics.

    The mapped canonical interval must be contiguous.  This prevents the
    native featurizer's gap-elimination behavior from silently compressing a
    true UniProt gap inside the model domain.  Missing coordinates are allowed when at least one state has
    a coordinate at that position; a state with no coordinates at all is not a
    valid two-state input.
    """

    if not case.protein_id or not case.pair_id:
        _raise("missing_identity", "protein and pair identities are required")
    positions = tuple(case.canonical_positions)
    if any(right != left + 1 for left, right in zip(positions, positions[1:])):
        _raise(
            "canonical_axis_mismatch",
            "DynamicMPNN requires the complete contiguous canonical residue axis",
        )
    if len(case.sequences) != 2:
        _raise("conformation_count_mismatch", "exactly APO and HOLO sequences are required")
    if any(type(seq) is not str or not seq for seq in case.sequences):
        _raise("invalid_sequence", "both conformation sequences must be non-empty strings")
    if any(set(seq).difference(STANDARD_AMINO_ACIDS) for seq in case.sequences):
        _raise("nonstandard_sequence", "sequences must use the standard uppercase 20-AA alphabet")
    if case.sequences[0] != case.sequences[1]:
        _raise(
            "conformation_sequence_mismatch",
            "APO and HOLO must share the frozen canonical sequence",
        )
    length = len(positions)
    if any(len(seq) != length for seq in case.sequences):
        _raise("sequence_axis_length_mismatch", "sequence length differs from canonical axis")

    coordinates = np.asarray(case.coordinates)
    present = np.asarray(case.coordinate_present)
    if coordinates.shape != (length, 2, 3, 3):
        _raise("invalid_backbone_coordinates", "coordinates must have shape [L,2,3,3] for N/CA/C")
    if present.shape != (length, 2) or present.dtype != np.bool_:
        _raise("invalid_coordinate_mask", "coordinate_present must be a boolean [L,2] mask")
    for state in range(2):
        if not bool(present[:, state].any()):
            _raise(
                "conformation_coordinate_state_missing",
                "each conformation must contribute at least one coordinate-bearing residue",
            )
    if np.any(~present[:, 0] & ~present[:, 1]):
        _raise(
            "both_conformations_missing_coordinate",
            "a canonical position missing in both states cannot be represented without dropping the axis",
        )
    finite = np.isfinite(coordinates).all(axis=(2, 3))
    if np.any(present & ~finite) or np.any(~present & finite):
        _raise(
            "coordinate_mask_mismatch",
            "coordinate presence must exactly agree with finite N/CA/C coordinates",
        )
    for digest in case.structure_sha256:
        _check_digest(digest, "structure_sha256")
    if len(case.chain_ids) != 2 or any(not isinstance(chain, str) or not chain for chain in case.chain_ids):
        _raise("chain_identity_missing", "APO and HOLO chain identities are required")
    return DynamicMPNNInputValidation(
        evaluable=True,
        reason=None,
        residue_count=length,
        missing_coordinate_count=int((~present).sum()),
    )


def _source_commit(source_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot resolve DynamicMPNN source commit") from exc


def _source_is_clean(source_root: Path) -> bool:
    status = subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--porcelain=v1", "--untracked-files=no"],
        text=True,
        stderr=subprocess.STDOUT,
    )
    return not status.strip()


class DynamicMPNNAdapter:
    """Lazy-loading official DynamicMPNN single-chain-k2 adapter."""

    def __init__(
        self,
        *,
        repository_root: Path | None = None,
        checkpoint_path: Path | None = None,
        device: str = "auto",
    ) -> None:
        self.repository_root = repository_root or Path(__file__).resolve().parents[3]
        self.source_root = self.repository_root / "third_party/DynamicMPNN"
        self.checkpoint_path = checkpoint_path or (
            self.repository_root / DEFAULT_CHECKPOINT_RELATIVE_PATH
        )
        self.device_name = device
        self._model: Any | None = None
        self._featuriser: Any | None = None
        self._torch: Any | None = None
        self._graph_cache: dict[str, Any] = {}
        self._graph_cache_residues = 0
        # Keep only a small validation prefix cached; feature tensors are
        # substantially larger than raw coordinates and full-cohort caching
        # can exhaust host RAM on the StructCal projection.
        self._graph_cache_residue_limit = 10_000
        self.checkpoint_sha256 = sha256_file(self.checkpoint_path)
        self.source_commit = _source_commit(self.source_root)
        if self.source_commit != OFFICIAL_SOURCE_COMMIT:
            raise RuntimeError(
                f"DynamicMPNN source drift: expected {OFFICIAL_SOURCE_COMMIT}, got {self.source_commit}"
            )
        if not _source_is_clean(self.source_root):
            raise RuntimeError("DynamicMPNN source worktree contains tracked-file drift")

    @property
    def checkpoint_relative_path(self) -> str:
        return self.checkpoint_path.relative_to(self.repository_root).as_posix()

    def _load(self, n_samples: int) -> None:
        if self._model is not None and getattr(self._model, "n_samples", None) == n_samples:
            return
        source_path = str(self.source_root / "src")
        if source_path not in sys.path:
            sys.path.insert(0, source_path)
        import torch
        from omegaconf import OmegaConf

        # The pilot compares paired views at probability level.  Deterministic
        # scatter/reduction kernels keep repeated equivalent forwards from
        # turning numerical reduction order into apparent structural response.
        torch.use_deterministic_algorithms(True)

        # The official package initializer imports the training stack
        # (PyTorch Lightning and Hydra callbacks).  Sampling needs only the
        # model/features subpackages, so expose the official source as a
        # namespace package and avoid importing training-only side effects.
        package = sys.modules.get("dynamicmpnn")
        if package is None or not hasattr(package, "__path__"):
            package = types.ModuleType("dynamicmpnn")
            package.__path__ = [str(self.source_root / "src" / "dynamicmpnn")]
            sys.modules["dynamicmpnn"] = package
        ProteinGraphFeaturiserSingleChain = importlib.import_module(
            "dynamicmpnn.features.featurizer"
        ).ProteinGraphFeaturiserSingleChain
        DynamicMPNN = importlib.import_module(
            "dynamicmpnn.modules.models.models"
        ).DynamicMPNN

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        cfg = checkpoint["hyper_parameters"]["cfg"]
        cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        feature_cfg = cfg.features
        self._featuriser = ProteinGraphFeaturiserSingleChain(
            representation=feature_cfg.representation,
            scalar_node_features=list(feature_cfg.scalar_node_features),
            vector_node_features=list(feature_cfg.vector_node_features),
            edge_types=list(feature_cfg.edge_types),
            scalar_edge_features=list(feature_cfg.scalar_edge_features),
            vector_edge_features=list(feature_cfg.vector_edge_features),
            k=2,
            split="test",
            noise_scale=0.0,
            device="cpu",
        )
        model_cfg = dict(OmegaConf.to_container(cfg.model, resolve=True))
        model_cfg.pop("_target_", None)
        model_cfg["pooling_strategy"] = "single_chain_k"
        model_cfg["n_samples"] = n_samples
        model = DynamicMPNN(**model_cfg)
        state_dict = {
            key.removeprefix("GNN_model."): value
            for key, value in checkpoint["state_dict"].items()
            if key.startswith("GNN_model.")
        }
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        device = torch.device(
            "cuda" if self.device_name == "auto" and torch.cuda.is_available() else self.device_name
            if self.device_name != "auto"
            else "cpu"
        )
        self._model = model.to(device)
        self._torch = torch

    def _build_graph_data(self, case: DynamicMPNNInputCase, *, cache: bool = True) -> Any:
        validate_dynamic_input(case)
        if self._torch is None or self._featuriser is None:
            raise RuntimeError("DynamicMPNN adapter has not been loaded")
        from dynamicmpnn.types import BASE_AMINO_ACIDS, FILL_VALUE
        from torch_geometric.data import Data

        torch = self._torch
        cache_key = None
        if cache:
            digest = hashlib.sha256()
            digest.update(case.sequences[0].encode())
            digest.update(np.ascontiguousarray(case.coordinates).tobytes())
            digest.update(np.ascontiguousarray(case.coordinate_present).tobytes())
            cache_key = digest.hexdigest()
            cached = self._graph_cache.get(cache_key)
            if cached is not None:
                return cached
        aa_index = {aa: index for index, aa in enumerate(BASE_AMINO_ACIDS)}
        coords = np.asarray(case.coordinates, dtype=np.float32).copy()
        present = np.asarray(case.coordinate_present, dtype=bool)
        coords[~present] = FILL_VALUE
        residue_type = torch.tensor(
            [aa_index[aa] for aa in case.sequences[0]], dtype=torch.long
        )
        conformations = []
        for state in range(2):
            conformations.append(
                Data(
                    coords=torch.from_numpy(coords[:, state]),
                    residue_type=residue_type.clone(),
                    residue_index=torch.arange(1, case.residue_count + 1, dtype=torch.long),
                )
            )
        protein = type(
            "DynamicProteinInput",
            (),
            {
                "cluster_members": ["APO_A", "HOLO_A"],
                "pyg_dict": {"APO_A": conformations[0], "HOLO_A": conformations[1]},
            },
        )()
        graph = self._featuriser(protein, pdb_code=case.pair_id)
        if graph is None:
            raise DynamicMPNNInputError(
                "dynamicmpnn_featurizer_unavailable",
                "official DynamicMPNN featurizer rejected the aligned two-state input",
            )
        if cache_key is not None and self._graph_cache_residues + case.residue_count <= self._graph_cache_residue_limit:
            self._graph_cache[cache_key] = graph
            self._graph_cache_residues += case.residue_count
        return graph

    def _build_graph(self, case: DynamicMPNNInputCase) -> Any:
        """Build the one-case batch used by the public scoring methods."""

        from torch_geometric.data import Batch

        return Batch.from_data_list([self._build_graph_data(case)])

    def _teacher_forced_forward(self, graph: Any, target: Any) -> tuple[Any, Any]:
        """Run one vectorized native-prefix decode on an already-built graph."""

        if self._torch is None or self._model is None:
            raise RuntimeError("DynamicMPNN adapter has not been loaded")
        from dynamicmpnn.modules.models.models import (
            pool_conformations,
            pool_edges_conformations,
        )

        torch = self._torch
        model = self._model
        edge_index = graph.edge_index
        h_V = (graph.node_s, graph.node_v)
        h_E = (graph.edge_s, graph.edge_v)
        virtual_mask = graph.virtual_mask
        k = h_V[0].shape[1]
        h_V_s, h_V_v, h_E_s, h_E_v = [], [], [], []
        for index in range(k):
            node = model.W_v((h_V[0][:, index], h_V[1][:, index]))
            edge = model.W_e((h_E[0][:, index], h_E[1][:, index]))
            h_V_s.append(node[0])
            h_V_v.append(node[1])
            h_E_s.append(edge[0])
            h_E_v.append(edge[1])
        h_V = (torch.stack(h_V_s, dim=1), torch.stack(h_V_v, dim=1))
        h_E = (torch.stack(h_E_s, dim=1), torch.stack(h_E_v, dim=1))
        for layer in model.encoder_layers:
            h_V, h_E = layer(h_V, edge_index, h_E)
        h_V_decoder = pool_conformations(h_V, virtual_mask)
        src, dst = edge_index
        edge_mask = virtual_mask[src] & virtual_mask[dst]
        h_E_decoder = pool_edges_conformations(h_E, edge_mask)
        for layer in model.pooled_encoder_layers:
            h_V_decoder, h_E_decoder = layer(h_V_decoder, edge_index, h_E_decoder)
        encoder_embeddings = h_V_decoder
        target = target.to(device=edge_index.device, dtype=torch.long)
        if target.ndim != 1 or target.shape[0] != h_V_decoder[0].shape[0]:
            raise ValueError("teacher-forcing target length does not match graph")
        # ``edge_index`` is global for a PyG batch.  Compare local residue
        # indices so disconnected examples retain the exact natural-order
        # ``src < dst`` causal convention used by the official decoder.
        batch_index = getattr(graph, "batch", None)
        if batch_index is None:
            batch_index = torch.zeros(h_V_decoder[0].shape[0], dtype=torch.long, device=src.device)
        ptr = getattr(graph, "ptr", None)
        if ptr is None:
            ptr = torch.tensor([0, h_V_decoder[0].shape[0]], dtype=torch.long, device=src.device)
        local_src = src - ptr[batch_index[src]]
        local_dst = dst - ptr[batch_index[dst]]
        h_S = model.W_s(target)[src]
        h_S = h_S.masked_fill((local_src >= local_dst).unsqueeze(-1), 0)
        h_E_decoder = (torch.cat([h_E_decoder[0], h_S], dim=-1), h_E_decoder[1])
        for layer in model.decoder_layers:
            h_V_decoder, h_E_decoder = layer(
                h_V_decoder,
                edge_index,
                h_E_decoder,
                autoregressive_x=encoder_embeddings,
            )
        return model.W_out(h_V_decoder), torch.ones(
            target.shape[0], dtype=torch.bool, device=target.device
        )

    def teacher_forced_batch(
        self,
        cases: tuple[DynamicMPNNInputCase, ...] | list[DynamicMPNNInputCase],
        *,
        sequences: tuple[str, ...] | list[str] | None = None,
        inference: bool = False,
        cache_graph: bool | tuple[bool, ...] | list[bool] = True,
    ) -> tuple[DynamicMPNNScoringResult, ...]:
        """Score several independent cases in one PyG/decoder forward.

        Cases remain disconnected graphs and are split back into per-case
        logits.  This is a throughput-only adapter operation: every case
        uses the same official layers, checkpoint parameters, native prefix
        tokens and natural residue-index causal edge rule as ``teacher_forced``.
        """

        cases = tuple(cases)
        if not cases:
            raise ValueError("teacher-forcing batch must not be empty")
        if sequences is None:
            sequences = tuple(case.sequences[0] for case in cases)
        else:
            sequences = tuple(sequences)
        if len(sequences) != len(cases):
            raise ValueError("one teacher-forcing sequence is required per case")
        self._load(1)
        assert self._torch is not None and self._model is not None
        from dynamicmpnn.types import BASE_AMINO_ACIDS
        from torch_geometric.data import Batch

        aa_index = {aa: index for index, aa in enumerate(BASE_AMINO_ACIDS)}
        if isinstance(cache_graph, bool):
            cache_flags = (cache_graph,) * len(cases)
        else:
            cache_flags = tuple(cache_graph)
            if len(cache_flags) != len(cases):
                raise ValueError("one graph-cache flag is required per case")
        graphs = [
            self._build_graph_data(case, cache=flag)
            for case, flag in zip(cases, cache_flags)
        ]
        target_values: list[int] = []
        for case, sequence in zip(cases, sequences):
            if len(sequence) != case.residue_count or any(aa not in STANDARD_AMINO_ACIDS for aa in sequence):
                raise ValueError("teacher-forcing sequence must be a standard-AA sequence of case length")
            target_values.extend(aa_index[aa] for aa in sequence)
        graph = Batch.from_data_list(graphs).to(next(self._model.parameters()).device)
        target = self._torch.tensor(target_values, dtype=self._torch.long, device=graph.edge_index.device)
        if inference:
            with self._torch.inference_mode():
                logits, valid_mask = self._teacher_forced_forward(graph, target)
        else:
            logits, valid_mask = self._teacher_forced_forward(graph, target)
        lengths = tuple(case.residue_count for case in cases)
        if int(valid_mask.shape[0]) != sum(lengths):
            raise RuntimeError("batched teacher-forcing output length does not match cases")
        outputs: list[DynamicMPNNScoringResult] = []
        offset = 0
        for length in lengths:
            outputs.append(
                DynamicMPNNScoringResult(
                    logits=logits[offset : offset + length],
                    valid_mask=valid_mask[offset : offset + length],
                    decoding_order=(build_natural_decoding_order(length),),
                )
            )
            offset += length
        return tuple(outputs)

    def set_training_mode(self, training: bool) -> None:
        """Set model mode; pilot training deliberately disables dropout."""

        self._load(1)
        assert self._model is not None
        # Both scoring/training paths use deterministic LayerNorm-only behavior;
        # paired-view JSD must not measure independent dropout masks.
        del training
        self._model.eval()

    def trainable_parameters(self) -> tuple[Any, ...]:
        self._load(1)
        assert self._model is not None
        return tuple(self._model.parameters())

    def state_dict(self) -> Any:
        self._load(1)
        assert self._model is not None
        return self._model.state_dict()

    def load_state_dict(self, state_dict: Any) -> None:
        self._load(1)
        assert self._model is not None
        self._model.load_state_dict(state_dict, strict=True)

    def official_forward(self, case: DynamicMPNNInputCase) -> DynamicMPNNScoringResult:
        """Reproduce the exact upstream single-chain-k2 forward semantics."""

        self._load(1)
        graph = self._build_graph(case)
        assert self._torch is not None and self._model is not None
        graph = graph.to(next(self._model.parameters()).device)
        with self._torch.inference_mode():
            logits, valid_mask = self._model(graph)
        length = int(valid_mask.shape[0])
        return DynamicMPNNScoringResult(
            logits=logits,
            valid_mask=valid_mask,
            decoding_order=(build_natural_decoding_order(length),),
        )

    def teacher_forced(
        self,
        case: DynamicMPNNInputCase,
        *,
        sequence: str | None = None,
        inference: bool = False,
        cache_graph: bool = True,
    ) -> DynamicMPNNScoringResult:
        """Score native sequence prefixes in one causal vectorized pass."""

        self._load(1)
        graph = self._build_graph_data(case, cache=cache_graph)
        from torch_geometric.data import Batch

        graph = Batch.from_data_list([graph])
        assert self._torch is not None and self._model is not None
        target_sequence = sequence or case.sequences[0]
        if len(target_sequence) != case.residue_count or any(
            aa not in STANDARD_AMINO_ACIDS for aa in target_sequence
        ):
            raise ValueError("teacher-forcing sequence must be a standard-AA sequence of case length")
        from dynamicmpnn.types import BASE_AMINO_ACIDS

        aa_index = {aa: index for index, aa in enumerate(BASE_AMINO_ACIDS)}
        target = self._torch.tensor(
            [aa_index[aa] for aa in target_sequence], dtype=self._torch.long
        )
        graph = graph.to(next(self._model.parameters()).device)
        if inference:
            with self._torch.inference_mode():
                logits, valid_mask = self._teacher_forced_forward(graph, target)
        else:
            logits, valid_mask = self._teacher_forced_forward(graph, target)
        length = int(valid_mask.shape[0])
        return DynamicMPNNScoringResult(
            logits=logits,
            valid_mask=valid_mask,
            decoding_order=(build_natural_decoding_order(length),),
        )

    def generate(self, case: DynamicMPNNInputCase, *, seed: int, temperature: float = 0.1) -> str:
        """Generate one sequence with the official natural-order sampler."""
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self._load(1)
        assert self._model is not None
        original_temperature = float(self._model.temperature)
        self._model.temperature = float(temperature)
        try:
            result = self.sample(case, n_samples=1, seed=seed)
        finally:
            self._model.temperature = original_temperature
        if not result.sequences:
            raise RuntimeError("DynamicMPNN generated no sequences")
        return result.sequences[0]

    def sample(
        self,
        case: DynamicMPNNInputCase,
        *,
        n_samples: int = 64,
        seed: int,
    ) -> DynamicMPNNSampleResult:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self._load(n_samples)
        graph = self._build_graph(case)
        torch = self._torch
        assert torch is not None and self._model is not None
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        graph = graph.to(next(self._model.parameters()).device)
        with torch.inference_mode():
            sampled = self._model.sample(graph)
        from dynamicmpnn.types import BASE_AMINO_ACIDS

        sequences = tuple(
            "".join(BASE_AMINO_ACIDS[int(index)] for index in row.detach().cpu().tolist())
            for row in sampled
        )
        if len(sequences) != n_samples or any(
            len(sequence) != case.residue_count
            or set(sequence).difference(STANDARD_AMINO_ACIDS)
            for sequence in sequences
        ):
            raise RuntimeError("DynamicMPNN returned invalid standard-AA sequence output")
        return DynamicMPNNSampleResult(
            protein_id=case.protein_id,
            pair_id=case.pair_id,
            seed=seed,
            sequences=sequences,
            residue_count=case.residue_count,
            checkpoint_sha256=self.checkpoint_sha256,
            source_commit=self.source_commit,
        )
