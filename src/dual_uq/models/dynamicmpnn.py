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

STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
OFFICIAL_SOURCE_COMMIT = "1f3e326c0f4d275ee8b3918e4726e19d3eef6c3f"
DEFAULT_CHECKPOINT_RELATIVE_PATH = "third_party/DynamicMPNN/checkpoints/single_chain_k2.ckpt"
_SHA256 = re.compile(r"[0-9a-f]{64}")


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        self.checkpoint_sha256 = _sha256_file(self.checkpoint_path)
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

    def _build_graph(self, case: DynamicMPNNInputCase) -> Any:
        validate_dynamic_input(case)
        if self._torch is None or self._featuriser is None:
            raise RuntimeError("DynamicMPNN adapter has not been loaded")
        from torch_geometric.data import Batch, Data
        from dynamicmpnn.types import BASE_AMINO_ACIDS, FILL_VALUE

        torch = self._torch
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
        return Batch.from_data_list([graph])

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
