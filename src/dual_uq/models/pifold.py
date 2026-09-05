"""Official PiFold adapter for the StructCal L0 local-response probe."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from dual_uq.core.hashing import sha256_file

STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AUTHORIZED_IMPLEMENTATION_COMMIT = "b28a0994ae02d7770733bf0aee792ffab99bdf68"
AUTHORIZED_CHECKPOINT_SHA256 = "1877a1dfbec870d4c1c14011cf629a83ffa37fa1bf24f34d06f6f464bbbb3fde"


class PiFoldModelError(ValueError):
    """Raised when the authorized PiFold boundary cannot be satisfied."""


@dataclass(frozen=True)
class PiFoldStructureInput:
    """Complete single-chain backbone coordinates for the geometry-only probe."""

    coordinates: np.ndarray
    sequence_length: int


def validate_pifold_structure(structure: PiFoldStructureInput) -> None:
    coordinates = np.asarray(structure.coordinates)
    if isinstance(structure.sequence_length, bool) or not isinstance(structure.sequence_length, int):
        raise PiFoldModelError("sequence_length must be an integer")
    if structure.sequence_length <= 0 or coordinates.shape != (structure.sequence_length, 4, 3):
        raise PiFoldModelError("PiFold coordinates must have shape [L,4,3]")
    if not np.isfinite(coordinates).all():
        raise PiFoldModelError("PiFold coordinates must be finite")


def normalize_pifold_logits(logits: Any) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(STANDARD_AMINO_ACIDS):
        raise PiFoldModelError("PiFold logits must have shape [L,20]")
    if values.shape[0] == 0 or not np.isfinite(values).all():
        raise PiFoldModelError("PiFold logits must be finite and non-empty")
    shifted = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    normalizer = probabilities.sum(axis=1, keepdims=True)
    if not np.isfinite(normalizer).all() or (normalizer <= 0).any():
        raise PiFoldModelError("PiFold logits have invalid normalization")
    result = probabilities / normalizer
    if not np.isfinite(result).all() or not np.allclose(result.sum(axis=1), 1.0, atol=1e-6):
        raise PiFoldModelError("PiFold probabilities are not normalized")
    return result


def call_legacy_pifold_featurizer(
    featurizer: Callable[[list[dict[str, Any]]], tuple[Any, ...]],
    batch: list[dict[str, Any]],
) -> tuple[Any, ...]:
    """Call the frozen source featurizer across modern NumPy versions."""

    had_numpy_int = hasattr(np, "int")
    previous_numpy_int = getattr(np, "int", None)
    if not had_numpy_int:
        np.int = int
    try:
        return featurizer(batch)
    finally:
        if had_numpy_int:
            np.int = previous_numpy_int
        else:
            delattr(np, "int")


@dataclass
class PiFoldAdapter:
    """Thin wrapper around the official PiFold geometry-only forward path."""

    model: Any
    torch_module: Any
    device: Any
    featurizer: Callable[[list[dict[str, Any]]], tuple[Any, ...]]
    implementation_id: str
    checkpoint_id: str

    def binding(self) -> dict[str, Any]:
        return {
            "model_id": "pifold_official_checkpoint_pth",
            "implementation_id": self.implementation_id,
            "checkpoint_id": self.checkpoint_id,
            "semantic_class": "L0",
            "probe_semantics": "geometry_only",
            "native_sequence_leakage": False,
            "aa_order": STANDARD_AMINO_ACIDS,
            "output_units": "probability",
        }

    def probability_distributions(self, structure: PiFoldStructureInput) -> np.ndarray:
        validate_pifold_structure(structure)
        coordinates = np.asarray(structure.coordinates, dtype=np.float32)
        batch = [
            {
                "seq": "A" * structure.sequence_length,
                "N": coordinates[:, 0, :],
                "CA": coordinates[:, 1, :],
                "C": coordinates[:, 2, :],
                "O": coordinates[:, 3, :],
            }
        ]
        X, S, score, mask, _lengths = call_legacy_pifold_featurizer(self.featurizer, batch)
        X = X.to(self.device)
        S = S.to(self.device)
        score = score.to(self.device)
        mask = mask.to(self.device)
        with self.torch_module.no_grad():
            features = self.model._get_features(S, score, X=X, mask=mask)
            log_probs = self.model(features[3], features[4], features[5], features[6])
        if hasattr(log_probs, "detach"):
            log_probs = log_probs.detach().cpu().numpy()
        values = np.asarray(log_probs, dtype=np.float64)
        if values.ndim == 3 and values.shape[0] == 1:
            values = values[0]
        if values.shape != (structure.sequence_length, len(STANDARD_AMINO_ACIDS)):
            raise PiFoldModelError("PiFold forward output length differs from structure")
        return normalize_pifold_logits(values)

def _official_args() -> SimpleNamespace:
    return SimpleNamespace(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        dropout=0.1,
        num_encoder_layers=10,
        k_neighbors=30,
        node_dist=1,
        node_angle=1,
        node_direct=1,
        edge_dist=1,
        edge_angle=1,
        edge_direct=1,
        virtual_num=3,
    )


def _git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PiFoldModelError("cannot verify PiFold source revision") from exc


def load_authorized_pifold_adapter(
    *, implementation_path: Path, checkpoint_path: Path, device_name: str
) -> PiFoldAdapter:
    """Load the exact frozen PiFold source/checkpoint in its own environment."""

    implementation_path = Path(implementation_path)
    checkpoint_path = Path(checkpoint_path)
    if _git_revision(implementation_path) != AUTHORIZED_IMPLEMENTATION_COMMIT:
        raise PiFoldModelError("PiFold source revision differs from the frozen capability card")
    try:
        if sha256_file(checkpoint_path) != AUTHORIZED_CHECKPOINT_SHA256:
            raise PiFoldModelError("PiFold checkpoint differs from the frozen capability card")
    except OSError as exc:
        raise PiFoldModelError("PiFold checkpoint is unavailable") from exc
    try:
        import sys

        import torch

        if device_name.startswith("cuda") and not torch.cuda.is_available():
            raise PiFoldModelError("requested PiFold CUDA device is unavailable")
        if str(implementation_path) not in sys.path:
            sys.path.insert(0, str(implementation_path))
        from API.featurizer import featurize_GTrans
        from methods.prodesign_model import ProDesign_Model

        device = torch.device(device_name)
        model = ProDesign_Model(_official_args()).to(device)
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
    except PiFoldModelError:
        raise
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise PiFoldModelError("PiFold model import or checkpoint load failed") from exc
    return PiFoldAdapter(
        model=model,
        torch_module=torch,
        device=device,
        featurizer=featurize_GTrans,
        implementation_id=AUTHORIZED_IMPLEMENTATION_COMMIT,
        checkpoint_id=checkpoint_path.name,
    )
