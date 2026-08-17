"""Official ESM-IF1 binding and model-native adapter.

The official ESM implementation is intentionally imported lazily.  The normal
Dual-UQ environment therefore remains usable without PyTorch/torch-geometric;
ESM-IF1 execution is restricted to the separately provisioned environment.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dual_uq.models.proteinmpnn import STANDARD_AMINO_ACIDS, validate_protein_sequence

OFFICIAL_MODEL_NAME = "esm_if1_gvp4_t16_142M_UR50"
OFFICIAL_SOURCE_REPOSITORY = "https://github.com/facebookresearch/esm"
STANDARD_AMINO_ACIDS_TUPLE = tuple(STANDARD_AMINO_ACIDS)


class ESMIF1ContractError(ValueError):
    """Raised when the official model/source contract cannot be validated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ESMIF1ContractError(f"checkpoint is unreadable: {path}") from exc
    return digest.hexdigest()


def _git_output(source_root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(source_root), *args],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ESMIF1ContractError("official ESM source is not a readable git checkout") from exc


def validate_esm_if1_source(
    source_root: Path,
    checkpoint_path: Path,
    *,
    expected_revision: str | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> dict[str, object]:
    """Validate the official checkout and exact local checkpoint bytes."""
    source_root = Path(source_root)
    checkpoint_path = Path(checkpoint_path)
    if not source_root.is_dir() or not (source_root / "esm").is_dir():
        raise ESMIF1ContractError("official ESM source directory is missing")
    remote = _git_output(source_root, "config", "--get", "remote.origin.url")
    if "facebookresearch/esm" not in remote.lower():
        raise ESMIF1ContractError("official ESM source remote is not facebookresearch/esm")
    revision = _git_output(source_root, "rev-parse", "HEAD")
    if expected_revision is not None and revision != expected_revision:
        raise ESMIF1ContractError(
            f"official ESM source revision mismatch: expected {expected_revision}, got {revision}"
        )
    if not checkpoint_path.is_file():
        raise ESMIF1ContractError(f"ESM-IF1 checkpoint is missing: {checkpoint_path}")
    checkpoint_sha256 = _sha256(checkpoint_path)
    if expected_checkpoint_sha256 is not None and checkpoint_sha256 != expected_checkpoint_sha256:
        raise ESMIF1ContractError(
            "ESM-IF1 checkpoint SHA256 mismatch: "
            f"expected {expected_checkpoint_sha256}, got {checkpoint_sha256}"
        )
    return {
        "model_name": OFFICIAL_MODEL_NAME,
        "implementation_revision": revision,
        "checkpoint_path": checkpoint_path.as_posix(),
        "checkpoint_sha256": checkpoint_sha256,
        "source_repository": OFFICIAL_SOURCE_REPOSITORY,
    }


@dataclass(frozen=True, slots=True)
class ESMIF1Binding:
    """Immutable model identity attached to every ESM-IF1 measurement."""

    model_name: str
    implementation_revision: str
    checkpoint_path: str
    checkpoint_sha256: str
    source_repository: str = OFFICIAL_SOURCE_REPOSITORY

    def as_dict(self) -> dict[str, str]:
        return {
            "model_name": self.model_name,
            "implementation_revision": self.implementation_revision,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "source_repository": self.source_repository,
        }


class ESMIF1Adapter:
    """Thin model-native ESM-IF1 score/generation adapter."""

    def __init__(self, model: Any, alphabet: Any, binding: ESMIF1Binding, *, device: str) -> None:
        self.model = model
        self.alphabet = alphabet
        self._binding = binding
        self.device = device

    def binding(self) -> dict[str, str]:
        return self._binding.as_dict()

    @staticmethod
    def _validate_coordinates(
        coordinates: np.ndarray, *, allow_missing: bool = False
    ) -> np.ndarray:
        coords = np.asarray(coordinates, dtype=np.float32)
        if coords.ndim != 3 or coords.shape[1:] != (3, 3) or not len(coords):
            raise ESMIF1ContractError("coordinates must have shape (L, 3, 3)")
        if allow_missing:
            finite_rows = np.isfinite(coords).all(axis=(1, 2))
            missing_rows = np.isnan(coords).all(axis=(1, 2))
            if not np.logical_or(finite_rows, missing_rows).all():
                raise ESMIF1ContractError(
                    "ESM-IF1 missing-coordinate rows must be fully NaN or finite"
                )
        elif not np.isfinite(coords).all():
            raise ESMIF1ContractError("ESM-IF1 coordinates must be finite")
        return coords

    def _converter(self) -> Any:
        try:
            from esm.inverse_folding.util import CoordBatchConverter
        except ImportError as exc:  # pragma: no cover - only isolated runtime
            raise ESMIF1ContractError(
                "official ESM-IF1 runtime dependencies are unavailable"
            ) from exc
        return CoordBatchConverter(self.alphabet)

    def score_teacher_forced(
        self,
        sequence: str,
        coordinates: np.ndarray,
        *,
        allow_missing_coordinates: bool = False,
    ) -> np.ndarray:
        """Return native 20-AA conditional probabilities for each residue."""
        sequence = validate_protein_sequence(sequence)
        coords = self._validate_coordinates(
            coordinates, allow_missing=allow_missing_coordinates
        )
        if len(sequence) != len(coords):
            raise ESMIF1ContractError("sequence and coordinate lengths differ")
        converter = self._converter()
        batch_coords, confidence, _, tokens, padding_mask = converter(
            [(coords, None, sequence)], device=self.device
        )
        prev_output_tokens = tokens[:, :-1]
        try:
            import torch
            with torch.no_grad():
                logits, _ = self.model.forward(
                    batch_coords, padding_mask, confidence, prev_output_tokens
                )
        except ImportError as exc:  # pragma: no cover - only isolated runtime
            raise ESMIF1ContractError("PyTorch is required for ESM-IF1 scoring") from exc
        aa_indices = [self.alphabet.get_idx(aa) for aa in STANDARD_AMINO_ACIDS_TUPLE]
        # The official decoder returns (batch, vocabulary, target_length),
        # matching its cross-entropy helper; expose the analysis-friendly
        # (target_length, standard_20_aa) categorical table.
        selected = logits[0, aa_indices, :].transpose(0, 1)
        try:
            import torch

            probabilities = torch.softmax(selected, dim=-1).detach().cpu().numpy()
        except ImportError as exc:  # pragma: no cover - only isolated runtime
            raise ESMIF1ContractError("PyTorch is required for ESM-IF1 scoring") from exc
        return np.asarray(probabilities, dtype=np.float64)

    def score_teacher_forced_batch(
        self,
        sequences: tuple[str, ...],
        coordinates: np.ndarray,
        *,
        batch_size: int = 32,
    ) -> tuple[np.ndarray, ...]:
        """Return teacher-forced 20-AA tables for equal-length sequence batches."""
        if not sequences:
            raise ESMIF1ContractError("sequence batch must not be empty")
        checked = tuple(validate_protein_sequence(sequence) for sequence in sequences)
        if len(set(map(len, checked))) != 1:
            raise ESMIF1ContractError("sequence batch lengths differ")
        coords = self._validate_coordinates(coordinates)
        if len(checked[0]) != len(coords):
            raise ESMIF1ContractError("sequence and coordinate lengths differ")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ESMIF1ContractError("batch_size must be positive")
        converter = self._converter()
        aa_indices = [self.alphabet.get_idx(aa) for aa in STANDARD_AMINO_ACIDS_TUPLE]
        result: list[np.ndarray] = []
        try:
            import torch
            with torch.no_grad():
                # The geometric encoder depends only on the backbone.  Encode
                # it once, then reuse the immutable encoder output for all
                # teacher-forced sequence batches.
                encoder_coords, encoder_confidence, _, _, encoder_padding = converter(
                    [(coords, None, checked[0])], device=self.device
                )
                encoded = self.model.encoder(
                    encoder_coords, encoder_padding, encoder_confidence
                )
                for start in range(0, len(checked), batch_size):
                    chunk = checked[start : start + batch_size]
                    _batch_coords, confidence, _, tokens, padding_mask = converter(
                        [(coords, None, sequence) for sequence in chunk], device=self.device
                    )
                    batch_n = len(chunk)
                    encoder_out = {
                        "encoder_out": [encoded["encoder_out"][0].repeat(1, batch_n, 1)],
                        "encoder_padding_mask": [encoded["encoder_padding_mask"][0].repeat(batch_n, 1)],
                        "encoder_embedding": [
                            {
                                key: value.repeat(batch_n, 1, 1)
                                for key, value in encoded["encoder_embedding"][0].items()
                            }
                        ],
                        "encoder_states": [],
                    }
                    logits, _ = self.model.decoder(
                        tokens[:, :-1], encoder_out=encoder_out
                    )
                    selected = logits[:, aa_indices, :].transpose(1, 2)
                    probabilities = torch.softmax(selected, dim=-1).detach().cpu().numpy()
                    result.extend(np.asarray(row, dtype=np.float64) for row in probabilities)
        except ImportError as exc:  # pragma: no cover - isolated runtime
            raise ESMIF1ContractError("PyTorch is required for ESM-IF1 scoring") from exc
        if any(not np.isfinite(row).all() for row in result):
            raise ESMIF1ContractError("ESM-IF1 batch returned non-finite probabilities")
        return tuple(result)

    def sample(
        self,
        coordinates: np.ndarray,
        temperature: float,
        seed: int,
        *,
        allow_missing_coordinates: bool = False,
    ) -> str:
        """Sample one sequence using the official ESM-IF1 autoregressive sampler."""
        if type(seed) is not int or seed < 0:
            raise ESMIF1ContractError("sampling seed must be a non-negative integer")
        if not np.isfinite(float(temperature)) or temperature <= 0:
            raise ESMIF1ContractError("sampling temperature must be positive")
        coords = self._validate_coordinates(
            coordinates, allow_missing=allow_missing_coordinates
        )
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - only isolated runtime
            raise ESMIF1ContractError("PyTorch is required for ESM-IF1 sampling") from exc
        torch.manual_seed(seed)
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            sequence = self.model.sample(
                coords, temperature=float(temperature), device=self.device
            )
        return validate_protein_sequence(sequence)

    def sample_batch(
        self,
        coordinates: np.ndarray,
        temperatures: tuple[float, ...],
        seeds: tuple[int, ...],
        *,
        allow_missing_coordinates: bool = False,
    ) -> tuple[str, ...]:
        """Batch the official autoregressive decoder while preserving per-seed RNGs."""
        coords = np.asarray(coordinates, dtype=np.float32)
        if coords.ndim != 4 or coords.shape[2:] != (3, 3) or not len(coords):
            raise ESMIF1ContractError("batched coordinates must have shape (B, L, 3, 3)")
        if len(temperatures) != len(coords) or len(seeds) != len(coords):
            raise ESMIF1ContractError("batched sampling metadata length differs from coordinates")
        self._validate_coordinates(
            coords[0], allow_missing=allow_missing_coordinates
        )
        if allow_missing_coordinates:
            for item in coords[1:]:
                self._validate_coordinates(item, allow_missing=True)
        if any(float(value) <= 0 for value in temperatures):
            raise ESMIF1ContractError("sampling temperatures must be positive")
        if any(type(seed) is not int or seed < 0 for seed in seeds):
            raise ESMIF1ContractError("sampling seeds must be non-negative integers")
        try:
            import torch
            from esm.inverse_folding.util import CoordBatchConverter
        except ImportError as exc:  # pragma: no cover - isolated runtime only
            raise ESMIF1ContractError("official ESM-IF1 runtime dependencies are unavailable") from exc
        converter = CoordBatchConverter(self.alphabet)
        batch = [(item, None, None) for item in coords]
        batch_coords, confidence, _, _, padding_mask = converter(batch, device=self.device)
        length = coords.shape[1]
        mask_idx = self.alphabet.get_idx("<mask>")
        cath_idx = self.alphabet.get_idx("<cath>")
        sampled_tokens = torch.full(
            (len(coords), 1 + length), mask_idx, dtype=torch.long, device=self.device
        )
        sampled_tokens[:, 0] = cath_idx
        generators = []
        for seed in seeds:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            generators.append(generator)
        with torch.no_grad():
            encoder_out = self.model.encoder(batch_coords, padding_mask, confidence)
            incremental_state: dict[str, Any] = {}
            allowed_indices = [self.alphabet.get_idx(aa) for aa in STANDARD_AMINO_ACIDS_TUPLE]
            for index in range(1, length + 1):
                logits, _ = self.model.decoder(
                    sampled_tokens[:, :index], encoder_out, incremental_state=incremental_state
                )
                forbidden = torch.ones(logits.shape[1], dtype=torch.bool, device=self.device)
                forbidden[allowed_indices] = False
                probabilities = torch.softmax(
                    logits[:, :, -1].masked_fill(
                        forbidden[None, :],
                        float("-inf"),
                    ) / torch.as_tensor(
                        temperatures, dtype=logits.dtype, device=logits.device
                    ).unsqueeze(1),
                    dim=-1,
                )
                for row, generator in enumerate(generators):
                    sampled_tokens[row, index] = torch.multinomial(
                        probabilities[row], 1, generator=generator
                    ).squeeze(0)
        sequences = tuple(
            validate_protein_sequence(
                "".join(self.alphabet.get_tok(token) for token in row[1:])
            )
            for row in sampled_tokens.detach().cpu()
        )
        return sequences


class FakeESMIF1Adapter:
    """Small deterministic adapter used by contract and pipeline tests."""

    def __init__(self, *, implementation_revision: str, checkpoint_sha256: str) -> None:
        self.implementation_revision = implementation_revision
        self.checkpoint_sha256 = checkpoint_sha256

    def binding(self) -> dict[str, str]:
        return {
            "model_name": OFFICIAL_MODEL_NAME,
            "implementation_revision": self.implementation_revision,
            "checkpoint_sha256": self.checkpoint_sha256,
        }

    def score_teacher_forced(
        self,
        sequence: str,
        coordinates: np.ndarray,
        *,
        allow_missing_coordinates: bool = False,
    ) -> np.ndarray:
        sequence = validate_protein_sequence(sequence)
        coords = np.asarray(coordinates)
        if coords.shape != (len(sequence), 3, 3):
            raise ESMIF1ContractError("sequence and coordinate lengths differ")
        logits = np.zeros((len(sequence), 20), dtype=np.float64)
        for index, residue in enumerate(sequence):
            logits[index, STANDARD_AMINO_ACIDS_TUPLE.index(residue)] = 1.0
            coordinate_value = float(coords[index, 1, 0])
            logits[index] += (0.0 if np.isnan(coordinate_value) else coordinate_value) * 0.001
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def sample(
        self,
        coordinates: np.ndarray,
        temperature: float,
        seed: int,
        *,
        allow_missing_coordinates: bool = False,
    ) -> str:
        coords = np.asarray(coordinates)
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(STANDARD_AMINO_ACIDS_TUPLE), size=len(coords))
        return "".join(STANDARD_AMINO_ACIDS_TUPLE[index] for index in indices)

    def sample_batch(
        self,
        coordinates: np.ndarray,
        temperatures: tuple[float, ...],
        seeds: tuple[int, ...],
        *,
        allow_missing_coordinates: bool = False,
    ) -> tuple[str, ...]:
        return tuple(
            self.sample(
                item,
                temperature,
                seed,
                allow_missing_coordinates=allow_missing_coordinates,
            )
            for item, temperature, seed in zip(coordinates, temperatures, seeds, strict=True)
        )


def load_esm_if1(
    source_root: Path,
    checkpoint_path: Path,
    *,
    expected_revision: str,
    expected_checkpoint_sha256: str,
    device: str = "cuda",
) -> ESMIF1Adapter:
    """Validate and load the official ESM-IF1 checkpoint from local bytes."""
    observed = validate_esm_if1_source(
        source_root,
        checkpoint_path,
        expected_revision=expected_revision,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    try:
        import esm
        import torch
    except ImportError as exc:  # pragma: no cover - isolated runtime
        raise ESMIF1ContractError("official ESM package is unavailable") from exc
    try:
        # The trusted official checkpoint contains an argparse.Namespace.  Newer
        # PyTorch releases default to weights_only=True; allowlist only this
        # historical container type before invoking the official local loader.
        torch.serialization.add_safe_globals([argparse.Namespace])
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(checkpoint_path))
    except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - model runtime
        raise ESMIF1ContractError("official ESM-IF1 checkpoint could not be loaded") from exc
    model = model.eval()
    model = model.to(device)
    binding = ESMIF1Binding(
        model_name=OFFICIAL_MODEL_NAME,
        implementation_revision=str(observed["implementation_revision"]),
        checkpoint_path=str(observed["checkpoint_path"]),
        checkpoint_sha256=str(observed["checkpoint_sha256"]),
    )
    return ESMIF1Adapter(model, alphabet, binding, device=device)
