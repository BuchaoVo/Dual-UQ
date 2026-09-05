"""Thin boundary around the official ProteinInvBench KW-Design model."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from dual_uq.core.hashing import sha256_file
from dual_uq.models.proteinmpnn import STANDARD_AMINO_ACIDS, validate_protein_sequence

AUTHORIZED_IMPLEMENTATION_COMMIT = "d676962822c3f8009d5100a231443634ae9ade42"
AUTHORIZED_BASE_SHA256 = "c220df7354ffe85bf641cc6c0e7d5994396ef6e32cbddce3f9c8ce30124589c5"
AUTHORIZED_TUNER_SHA256 = "f2c0f1fc45ae5657af4524e40eaf00eab0fa3a6d18c96eaa4fd2d2a1279e967d"


class KWDesignContractError(ValueError):
    """Raised when the official KW-Design execution contract is not met."""


@dataclass(frozen=True, slots=True)
class KWDesignStructureInput:
    """A complete single-chain backbone scored by KW-Design."""

    coordinates: np.ndarray
    sequence: str
    title: str


def validate_kwdesign_structure(structure: KWDesignStructureInput) -> None:
    sequence = validate_protein_sequence(structure.sequence)
    coordinates = np.asarray(structure.coordinates)
    if coordinates.shape != (len(sequence), 4, 3):
        raise KWDesignContractError("KW-Design coordinates must have shape [L,4,3]")
    if not np.isfinite(coordinates).all():
        raise KWDesignContractError("KW-Design coordinates must be finite")
    if not isinstance(structure.title, str) or not structure.title:
        raise KWDesignContractError("KW-Design title must be a non-empty string")


def make_kwdesign_record(structure: KWDesignStructureInput) -> dict[str, Any]:
    """Convert one structure to the official single-chain featurizer record."""

    validate_kwdesign_structure(structure)
    coordinates = np.asarray(structure.coordinates, dtype=np.float32)
    length = len(structure.sequence)
    return {
        "title": structure.title,
        "seq": structure.sequence,
        "N": coordinates[:, 0],
        "CA": coordinates[:, 1],
        "C": coordinates[:, 2],
        "O": coordinates[:, 3],
        "chain_mask": np.ones(length, dtype=np.float32),
        "chain_encoding": np.ones(length, dtype=np.float32),
    }


def project_kwdesign_log_probabilities(
    log_probabilities: object, token_to_id: Mapping[str, int]
) -> np.ndarray:
    """Project the official 33-token output onto the canonical 20 amino acids."""

    values = np.asarray(log_probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
        raise KWDesignContractError("KW-Design log probabilities must be finite [L,V]")
    if any(amino_acid not in token_to_id for amino_acid in STANDARD_AMINO_ACIDS):
        raise KWDesignContractError("KW-Design tokenizer lacks a canonical amino-acid token")
    indices = [int(token_to_id[amino_acid]) for amino_acid in STANDARD_AMINO_ACIDS]
    if min(indices) < 0 or max(indices) >= values.shape[1] or len(set(indices)) != len(indices):
        raise KWDesignContractError("KW-Design canonical token indices are invalid")
    selected = values[:, indices]
    shifted = selected - selected.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-6
    ):
        raise KWDesignContractError("KW-Design canonical probabilities are invalid")
    return probabilities


def select_recycle_log_probabilities(
    log_probabilities: object, confidences: object
) -> np.ndarray:
    """Apply the official per-residue maximum-confidence recycle rule."""

    values = np.asarray(log_probabilities)
    scores = np.asarray(confidences)
    if values.ndim != 3 or scores.shape != values.shape[:2]:
        raise KWDesignContractError("KW-Design recycle arrays have incompatible shapes")
    if not np.isfinite(values).all() or not np.isfinite(scores).all():
        raise KWDesignContractError("KW-Design recycle arrays must be finite")
    winners = scores.argmax(axis=0)
    return values[winners, np.arange(values.shape[1])]


@dataclass
class KWDesignAdapter:
    """Official KW-Design forward with caches disabled and a paired fixed seed."""

    model: Any
    torch_module: Any
    device: Any
    featurizer: Any
    token_to_id: Mapping[str, int]
    implementation_id: str
    base_checkpoint_id: str
    tuner_checkpoint_id: str
    inference_seed: int = 111

    def binding(self) -> dict[str, Any]:
        return {
            "model_id": "kwdesign_official",
            "implementation_id": self.implementation_id,
            "base_checkpoint_id": self.base_checkpoint_id,
            "tuner_checkpoint_id": self.tuner_checkpoint_id,
            "semantic_class": "L0_composite",
            "probe_semantics": "geometry_only_seeded_msa",
            "native_sequence_leakage": False,
            "inference_seed": self.inference_seed,
            "aa_order": STANDARD_AMINO_ACIDS,
            "output_units": "probability",
        }

    def _reset_rng(self) -> None:
        self.torch_module.manual_seed(self.inference_seed)
        if self.torch_module.cuda.is_available():
            self.torch_module.cuda.manual_seed_all(self.inference_seed)

    def _clear_runtime_state(self) -> None:
        """Prevent official memoization from leaking one structure into another."""

        for module in self.model.modules():
            memory = getattr(module, "memory", None)
            if isinstance(memory, dict):
                memory.clear()
            confidence = getattr(module, "confidence", None)
            if isinstance(confidence, list):
                confidence.clear()

    def probability_distributions(self, structure: KWDesignStructureInput) -> np.ndarray:
        """Return seeded official KW-Design probabilities in canonical AA order."""

        record = make_kwdesign_record(structure)
        protein = self.featurizer([record])
        protein = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in protein.items()
        }
        features = self.model.Design1.design_model.PretrainPiFold._get_features(protein)
        batch = {
            "title": features["title"],
            "h_V": features["_V"],
            "h_E": features["_E"],
            "E_idx": features["E_idx"],
            "batch_id": features["batch_id"],
            "alphabet": f"{STANDARD_AMINO_ACIDS}X",
            "S": features["S"],
            "position": features["X"],
            "seq_mask": None,
        }
        self._clear_runtime_state()
        self._reset_rng()
        with self.torch_module.no_grad():
            log_probabilities = self.model(batch)["log_probs"]
        values = log_probabilities.detach().cpu().numpy()
        if values.shape[0] != len(structure.sequence):
            raise KWDesignContractError("KW-Design forward output length differs from structure")
        return project_kwdesign_log_probabilities(values, self.token_to_id)


def validate_kwdesign_weights(
    *,
    base_checkpoint: Path,
    tuner_checkpoint: Path,
    expected_base_sha256: str,
    expected_tuner_sha256: str = AUTHORIZED_TUNER_SHA256,
) -> dict[str, str]:
    """Verify the two learned assets needed by the official composite model."""

    base_checkpoint = Path(base_checkpoint)
    tuner_checkpoint = Path(tuner_checkpoint)
    try:
        base_sha256 = sha256_file(base_checkpoint)
        tuner_sha256 = sha256_file(tuner_checkpoint)
    except OSError as exc:
        raise KWDesignContractError("KW-Design checkpoint is unavailable") from exc
    if base_sha256 != expected_base_sha256:
        raise KWDesignContractError("KW-Design base checkpoint SHA256 mismatch")
    if tuner_sha256 != expected_tuner_sha256:
        raise KWDesignContractError("KW-Design tuner checkpoint SHA256 mismatch")
    return {
        "base_checkpoint": base_checkpoint.as_posix(),
        "base_checkpoint_sha256": base_sha256,
        "tuner_checkpoint": tuner_checkpoint.as_posix(),
        "tuner_checkpoint_sha256": tuner_sha256,
    }


def validate_kwdesign_load_keys(
    missing_keys: list[str], unexpected_keys: list[str]
) -> dict[str, int]:
    """Fail closed except for obsolete rotary-position tensors ignored upstream."""

    allowed_suffixes = (
        "DesignEmbed.position_ids",
        "DesignEmbed.position_embeddings.weight",
        "ESMEmbed.position_ids",
        "ESMEmbed.position_embeddings.weight",
    )
    unsupported = [key for key in unexpected_keys if not key.endswith(allowed_suffixes)]
    if unsupported:
        raise KWDesignContractError(
            f"KW-Design tuner has unexpected parameters: {unsupported[:3]}"
        )
    missing_tuning = [key for key in missing_keys if ".GNNTuning." in key]
    if missing_tuning:
        raise KWDesignContractError(
            f"KW-Design tuner has missing tuning parameters: {missing_tuning[:3]}"
        )
    return {
        "missing_pretrained_parameter_count": len(missing_keys),
        "ignored_rotary_position_parameter_count": len(unexpected_keys),
    }


def _git_revision(source_path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(source_path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise KWDesignContractError("cannot verify ProteinInvBench source revision") from exc


def _ensure_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() and link.resolve() == target.resolve():
        return
    if link.exists() or link.is_symlink():
        raise KWDesignContractError(f"KW-Design runtime path already exists: {link}")
    link.symlink_to(target.resolve())


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextmanager
def _local_transformers_assets(model_path: Path):
    from transformers import AutoTokenizer, EsmForMaskedLM

    tokenizer_had_attribute = "from_pretrained" in AutoTokenizer.__dict__
    model_had_attribute = "from_pretrained" in EsmForMaskedLM.__dict__
    tokenizer_descriptor = inspect.getattr_static(AutoTokenizer, "from_pretrained")
    model_descriptor = inspect.getattr_static(EsmForMaskedLM, "from_pretrained")
    tokenizer_loader = AutoTokenizer.from_pretrained
    model_loader = EsmForMaskedLM.from_pretrained
    AutoTokenizer.from_pretrained = classmethod(  # type: ignore[method-assign]
        lambda cls, *_args, **_kwargs: tokenizer_loader(
            model_path.as_posix(), local_files_only=True
        )
    )
    EsmForMaskedLM.from_pretrained = classmethod(  # type: ignore[method-assign]
        lambda cls, *_args, **_kwargs: model_loader(model_path.as_posix(), local_files_only=True)
    )
    try:
        yield
    finally:
        if tokenizer_had_attribute:
            AutoTokenizer.from_pretrained = tokenizer_descriptor  # type: ignore[method-assign]
        else:
            delattr(AutoTokenizer, "from_pretrained")
        if model_had_attribute:
            EsmForMaskedLM.from_pretrained = model_descriptor  # type: ignore[method-assign]
        else:
            delattr(EsmForMaskedLM, "from_pretrained")


def load_authorized_kwdesign_adapter(
    *,
    implementation_path: Path,
    model_parameters_path: Path,
    base_checkpoint: Path,
    tuner_checkpoint: Path,
    esm2_model_path: Path,
    esm_if1_checkpoint: Path,
    runtime_root: Path,
    device_name: str,
    inference_seed: int = 111,
) -> KWDesignAdapter:
    """Load the released ProteinInvBench KW-Design composite without source edits."""

    implementation_path = Path(implementation_path)
    if _git_revision(implementation_path) != AUTHORIZED_IMPLEMENTATION_COMMIT:
        raise KWDesignContractError("ProteinInvBench source revision is not authorized")
    weight_binding = validate_kwdesign_weights(
        base_checkpoint=base_checkpoint,
        tuner_checkpoint=tuner_checkpoint,
        expected_base_sha256=AUTHORIZED_BASE_SHA256,
    )
    if not Path(esm2_model_path).is_dir() or not Path(esm_if1_checkpoint).is_file():
        raise KWDesignContractError("KW-Design pretrained ESM asset is unavailable")
    try:
        parameters = json.loads(Path(model_parameters_path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise KWDesignContractError("KW-Design model parameters are unreadable") from exc
    parameters.update(
        {
            "res_dir": Path(runtime_root).as_posix(),
            "data_name": "MPNN",
            "is_colab": True,
            "load_memory": False,
        }
    )
    args = SimpleNamespace(**parameters)
    runtime_root = Path(runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)
    _ensure_symlink(runtime_root / "MPNN/PiFold/checkpoint.pth", Path(base_checkpoint))
    _ensure_symlink(
        runtime_root / "main/results/esm_if1_gvp4_t16_142M_UR50.pt",
        Path(esm_if1_checkpoint),
    )
    try:
        import torch

        if device_name.startswith("cuda") and not torch.cuda.is_available():
            raise KWDesignContractError("requested KW-Design CUDA device is unavailable")
        if implementation_path.as_posix() not in sys.path:
            sys.path.insert(0, implementation_path.as_posix())
        original_torch_load = torch.load

        def compatible_torch_load(*load_args: Any, **load_kwargs: Any) -> Any:
            load_kwargs.setdefault("weights_only", False)
            return original_torch_load(*load_args, **load_kwargs)

        with _working_directory(runtime_root), _local_transformers_assets(
            Path(esm2_model_path)
        ):
            torch.load = compatible_torch_load
            try:
                from PInvBench.src.datasets.featurizer import featurize_GTrans
                from PInvBench.src.models.kwdesign_model import KWDesign_model

                model = KWDesign_model(args)
            finally:
                torch.load = original_torch_load
        device = torch.device(device_name)
        tuner_state = original_torch_load(
            tuner_checkpoint, map_location=device, weights_only=False
        )
        missing_keys, unexpected_keys = model.load_state_dict(tuner_state, strict=False)
        load_key_audit = validate_kwdesign_load_keys(missing_keys, unexpected_keys)
        model = model.to(device)
        model.eval()
        token_to_id = dict(model.Design1.LM_model.tokenizer.get_vocab())
    except KWDesignContractError:
        raise
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise KWDesignContractError("KW-Design model import or checkpoint load failed") from exc
    adapter = KWDesignAdapter(
        model=model,
        torch_module=torch,
        device=device,
        featurizer=featurize_GTrans,
        token_to_id=token_to_id,
        implementation_id=AUTHORIZED_IMPLEMENTATION_COMMIT,
        base_checkpoint_id=Path(base_checkpoint).name,
        tuner_checkpoint_id=Path(tuner_checkpoint).name,
        inference_seed=inference_seed,
    )
    adapter.load_audit = {  # type: ignore[attr-defined]
        **weight_binding,
        **load_key_audit,
        "esm_if1_checkpoint": Path(esm_if1_checkpoint).as_posix(),
        "esm2_model_path": Path(esm2_model_path).as_posix(),
    }
    return adapter
