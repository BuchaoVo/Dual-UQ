"""Orchestration for StructCal cross-model representation sensitivity."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dual_uq.evaluation.apo_holo_local_response import build_decoding_realizations
from dual_uq.evaluation.cross_model_representation_cases import make_degenerate_dynamic_case
from dual_uq.evaluation.cross_model_representation_sensitivity import build_response_rows
from dual_uq.models.dynamicmpnn import DynamicMPNNAdapter
from dual_uq.models.esm_if1 import load_esm_if1
from dual_uq.models.kwdesign import KWDesignStructureInput, load_authorized_kwdesign_adapter
from dual_uq.models.pifold import PiFoldStructureInput, load_authorized_pifold_adapter
from dual_uq.models.proteinmpnn import (
    ProteinMPNNStructureInput,
    load_authorized_proteinmpnn_adapter,
)

REGIMES = ("identical", "exact_se3", "controlled", "operational_pdb_afdb")
PRIMARY_MODELS = ("proteinmpnn", "esm_if1", "pifold", "dynamicmpnn")
DEFAULT_RUN_ROOT = Path("runs/structcal_cross_model_representation_sensitivity")
STRUCTCAL_RELEASE_ROOT = Path("artifacts/releases/structcal_v1")
CHECKPOINTS = {
    "proteinmpnn": "v_48_020",
    "esm_if1": "esm_if1_gvp4_t16_142M_UR50",
    "pifold": "official_checkpoint_pth",
    "dynamicmpnn": "single_chain_k2",
}
MAIN_MODELS = {model: CHECKPOINTS[model] for model in PRIMARY_MODELS}

ESM_IF1_REVISION = "2b369911bb5b4b0dda914521b9475cad1656b2ac"
ESM_IF1_SHA256 = "be4ba36edec22a9bfaa4946ff6b2815f1f19d8a3d7e0eada8b796d5a0eae9fd4"

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "proteinmpnn": {
        "atoms": ("N", "CA", "C", "O"),
        "semantic_class": "L2",
        "probe_semantics": "fixed_order_native_autoregressive_context",
    },
    "esm_if1": {
        "atoms": ("N", "CA", "C"),
        "semantic_class": "L2",
        "probe_semantics": "teacher_forced_native_autoregressive_context",
    },
    "pifold": {
        "atoms": ("N", "CA", "C", "O"),
        "semantic_class": "L0",
        "probe_semantics": "geometry_only",
    },
    "dynamicmpnn": {
        "atoms": ("N", "CA", "C"),
        "semantic_class": "L2",
        "probe_semantics": "teacher_forced_natural_order_native_prefix",
    },
    "kwdesign": {
        "atoms": ("N", "CA", "C", "O"),
        "semantic_class": "L0_composite",
        "probe_semantics": "geometry_only_seeded_msa",
    },
}


def filter_dynamicmpnn_contiguous_cases(
    cases: pd.DataFrame, *, regime: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep whole pairs satisfying DynamicMPNN's native contiguous-axis contract."""

    excluded: list[dict[str, Any]] = []
    excluded_ids: set[str] = set()
    for pair_id, group in cases.groupby("pair_id", sort=True):
        positions = tuple(sorted(group["canonical_position"].astype(int).unique()))
        if positions != tuple(range(positions[0], positions[-1] + 1)):
            excluded_ids.add(str(pair_id))
            excluded.append(
                {
                    "model": "dynamicmpnn",
                    "protein_id": str(group.iloc[0]["protein_id"]),
                    "pair_id": str(pair_id),
                    "structural_regime": regime,
                    "status": "DATA_UNRESOLVED",
                    "reason": "NONCONTIGUOUS_CANONICAL_AXIS",
                    "detail": "DynamicMPNN requires a complete contiguous canonical residue axis; the pair was not split or imputed.",
                }
            )
    evaluable = cases.loc[~cases["pair_id"].astype(str).isin(excluded_ids)].copy()
    return evaluable, pd.DataFrame(excluded)


def safe_output_path(output_root: Path, relative: str | Path) -> Path:
    """Resolve a task output while preventing writes outside its run root."""

    root = Path(output_root).resolve()
    result = (root / relative).resolve()
    if result != root and root not in result.parents:
        raise ValueError("output path would escape the task run root")
    return result


def response_shard_path(
    output_root: Path, *, model: str, checkpoint: str, regime: str
) -> Path:
    if model not in MODEL_SPECS:
        raise ValueError(f"unknown cross-model probe: {model}")
    if regime not in REGIMES:
        raise ValueError(f"unknown representation regime: {regime}")
    return safe_output_path(output_root, Path("shards") / model / checkpoint / regime)


def coordinates_from_case_group(group: pd.DataFrame) -> np.ndarray:
    for value in group.sort_values("canonical_position", kind="mergesort")["coordinates"]:
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            result = np.asarray(
                [
                    [np.asarray(atom, dtype=np.float32) for atom in residue]
                    for residue in value
                ],
                dtype=np.float32,
            )
            if result.ndim != 3 or result.shape[-1] != 3 or not np.isfinite(result).all():
                raise ValueError("case coordinates must be finite with shape [L, atoms, 3]")
            return result
    raise ValueError("case condition has no coordinates")


def _required_environment_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} is required for this model")
    return Path(value)


def load_structcal_model(
    model_name: str,
    checkpoint: str,
    project_root: Path,
    output_root: Path,
    device: str,
    *,
    checkpoint_path: Path | None = None,
    backbone_noise: float = 0.0,
) -> tuple[Any, str]:
    """Load one authorized StructCal model and return its canonical checkpoint id."""

    if model_name not in MODEL_SPECS:
        raise ValueError(f"unknown model: {model_name}")
    if model_name == "proteinmpnn":
        checkpoint_name = checkpoint if checkpoint.endswith(".pt") else f"{checkpoint}.pt"
        path = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else project_root
            / "third_party/ProteinMPNN/vanilla_model_weights"
            / checkpoint_name
        )
        return (
            load_authorized_proteinmpnn_adapter(
                implementation_path=project_root / "third_party/ProteinMPNN",
                checkpoint_path=path,
                device_name=device,
                backbone_noise=backbone_noise,
            ),
            path.stem,
        )
    if model_name == "pifold":
        return (
            load_authorized_pifold_adapter(
                implementation_path=_required_environment_path("PIFOLD_SOURCE_PATH"),
                checkpoint_path=(
                    Path(checkpoint_path)
                    if checkpoint_path is not None
                    else _required_environment_path("PIFOLD_CHECKPOINT_PATH")
                ),
                device_name=device,
            ),
            CHECKPOINTS["pifold"],
        )
    if model_name == "esm_if1":
        path = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else _required_environment_path("ESM_IF1_CHECKPOINT_PATH")
        )
        return (
            load_esm_if1(
                project_root / "third_party/esm",
                path,
                expected_revision=ESM_IF1_REVISION,
                expected_checkpoint_sha256=ESM_IF1_SHA256,
                device=device,
            ),
            CHECKPOINTS["esm_if1"],
        )
    if model_name == "dynamicmpnn":
        return (
            DynamicMPNNAdapter(repository_root=project_root, device=device),
            CHECKPOINTS["dynamicmpnn"],
        )
    checkpoint_root = _required_environment_path("KWDESIGN_CHECKPOINT_ROOT")
    return (
        load_authorized_kwdesign_adapter(
            implementation_path=_required_environment_path("KWDESIGN_SOURCE_PATH"),
            model_parameters_path=checkpoint_root / "model_param.json",
            base_checkpoint=_required_environment_path(
                "KWDESIGN_BASE_CHECKPOINT_PATH"
            ),
            tuner_checkpoint=checkpoint_root / "checkpoint.pth",
            esm2_model_path=_required_environment_path("KWDESIGN_ESM2_MODEL_PATH"),
            esm_if1_checkpoint=_required_environment_path("ESM_IF1_CHECKPOINT_PATH"),
            runtime_root=safe_output_path(output_root, "runtime/kwdesign"),
            device_name=device,
        ),
        "release_checkpoint_pth",
    )


def build_structcal_scorer(model_name: str, adapter: Any) -> Callable[..., np.ndarray]:
    """Bind native probability semantics for one authorized StructCal adapter."""

    if model_name not in MODEL_SPECS:
        raise ValueError(f"unknown model: {model_name}")

    def score(**kwargs: Any) -> np.ndarray:
        coordinates = np.asarray(kwargs["coordinates"], dtype=np.float32)
        sequence = str(kwargs["sequence"])
        positions = tuple(kwargs["positions"])
        if model_name == "pifold":
            return adapter.probability_distributions(
                PiFoldStructureInput(coordinates=coordinates, sequence_length=len(sequence))
            )
        if model_name == "esm_if1":
            return adapter.score_teacher_forced(sequence, coordinates[:, :3])
        if model_name == "proteinmpnn":
            structure = ProteinMPNNStructureInput(
                protein_id=str(kwargs["protein_id"]),
                backbone_condition=str(kwargs["condition"]),
                uniprot_positions=positions,
                wt_sequence_projection=sequence,
                coordinates=coordinates,
                structure_sha256=None,
            )
            realization = build_decoding_realizations(
                str(kwargs["protein_id"]), len(sequence), count=1
            )[0]
            return adapter.probability_distributions(
                structure, (sequence,), realization, batch_size=1
            )[0]
        if model_name == "dynamicmpnn":
            case = make_degenerate_dynamic_case(
                protein_id=str(kwargs["protein_id"]),
                pair_id=f"{kwargs['pair_id']}::{kwargs['condition']}",
                canonical_positions=positions,
                sequence=sequence,
                coordinates=coordinates[:, :3],
                chain_id="A",
            )
            result = adapter.teacher_forced(case, sequence=sequence, inference=True)
            return result.logits.softmax(dim=-1).detach().cpu().numpy()
        if model_name == "kwdesign":
            return adapter.probability_distributions(
                KWDesignStructureInput(
                    coordinates=coordinates,
                    sequence=sequence,
                    title=f"{kwargs['pair_id']}::{kwargs['condition']}",
                )
            )
        raise ValueError(f"unknown model: {model_name}")

    return score


def score_cases_with_callback(
    cases: pd.DataFrame,
    scorer: Callable[..., np.ndarray],
    *,
    model_id: str,
    checkpoint_id: str,
    semantic_class: str,
    regime: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score every paired view independently, then apply shared metrics."""

    distributions: dict[tuple[str, str], np.ndarray] = {}
    for (pair_id, condition), group in cases.groupby(["pair_id", "condition"], sort=True):
        ordered = group.sort_values("canonical_position", kind="mergesort")
        positions = tuple(int(value) for value in ordered["canonical_position"])
        sequences = ordered["wt_sequence_projection"].astype(str).drop_duplicates()
        if len(sequences) != 1:
            raise ValueError(f"case sequence is not constant: {pair_id}/{condition}")
        distributions[(str(pair_id), str(condition))] = scorer(
            protein_id=str(ordered.iloc[0]["protein_id"]),
            pair_id=str(pair_id),
            condition=str(condition),
            sequence=str(sequences.iloc[0]),
            coordinates=coordinates_from_case_group(ordered),
            positions=positions,
        )
    return build_response_rows(
        cases,
        distributions,
        model_id=model_id,
        checkpoint_id=checkpoint_id,
        semantic_class=semantic_class,
        regime=regime,
    )
