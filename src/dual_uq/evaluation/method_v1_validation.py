"""Validation-only generation and cross-evaluator endpoint calculation for V1."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable

import pandas as pd
import torch

from dual_uq.core.atomic_io import atomic_write_new_bytes
from dual_uq.evaluation.method_v1_data import ProteinStateExample
from dual_uq.evaluation.method_v1_training import _prediction_logits, _to_inputs
from dual_uq.models.paired_state import PairedStateConfig, PairedStateModel
from dual_uq.evaluation.multi_state_baseline import compatibility_endpoints


class V1ValidationError(ValueError):
    """Raised when validation generation/evaluation violates the frozen scope."""


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Nested validation sequence ensembles and convergence counts."""

    sequences: pd.DataFrame
    convergence: pd.DataFrame
    locked_test_accessed: bool


@dataclass(frozen=True, slots=True)
class ValidationEvaluation:
    """Evaluator-separated raw scores and protein-level summaries."""

    scores: pd.DataFrame
    protein_summary: pd.DataFrame


def _stable_seed(seed: int, protein_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{protein_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _load_model(checkpoint_path: str, example: ProteinStateExample, device: torch.device) -> tuple[PairedStateModel, PairedStateConfig]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = PairedStateConfig(**payload["model_config"])
    model = PairedStateModel(config).to(device)
    with torch.no_grad():
        model(_to_inputs(example, device))
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, config


def _safe_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Convert model logits to valid sampling probabilities without target leakage.

    A single-state ablation intentionally has no prediction for residues that are
    unobserved in its selected state.  Those rows are represented by NaNs in the
    model output; generation uses a uniform distribution there rather than
    imputing the canonical target residue or allowing CUDA multinomial to fail.
    Non-finite rows are treated the same way, while finite model rows retain their
    learned distribution unchanged.
    """

    finite_rows = torch.isfinite(logits).all(dim=-1)
    safe_logits = torch.where(finite_rows.unsqueeze(-1), logits, torch.zeros_like(logits))
    probabilities = torch.softmax(safe_logits, dim=-1)
    if not torch.isfinite(probabilities).all():
        raise V1ValidationError("validation generation produced non-finite probabilities")
    return probabilities


def generate_validation_sequences(
    checkpoint_path: str,
    examples: tuple[ProteinStateExample, ...],
    *,
    count: int = 64,
    checkpoints: tuple[int, ...] = (16, 32, 64),
    seed: int = 20260817,
    device: str = "cpu",
) -> GenerationResult:
    """Generate deterministic consensus sequences for VALIDATION only."""

    if count <= 0 or not checkpoints or any(value <= 0 or value > count for value in checkpoints):
        raise V1ValidationError("validation generation checkpoints must be within the requested count")
    if tuple(sorted(set(checkpoints))) != checkpoints:
        raise V1ValidationError("validation checkpoints must be strictly increasing")
    validation = tuple(example for example in examples if example.split == "VALIDATION")
    if any(example.split == "LOCKED_TEST" for example in examples):
        raise V1ValidationError("LOCKED_TEST examples are inaccessible during validation")
    if not validation:
        raise V1ValidationError("validation generation requires non-empty VALIDATION examples")
    torch_device = torch.device(device)
    model, model_config = _load_model(checkpoint_path, validation[0], torch_device)
    rows: list[dict[str, object]] = []
    for example in validation:
        with torch.no_grad():
            output = model(_to_inputs(example, torch_device))
            logits = _prediction_logits(output, model_config)[0]
            probabilities = _safe_probabilities(logits)
        generator = torch.Generator(device=torch_device).manual_seed(_stable_seed(seed, example.protein_id))
        samples = torch.multinomial(probabilities, count, replacement=True, generator=generator).transpose(0, 1)
        for sample_index, target in enumerate(samples.tolist()):
            sequence = "".join(STANDARD_AMINO_ACIDS[index] for index in target)
            rows.append(
                {
                    "protein_id": example.protein_id,
                    "pair_id": example.pair_id,
                    "split": "VALIDATION",
                    "sample_index": sample_index,
                    "sequence": sequence,
                    "sequence_hash": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
                    "source_state": "CONSENSUS",
                    "seed": _stable_seed(seed, example.protein_id),
                    "entropy": float((-(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)).mean()),
                }
            )
    frame = pd.DataFrame(rows).sort_values(["protein_id", "sample_index"], kind="mergesort").reset_index(drop=True)
    convergence = pd.DataFrame(
        [
            {"protein_id": protein, "checkpoint": checkpoint, "sequence_count": checkpoint}
            for protein in frame["protein_id"].drop_duplicates()
            for checkpoint in checkpoints
        ]
    )
    return GenerationResult(sequences=frame, convergence=convergence, locked_test_accessed=False)


def materialize_validation_generation_shards(
    generated: pd.DataFrame,
    output_root: str | Path,
) -> dict[str, object]:
    """Materialize V1 validation sequences in the existing scoring-shard schema.

    The scorer consumes one immutable shard per protein.  This adapter changes
    only storage shape; it does not alter sequences, sample indices, or hashes.
    """

    required = {"protein_id", "pair_id", "split", "sample_index", "sequence", "sequence_hash"}
    if not required.issubset(generated.columns):
        raise V1ValidationError("generated validation table lacks shard fields")
    if generated.empty or not generated["split"].eq("VALIDATION").all():
        raise V1ValidationError("only VALIDATION sequences can be materialized")
    if generated.duplicated(["protein_id", "sample_index"]).any():
        raise V1ValidationError("validation sequences contain duplicate sample indices")
    root = Path(output_root)
    protein_count = 0
    record_count = 0
    for protein_id, group in generated.groupby("protein_id", sort=True):
        ordered = group.sort_values("sample_index", kind="mergesort")
        if ordered["pair_id"].nunique() != 1:
            raise V1ValidationError(f"validation shard has multiple pair IDs: {protein_id}")
        records = []
        for row in ordered.itertuples(index=False):
            sequence = str(row.sequence)
            sequence_hash = str(row.sequence_hash)
            expected_hash = hashlib.sha256(sequence.encode("ascii")).hexdigest()
            if expected_hash != sequence_hash:
                raise V1ValidationError(f"validation sequence hash mismatch: {protein_id}")
            records.append(
                {
                    "sample_index": int(row.sample_index),
                    "seed": int(row.seed) if hasattr(row, "seed") else 0,
                    "decoding_realization": "v1_consensus",
                    "sequence": sequence,
                    "sequence_hash": sequence_hash,
                }
            )
        payload = {
            "schema_version": "apo_holo_multistate_generation_v1",
            "protein_id": str(protein_id),
            "pair_id": str(ordered["pair_id"].iloc[0]),
            "n_samples": len(records),
            "model": "Dual-UQ-Method-V1",
            "source_state": "CONSENSUS",
            "records": records,
        }
        rendered = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        shard_path = root / "shards" / str(protein_id)[:2] / f"{protein_id}.json"
        atomic_write_new_bytes(shard_path, rendered)
        protein_count += 1
        record_count += len(records)
    return {"protein_count": protein_count, "record_count": record_count, "locked_test_accessed": False}


def project_validation_sequences_to_cases(
    generated: pd.DataFrame,
    cases: pd.DataFrame,
    *,
    allow_missing: bool = False,
) -> pd.DataFrame:
    """Project full-canonical V1 sequences onto a frozen evaluator case axis."""

    required_cases = {"protein_id", "canonical_positions", "wt_sequence", "wt_sequence_projection"}
    if not required_cases.issubset(cases.columns):
        raise V1ValidationError("case cache lacks sequence projection fields")
    if "sequence" not in generated.columns or "protein_id" not in generated.columns:
        raise V1ValidationError("generated validation table lacks sequence fields")
    projected = generated.copy()
    projected["source_sequence_hash"] = projected["sequence_hash"].astype(str)
    lookup: dict[str, tuple[str, str, tuple[int, ...]]] = {}
    for protein_id, group in cases.groupby("protein_id", sort=False):
        row = group.iloc[0]
        full = str(row.wt_sequence)
        projection = str(row.wt_sequence_projection)
        positions = tuple(int(value) for value in row.canonical_positions)
        if "condition" in group.columns:
            state_sequences = group.groupby("condition")["wt_sequence"].first().astype(str).to_dict()
            if len(set(state_sequences.values())) > 1:
                raise V1ValidationError(f"case cache state sequence mismatch: {protein_id}")
        if len(projection) != len(positions):
            raise V1ValidationError(f"case cache projection axis mismatch: {protein_id}")
        lookup[str(protein_id)] = (full, projection, positions)
    for index, row in projected.iterrows():
        protein_id = str(row["protein_id"])
        if protein_id not in lookup:
            if allow_missing:
                continue
            raise V1ValidationError(f"generated protein missing from case cache: {protein_id}")
        full, projection, positions = lookup[protein_id]
        sequence = str(row["sequence"])
        if len(sequence) == len(full):
            sequence = "".join(sequence[position - 1] for position in positions)
        elif len(sequence) == len(projection):
            pass
        else:
            raise V1ValidationError(f"generated sequence length is incompatible with case axis: {protein_id}")
        projected.at[index, "sequence"] = sequence
        projected.at[index, "sequence_hash"] = hashlib.sha256(sequence.encode("ascii")).hexdigest()
    if allow_missing:
        projected = projected.loc[projected["protein_id"].astype(str).isin(lookup)].copy()
    return projected


STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
ScoreCallback = Callable[[str, str, str, str], float]


def evaluate_validation_outputs(
    generated: pd.DataFrame,
    examples: tuple[ProteinStateExample, ...],
    scorer: ScoreCallback,
) -> ValidationEvaluation:
    """Evaluate validation sequences with separately labelled model callbacks."""

    required = {"protein_id", "pair_id", "split", "sample_index", "sequence", "sequence_hash"}
    if not required.issubset(generated.columns):
        raise V1ValidationError("generated validation table lacks required fields")
    if generated.empty or not generated["split"].eq("VALIDATION").all():
        raise V1ValidationError("validation evaluator received non-validation rows")
    validation_ids = {example.protein_id for example in examples if example.split == "VALIDATION"}
    if not set(generated["protein_id"]).issubset(validation_ids):
        raise V1ValidationError("generated rows are not a subset of VALIDATION proteins")
    by_id = {example.protein_id: example for example in examples}
    rows: list[dict[str, object]] = []
    for evaluator in ("ProteinMPNN", "ESM-IF1"):
        for protein_id, group in generated.groupby("protein_id", sort=True):
            example = by_id[str(protein_id)]
            wt_apo = float(scorer(evaluator, str(protein_id), example.sequence, "APO"))
            wt_holo = float(scorer(evaluator, str(protein_id), example.sequence, "HOLO"))
            for record in group.sort_values("sample_index", kind="mergesort").itertuples(index=False):
                score_apo = float(scorer(evaluator, str(protein_id), str(record.sequence), "APO"))
                score_holo = float(scorer(evaluator, str(protein_id), str(record.sequence), "HOLO"))
                endpoint = compatibility_endpoints(
                    apo_sequence_score=score_apo,
                    holo_sequence_score=score_holo,
                    apo_wt_score=wt_apo,
                    holo_wt_score=wt_holo,
                )
                rows.append(
                    {
                        "protein_id": str(protein_id),
                        "pair_id": str(record.pair_id),
                        "evaluator": evaluator,
                        "source_state": "CONSENSUS",
                        "sample_index": int(record.sample_index),
                        "sequence_hash": str(record.sequence_hash),
                        "score_apo": score_apo,
                        "score_holo": score_holo,
                        "wt_score_apo": wt_apo,
                        "wt_score_holo": wt_holo,
                        "c_apo": endpoint.c_apo,
                        "c_holo": endpoint.c_holo,
                        "mean_compat": endpoint.mean_compat,
                        "worst_compat": endpoint.worst_compat,
                        "state_gap": endpoint.state_gap,
                    }
                )
    scores = pd.DataFrame(rows).sort_values(["evaluator", "protein_id", "sample_index"], kind="mergesort").reset_index(drop=True)
    summaries: list[dict[str, object]] = []
    for (evaluator, protein_id), group in scores.groupby(["evaluator", "protein_id"], sort=True):
        summaries.append(
            {
                "evaluator": evaluator,
                "protein_id": protein_id,
                "n_sequences": int(len(group)),
                "mean_compat_median": float(group["mean_compat"].median()),
                "worst_compat_median": float(group["worst_compat"].median()),
                "state_gap_median": float(group["state_gap"].median()),
                "worst_compat_q10": float(group["worst_compat"].quantile(0.10)),
                "worst_compat_q90": float(group["worst_compat"].quantile(0.90)),
                "positive_worst_fraction": float((group["worst_compat"] > 0).mean()),
            }
        )
    return ValidationEvaluation(scores=scores, protein_summary=pd.DataFrame(summaries))
