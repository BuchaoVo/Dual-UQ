"""Deterministic TRAIN/VALIDATION training for the paired-state V1 model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from dual_uq.models.paired_state import (
    PairedStateConfig,
    PairedStateInputs,
    PairedStateModel,
    compute_paired_state_loss,
    make_ablation_config,
)
from dual_uq.evaluation.method_v1_data import ProteinStateExample


class V1TrainingError(ValueError):
    """Raised when a V1 training run cannot satisfy its frozen contract."""


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Shared training budget and deterministic runtime settings."""

    seed: int = 20260817
    epochs: int = 8
    learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    hidden_dim: int = 64
    lambda_worst: float = 0.5
    lambda_consistency: float = 0.5
    temperature: float = 0.5
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.seed < 0 or self.epochs <= 0 or self.learning_rate <= 0 or self.hidden_dim <= 0:
            raise ValueError("training seed, epochs, learning rate, and hidden dimension must be positive")


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Structured output for one A--F training run."""

    ablation: str
    checkpoint_path: str
    history_path: str
    best_epoch: int
    best_validation_loss: float
    train_count: int
    validation_count: int
    config: dict[str, Any]


def _seed_for(config: TrainingConfig, ablation: str, epoch: int) -> int:
    return int(config.seed + (ord(ablation) - ord("A")) * 100_003 + epoch)


def _to_inputs(example: ProteinStateExample, device: torch.device) -> PairedStateInputs:
    return PairedStateInputs(
        state_features=torch.as_tensor(example.state_features, dtype=torch.float32, device=device).unsqueeze(0),
        observability=torch.as_tensor(example.observability, dtype=torch.bool, device=device).unsqueeze(0),
        disagreement=torch.as_tensor(example.disagreement, dtype=torch.float32, device=device).unsqueeze(0),
        targets=torch.as_tensor(example.targets, dtype=torch.long, device=device).unsqueeze(0),
    )


def _prediction_logits(output: Any, config: PairedStateConfig) -> torch.Tensor:
    if config.use_single_state:
        return output.apo_logits
    if config.use_consensus:
        return output.consensus_logits
    return torch.nan_to_num(output.apo_logits, nan=0.0) * 0.5 + torch.nan_to_num(output.holo_logits, nan=0.0) * 0.5


def _run_epoch(
    model: PairedStateModel,
    examples: tuple[ProteinStateExample, ...],
    config: PairedStateConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    if optimizer is None:
        model.eval()
    else:
        model.train()
    totals = {"total": 0.0, "state": 0.0, "worst": 0.0, "consistency": 0.0, "nll": 0.0, "recovery": 0.0, "mask_utilization": 0.0, "entropy": 0.0}
    with torch.set_grad_enabled(optimizer is not None):
        for example in examples:
            inputs = _to_inputs(example, device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            output = model(inputs)
            losses = compute_paired_state_loss(output, inputs, config)
            if optimizer is not None:
                losses.total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            logits = _prediction_logits(output, config)
            target = inputs.targets
            assert target is not None
            valid = torch.isfinite(logits).all(dim=-1)
            log_prob = torch.log_softmax(logits[valid], dim=-1)
            target_values = target[valid]
            nll = torch.nn.functional.nll_loss(log_prob, target_values) if valid.any() else losses.total * 0.0
            predictions = log_prob.argmax(dim=-1) if valid.any() else torch.empty(0, device=device, dtype=torch.long)
            recovery = (predictions == target_values).float().mean() if valid.any() else losses.total * 0.0
            probabilities = log_prob.exp() if valid.any() else torch.empty((0, config.alphabet_size), device=device)
            entropy = -(probabilities * log_prob).sum(dim=-1).mean() if valid.any() else losses.total * 0.0
            totals["total"] += float(losses.total.detach())
            totals["state"] += float(losses.state.detach())
            totals["worst"] += float(losses.worst.detach())
            totals["consistency"] += float(losses.consistency.detach())
            totals["nll"] += float(nll.detach())
            totals["recovery"] += float(recovery.detach())
            totals["entropy"] += float(entropy.detach())
            totals["mask_utilization"] += float(inputs.observability.float().mean().detach())
    count = max(len(examples), 1)
    return {key: value / count for key, value in totals.items()}


def _make_model_config(name: str, config: TrainingConfig) -> PairedStateConfig:
    base = PairedStateConfig(
        hidden_dim=config.hidden_dim,
        alphabet_size=len("ACDEFGHIKLMNPQRSTVWY"),
        lambda_worst=config.lambda_worst,
        lambda_consistency=config.lambda_consistency,
        temperature=config.temperature,
    )
    return make_ablation_config(name, base)


def train_ablation(
    name: str,
    examples: tuple[ProteinStateExample, ...],
    config: TrainingConfig,
    output_dir: Path,
) -> TrainingResult:
    """Train one preregistered ablation with shared train/validation budget."""

    if name not in set("ABCDEF"):
        raise V1TrainingError("ablation name must be one of A, B, C, D, E, or F")
    train = tuple(example for example in examples if example.split == "TRAIN")
    validation = tuple(example for example in examples if example.split == "VALIDATION")
    if not train or not validation:
        raise V1TrainingError("V1 training requires non-empty TRAIN and VALIDATION examples")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise V1TrainingError("requested CUDA device is unavailable")
    torch.set_num_threads(1)
    torch.manual_seed(_seed_for(config, name, 0))
    np.random.seed(_seed_for(config, name, 0) % (2**32 - 1))
    model_config = _make_model_config(name, config)
    model = PairedStateModel(model_config).to(device)
    # Initialize LazyLinear before optimizer construction.
    with torch.no_grad():
        model(_to_inputs(train[0], device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    rows: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"ablation_{name}_best.pt"
    history_path = output_dir / f"ablation_{name}_history.parquet"
    for epoch in range(config.epochs):
        torch.manual_seed(_seed_for(config, name, epoch))
        train_metrics = _run_epoch(model, train, model_config, device, optimizer)
        validation_metrics = _run_epoch(model, validation, model_config, device, None)
        row: dict[str, Any] = {"ablation": name, "epoch": epoch + 1}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"validation_{key}": value for key, value in validation_metrics.items()})
        rows.append(row)
        if validation_metrics["total"] < best_loss:
            best_loss = validation_metrics["total"]
            best_epoch = epoch + 1
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": asdict(model_config),
                    "training_config": asdict(config),
                    "ablation": name,
                    "best_epoch": best_epoch,
                    "validation_metrics": validation_metrics,
                },
                checkpoint,
            )
    history = pd.DataFrame(rows)
    history.to_parquet(history_path, index=False)
    manifest_path = output_dir / f"ablation_{name}_run.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "dual_uq_method_v1_training_v1",
                "ablation": name,
                "train_count": len(train),
                "validation_count": len(validation),
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                "training_config": asdict(config),
                "model_config": asdict(model_config),
                "locked_test_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return TrainingResult(
        ablation=name,
        checkpoint_path=checkpoint.as_posix(),
        history_path=history_path.as_posix(),
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        train_count=len(train),
        validation_count=len(validation),
        config=asdict(config),
    )


def select_validation_checkpoint(results: tuple[TrainingResult, ...]) -> TrainingResult:
    """Select only by validation loss, with deterministic A--F tie breaking."""

    if not results:
        raise V1TrainingError("no V1 training results available for selection")
    if any(result.validation_count <= 0 for result in results):
        raise V1TrainingError("checkpoint selection requires validation results")
    return min(results, key=lambda result: (result.best_validation_loss, result.ablation))
