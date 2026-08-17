"""CPU-only synthetic smoke for the paired-state method contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch

from dual_uq.models.paired_state import (
    PairedStateConfig,
    PairedStateInputs,
    PairedStateModel,
    compute_paired_state_loss,
    make_ablation_config,
)


@dataclass(frozen=True, slots=True)
class SmokeResult:
    """Structured synthetic smoke result, independent of production artifacts."""

    ablation_table: pd.DataFrame
    mask_checks: dict[str, bool]
    gradient_checks: dict[str, bool]
    summary: dict[str, Any]


def _synthetic_inputs(*, seed: int, n_proteins: int, length: int, disagreement: float) -> PairedStateInputs:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    features = torch.randn(n_proteins, 2, length, 6, generator=generator)
    observability = torch.ones(n_proteins, 2, length, dtype=torch.bool)
    if n_proteins > 0 and length > 2:
        observability[0, 1, 2] = False
    disagreement_values = torch.full((n_proteins, length, 3), disagreement)
    targets = torch.randint(0, 20, (n_proteins, length), generator=generator)
    return PairedStateInputs(
        state_features=features,
        observability=observability,
        disagreement=disagreement_values,
        targets=targets,
    )


def run_paired_state_smoke(*, seed: int, n_proteins: int, length: int) -> SmokeResult:
    """Run deterministic CPU checks without loading a model or production input."""

    if seed < 0 or n_proteins <= 0 or length <= 0:
        raise ValueError("seed, n_proteins, and length must be positive")
    torch.set_num_threads(1)
    base = PairedStateConfig(hidden_dim=16, alphabet_size=20, lambda_worst=0.5, lambda_consistency=0.5)
    rows: list[dict[str, Any]] = []
    gradient_finite = True
    for offset, name in enumerate("ABCDEF"):
        config = make_ablation_config(name, base)
        inputs = _synthetic_inputs(seed=seed, n_proteins=n_proteins, length=length, disagreement=0.2)
        torch.manual_seed(seed + offset)
        model = PairedStateModel(config)
        output = model(inputs)
        losses = compute_paired_state_loss(output, inputs, config)
        losses.total.backward()
        finite_grads = all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        gradient_finite = gradient_finite and bool(finite_grads)
        rows.append(
            {
                "ablation": name,
                "state": float(losses.state.detach()),
                "worst": float(losses.worst.detach()),
                "consistency": float(losses.consistency.detach()),
                "total": float(losses.total.detach()),
                "finite_forward": bool(torch.isfinite(output.apo_logits).all() and torch.isfinite(output.consensus).all()),
                "finite_loss": bool(torch.isfinite(losses.total)),
                "finite_gradient": bool(finite_grads),
            }
        )

    full_model = PairedStateModel(base)
    full_inputs = _synthetic_inputs(seed=seed, n_proteins=n_proteins, length=length, disagreement=0.2)
    full_output = full_model(full_inputs)
    low_output = full_model(_synthetic_inputs(seed=seed, n_proteins=n_proteins, length=length, disagreement=0.1))
    high_output = full_model(_synthetic_inputs(seed=seed, n_proteins=n_proteins, length=length, disagreement=1.0))
    mask_checks = {
        "missing_state_logits_nan": bool(torch.isnan(full_output.holo_logits[0, 2]).all()),
        "consensus_finite": bool(torch.isfinite(full_output.consensus).all()),
        "no_coordinate_imputation": bool(torch.isnan(full_output.holo_logits[0, 2]).all()),
    }
    low_weight = float(low_output.weights.mean())
    high_weight = float(high_output.weights.mean())
    gradient_checks = {"all_finite": gradient_finite}
    table = pd.DataFrame(rows)
    summary: dict[str, Any] = {
        "schema_version": "dual_uq_paired_state_smoke_v1",
        "seed": int(seed),
        "n_proteins": int(n_proteins),
        "length": int(length),
        "ablations": list("ABCDEF"),
        "mask_checks": mask_checks,
        "gradient_checks": gradient_checks,
        "low_disagreement_weight": round(low_weight, 12),
        "high_disagreement_weight": round(high_weight, 12),
        "no_production_inputs": True,
    }
    return SmokeResult(
        ablation_table=table,
        mask_checks=mask_checks,
        gradient_checks=gradient_checks,
        summary=summary,
    )
