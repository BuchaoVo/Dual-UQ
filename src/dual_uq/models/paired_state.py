"""Minimal masked paired-state inverse-folding model contract.

This module owns only the reusable two-state representation and objective.  It
does not load a checkpoint, call an evaluator, or define a scientific endpoint.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
import torch
from torch import Tensor, nn
from torch.nn import functional as functional


@dataclass(frozen=True, slots=True)
class PairedStateInputs:
    """Aligned two-state features, observability, disagreement, and targets."""

    state_features: Tensor
    observability: Tensor
    disagreement: Tensor
    targets: Tensor | None = None


@dataclass(frozen=True, slots=True)
class PairedStateConfig:
    """Configuration for the small paired-state representation and loss."""

    hidden_dim: int = 128
    alphabet_size: int = 20
    lambda_worst: float = 0.5
    lambda_consistency: float = 0.5
    temperature: float = 0.5
    use_single_state: bool = False
    use_consensus: bool = True
    use_worst_state: bool = True
    use_selective_consistency: bool = True
    ablation: str = "F"

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.alphabet_size <= 1:
            raise ValueError("hidden_dim must be positive and alphabet_size must exceed one")
        if self.lambda_worst < 0 or self.lambda_consistency < 0 or self.temperature <= 0:
            raise ValueError("loss weights must be non-negative and temperature must be positive")


@dataclass(frozen=True, slots=True)
class PairedStateOutput:
    """State-specific logits plus masked consensus and consistency weights."""

    apo_logits: Tensor
    holo_logits: Tensor
    consensus: Tensor
    consensus_logits: Tensor
    weights: Tensor


class LossComponents(Mapping[str, Tensor]):
    """Named loss components that also support mapping-style iteration."""

    __slots__ = ("state", "worst", "consistency", "total")

    def __init__(self, *, state: Tensor, worst: Tensor, consistency: Tensor, total: Tensor) -> None:
        self.state = state
        self.worst = worst
        self.consistency = consistency
        self.total = total

    def __getitem__(self, key: str) -> Tensor:
        if key not in self:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(("state", "worst", "consistency", "total"))

    def __len__(self) -> int:
        return 4


def _check_inputs(inputs: PairedStateInputs) -> tuple[Tensor, Tensor, Tensor]:
    features = inputs.state_features
    observed = inputs.observability
    disagreement = inputs.disagreement
    if features.ndim != 4 or features.shape[1] != 2:
        raise ValueError("state_features must have shape [batch, 2, length, features]")
    if observed.shape != features.shape[:3] or observed.dtype is not torch.bool:
        raise ValueError("observability must be boolean [batch, 2, length]")
    if disagreement.ndim != 3 or disagreement.shape[:2] != features.shape[:1] + features.shape[2:3]:
        raise ValueError("disagreement must have shape [batch, length, descriptors]")
    if not torch.isfinite(features).all() or not torch.isfinite(disagreement).all():
        raise ValueError("state features and disagreement must be finite")
    if inputs.targets is not None:
        if inputs.targets.shape != features.shape[:1] + features.shape[2:3]:
            raise ValueError("targets must have shape [batch, length]")
        if inputs.targets.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8):
            raise ValueError("targets must be an integer tensor")
    return features, observed, disagreement


class PairedStateModel(nn.Module):
    """Small state-specific encoder with explicit missing-coordinate masks."""

    def __init__(self, config: PairedStateConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.LazyLinear(config.hidden_dim)
        self.activation = nn.GELU()
        self.head = nn.Linear(config.hidden_dim, config.alphabet_size)

    def forward(self, inputs: PairedStateInputs) -> PairedStateOutput:
        features, observed, disagreement = _check_inputs(inputs)
        batch, _, length, feature_count = features.shape
        encoded = self.activation(self.encoder(features.reshape(batch * 2 * length, feature_count)))
        encoded = encoded.reshape(batch, 2, length, self.config.hidden_dim)
        logits = self.head(encoded)
        state_reliability = torch.exp(-disagreement.abs().mean(dim=-1)).unsqueeze(1)
        observed_float = observed.to(dtype=encoded.dtype)
        if self.config.use_single_state:
            consensus = encoded[:, 0]
            consensus_logits = logits[:, 0]
        elif self.config.use_consensus:
            weighted = encoded * observed_float.unsqueeze(-1) * state_reliability.unsqueeze(-1)
            denominator = (observed_float * state_reliability).sum(dim=1, keepdim=True).clamp_min(1e-8)
            consensus = weighted.sum(dim=1) / denominator.squeeze(1).unsqueeze(-1)
            logit_weighted = logits * observed_float.unsqueeze(-1) * state_reliability.unsqueeze(-1)
            consensus_logits = logit_weighted.sum(dim=1) / denominator.squeeze(1).unsqueeze(-1)
        else:
            weighted = encoded * observed_float.unsqueeze(-1)
            denominator = observed_float.sum(dim=1, keepdim=True).clamp_min(1.0)
            consensus = weighted.sum(dim=1) / denominator.squeeze(1).unsqueeze(-1)
            consensus_logits = (logits * observed_float.unsqueeze(-1)).sum(dim=1) / denominator.squeeze(1).unsqueeze(-1)
        both_observed = observed[:, 0] & observed[:, 1]
        weights = torch.exp(-disagreement.abs().mean(dim=-1)) * both_observed.to(encoded.dtype)
        missing_value = torch.full_like(logits, float("nan"))
        apo_logits = torch.where(observed[:, 0].unsqueeze(-1), logits[:, 0], missing_value[:, 0])
        if self.config.use_single_state:
            holo_logits = missing_value[:, 1]
        else:
            holo_logits = torch.where(observed[:, 1].unsqueeze(-1), logits[:, 1], missing_value[:, 1])
        return PairedStateOutput(
            apo_logits=apo_logits,
            holo_logits=holo_logits,
            consensus=consensus,
            consensus_logits=consensus_logits,
            weights=weights,
        )


def _state_loss(logits: Tensor, observed: Tensor, targets: Tensor) -> Tensor:
    valid = observed & torch.isfinite(logits).all(dim=-1)
    if not valid.any():
        return logits.nan_to_num().sum() * 0.0
    return functional.cross_entropy(logits[valid], targets[valid])


def _js_loss(apo_logits: Tensor, holo_logits: Tensor, mask: Tensor) -> Tensor:
    if not mask.any():
        return apo_logits.nan_to_num().sum() * 0.0
    apo_logp = functional.log_softmax(apo_logits[mask], dim=-1)
    holo_logp = functional.log_softmax(holo_logits[mask], dim=-1)
    apo_p = apo_logp.exp()
    holo_p = holo_logp.exp()
    midpoint = (apo_p + holo_p).mul(0.5)
    log_midpoint = midpoint.clamp_min(torch.finfo(midpoint.dtype).tiny).log()
    js = 0.5 * ((apo_p * (apo_logp - log_midpoint)).sum(-1) + (holo_p * (holo_logp - log_midpoint)).sum(-1))
    return js.mean()


def compute_paired_state_loss(
    output: PairedStateOutput,
    inputs: PairedStateInputs,
    config: PairedStateConfig,
) -> LossComponents:
    """Compute state mean, differentiable soft-worst, and masked JS consistency."""

    _, observed, _ = _check_inputs(inputs)
    if inputs.targets is None:
        raise ValueError("targets are required to compute supervised paired-state loss")
    targets = inputs.targets.to(dtype=torch.long)
    apo_loss = _state_loss(output.apo_logits, observed[:, 0], targets)
    if config.use_single_state:
        state_values = apo_loss.unsqueeze(0)
        available = torch.ones(1, dtype=torch.bool, device=apo_loss.device)
    else:
        holo_loss = _state_loss(output.holo_logits, observed[:, 1], targets)
        state_values = torch.stack([apo_loss, holo_loss])
        available = torch.stack(
            [observed[:, 0].any(), observed[:, 1].any()], dim=0
        )
    state = state_values[available].mean()
    if config.use_worst_state:
        worst = config.temperature * torch.logsumexp(state_values[available] / config.temperature, dim=0)
    else:
        worst = state.detach() * 0.0
    if config.use_single_state:
        consistency = state.detach() * 0.0
    else:
        consistency = _js_loss(output.apo_logits, output.holo_logits, observed[:, 0] & observed[:, 1])
    if config.use_selective_consistency and not config.use_single_state:
        both = observed[:, 0] & observed[:, 1]
        if both.any():
            # Reweight the per-residue JS by the frozen disagreement-derived weight.
            apo_logp = functional.log_softmax(output.apo_logits[both], dim=-1)
            holo_logp = functional.log_softmax(output.holo_logits[both], dim=-1)
            apo_p = apo_logp.exp()
            holo_p = holo_logp.exp()
            midpoint = (apo_p + holo_p).mul(0.5)
            log_midpoint = midpoint.clamp_min(torch.finfo(midpoint.dtype).tiny).log()
            per_position = 0.5 * ((apo_p * (apo_logp - log_midpoint)).sum(-1) + (holo_p * (holo_logp - log_midpoint)).sum(-1))
            weights = output.weights[both]
            consistency = (per_position * weights).sum() / weights.sum().clamp_min(1e-8)
    else:
        consistency = consistency.detach() * 0.0
    total = state + config.lambda_worst * worst + config.lambda_consistency * consistency
    return LossComponents(state=state, worst=worst, consistency=consistency, total=total)


def make_ablation_config(name: str, base: PairedStateConfig) -> PairedStateConfig:
    """Select one of the frozen A--F component combinations."""

    if name not in set("ABCDEF"):
        raise ValueError("ablation name must be one of A, B, C, D, E, or F")
    settings: dict[str, dict[str, object]] = {
        "A": {"use_single_state": True, "use_consensus": False, "use_worst_state": False, "use_selective_consistency": False, "lambda_worst": 0.0, "lambda_consistency": 0.0},
        "B": {"use_single_state": False, "use_consensus": False, "use_worst_state": False, "use_selective_consistency": False, "lambda_worst": 0.0, "lambda_consistency": 0.0},
        "C": {"use_single_state": False, "use_consensus": True, "use_worst_state": False, "use_selective_consistency": False, "lambda_worst": 0.0, "lambda_consistency": 0.0},
        "D": {"use_single_state": False, "use_consensus": True, "use_worst_state": True, "use_selective_consistency": False, "lambda_consistency": 0.0},
        "E": {"use_single_state": False, "use_consensus": True, "use_worst_state": False, "use_selective_consistency": True, "lambda_worst": 0.0},
        "F": {"use_single_state": False, "use_consensus": True, "use_worst_state": True, "use_selective_consistency": True},
    }
    return replace(base, ablation=name, **settings[name])
