"""Model adapters and model-independent scoring request/result contracts."""

from dual_uq.models.capabilities import ModelCapability
from dual_uq.models.interface import InverseFoldingAdapter
from dual_uq.models.registry import ModelRegistry
from dual_uq.models.scoring import (
    CandidateCollection,
    ScoreDispatchError,
    ScorerBinding,
    ScoreRecord,
    ScoreRequest,
    ScoringVariant,
    SequenceScorer,
    VariantKind,
    execute_score_request,
)

__all__ = [
    "CandidateCollection", "InverseFoldingAdapter", "ModelCapability", "ModelRegistry",
    "ScoreDispatchError", "ScoreRecord", "ScoreRequest", "ScorerBinding", "ScoringVariant",
    "SequenceScorer", "VariantKind", "execute_score_request",
]
