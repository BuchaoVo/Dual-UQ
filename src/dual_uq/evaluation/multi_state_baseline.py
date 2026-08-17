"""Pure compatibility calculations for the multi-state baseline experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class CompatibilityEndpoints:
    c_apo: float
    c_holo: float
    mean_compat: float
    worst_compat: float
    state_gap: float


def add_compatibility_endpoints(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Attach WT-normalized endpoints to raw sequence/state score rows.

    Rows must contain ``protein_id``, ``source_state``, ``evaluated_state``,
    ``score`` and ``wt_score``.  The function is intentionally evaluator-agnostic;
    caller-provided rows remain separated by evaluator identity.
    """
    result: list[dict[str, object]] = []
    for row in rows:
        copied = dict(row)
        endpoint = compatibility_endpoints(
            apo_sequence_score=float(row["score_apo"]),
            holo_sequence_score=float(row["score_holo"]),
            apo_wt_score=float(row["wt_score_apo"]),
            holo_wt_score=float(row["wt_score_holo"]),
        )
        copied.update(
            c_apo=endpoint.c_apo,
            c_holo=endpoint.c_holo,
            mean_compat=endpoint.mean_compat,
            worst_compat=endpoint.worst_compat,
            state_gap=endpoint.state_gap,
        )
        result.append(copied)
    return result


def compatibility_endpoints(
    *, apo_sequence_score: float, holo_sequence_score: float,
    apo_wt_score: float, holo_wt_score: float,
) -> CompatibilityEndpoints:
    """Return WT-normalized compatibility endpoints for one sequence."""
    values = (apo_sequence_score, holo_sequence_score, apo_wt_score, holo_wt_score)
    if not np.isfinite(values).all():
        raise ValueError("compatibility scores must be finite")
    c_apo = float(apo_sequence_score - apo_wt_score)
    c_holo = float(holo_sequence_score - holo_wt_score)
    return CompatibilityEndpoints(
        c_apo=c_apo,
        c_holo=c_holo,
        mean_compat=(c_apo + c_holo) / 2.0,
        worst_compat=min(c_apo, c_holo),
        state_gap=abs(c_apo - c_holo),
    )


def aggregate_equal_weight_log_probabilities(
    apo_log_probabilities: np.ndarray, holo_log_probabilities: np.ndarray
) -> np.ndarray:
    """Combine two model-native log distributions with equal log weights."""
    apo = np.asarray(apo_log_probabilities, dtype=np.float64)
    holo = np.asarray(holo_log_probabilities, dtype=np.float64)
    if apo.shape != holo.shape or apo.ndim != 1 or apo.size != 20:
        raise ValueError("APO and HOLO probabilities must be matching 20-AA vectors")
    if not np.isfinite(apo).all() or not np.isfinite(holo).all():
        raise ValueError("log probabilities must be finite")
    combined = 0.5 * apo + 0.5 * holo
    combined -= np.max(combined)
    normalizer = np.exp(combined).sum()
    if not np.isfinite(normalizer) or normalizer <= 0:
        raise ValueError("combined probability distribution is invalid")
    return combined - np.log(normalizer)
