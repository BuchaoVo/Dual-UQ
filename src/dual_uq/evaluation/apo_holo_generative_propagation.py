"""Analysis of independent APO/HOLO generated-sequence ensembles.

The generator owns model execution and immutable shards.  This module only
consumes validated records and derives one canonical set of protein,
position, and convergence summaries.  All distances are descriptive
within-protein quantities; the protein remains the cohort-level unit.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from dual_uq.inference.apo_holo_generation import (
    GENERATION_SAMPLE_COUNT,
    ApoHoloGenerationError,
    ApoHoloGenerationRecord,
    validate_generation_records,
)
from dual_uq.models.proteinmpnn import STANDARD_AMINO_ACIDS


class ApoHoloGenerativePropagationError(ValueError):
    """Raised when generated ensembles cannot be compared on a common axis."""


@dataclass(frozen=True, slots=True)
class GenerativePropagationResult:
    """Canonical summaries derived from one validated generation record set."""

    protein_summary: pd.DataFrame
    position_shift: pd.DataFrame
    convergence: pd.DataFrame


def _hamming_distance(left: str, right: str) -> float:
    if len(left) != len(right):
        raise ApoHoloGenerativePropagationError("generated sequence lengths differ")
    return float(sum(a != b for a, b in zip(left, right))) / float(len(left))


def _mean_pairwise(sequences: tuple[str, ...]) -> float:
    values = [_hamming_distance(sequences[i], sequences[j]) for i, j in combinations(range(len(sequences)), 2)]
    return float(np.mean(values)) if values else float("nan")


def _mean_cross(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    if not left or not right:
        return float("nan")
    return float(np.mean([_hamming_distance(a, b) for a in left for b in right]))


def _js_bits(left: np.ndarray, right: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits for two probability vectors."""
    midpoint = 0.5 * (left + right)

    def _kl(probability: np.ndarray, reference: np.ndarray) -> float:
        mask = probability > 0
        return float(np.sum(probability[mask] * np.log2(probability[mask] / reference[mask])))

    return 0.5 * (_kl(left, midpoint) + _kl(right, midpoint))


def _position_distribution(sequences: tuple[str, ...], index: int) -> np.ndarray:
    counts = np.zeros(len(STANDARD_AMINO_ACIDS), dtype=np.float64)
    lookup = {aa: offset for offset, aa in enumerate(STANDARD_AMINO_ACIDS)}
    for sequence in sequences:
        try:
            counts[lookup[sequence[index]]] += 1.0
        except KeyError as exc:
            raise ApoHoloGenerativePropagationError("generated sequence contains non-standard amino acid") from exc
    return counts / float(len(sequences))


def _record_groups(
    records: tuple[ApoHoloGenerationRecord, ...],
) -> dict[str, dict[str, tuple[ApoHoloGenerationRecord, ...]]]:
    groups: dict[str, dict[str, list[ApoHoloGenerationRecord]]] = {}
    for record in records:
        groups.setdefault(record.condition.protein_id, {}).setdefault(record.condition.state, []).append(record)
    return {
        protein: {
            state: tuple(sorted(values, key=lambda item: item.sample_index))
            for state, values in states.items()
        }
        for protein, states in groups.items()
    }


def summarize_generation_propagation(
    records: Iterable[ApoHoloGenerationRecord],
    *,
    prefixes: tuple[int, ...] = (16, 32, 64),
) -> GenerativePropagationResult:
    """Derive nested diversity and position-wise JS summaries.

    ``records`` must contain exactly two conditions and 64 samples per protein;
    prefixes are nested prefixes of that fixed ensemble.  No cross-protein
    pooling is performed.
    """
    try:
        prefix_values = tuple(prefixes)
        if not prefix_values or any(type(value) is not int for value in prefix_values):
            raise ApoHoloGenerativePropagationError("prefixes must be integer sample counts")
        if tuple(sorted(set(prefix_values))) != prefix_values or any(
            value <= 0 or value > GENERATION_SAMPLE_COUNT for value in prefix_values
        ):
            raise ApoHoloGenerativePropagationError("prefixes must be increasing values in 1..64")
        materialized = tuple(records)
        proteins = {record.condition.protein_id for record in materialized}
        validated = validate_generation_records(materialized, expected_protein_count=len(proteins))
    except (ApoHoloGenerationError, ValueError) as exc:
        if isinstance(exc, ApoHoloGenerativePropagationError):
            raise
        raise ApoHoloGenerativePropagationError(str(exc)) from exc
    groups = _record_groups(validated)

    protein_rows: list[dict[str, object]] = []
    convergence_rows: list[dict[str, object]] = []
    position_rows: list[dict[str, object]] = []
    for protein_id in sorted(groups):
        states = groups[protein_id]
        if set(states) != {"APO", "HOLO"}:
            raise ApoHoloGenerativePropagationError(f"APO/HOLO states are incomplete: {protein_id}")
        apo_records = states["APO"]
        holo_records = states["HOLO"]
        apo_condition = apo_records[0].condition
        holo_condition = holo_records[0].condition
        if (
            apo_condition.canonical_positions != holo_condition.canonical_positions
            or apo_condition.wt_sequence_projection != holo_condition.wt_sequence_projection
        ):
            raise ApoHoloGenerativePropagationError(f"APO/HOLO sequence axes differ: {protein_id}")
        positions = apo_condition.canonical_positions
        protein_position_rows: list[dict[str, object]] = [
            {
                "protein_id": protein_id,
                "canonical_position": canonical_position,
            }
            for canonical_position in positions
        ]
        for prefix in prefix_values:
            apo = tuple(item.sequence for item in apo_records[:prefix])
            holo = tuple(item.sequence for item in holo_records[:prefix])
            d_apo = _mean_pairwise(apo)
            d_holo = _mean_pairwise(holo)
            d_ah = _mean_cross(apo, holo)
            d_excess = d_ah - 0.5 * (d_apo + d_holo)
            convergence_rows.append(
                {
                    "protein_id": protein_id,
                    "sample_count": prefix,
                    "d_apo": d_apo,
                    "d_holo": d_holo,
                    "d_ah": d_ah,
                    "d_excess": d_excess,
                }
            )
            for index, canonical_position in enumerate(positions):
                apo_distribution = _position_distribution(apo, index)
                holo_distribution = _position_distribution(holo, index)
                protein_position_rows[index][f"js_bits_{prefix}"] = _js_bits(
                    apo_distribution, holo_distribution
                )
                protein_position_rows[index][f"apo_entropy_bits_{prefix}"] = float(
                    -np.sum(
                        apo_distribution[apo_distribution > 0]
                        * np.log2(apo_distribution[apo_distribution > 0])
                    )
                )
                protein_position_rows[index][f"holo_entropy_bits_{prefix}"] = float(
                    -np.sum(
                        holo_distribution[holo_distribution > 0]
                        * np.log2(holo_distribution[holo_distribution > 0])
                    )
                )
            if prefix == prefix_values[-1]:
                position_rows.extend(protein_position_rows)
                final_js = [row[f"js_bits_{prefix}"] for row in protein_position_rows]
                protein_rows.append(
                    {
                        "protein_id": protein_id,
                        "d_apo_64": d_apo,
                        "d_holo_64": d_holo,
                        "d_ah_64": d_ah,
                        "d_excess_64": d_excess,
                        "js_bits_mean_64": float(np.mean(final_js)),
                        "js_bits_q90_64": float(np.quantile(final_js, 0.9, method="linear")),
                    }
                )
    protein_summary = pd.DataFrame(protein_rows).sort_values("protein_id", kind="mergesort").reset_index(drop=True)
    convergence = pd.DataFrame(convergence_rows).sort_values(["protein_id", "sample_count"], kind="mergesort").reset_index(drop=True)
    position_shift = pd.DataFrame(position_rows).sort_values(["protein_id", "canonical_position"], kind="mergesort").reset_index(drop=True)
    return GenerativePropagationResult(
        protein_summary=protein_summary,
        position_shift=position_shift,
        convergence=convergence,
    )
