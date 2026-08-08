"""Pairwise decision-boundary audit for the frozen Stage0-2A scores.

This module consumes only frozen score and mechanism releases.  It never runs
ProteinMPNN, generates sequences, or creates a repeat-pair artifact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dual_uq.core.hashing import sha256_file
from dual_uq.dataset.fixed_probe_decision_sensitivity import (
    REPEAT_COUNT,
    STANDARD_AMINO_ACIDS,
    DecisionInputs,
    rank_local_amino_acid_scores,
)
from dual_uq.dataset.fixed_probe_mechanism import (
    EXPECTED_POSITION_COUNT,
    EXPECTED_PROTEIN_COUNT,
    load_frozen_mechanism_inputs,
)
from dual_uq.dataset.fixed_probe_sensitivity import (
    write_immutable_json,
    write_immutable_parquet,
)

MECHANISM_MANIFEST_PATH = Path(
    "experiments/p2_design_baseline/stage0/fixed_probe_mechanism_manifest.json"
)
MECHANISM_MANIFEST_SHA256 = (
    "caab12fe4e290164d513c78decbfe07269ebce8ec207c0ca02c7cd1b2f85f14b"
)
MECHANISM_POSITION_SHA256 = (
    "4d295053103ec49522613cc8aefc8d1a1c30c60d5406bcfa235eaa61ed5d665e"
)
MECHANISM_ASSOCIATION_SHA256 = (
    "65db52c1c72a2b0eae1b5cd665822c8c0a0e38e4550845fb2ec29d47e4b72f6b"
)
PER_RESIDUE_DECOMPOSITION_AVAILABLE = "PER_RESIDUE_DECOMPOSITION_AVAILABLE"
PER_RESIDUE_DECOMPOSITION_NOT_AVAILABLE = (
    "PER_RESIDUE_DECOMPOSITION_NOT_AVAILABLE"
)
TOLERANCE = 1.0e-12


class RepresentationAuditError(ValueError):
    """Structured boundary-audit failure."""

    def __init__(self, code: str, message: str, *, outcome: str = "BLOCKED") -> None:
        self.code = code
        self.outcome = outcome
        super().__init__(message)


@dataclass(frozen=True)
class BoundaryAuditResult:
    """One canonical result rendered into the two tables and manifest."""

    project_root: Path
    position_boundary: pd.DataFrame
    protein_associations: pd.DataFrame
    manifest_fields: dict[str, Any]
    input_provenance: dict[str, Any]


def pairwise_gap_sd(sd_pdb: float, sd_afdb: float) -> float:
    """Use a symmetric RMS of same-state population SDs."""
    values = np.asarray([sd_pdb, sd_afdb], dtype=np.float64)
    if not np.isfinite(values).all():
        raise RepresentationAuditError(
            "nonfinite_pairwise_gap_sd", "Pairwise gap SD inputs are nonfinite"
        )
    if (values < 0).any():
        raise RepresentationAuditError(
            "invalid_pairwise_gap_sd", "Pairwise gap SD inputs must be finite and nonnegative"
        )
    return float(np.sqrt(np.mean(values**2)))


def assess_representation_identifiability(columns: set[str]) -> dict[str, str]:
    """Report whether frozen artifacts retain per-residue score contributions."""
    required = {"per_residue_log_probability", "target_position_log_probability"}
    if required.issubset(columns):
        return {
            "status": PER_RESIDUE_DECOMPOSITION_AVAILABLE,
            "statement": "Per-residue score contributions are present in the frozen schema.",
        }
    return {
        "status": PER_RESIDUE_DECOMPOSITION_NOT_AVAILABLE,
        "statement": (
            "Direct-versus-propagated autoregressive contribution is not identifiable "
            "from the frozen Stage0 artifacts."
        ),
    }


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RepresentationAuditError(
            "boundary_schema_mismatch", f"{label} is missing columns: {missing}"
        )


def _score_map(group: pd.DataFrame) -> dict[str, float]:
    scores = pd.to_numeric(group["score_mean_logp_mask"], errors="coerce")
    if not np.isfinite(scores.to_numpy(dtype=np.float64)).all():
        raise RepresentationAuditError(
            "nonfinite_pairwise_score", "Boundary landscape contains a nonfinite score"
        )
    return dict(zip(group["aa"].astype(str), scores.astype(float), strict=True))


def _build_fast_local_scores(inputs: DecisionInputs) -> pd.DataFrame:
    """Expand the already validated raw/WT tables without analytical groupby work."""
    scoring = inputs.scoring
    fixed = scoring.fixed_probes[["protein_id", "position", "wt_aa"]].drop_duplicates()
    raw = scoring.raw_scores
    mutant = pd.DataFrame(
        {
            "protein_id": raw["protein_id"].astype(str),
            "position": raw["position"].astype(int),
            "wt_aa": raw["wt_aa"].astype(str),
            "aa": raw["mut_aa"].astype(str),
            "is_wt": False,
            "sequence_hash": raw["sequence_hash"].astype(str),
            "backbone_condition": raw["backbone_condition"].astype(str),
            "backbone_sha256": raw["backbone_sha256"].astype(str),
            "repeat_index": raw["repeat_index"].astype(int),
            "seed": raw["seed"].astype(int),
            "decoding_realization_sha256": raw["decoding_realization_sha256"].astype(str),
            "score_mean_logp_mask": raw["score_mean_logp_mask"].astype(float),
        }
    )
    wt = fixed.merge(
        scoring.wt_scores,
        on="protein_id",
        how="inner",
        validate="many_to_many",
        sort=False,
    )
    wild_type = pd.DataFrame(
        {
            "protein_id": wt["protein_id"].astype(str),
            "position": wt["position"].astype(int),
            "wt_aa": wt["wt_aa"].astype(str),
            "aa": wt["wt_aa"].astype(str),
            "is_wt": True,
            "sequence_hash": pd.Series([pd.NA] * len(wt), dtype="string"),
            "backbone_condition": wt["backbone_condition"].astype(str),
            "backbone_sha256": wt["backbone_sha256"].astype(str),
            "repeat_index": wt["repeat_index"].astype(int),
            "seed": wt["seed"].astype(int),
            "decoding_realization_sha256": wt["decoding_realization_sha256"].astype(str),
            "score_mean_logp_mask": wt["score_mean_logp_mask"].astype(float),
        }
    )
    local = pd.concat([wild_type, mutant], ignore_index=True)
    if not np.isfinite(local["score_mean_logp_mask"].to_numpy(dtype=np.float64)).all():
        raise RepresentationAuditError("nonfinite_local_score", "Frozen local scores contain a nonfinite value")
    return local


def _pair_stats(
    ranked_scores: pd.DataFrame,
    *,
    protein_id: str,
    position: int,
    pair_a: str,
    pair_b: str,
    condition: str,
    repeat_count: int,
) -> dict[str, float]:
    frame = ranked_scores.loc[
        (ranked_scores["protein_id"].astype(str) == protein_id)
        & (ranked_scores["position"].astype(int) == position)
        & (ranked_scores["backbone_condition"].astype(str) == condition)
        & (ranked_scores["aa"].astype(str).isin([pair_a, pair_b]))
    ].copy()
    grouped = frame.groupby("repeat_index", sort=True)
    expected = set(range(repeat_count))
    if set(grouped.groups) != expected or len(frame) != repeat_count * 2:
        raise RepresentationAuditError(
            "pairwise_repeat_grid_mismatch",
            f"Fixed pair {pair_a}>{pair_b} lacks the complete {condition} repeat grid",
        )
    values: list[float] = []
    for _, repeat in grouped:
        scores = _score_map(repeat)
        values.append(scores[pair_a] - scores[pair_b])
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise RepresentationAuditError(
            "nonfinite_pairwise_gap", "Pairwise gap values are not finite"
        )
    mean = float(array.mean())
    sd = float(array.std(ddof=0))
    return {
        "mean": mean,
        "sd": sd,
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "sign_change_fraction": float(np.mean(array < 0.0)),
    }


def build_pairwise_boundary_records(
    ranked_scores: pd.DataFrame, *, repeat_count: int = REPEAT_COUNT
) -> pd.DataFrame:
    """Build an in-memory explicit-pair table; callers aggregate and discard it."""
    required = {
        "protein_id",
        "position",
        "backbone_condition",
        "repeat_index",
        "seed",
        "decoding_realization_sha256",
        "aa",
        "rank",
        "score_mean_logp_mask",
    }
    _require_columns(ranked_scores, required, "ranked scores")
    if set(ranked_scores["aa"].astype(str)) != set(STANDARD_AMINO_ACIDS):
        raise RepresentationAuditError(
            "local_amino_acid_grid_mismatch",
            "Boundary landscapes must use the standard 20 amino acids",
        )
    group_keys = ["protein_id", "position", "backbone_condition", "repeat_index"]
    if ranked_scores.duplicated(group_keys + ["aa"]).any():
        raise RepresentationAuditError(
            "duplicate_pairwise_score", "Boundary landscape repeats an amino acid"
        )
    groups = ranked_scores.groupby(group_keys, sort=False)
    if not (groups.size() == len(STANDARD_AMINO_ACIDS)).all():
        raise RepresentationAuditError(
            "local_amino_acid_grid_mismatch", "Each boundary landscape must have 20 amino acids"
        )
    rows: list[dict[str, Any]] = []
    # rank_local_amino_acid_scores emits complete 20-row landscapes contiguously;
    # use that invariant to avoid repeatedly copying large pandas groups.
    if len(ranked_scores) % 20:
        raise RepresentationAuditError(
            "local_amino_acid_grid_mismatch", "Ranked score table is not divisible into 20-AA blocks"
        )
    block_count = len(ranked_scores) // 20
    score_array = ranked_scores["score_mean_logp_mask"].to_numpy(dtype=np.float64).reshape(
        block_count, 20
    )
    aa_array = ranked_scores["aa"].astype(str).to_numpy().reshape(block_count, 20)
    realization_array = ranked_scores["decoding_realization_sha256"].astype(str).to_numpy()[::20]
    metadata = ranked_scores[group_keys].to_numpy()[::20]
    block_lookup: dict[tuple[str, int, str, int], int] = {}
    landscape_blocks: dict[tuple[str, int, str], list[tuple[int, int]]] = {}
    for block_index, (protein_id, position, condition, repeat_index) in enumerate(metadata):
        key = (str(protein_id), int(position), str(condition), int(repeat_index))
        block_lookup[key] = block_index
        landscape_blocks.setdefault(key[:3], []).append((key[3], block_index))
    landscape_arrays: dict[tuple[str, int, str], tuple[dict[str, int], np.ndarray]] = {}
    for key, blocks in landscape_blocks.items():
        blocks.sort()
        if [repeat for repeat, _ in blocks] != list(range(repeat_count)):
            raise RepresentationAuditError(
                "pairwise_repeat_grid_mismatch", f"Incomplete repeat grid for {key}"
            )
        aa_order = {aa: index for index, aa in enumerate(aa_array[blocks[0][1]])}
        landscape_arrays[key] = (
            aa_order,
            np.stack([score_array[index] for _, index in blocks]),
        )
    for block_index, (protein_id, position, condition, repeat_index) in enumerate(metadata):
        protein_id, position, condition, repeat_index = (
            str(protein_id), int(position), str(condition), int(repeat_index)
        )
        block_aas = aa_array[block_index]
        block_scores = score_array[block_index]
        if list(ranked_scores.iloc[block_index * 20 : block_index * 20 + 2]["rank"].astype(int)) != [1, 2]:
            raise RepresentationAuditError(
                "local_rank_grid_mismatch", "Top-1 and Top-2 ranks are not available"
            )
        pair_a, pair_b = str(block_aas[0]), str(block_aas[1])
        source_map = {str(aa): float(score) for aa, score in zip(block_aas, block_scores, strict=True)}
        other_condition = "AFDB" if str(condition) == "PDB" else "PDB"
        other_block = block_lookup[(protein_id, position, other_condition, repeat_index)]
        other_aas = aa_array[other_block]
        other_scores = score_array[other_block]
        other_map = {str(aa): float(score) for aa, score in zip(other_aas, other_scores, strict=True)}
        baseline_gap = source_map[pair_a] - source_map[pair_b]
        other_gap = other_map[pair_a] - other_map[pair_b]
        shift = other_gap - baseline_gap if str(condition) == "PDB" else baseline_gap - other_gap
        try:
            pdb_order, pdb_matrix = landscape_arrays[(protein_id, position, "PDB")]
            afdb_order, afdb_matrix = landscape_arrays[(protein_id, position, "AFDB")]
            pdb_values = pdb_matrix[:, pdb_order[pair_a]] - pdb_matrix[:, pdb_order[pair_b]]
            afdb_values = afdb_matrix[:, afdb_order[pair_a]] - afdb_matrix[:, afdb_order[pair_b]]
        except (KeyError, IndexError) as exc:
            raise RepresentationAuditError(
                "pairwise_repeat_grid_mismatch",
                f"Fixed pair {pair_a}>{pair_b} is missing from one condition",
            ) from exc
        if len(pdb_values) != repeat_count or len(afdb_values) != repeat_count:
            raise RepresentationAuditError(
                "pairwise_repeat_grid_mismatch",
                f"Fixed pair {pair_a}>{pair_b} lacks the complete repeat grid",
            )
        pdb_array = np.asarray(pdb_values, dtype=np.float64)
        afdb_array = np.asarray(afdb_values, dtype=np.float64)
        pdb_stats = {
            "mean": float(pdb_array.mean()),
            "sd": float(pdb_array.std(ddof=0)),
            "sign_change_fraction": float(np.mean(pdb_array < 0.0)),
        }
        afdb_stats = {
            "mean": float(afdb_array.mean()),
            "sd": float(afdb_array.std(ddof=0)),
            "sign_change_fraction": float(np.mean(afdb_array < 0.0)),
        }
        scale = pairwise_gap_sd(pdb_stats["sd"], afdb_stats["sd"])
        rows.append(
            {
                "protein_id": str(protein_id),
                "position": int(position),
                "repeat_index": int(repeat_index),
                "boundary_source": str(condition),
                "pair_id": f"{pair_a}>{pair_b}",
                "winner_aa": pair_a,
                "runner_up_aa": pair_b,
                "baseline_pairwise_margin": float(baseline_gap),
                "structural_gap_shift": float(shift),
                "absolute_structural_gap_shift": abs(float(shift)),
                "pairwise_gap_sd_pdb": pdb_stats["sd"],
                "pairwise_gap_sd_afdb": afdb_stats["sd"],
                "pairwise_gap_sd": scale,
                "pairwise_fragility": (
                    abs(float(shift)) / scale if scale > 0 else np.nan
                ),
                "pairwise_fragility_status": "defined" if scale > 0 else "zero_scale",
                "pdb_pairwise_gap_mean": pdb_stats["mean"],
                "afdb_pairwise_gap_mean": afdb_stats["mean"],
                "pdb_pairwise_gap_sign_change_fraction": pdb_stats[
                    "sign_change_fraction"
                ],
                "afdb_pairwise_gap_sign_change_fraction": afdb_stats[
                    "sign_change_fraction"
                ],
                "decoding_realization_sha256": str(realization_array[block_index]),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise RepresentationAuditError("empty_boundary_table", "No pairwise boundary records were built")
    return result


def _correlation(x: pd.Series, y: pd.Series, method: str) -> float | None:
    values = pd.DataFrame({"x": x, "y": y}).apply(pd.to_numeric, errors="coerce").dropna()
    if len(values) < 2 or values["x"].nunique() < 2 or values["y"].nunique() < 2:
        return None
    return float(values["x"].corr(values["y"], method=method))


def _association_fields(
    frame: pd.DataFrame,
    x_field: str,
    y_field: str,
    prefix: str,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for method in ("pearson", "spearman"):
        value = _correlation(frame[x_field], frame[y_field], method)
        output[f"{prefix}_{method}"] = value
        output[f"{prefix}_{method}_status"] = "defined" if value is not None else "undefined"
        output[f"{prefix}_{method}_reason"] = None if value is not None else "constant_or_insufficient"
    return output


def _aggregate_position_boundaries(
    records: pd.DataFrame,
    mechanism_positions: pd.DataFrame,
    *,
    protein_order: tuple[str, ...],
) -> pd.DataFrame:
    grouped = records.groupby(["protein_id", "position"], sort=False)
    rows: list[dict[str, Any]] = []
    for (protein_id, position), group in grouped:
        mechanism = mechanism_positions.loc[
            (mechanism_positions["protein_id"].astype(str) == str(protein_id))
            & (mechanism_positions["position"].astype(int) == int(position))
        ]
        if len(mechanism) != 1:
            raise RepresentationAuditError("position_key_mismatch", "Mechanism position key is not unique")
        m = mechanism.iloc[0]
        defined_fragility = pd.to_numeric(group["pairwise_fragility"], errors="coerce").dropna()
        rows.append(
            {
                "protein_id": str(protein_id),
                "position": int(position),
                "position_mean_abs_interaction": float(m["position_mean_abs_interaction"]),
                "margin_mean_both": float(m["margin_mean_both"]),
                "perturbation_to_margin": m["perturbation_to_margin"],
                "top1_disagreement_fraction": float(m["top1_disagreement_fraction"]),
                "symmetric_regret_mean": float(m["symmetric_regret_mean"]),
                "mean_normalized_rank_displacement_mean": float(
                    m["mean_normalized_rank_displacement_mean"]
                ),
                "boundary_record_count": len(group),
                "boundary_pair_agreement_fraction": float(
                    group["pair_id"].nunique() == 1
                ),
                "baseline_pairwise_margin_mean": float(group["baseline_pairwise_margin"].mean()),
                "baseline_pairwise_margin_median": float(group["baseline_pairwise_margin"].median()),
                "absolute_structural_gap_shift_mean": float(
                    group["absolute_structural_gap_shift"].mean()
                ),
                "absolute_structural_gap_shift_median": float(
                    group["absolute_structural_gap_shift"].median()
                ),
                "absolute_structural_gap_shift_max": float(
                    group["absolute_structural_gap_shift"].max()
                ),
                "pairwise_gap_sd_mean": float(group["pairwise_gap_sd"].mean()),
                "pairwise_gap_sd_median": float(group["pairwise_gap_sd"].median()),
                "pairwise_gap_sd_max": float(group["pairwise_gap_sd"].max()),
                "pairwise_fragility_mean": (
                    float(defined_fragility.mean()) if len(defined_fragility) else np.nan
                ),
                "pairwise_fragility_median": (
                    float(defined_fragility.median()) if len(defined_fragility) else np.nan
                ),
                "pairwise_fragility_max": (
                    float(defined_fragility.max()) if len(defined_fragility) else np.nan
                ),
                "pairwise_fragility_defined_fraction": float(
                    len(defined_fragility) / len(group)
                ),
                "zero_scale_record_count": int((group["pairwise_gap_sd"] == 0).sum()),
            }
        )
    result = pd.DataFrame(rows)
    rank = {protein: index for index, protein in enumerate(protein_order)}
    result["_order"] = result["protein_id"].map(rank)
    return result.sort_values(["_order", "position"], kind="stable").drop(columns="_order").reset_index(drop=True)


def _build_associations(
    positions: pd.DataFrame, *, protein_order: tuple[str, ...]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for protein_id in protein_order:
        frame = positions.loc[positions["protein_id"].astype(str) == protein_id]
        row: dict[str, Any] = {"protein_id": protein_id, "position_count": len(frame)}
        for x_field, prefix in (
            ("perturbation_to_margin", "ratio"),
            ("pairwise_fragility_mean", "pairwise_fragility"),
            ("absolute_structural_gap_shift_mean", "absolute_shift"),
        ):
            for y_field, outcome in (
                ("top1_disagreement_fraction", "top1_disagreement"),
                ("symmetric_regret_mean", "symmetric_regret"),
            ):
                row.update(_association_fields(frame, x_field, y_field, f"{prefix}_{outcome}"))
        rows.append(row)
    return pd.DataFrame(rows)


def _conclusion(
    associations: pd.DataFrame, *, protein_count: int
) -> tuple[str, dict[str, Any]]:
    recurrence_target = protein_count // 2 + 1
    outcomes = ("top1_disagreement", "symmetric_regret")
    methods = ("pearson", "spearman")
    directional: dict[str, int] = {}
    clearer: dict[str, int] = {}
    for outcome in outcomes:
        values = []
        for method in methods:
            pair_field = f"pairwise_fragility_{outcome}_{method}"
            values.extend(pd.to_numeric(associations[pair_field], errors="coerce").dropna().tolist())
            directional[f"pairwise_fragility_{outcome}_{method}"] = sum(v > 0 for v in values[-len(pd.to_numeric(associations[pair_field], errors="coerce").dropna()):])
            ratio_field = f"ratio_{outcome}_{method}"
            pair_series = pd.to_numeric(associations[pair_field], errors="coerce")
            ratio_series = pd.to_numeric(associations[ratio_field], errors="coerce")
            clearer[f"{outcome}_{method}"] = int(
                ((pair_series > 0) & pair_series.abs().ge(ratio_series.abs())).sum()
            )
    pairwise_coherent = all(
        directional[f"pairwise_fragility_{outcome}_{method}"] >= recurrence_target
        for outcome in outcomes
        for method in methods
    )
    clearer_count = sum(clearer.values())
    single_protein_share = max(clearer.values(), default=0) / max(clearer_count, 1)
    ratio_coherent = all(
        (
            pd.to_numeric(associations[f"ratio_{outcome}_{method}"], errors="coerce")
            .dropna()
            .gt(0)
            .sum()
            >= recurrence_target
        )
        for outcome in outcomes
        for method in methods
    )
    if pairwise_coherent and clearer_count >= recurrence_target * 2 and single_protein_share <= 0.5:
        conclusion = "PAIRWISE_BOUNDARY_REPRESENTATION_PREFERRED"
    elif ratio_coherent:
        conclusion = "CURRENT_REPRESENTATION_ADEQUATE"
    else:
        conclusion = "CURRENT_REPRESENTATION_INSUFFICIENT"
    return conclusion, {
        "recurrence_target": recurrence_target,
        "recurrence_target_rule": "floor(protein_count / 2) + 1",
        "directional_positive_counts": directional,
        "clearer_pairwise_counts": clearer,
        "clearer_comparison_count": clearer_count,
        "single_protein_clearer_share": single_protein_share,
        "single_protein_dominance_threshold": 0.5,
        "semantics": "descriptive_audit_rule_not_hypothesis_test_or_calibrated_threshold",
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RepresentationAuditError("manifest_unreadable", f"Unable to read manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise RepresentationAuditError("manifest_schema_mismatch", "Manifest must be an object")
    return payload


def _load_bound_output(root: Path, path_value: str, expected_sha: str, expected_rows: int, label: str) -> pd.DataFrame:
    path = (root / path_value).resolve()
    if not path.is_file():
        raise RepresentationAuditError("upstream_artifact_missing", f"Missing {label}: {path}")
    if sha256_file(path) != expected_sha:
        raise RepresentationAuditError("upstream_hash_mismatch", f"SHA mismatch for {label}")
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        raise RepresentationAuditError("upstream_artifact_unreadable", f"Unable to parse {label}") from exc
    if len(frame) != expected_rows:
        raise RepresentationAuditError("upstream_row_count_mismatch", f"Invalid row count for {label}")
    return frame


def _load_mechanism_outputs(root: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    path = (root / MECHANISM_MANIFEST_PATH).resolve()
    if not path.is_file() or sha256_file(path) != MECHANISM_MANIFEST_SHA256:
        raise RepresentationAuditError("mechanism_manifest_hash_mismatch", "Frozen mechanism manifest is absent or changed")
    manifest = _load_json(path)
    if manifest.get("status") != "complete" or manifest.get("schema_version") != "stage0_fixed_probe_mechanism_manifest_v1":
        raise RepresentationAuditError("mechanism_manifest_contract_mismatch", "Frozen mechanism manifest is not complete")
    outputs = manifest.get("outputs", {})
    position = outputs.get("position_mechanism", {})
    association = outputs.get("protein_associations", {})
    if position.get("sha256") != MECHANISM_POSITION_SHA256 or association.get("sha256") != MECHANISM_ASSOCIATION_SHA256:
        raise RepresentationAuditError("mechanism_manifest_contract_mismatch", "Mechanism output bindings differ")
    positions = _load_bound_output(root, position["path"], position["sha256"], EXPECTED_POSITION_COUNT, "mechanism positions")
    associations = _load_bound_output(root, association["path"], association["sha256"], EXPECTED_PROTEIN_COUNT, "mechanism associations")
    return manifest, positions, associations


def run_fixed_probe_boundary_audit(project_root: Path) -> BoundaryAuditResult:
    """Load frozen releases, compute the in-memory pair table, and aggregate it."""
    root = project_root.resolve()
    mechanism_manifest, mechanism_positions, _mechanism_associations = _load_mechanism_outputs(root)
    inputs = load_frozen_mechanism_inputs(root)
    local = _build_fast_local_scores(inputs.decision_inputs)
    ranked = rank_local_amino_acid_scores(local, protein_order=inputs.protein_order)
    identifiability = assess_representation_identifiability(set(inputs.decision_inputs.scoring.raw_scores.columns))
    records = build_pairwise_boundary_records(ranked, repeat_count=REPEAT_COUNT)
    positions = _aggregate_position_boundaries(records, mechanism_positions, protein_order=inputs.protein_order)
    if len(positions) != EXPECTED_POSITION_COUNT:
        raise RepresentationAuditError("position_count_mismatch", "Boundary position table must have 1790 rows")
    associations = _build_associations(positions, protein_order=inputs.protein_order)
    if len(associations) != EXPECTED_PROTEIN_COUNT:
        raise RepresentationAuditError("protein_count_mismatch", "Boundary protein table must have 8 rows")
    conclusion, rule = _conclusion(associations, protein_count=len(inputs.protein_order))
    manifest_fields = {
        "schema_version": "stage0_fixed_probe_boundary_audit_manifest_v1",
        "status": "complete",
        "scientific_scope": "stage0_fixed_probe_pairwise_decision_boundary_audit_only",
        "checkpoint_commit": mechanism_manifest["checkpoint_commit"],
        "stage0_2b_status": "HOLD_STAGE0_2B",
        "representation_identifiability": identifiability,
        "analysis_protocol": {
            "identity": "stage0_fixed_probe_pairwise_boundary_v1",
            "position_count": len(positions),
            "protein_count": len(associations),
            "repeat_count": REPEAT_COUNT,
            "candidate_count": 20,
            "pairwise_gap_sd": "sqrt((gap_sd_pdb**2 + gap_sd_afdb**2) / 2), population SD ddof=0",
            "pairwise_fragility": "absolute_structural_gap_shift / pairwise_gap_sd when positive; otherwise null",
            "orientation": "structural_gap_shift is AFDB gap minus PDB gap for each explicitly retained pair",
            "internal_nonemitted_diagnostics": [
                "explicit_boundary_pair_identity",
                "backbone_specific_gap_sd_pdb",
                "backbone_specific_gap_sd_afdb",
                "backbone_specific_sign_change_fraction",
            ],
            "rank_tie_policy": mechanism_manifest["analysis_protocol"]["ranking"],
            "association_semantics": "within_protein descriptive Pearson/Spearman only; no p-values",
            "repeat_pair_artifact": False,
        },
        "protein_order": list(inputs.protein_order),
        "conclusion": conclusion,
        "decision_rule": rule,
        "inputs": {
            "mechanism_manifest": {"path": MECHANISM_MANIFEST_PATH.as_posix(), "sha256": MECHANISM_MANIFEST_SHA256},
            "mechanism_position": {"path": mechanism_manifest["outputs"]["position_mechanism"]["path"], "sha256": MECHANISM_POSITION_SHA256, "rows": EXPECTED_POSITION_COUNT},
            "mechanism_association": {"path": mechanism_manifest["outputs"]["protein_associations"]["path"], "sha256": MECHANISM_ASSOCIATION_SHA256, "rows": EXPECTED_PROTEIN_COUNT},
            "decision_manifest": {"path": inputs.decision_manifest_path.relative_to(root).as_posix(), "sha256": inputs.decision_manifest_sha256},
            "decision_position": {"path": inputs.decision_position_path.relative_to(root).as_posix(), "sha256": inputs.decision_position_sha256, "rows": len(inputs.decision_positions)},
            "scoring_manifest": {
                "path": inputs.decision_inputs.scoring.scoring_manifest_path.relative_to(root).as_posix(),
                "sha256": inputs.decision_inputs.scoring.scoring_manifest_sha256,
            },
            "wt_scores": {
                "path": inputs.decision_inputs.scoring.wt_scores_path.relative_to(root).as_posix(),
                "sha256": inputs.decision_inputs.scoring.wt_scores_sha256,
                "rows": len(inputs.decision_inputs.scoring.wt_scores),
            },
            "raw_scores": {
                "path": inputs.decision_inputs.scoring.raw_scores_path.relative_to(root).as_posix(),
                "sha256": inputs.decision_inputs.scoring.raw_scores_sha256,
                "rows": len(inputs.decision_inputs.scoring.raw_scores),
            },
            "mechanism_manifest_inputs": mechanism_manifest["inputs"],
        },
        "scope_declarations": {
            "proteinmpnn_executed": False,
            "sequence_generation_performed": False,
            "new_model_used": False,
            "new_data_acquired": False,
            "hypothesis_testing_performed": False,
            "p_values_computed": False,
            "binary_uncertainty_classification_performed": False,
            "publication_figures_created": False,
            "arm_b_started": False,
            "external_evaluator_executed": False,
            "stage0_2b_started": False,
        },
        "outputs": {},
    }
    return BoundaryAuditResult(root, positions, associations, manifest_fields, manifest_fields["inputs"])


def materialize_fixed_probe_boundary_audit(result: BoundaryAuditResult, output_root: Path) -> dict[str, Any]:
    """Validate and immutably render the two tables, then the manifest last."""
    if len(result.position_boundary) != EXPECTED_POSITION_COUNT or len(result.protein_associations) != EXPECTED_PROTEIN_COUNT:
        raise RepresentationAuditError("output_row_count_mismatch", "Boundary output row counts are invalid")
    output_root.mkdir(parents=True, exist_ok=True)
    position_path = output_root / "fixed_probe_position_boundary_audit.parquet"
    association_path = output_root / "fixed_probe_protein_boundary_association.parquet"
    write_status = {
        "position_boundary": write_immutable_parquet(position_path, result.position_boundary),
        "protein_associations": write_immutable_parquet(association_path, result.protein_associations),
    }
    manifest = dict(result.manifest_fields)
    manifest["outputs"] = {
        "position_boundary": {"path": position_path.relative_to(result.project_root).as_posix(), "sha256": sha256_file(position_path), "rows": len(result.position_boundary)},
        "protein_associations": {"path": association_path.relative_to(result.project_root).as_posix(), "sha256": sha256_file(association_path), "rows": len(result.protein_associations)},
    }
    manifest_path = output_root / "fixed_probe_boundary_audit_manifest.json"
    write_status["manifest"] = write_immutable_json(manifest_path, manifest)
    return {"status": "PASS", "manifest_path": manifest_path, "write_status": write_status, "conclusion": manifest["conclusion"]}


__all__ = [
    "PER_RESIDUE_DECOMPOSITION_AVAILABLE",
    "PER_RESIDUE_DECOMPOSITION_NOT_AVAILABLE",
    "BoundaryAuditResult",
    "RepresentationAuditError",
    "assess_representation_identifiability",
    "build_pairwise_boundary_records",
    "materialize_fixed_probe_boundary_audit",
    "pairwise_gap_sd",
    "run_fixed_probe_boundary_audit",
]
