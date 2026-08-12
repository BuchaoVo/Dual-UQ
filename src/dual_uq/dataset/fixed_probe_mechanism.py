"""Frozen Stage0-2A local mechanism analysis.

This module is analysis-only. It never executes ProteinMPNN or constructs new
candidate sequences.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dual_uq.core.hashing import sha256_file
from dual_uq.dataset.fixed_probe_decision_sensitivity import (
    STANDARD_AMINO_ACIDS,
    DecisionInputs,
    DecisionSensitivityError,
    build_local_amino_acid_scores,
    load_frozen_decision_inputs,
    rank_local_amino_acid_scores,
)
from dual_uq.dataset.fixed_probe_sensitivity import (
    FixedProbeSensitivityError,
    write_immutable_json,
    write_immutable_parquet,
)

STAGE0_CHECKPOINT_COMMIT = "9cf144656b96ef26a14c42d7a7ff6d400e2c8f15"
DECISION_MANIFEST_PATH = Path(
    "experiments/p2_design_baseline/stage0/"
    "fixed_probe_decision_sensitivity_manifest.json"
)
DECISION_MANIFEST_SHA256 = (
    "f7e0df9a6636864207a849e659d68aad077dbaf6aaaf5aa1fb2d943480986e98"
)
EXPECTED_POSITION_COUNT = 1_790
EXPECTED_PROTEIN_COUNT = 8
REPEAT_COUNT = 30
ASSOCIATION_SPECS = (
    (
        "perturbation_top1_disagreement",
        "position_mean_abs_interaction",
        "top1_disagreement_fraction",
    ),
    (
        "perturbation_symmetric_regret",
        "position_mean_abs_interaction",
        "symmetric_regret_mean",
    ),
    (
        "perturbation_rank_displacement",
        "position_mean_abs_interaction",
        "mean_normalized_rank_displacement_mean",
    ),
    (
        "margin_top1_disagreement",
        "margin_mean_both",
        "top1_disagreement_fraction",
    ),
    (
        "margin_symmetric_regret",
        "margin_mean_both",
        "symmetric_regret_mean",
    ),
    (
        "ratio_top1_disagreement",
        "perturbation_to_margin",
        "top1_disagreement_fraction",
    ),
    (
        "ratio_symmetric_regret",
        "perturbation_to_margin",
        "symmetric_regret_mean",
    ),
    (
        "ratio_rank_displacement",
        "perturbation_to_margin",
        "mean_normalized_rank_displacement_mean",
    ),
)


class MechanismError(ValueError):
    """Structured input, analysis, or immutable-release failure."""

    def __init__(self, code: str, message: str, *, outcome: str = "BLOCKED") -> None:
        self.code = code
        self.outcome = outcome
        super().__init__(message)


@dataclass(frozen=True)
class MechanismInputs:
    """Checkpoint-bound inputs for the frozen local mechanism analysis."""

    project_root: Path
    checkpoint_commit: str
    decision_inputs: DecisionInputs
    decision_manifest_path: Path
    decision_manifest_sha256: str
    decision_manifest: dict[str, Any]
    continuous_positions: pd.DataFrame
    decision_position_path: Path
    decision_position_sha256: str
    decision_positions: pd.DataFrame
    protein_order: tuple[str, ...]


@dataclass(frozen=True)
class FixedProbeMechanismResult:
    """Single canonical result rendered into every mechanism release artifact."""

    project_root: Path
    checkpoint_commit: str
    input_provenance: dict[str, dict[str, Any]]
    position_mechanism: pd.DataFrame
    protein_associations: pd.DataFrame
    project_fork: dict[str, Any]
    repeat_count: int
    protein_order: tuple[str, ...]


def _read_json(path: Path, *, code: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MechanismError(code, f"Unable to read JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise MechanismError(code, f"JSON is not an object: {path}")
    return payload


def _checkpoint_bound_bytes(project_root: Path, logical_path: Path) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "show", f"{STAGE0_CHECKPOINT_COMMIT}:{logical_path.as_posix()}"],
            cwd=project_root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise MechanismError(
            "checkpoint_binding_unresolved",
            f"Unable to resolve {logical_path} at checkpoint "
            f"{STAGE0_CHECKPOINT_COMMIT}",
        ) from exc
    return completed.stdout


def _manifest_bound_path(
    project_root: Path, record: Any, *, label: str
) -> Path:
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise MechanismError(
            "decision_manifest_contract_mismatch",
            f"Missing manifest path for {label}",
        )
    path = (project_root / record["path"]).resolve()
    try:
        path.relative_to(project_root)
    except ValueError as exc:
        raise MechanismError(
            "decision_manifest_contract_mismatch",
            f"Manifest path escapes project root for {label}",
        ) from exc
    return path


def load_frozen_mechanism_inputs(project_root: Path) -> MechanismInputs:
    """Resolve all mechanism inputs from the checkpointed upstream manifests."""
    root = project_root.resolve()
    manifest_path = (root / DECISION_MANIFEST_PATH).resolve()
    if not manifest_path.is_file():
        raise MechanismError(
            "decision_manifest_missing",
            f"Frozen decision manifest is missing: {manifest_path}",
        )
    try:
        observed_manifest_sha = sha256_file(manifest_path)
    except OSError as exc:
        raise MechanismError(
            "decision_manifest_unreadable",
            f"Frozen decision manifest is unreadable: {manifest_path}",
        ) from exc
    if observed_manifest_sha != DECISION_MANIFEST_SHA256:
        raise MechanismError(
            "decision_manifest_hash_mismatch",
            "Frozen decision manifest SHA256 differs from the checkpoint binding",
        )
    if manifest_path.read_bytes() != _checkpoint_bound_bytes(
        root, DECISION_MANIFEST_PATH
    ):
        raise MechanismError(
            "checkpoint_binding_mismatch",
            "Decision manifest bytes differ from the Stage0 checkpoint",
        )
    manifest = _read_json(
        manifest_path, code="decision_manifest_contract_mismatch"
    )
    if (
        manifest.get("status") != "complete"
        or manifest.get("schema_version")
        != "stage0_fixed_probe_decision_sensitivity_manifest_v1"
    ):
        raise MechanismError(
            "decision_manifest_contract_mismatch",
            "Decision manifest is not the completed v1 release",
        )
    protein_order = tuple(str(value) for value in manifest.get("protein_order", []))
    if len(protein_order) != EXPECTED_PROTEIN_COUNT or len(set(protein_order)) != len(
        protein_order
    ):
        raise MechanismError(
            "protein_membership_mismatch", "Decision manifest protein order is invalid"
        )
    try:
        decision_inputs = load_frozen_decision_inputs(root)
    except DecisionSensitivityError as exc:
        raise MechanismError(exc.code, str(exc), outcome=exc.outcome) from exc
    if decision_inputs.protein_order != protein_order:
        raise MechanismError(
            "protein_membership_mismatch",
            "Decision and continuous releases use different protein order",
        )

    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise MechanismError(
            "decision_manifest_contract_mismatch", "Decision outputs are absent"
        )
    position_record = outputs.get("position_summary")
    if not isinstance(position_record, dict):
        raise MechanismError(
            "decision_manifest_contract_mismatch",
            "Decision position output binding is absent",
        )
    expected_sha = position_record.get("sha256")
    expected_rows = position_record.get("rows")
    if not isinstance(expected_sha, str) or expected_rows != EXPECTED_POSITION_COUNT:
        raise MechanismError(
            "decision_manifest_contract_mismatch",
            "Decision position output SHA or row count is invalid",
        )
    position_path = _manifest_bound_path(
        root, position_record, label="decision position summary"
    )
    decision_positions = load_bound_parquet(
        position_path,
        expected_sha256=expected_sha,
        expected_rows=EXPECTED_POSITION_COUNT,
        required_columns={
            "protein_id",
            "position",
            "wt_aa",
            "top1_disagreement_fraction",
            "symmetric_regret_mean",
            "mean_normalized_rank_displacement_mean",
        },
        label="decision position summary",
    )
    join_position_mechanism_inputs(
        decision_inputs.position_sensitivity,
        decision_positions,
        protein_order=protein_order,
        expected_position_count=EXPECTED_POSITION_COUNT,
    )
    return MechanismInputs(
        project_root=root,
        checkpoint_commit=STAGE0_CHECKPOINT_COMMIT,
        decision_inputs=decision_inputs,
        decision_manifest_path=manifest_path,
        decision_manifest_sha256=observed_manifest_sha,
        decision_manifest=manifest,
        continuous_positions=decision_inputs.position_sensitivity,
        decision_position_path=position_path,
        decision_position_sha256=expected_sha,
        decision_positions=decision_positions,
        protein_order=protein_order,
    )


def load_bound_parquet(
    path: Path,
    *,
    expected_sha256: str,
    expected_rows: int,
    required_columns: set[str],
    label: str,
) -> pd.DataFrame:
    """Load one manifest-bound Parquet only after byte and schema validation."""
    if not path.is_file():
        raise MechanismError(
            "upstream_artifact_missing", f"Frozen {label} is missing: {path}"
        )
    try:
        observed_sha256 = sha256_file(path)
    except OSError as exc:
        raise MechanismError(
            "upstream_artifact_unreadable", f"Frozen {label} is unreadable: {path}"
        ) from exc
    if observed_sha256 != expected_sha256:
        raise MechanismError(
            "upstream_hash_mismatch",
            f"Frozen {label} SHA256 differs: expected={expected_sha256} "
            f"observed={observed_sha256}",
        )
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        raise MechanismError(
            "upstream_artifact_unreadable", f"Unable to parse frozen {label}: {path}"
        ) from exc
    if len(frame) != expected_rows:
        raise MechanismError(
            "upstream_row_count_mismatch",
            f"Frozen {label} row count differs: expected={expected_rows} "
            f"observed={len(frame)}",
        )
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise MechanismError(
            "upstream_schema_mismatch",
            f"Frozen {label} is missing columns: {missing}",
        )
    return frame


def _require_position_table(
    frame: pd.DataFrame,
    *,
    label: str,
    required_columns: set[str],
    expected_position_count: int,
) -> None:
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise MechanismError(
            "upstream_schema_mismatch", f"{label} is missing columns: {missing}"
        )
    if frame.duplicated(["protein_id", "position"]).any():
        raise MechanismError(
            "duplicate_position_key", f"{label} contains duplicate scientific keys"
        )
    if len(frame) != expected_position_count:
        raise MechanismError(
            "upstream_row_count_mismatch",
            f"{label} row count differs: expected={expected_position_count} "
            f"observed={len(frame)}",
        )


def join_position_mechanism_inputs(
    continuous: pd.DataFrame,
    decision: pd.DataFrame,
    *,
    protein_order: tuple[str, ...],
    expected_position_count: int,
) -> pd.DataFrame:
    """Join the full frozen cohort exactly on protein and canonical position."""
    continuous_required = {
        "protein_id",
        "position",
        "wt_aa",
        "position_mean_abs_interaction",
        "position_rms_interaction",
        "position_max_abs_interaction",
    }
    decision_required = {
        "protein_id",
        "position",
        "wt_aa",
        "top1_disagreement_fraction",
        "symmetric_regret_mean",
        "structural_minus_mean_same_state_symmetric_regret",
        "mean_normalized_rank_displacement_mean",
    }
    _require_position_table(
        continuous,
        label="continuous position table",
        required_columns=continuous_required,
        expected_position_count=expected_position_count,
    )
    _require_position_table(
        decision,
        label="decision position table",
        required_columns=decision_required,
        expected_position_count=expected_position_count,
    )

    keys = ["protein_id", "position"]
    overlap = [
        column
        for column in continuous.columns
        if column in decision.columns and column not in keys + ["wt_aa"]
    ]
    joined = continuous.merge(
        decision,
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("", "__decision"),
        sort=False,
    )
    left_only = int((joined["_merge"] == "left_only").sum())
    right_only = int((joined["_merge"] == "right_only").sum())
    if left_only or right_only:
        raise MechanismError(
            "position_key_mismatch",
            f"Position tables do not define the same cohort: "
            f"left_only={left_only} right_only={right_only}",
        )
    if not (joined["wt_aa"] == joined["wt_aa__decision"]).all():
        raise MechanismError(
            "position_identity_mismatch", "WT identity differs across position tables"
        )
    for column in overlap:
        other = f"{column}__decision"
        if not joined[column].equals(joined[other]):
            raise MechanismError(
                "position_evidence_mismatch",
                f"Duplicated frozen field differs across position tables: {column}",
            )

    observed_proteins = set(joined["protein_id"].astype(str))
    if observed_proteins != set(protein_order) or len(protein_order) != len(
        observed_proteins
    ):
        raise MechanismError(
            "protein_membership_mismatch", "Position cohort differs from frozen proteins"
        )
    protein_rank = {protein: index for index, protein in enumerate(protein_order)}
    joined = joined.assign(
        _protein_order=joined["protein_id"].astype(str).map(protein_rank)
    ).sort_values(["_protein_order", "position"], kind="stable")
    drop_columns = ["_merge", "_protein_order", "wt_aa__decision"] + [
        f"{column}__decision" for column in overlap
    ]
    return joined.drop(columns=drop_columns).reset_index(drop=True)


def build_repeat_margins(ranked_scores: pd.DataFrame) -> pd.DataFrame:
    """Derive Top-1/Top-2 score margins from frozen ranked 20-AA landscapes."""
    required = {
        "protein_id",
        "position",
        "backbone_condition",
        "repeat_index",
        "aa",
        "rank",
        "score_mean_logp_mask",
        "practical_top_tie",
    }
    missing = sorted(required - set(ranked_scores.columns))
    if missing:
        raise MechanismError(
            "local_score_schema_mismatch", f"Ranked scores are missing columns: {missing}"
        )
    group_keys = [
        "protein_id",
        "position",
        "backbone_condition",
        "repeat_index",
    ]
    if ranked_scores.duplicated(group_keys + ["aa"]).any():
        raise MechanismError(
            "local_amino_acid_grid_mismatch",
            "A local amino acid repeats within a ranked landscape",
            outcome="FAIL",
        )
    groups = ranked_scores.groupby(group_keys, sort=False)
    alphabet = set(STANDARD_AMINO_ACIDS)
    if (
        not (groups.size() == len(alphabet)).all()
        or not (groups["aa"].nunique() == len(alphabet)).all()
        or any(set(group["aa"].astype(str)) != alphabet for _, group in groups)
    ):
        raise MechanismError(
            "local_amino_acid_grid_mismatch",
            "Each ranked landscape must contain exactly 20 standard amino acids",
            outcome="FAIL",
        )
    if any(
        set(group["rank"].astype(int)) != set(range(1, len(alphabet) + 1))
        for _, group in groups
    ):
        raise MechanismError(
            "local_rank_grid_mismatch",
            "Each ranked landscape must contain ranks 1 through 20 exactly once",
            outcome="FAIL",
        )

    top = ranked_scores.loc[
        ranked_scores["rank"].isin([1, 2]),
        group_keys
        + ["aa", "rank", "score_mean_logp_mask", "practical_top_tie"],
    ].copy()
    top1 = top.loc[top["rank"] == 1].rename(
        columns={
            "aa": "top1_aa",
            "score_mean_logp_mask": "top1_score",
        }
    )
    top2 = top.loc[top["rank"] == 2].rename(
        columns={
            "aa": "top2_aa",
            "score_mean_logp_mask": "top2_score",
        }
    )
    top1 = top1.drop(columns=["rank"])
    top2 = top2.drop(columns=["rank", "practical_top_tie"])
    margins = top1.merge(top2, on=group_keys, validate="one_to_one")
    values = margins["top1_score"].to_numpy(dtype=np.float64) - margins[
        "top2_score"
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise MechanismError(
            "nonfinite_top1_top2_margin", "Top-1/Top-2 margins are nonfinite", outcome="FAIL"
        )
    if (values < -1.0e-12).any():
        raise MechanismError(
            "negative_top1_top2_margin",
            "A frozen rank-1 score is below the rank-2 score",
            outcome="FAIL",
        )
    margins["top1_top2_margin"] = np.maximum(values, 0.0)
    condition_rank = margins["backbone_condition"].map({"PDB": 0, "AFDB": 1})
    if condition_rank.isna().any():
        raise MechanismError(
            "backbone_condition_mismatch", "Margins contain an unknown condition", outcome="FAIL"
        )
    margins = margins.assign(_condition_order=condition_rank).sort_values(
        ["protein_id", "position", "_condition_order", "repeat_index"],
        kind="stable",
    )
    return margins.drop(columns="_condition_order").reset_index(drop=True)


def summarize_position_margins(
    repeat_margins: pd.DataFrame,
    *,
    repeat_count: int,
    protein_order: tuple[str, ...],
) -> pd.DataFrame:
    """Aggregate PDB, AFDB, and combined margins for every position."""
    required = {
        "protein_id",
        "position",
        "backbone_condition",
        "repeat_index",
        "top1_top2_margin",
        "practical_top_tie",
    }
    missing = sorted(required - set(repeat_margins.columns))
    if missing:
        raise MechanismError(
            "margin_schema_mismatch", f"Repeat margins are missing columns: {missing}"
        )
    condition_keys = ["protein_id", "position", "backbone_condition"]
    groups = repeat_margins.groupby(condition_keys, sort=False)
    expected_repeats = set(range(repeat_count))
    if any(
        len(group) != repeat_count
        or set(group["repeat_index"].astype(int)) != expected_repeats
        for _, group in groups
    ):
        raise MechanismError(
            "margin_repeat_coverage_mismatch",
            "Each position-condition margin requires all frozen repeats",
            outcome="FAIL",
        )
    position_groups = repeat_margins.groupby(["protein_id", "position"], sort=False)
    if any(
        set(group["backbone_condition"].astype(str)) != {"PDB", "AFDB"}
        or len(group) != repeat_count * 2
        for _, group in position_groups
    ):
        raise MechanismError(
            "margin_condition_coverage_mismatch",
            "Each position requires PDB and AFDB margin coverage",
            outcome="FAIL",
        )

    records: list[dict[str, Any]] = []
    for (protein_id, position), group in position_groups:
        pdb = group.loc[group["backbone_condition"] == "PDB", "top1_top2_margin"]
        afdb = group.loc[group["backbone_condition"] == "AFDB", "top1_top2_margin"]
        combined = group["top1_top2_margin"]
        records.append(
            {
                "protein_id": str(protein_id),
                "position": int(position),
                "margin_mean_pdb": float(pdb.mean()),
                "margin_median_pdb": float(pdb.median()),
                "margin_mean_afdb": float(afdb.mean()),
                "margin_median_afdb": float(afdb.median()),
                "margin_mean_both": float(combined.mean()),
                "margin_median_both": float(combined.median()),
                "margin_min": float(combined.min()),
                "practical_tie_fraction": float(
                    group["practical_top_tie"].astype(bool).mean()
                ),
            }
        )
    result = pd.DataFrame.from_records(records)
    protein_rank = {protein: index for index, protein in enumerate(protein_order)}
    if set(result["protein_id"]) != set(protein_order):
        raise MechanismError(
            "protein_membership_mismatch", "Margin proteins differ from frozen order"
        )
    return (
        result.assign(_protein_order=result["protein_id"].map(protein_rank))
        .sort_values(["_protein_order", "position"], kind="stable")
        .drop(columns="_protein_order")
        .reset_index(drop=True)
    )


def attach_margin_diagnostics(
    positions: pd.DataFrame, margins: pd.DataFrame
) -> pd.DataFrame:
    """Attach exact margin keys and define the no-epsilon perturbation ratio."""
    keys = ["protein_id", "position"]
    if positions.duplicated(keys).any() or margins.duplicated(keys).any():
        raise MechanismError(
            "duplicate_position_key", "Position or margin table contains duplicate keys"
        )
    ordered_positions = positions.assign(_canonical_row_order=np.arange(len(positions)))
    result = ordered_positions.merge(
        margins,
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
        sort=False,
    )
    if not (result["_merge"] == "both").all():
        raise MechanismError(
            "position_margin_key_mismatch", "Position and margin keys differ"
        )
    margin = result["margin_mean_both"].to_numpy(dtype=np.float64)
    perturbation = result["position_mean_abs_interaction"].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(margin).all() or not np.isfinite(perturbation).all():
        raise MechanismError(
            "nonfinite_mechanism_value", "Margin or perturbation is nonfinite", outcome="FAIL"
        )
    if (margin < 0).any():
        raise MechanismError(
            "negative_top1_top2_margin", "Aggregated margin is negative", outcome="FAIL"
        )
    defined = margin > 0
    ratio = np.full(len(result), np.nan, dtype=np.float64)
    ratio[defined] = perturbation[defined] / margin[defined]
    result["perturbation_to_margin"] = ratio
    result["margin_ratio_status"] = np.where(defined, "defined", "zero_margin")
    return (
        result.sort_values("_canonical_row_order", kind="stable")
        .drop(columns=["_merge", "_canonical_row_order"])
        .reset_index(drop=True)
    )


def descriptive_correlation(x: pd.Series, y: pd.Series) -> dict[str, Any]:
    """Compute pairwise-complete descriptive Pearson and Spearman values."""
    x_values = pd.to_numeric(x, errors="coerce").to_numpy(dtype=np.float64)
    y_values = pd.to_numeric(y, errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    x_valid = x_values[valid]
    y_valid = y_values[valid]
    n = int(valid.sum())
    result: dict[str, Any] = {"n": n}
    if n < 2:
        for method in ("pearson", "spearman"):
            result[method] = None
            result[f"{method}_status"] = "undefined"
            result[f"{method}_reason"] = "insufficient_valid_pairs"
        return result

    constant_x = bool(np.all(x_valid == x_valid[0]))
    constant_y = bool(np.all(y_valid == y_valid[0]))
    if constant_x or constant_y:
        reason = (
            "constant_x_and_y"
            if constant_x and constant_y
            else "constant_x"
            if constant_x
            else "constant_y"
        )
        for method in ("pearson", "spearman"):
            result[method] = None
            result[f"{method}_status"] = "undefined"
            result[f"{method}_reason"] = reason
        return result

    pearson = float(np.corrcoef(x_valid, y_valid)[0, 1])
    x_rank = pd.Series(x_valid).rank(method="average").to_numpy(dtype=np.float64)
    y_rank = pd.Series(y_valid).rank(method="average").to_numpy(dtype=np.float64)
    spearman = float(np.corrcoef(x_rank, y_rank)[0, 1])
    for method, value in (("pearson", pearson), ("spearman", spearman)):
        if not np.isfinite(value) or value < -1.0 - 1.0e-12 or value > 1.0 + 1.0e-12:
            raise MechanismError(
                "invalid_descriptive_correlation",
                f"{method} correlation is outside its mathematical bounds",
                outcome="FAIL",
            )
        result[method] = float(np.clip(value, -1.0, 1.0))
        result[f"{method}_status"] = "defined"
        result[f"{method}_reason"] = None
    return result


def summarize_protein_mechanism_associations(
    position_mechanism: pd.DataFrame,
    *,
    protein_order: tuple[str, ...],
) -> pd.DataFrame:
    """Compute the frozen descriptive association matrix within each protein."""
    required = {"protein_id", "position"} | {
        field for _, x_field, y_field in ASSOCIATION_SPECS for field in (x_field, y_field)
    }
    missing = sorted(required - set(position_mechanism.columns))
    if missing:
        raise MechanismError(
            "position_mechanism_schema_mismatch",
            f"Position mechanism table is missing columns: {missing}",
        )
    observed_proteins = set(position_mechanism["protein_id"].astype(str))
    if observed_proteins != set(protein_order) or len(protein_order) != len(
        observed_proteins
    ):
        raise MechanismError(
            "protein_membership_mismatch",
            "Association cohort differs from frozen proteins",
        )
    records: list[dict[str, Any]] = []
    for protein_id in protein_order:
        group = position_mechanism.loc[
            position_mechanism["protein_id"].astype(str) == protein_id
        ]
        record: dict[str, Any] = {
            "protein_id": protein_id,
            "position_count": len(group),
        }
        for association_id, x_field, y_field in ASSOCIATION_SPECS:
            association = descriptive_correlation(group[x_field], group[y_field])
            for suffix, value in association.items():
                record[f"{association_id}_{suffix}"] = value
        records.append(record)
    return pd.DataFrame.from_records(records)


def _direction_count(
    associations: pd.DataFrame,
    association_id: str,
    *,
    method: str,
    direction: str,
) -> tuple[int, int]:
    field = f"{association_id}_{method}"
    values = pd.to_numeric(associations[field], errors="coerce").to_numpy(
        dtype=np.float64
    )
    defined = values[np.isfinite(values)]
    if direction == "positive":
        count = int((defined > 0).sum())
    elif direction == "negative":
        count = int((defined < 0).sum())
    else:  # pragma: no cover - private contract guard
        raise ValueError(direction)
    return count, len(defined)


def _ratio_improvement_count(
    associations: pd.DataFrame,
    ratio_id: str,
    perturbation_id: str,
    *,
    method: str,
) -> tuple[int, int]:
    ratio = pd.to_numeric(
        associations[f"{ratio_id}_{method}"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    perturbation = pd.to_numeric(
        associations[f"{perturbation_id}_{method}"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    defined = np.isfinite(ratio) & np.isfinite(perturbation)
    return int((np.abs(ratio[defined]) > np.abs(perturbation[defined])).sum()), int(
        defined.sum()
    )


def assess_project_fork(
    protein_associations: pd.DataFrame,
    position_mechanism: pd.DataFrame,
) -> dict[str, Any]:
    """Render the frozen multi-criterion descriptive Stage0 project fork."""
    expected_associations = {
        association_id for association_id, _, _ in ASSOCIATION_SPECS
    }
    required_association_fields = {"protein_id"} | {
        f"{association_id}_{method}"
        for association_id in expected_associations
        for method in ("pearson", "spearman")
    }
    missing = sorted(required_association_fields - set(protein_associations.columns))
    if missing:
        raise MechanismError(
            "protein_association_schema_mismatch",
            f"Project-fork associations are missing columns: {missing}",
        )
    position_required = {
        "top1_disagreement_fraction",
        "symmetric_regret_mean",
        "structural_minus_mean_same_state_symmetric_regret",
        "margin_mean_both",
        "position_mean_abs_interaction",
        "perturbation_to_margin",
    }
    position_missing = sorted(position_required - set(position_mechanism.columns))
    if position_missing:
        raise MechanismError(
            "position_mechanism_schema_mismatch",
            f"Project-fork positions are missing columns: {position_missing}",
        )

    perturbation_ids = (
        "perturbation_top1_disagreement",
        "perturbation_symmetric_regret",
        "perturbation_rank_displacement",
    )
    margin_ids = ("margin_top1_disagreement", "margin_symmetric_regret")
    ratio_pairs = (
        ("ratio_top1_disagreement", "perturbation_top1_disagreement"),
        ("ratio_symmetric_regret", "perturbation_symmetric_regret"),
        ("ratio_rank_displacement", "perturbation_rank_displacement"),
    )
    recurrence_target = protein_associations["protein_id"].nunique() // 2 + 1
    q1_counts = {
        association_id: {
            method: dict(
                zip(
                    ("expected_direction_count", "defined_count"),
                    _direction_count(
                        protein_associations,
                        association_id,
                        method=method,
                        direction="positive",
                    ),
                    strict=True,
                )
            )
            for method in ("pearson", "spearman")
        }
        for association_id in perturbation_ids
    }
    q1_coherent = all(
        q1_counts[association_id][method]["expected_direction_count"]
        >= recurrence_target
        for association_id in perturbation_ids
        for method in ("pearson", "spearman")
    )
    q2_counts = {
        association_id: {
            method: dict(
                zip(
                    ("protective_direction_count", "defined_count"),
                    _direction_count(
                        protein_associations,
                        association_id,
                        method=method,
                        direction="negative",
                    ),
                    strict=True,
                )
            )
            for method in ("pearson", "spearman")
        }
        for association_id in margin_ids
    }
    q2_protective = all(
        q2_counts[association_id][method]["protective_direction_count"]
        >= recurrence_target
        for association_id in margin_ids
        for method in ("pearson", "spearman")
    )

    q3_counts: dict[str, Any] = {}
    clearer_outcomes = 0
    for ratio_id, perturbation_id in ratio_pairs:
        q3_counts[ratio_id] = {}
        outcome_clearer = True
        for method in ("pearson", "spearman"):
            improvement_count, comparison_count = _ratio_improvement_count(
                protein_associations,
                ratio_id,
                perturbation_id,
                method=method,
            )
            positive_count, defined_count = _direction_count(
                protein_associations,
                ratio_id,
                method=method,
                direction="positive",
            )
            q3_counts[ratio_id][method] = {
                "stronger_absolute_count": improvement_count,
                "comparison_count": comparison_count,
                "expected_direction_count": positive_count,
                "defined_count": defined_count,
            }
            outcome_clearer = outcome_clearer and (
                improvement_count >= recurrence_target
                and positive_count >= recurrence_target
            )
        clearer_outcomes += int(outcome_clearer)

    dominance_values: list[float] = []
    for ratio_id, _ in ratio_pairs:
        values = np.abs(
            pd.to_numeric(
                protein_associations[f"{ratio_id}_spearman"], errors="coerce"
            ).to_numpy(dtype=np.float64)
        )
        values = values[np.isfinite(values)]
        total = float(values.sum())
        dominance_values.append(float(values.max() / total) if total > 0 else 1.0)
    maximum_single_protein_share = max(dominance_values)
    dominated_by_single_protein = maximum_single_protein_share > 0.5
    ratio_clearer = clearer_outcomes >= 2 and not dominated_by_single_protein

    top1 = pd.to_numeric(
        position_mechanism["top1_disagreement_fraction"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    regret = pd.to_numeric(
        position_mechanism["symmetric_regret_mean"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    margin = pd.to_numeric(
        position_mechanism["margin_mean_both"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    technical_contrast = pd.to_numeric(
        position_mechanism[
            "structural_minus_mean_same_state_symmetric_regret"
        ],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    finite = (
        np.isfinite(top1)
        & np.isfinite(regret)
        & np.isfinite(margin)
        & np.isfinite(technical_contrast)
    )
    crossing = finite & (top1 > 0)
    crossing_regret = regret[crossing]
    crossing_margin = margin[crossing]
    crossing_contrast = technical_contrast[crossing]
    if crossing_regret.size:
        regret_quantiles = {
            "median": float(np.quantile(crossing_regret, 0.5, method="linear")),
            "q75": float(np.quantile(crossing_regret, 0.75, method="linear")),
            "q95": float(np.quantile(crossing_regret, 0.95, method="linear")),
            "max": float(crossing_regret.max()),
        }
        margin_quantiles = {
            "median": float(np.quantile(crossing_margin, 0.5, method="linear")),
            "q25": float(np.quantile(crossing_margin, 0.25, method="linear")),
            "min": float(crossing_margin.min()),
        }
        contrast_quantiles = {
            "median": float(np.quantile(crossing_contrast, 0.5, method="linear")),
            "q75": float(np.quantile(crossing_contrast, 0.75, method="linear")),
            "q95": float(np.quantile(crossing_contrast, 0.95, method="linear")),
            "max": float(crossing_contrast.max()),
        }
        structural_exceeds_count = int((crossing_contrast > 0).sum())
        substantive_tail_present = contrast_quantiles["q95"] > 0
    else:
        regret_quantiles = {key: None for key in ("median", "q75", "q95", "max")}
        margin_quantiles = {key: None for key in ("median", "q25", "min")}
        contrast_quantiles = {key: None for key in ("median", "q75", "q95", "max")}
        structural_exceeds_count = 0
        substantive_tail_present = False

    go = q1_coherent and q2_protective and ratio_clearer
    return {
        "recommendation": "GO_STAGE0_2B" if go else "HOLD_STAGE0_2B",
        "assessment_semantics": (
            "multi_criterion_descriptive_gate_without_universal_coefficient_threshold"
        ),
        "recurrence_target": int(recurrence_target),
        "decision_rule": {
            "recurrence_target_rule": "floor(protein_count / 2) + 1",
            "recurrence_target": int(recurrence_target),
            "ratio_clearer_minimum_outcomes": 2,
            "ratio_clearer_total_outcomes": len(ratio_pairs),
            "single_protein_dominance_threshold": 0.5,
            "go_requires": [
                "continuous_perturbation_coherent",
                "margin_protective",
                "ratio_clearer",
            ],
            "semantics": (
                "descriptive_project_fork_not_calibrated_risk_threshold"
            ),
        },
        "evidence_summary": {
            "q1": {
                "continuous_perturbation_coherent": q1_coherent,
                "direction_counts": q1_counts,
            },
            "q2": {
                "margin_protective": q2_protective,
                "direction_counts": q2_counts,
            },
            "q3": {
                "ratio_clearer": ratio_clearer,
                "clearer_outcome_count": int(clearer_outcomes),
                "association_counts": q3_counts,
                "maximum_single_protein_share": maximum_single_protein_share,
                "dominated_by_single_protein": dominated_by_single_protein,
            },
            "q4": {
                "positions_with_top1_disagreement": int(crossing.sum()),
                "crossing_regret_quantiles": regret_quantiles,
                "crossing_margin_quantiles": margin_quantiles,
                "structural_minus_same_state_regret_quantiles": contrast_quantiles,
                "structural_exceeds_mean_same_state_regret_count": (
                    structural_exceeds_count
                ),
                "structural_exceeds_mean_same_state_regret_fraction": (
                    float(structural_exceeds_count / crossing_regret.size)
                    if crossing_regret.size
                    else None
                ),
                "substantive_regret_tail_present": substantive_tail_present,
                "substantive_regret_tail_definition": (
                    "q95_structural_minus_mean_same_state_regret_above_zero"
                ),
                "substantive_regret_tail_semantics": (
                    "descriptive_upper_tail_excess_not_biological_calibration"
                ),
            },
        },
        "stage0_2b_executed": False,
    }


def _logical_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError as exc:
        raise MechanismError(
            "nonportable_release_path", f"Path is outside project root: {resolved}"
        ) from exc


def _input_record(
    path: Path,
    sha256: str,
    rows: int | None,
    *,
    project_root: Path,
) -> dict[str, Any]:
    return {
        "path": _logical_path(path, project_root),
        "sha256": sha256,
        "rows": rows,
    }


def build_fixed_probe_mechanism_result(
    inputs: MechanismInputs,
) -> FixedProbeMechanismResult:
    """Run the canonical frozen-input-to-mechanism-result dataflow once."""
    joined_positions = join_position_mechanism_inputs(
        inputs.continuous_positions,
        inputs.decision_positions,
        protein_order=inputs.protein_order,
        expected_position_count=EXPECTED_POSITION_COUNT,
    )
    local_scores = build_local_amino_acid_scores(
        inputs.decision_inputs, repeat_count=REPEAT_COUNT
    )
    ranked_scores = rank_local_amino_acid_scores(
        local_scores, protein_order=inputs.protein_order
    )
    repeat_margins = build_repeat_margins(ranked_scores)
    position_margins = summarize_position_margins(
        repeat_margins,
        repeat_count=REPEAT_COUNT,
        protein_order=inputs.protein_order,
    )
    position_mechanism = attach_margin_diagnostics(
        joined_positions, position_margins
    )
    protein_associations = summarize_protein_mechanism_associations(
        position_mechanism, protein_order=inputs.protein_order
    )
    project_fork = assess_project_fork(protein_associations, position_mechanism)

    provenance = {
        label: dict(record)
        for label, record in inputs.decision_manifest["inputs"].items()
    }
    provenance["decision_sensitivity_manifest"] = _input_record(
        inputs.decision_manifest_path,
        inputs.decision_manifest_sha256,
        None,
        project_root=inputs.project_root,
    )
    provenance["decision_position_summary"] = _input_record(
        inputs.decision_position_path,
        inputs.decision_position_sha256,
        len(inputs.decision_positions),
        project_root=inputs.project_root,
    )
    result = FixedProbeMechanismResult(
        project_root=inputs.project_root,
        checkpoint_commit=inputs.checkpoint_commit,
        input_provenance=provenance,
        position_mechanism=position_mechanism,
        protein_associations=protein_associations,
        project_fork=project_fork,
        repeat_count=REPEAT_COUNT,
        protein_order=inputs.protein_order,
    )
    validate_mechanism_result(result)
    return result


def _forbidden_fields(frames: tuple[pd.DataFrame, ...]) -> list[str]:
    forbidden_tokens = (
        "p_value",
        "pvalue",
        "q_value",
        "confidence_interval",
        "uq_label",
        "mechanism_label",
        "is_uncertain",
    )
    return sorted(
        {
            column
            for frame in frames
            for column in frame.columns
            if any(token in column.lower() for token in forbidden_tokens)
        }
    )


def validate_mechanism_result(
    result: FixedProbeMechanismResult,
    *,
    expected_position_rows: int = EXPECTED_POSITION_COUNT,
    expected_protein_rows: int = EXPECTED_PROTEIN_COUNT,
) -> None:
    """Validate all scientific and release invariants before any write."""
    positions = result.position_mechanism
    proteins = result.protein_associations
    if len(positions) != expected_position_rows:
        raise MechanismError(
            "position_mechanism_count_mismatch",
            f"Expected {expected_position_rows} position rows, observed {len(positions)}",
            outcome="FAIL",
        )
    if len(proteins) != expected_protein_rows:
        raise MechanismError(
            "protein_association_count_mismatch",
            f"Expected {expected_protein_rows} protein rows, observed {len(proteins)}",
            outcome="FAIL",
        )
    required_position = {
        "protein_id",
        "position",
        "position_mean_abs_interaction",
        "position_rms_interaction",
        "position_max_abs_interaction",
        "top1_disagreement_fraction",
        "symmetric_regret_mean",
        "mean_normalized_rank_displacement_mean",
        "margin_mean_pdb",
        "margin_median_pdb",
        "margin_mean_afdb",
        "margin_median_afdb",
        "margin_mean_both",
        "margin_median_both",
        "margin_min",
        "practical_tie_fraction",
        "perturbation_to_margin",
        "margin_ratio_status",
    }
    missing = sorted(required_position - set(positions.columns))
    if missing:
        raise MechanismError(
            "position_mechanism_schema_mismatch",
            f"Position mechanism table is missing columns: {missing}",
            outcome="FAIL",
        )
    if positions.duplicated(["protein_id", "position"]).any():
        raise MechanismError(
            "duplicate_position_key",
            "Position mechanism scientific keys repeat",
            outcome="FAIL",
        )
    if tuple(proteins["protein_id"].astype(str)) != result.protein_order:
        raise MechanismError(
            "protein_membership_mismatch",
            "Protein association order differs from the frozen order",
            outcome="FAIL",
        )
    if set(positions["protein_id"].astype(str)) != set(result.protein_order):
        raise MechanismError(
            "protein_membership_mismatch",
            "Position mechanism proteins differ from the frozen order",
            outcome="FAIL",
        )
    observed_position_order = tuple(
        dict.fromkeys(positions["protein_id"].astype(str))
    )
    if observed_position_order != result.protein_order or any(
        not group["position"].is_monotonic_increasing
        for _, group in positions.groupby("protein_id", sort=False)
    ):
        raise MechanismError(
            "position_order_mismatch",
            "Position mechanism rows do not preserve frozen protein/position order",
            outcome="FAIL",
        )
    forbidden = _forbidden_fields((positions, proteins))
    if forbidden:
        raise MechanismError(
            "forbidden_mechanism_field",
            f"Mechanism outputs contain forbidden fields: {forbidden}",
            outcome="FAIL",
        )

    margin_fields = [
        "margin_mean_pdb",
        "margin_median_pdb",
        "margin_mean_afdb",
        "margin_median_afdb",
        "margin_mean_both",
        "margin_median_both",
        "margin_min",
    ]
    margins = positions[margin_fields].to_numpy(dtype=np.float64)
    if not np.isfinite(margins).all() or (margins < 0).any():
        raise MechanismError(
            "invalid_margin_value", "Margins must be finite and nonnegative", outcome="FAIL"
        )
    tie_fraction = positions["practical_tie_fraction"].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(tie_fraction).all()
        or (tie_fraction < 0).any()
        or (tie_fraction > 1).any()
    ):
        raise MechanismError(
            "invalid_practical_tie_fraction",
            "Practical-tie fractions must lie in [0, 1]",
            outcome="FAIL",
        )
    status = positions["margin_ratio_status"].astype(str)
    if not set(status).issubset({"defined", "zero_margin"}):
        raise MechanismError(
            "invalid_margin_ratio_status", "Margin-ratio status is invalid", outcome="FAIL"
        )
    defined = status == "defined"
    zero = status == "zero_margin"
    ratios = pd.to_numeric(
        positions["perturbation_to_margin"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    expected_ratios = (
        positions.loc[defined, "position_mean_abs_interaction"].to_numpy(
            dtype=np.float64
        )
        / positions.loc[defined, "margin_mean_both"].to_numpy(dtype=np.float64)
    )
    if (
        not (positions.loc[defined, "margin_mean_both"] > 0).all()
        or not np.isfinite(ratios[defined]).all()
        or not np.allclose(ratios[defined], expected_ratios, rtol=0, atol=1.0e-15)
        or not (positions.loc[zero, "margin_mean_both"] == 0).all()
        or not np.isnan(ratios[zero]).all()
    ):
        raise MechanismError(
            "margin_ratio_contract_mismatch",
            "Perturbation-to-margin ratio violates zero-margin semantics",
            outcome="FAIL",
        )

    for association_id, _, _ in ASSOCIATION_SPECS:
        for method in ("pearson", "spearman"):
            value_field = f"{association_id}_{method}"
            status_field = f"{association_id}_{method}_status"
            reason_field = f"{association_id}_{method}_reason"
            for field in (value_field, status_field, reason_field):
                if field not in proteins.columns:
                    raise MechanismError(
                        "protein_association_schema_mismatch",
                        f"Protein associations are missing {field}",
                        outcome="FAIL",
                    )
            metric_status = proteins[status_field].astype(str)
            values = pd.to_numeric(proteins[value_field], errors="coerce").to_numpy(
                dtype=np.float64
            )
            is_defined = metric_status == "defined"
            if (
                not np.isfinite(values[is_defined]).all()
                or (values[is_defined] < -1).any()
                or (values[is_defined] > 1).any()
                or not np.isnan(values[~is_defined]).all()
                or not set(metric_status).issubset({"defined", "undefined"})
            ):
                raise MechanismError(
                    "invalid_descriptive_correlation",
                    f"Invalid {association_id} {method} coefficient/status",
                    outcome="FAIL",
                )
    recommendation = result.project_fork.get("recommendation")
    if recommendation not in {"GO_STAGE0_2B", "HOLD_STAGE0_2B"}:
        raise MechanismError(
            "invalid_project_fork", "Project fork must choose one frozen outcome", outcome="FAIL"
        )
    if result.project_fork.get("stage0_2b_executed") is not False:
        raise MechanismError(
            "scope_violation", "Mechanism closure must not execute Stage0-2B", outcome="FAIL"
        )


def _rehash_inputs(result: FixedProbeMechanismResult) -> None:
    for label, record in result.input_provenance.items():
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise MechanismError(
                "input_provenance_mismatch", f"Input provenance is invalid: {label}"
            )
        path = (result.project_root / record["path"]).resolve()
        if not path.is_file():
            raise MechanismError(
                "upstream_input_changed", f"Frozen input is missing: {label}"
            )
        try:
            observed = sha256_file(path)
        except OSError as exc:
            raise MechanismError(
                "upstream_input_changed", f"Frozen input is unreadable: {label}"
            ) from exc
        if observed != record.get("sha256"):
            raise MechanismError(
                "upstream_input_changed", f"Frozen input changed during analysis: {label}"
            )


def _write_immutable_parquet(path: Path, frame: pd.DataFrame) -> str:
    try:
        return write_immutable_parquet(path, frame)
    except FixedProbeSensitivityError as exc:
        raise MechanismError(exc.code, str(exc), outcome=exc.outcome) from exc


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> str:
    try:
        return write_immutable_json(path, payload)
    except FixedProbeSensitivityError as exc:
        raise MechanismError(exc.code, str(exc), outcome=exc.outcome) from exc


def _mechanism_manifest(
    result: FixedProbeMechanismResult,
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    frames = {
        "position_mechanism": result.position_mechanism,
        "protein_associations": result.protein_associations,
    }
    outputs = {
        label: {
            "path": _logical_path(path, result.project_root),
            "sha256": sha256_file(path),
            "rows": len(frames[label]),
            "bytes": path.stat().st_size,
        }
        for label, path in output_paths.items()
    }
    return {
        "schema_version": "stage0_fixed_probe_mechanism_manifest_v1",
        "status": "complete",
        "scientific_scope": "stage0_fixed_probe_local_mechanism_closure_only",
        "checkpoint_commit": result.checkpoint_commit,
        "inputs": result.input_provenance,
        "analysis_protocol": {
            "identity": "stage0_fixed_probe_local_mechanism_v1",
            "cohort": {
                "definition": "exact_full_join_on_protein_id_and_canonical_position",
                "position_count": len(result.position_mechanism),
                "preselection_performed": False,
            },
            "continuous_variables": [
                "position_mean_abs_interaction",
                "position_rms_interaction",
                "position_max_abs_interaction",
            ],
            "decision_variables": [
                "top1_disagreement_fraction",
                "symmetric_regret_mean",
                "mean_normalized_rank_displacement_mean",
            ],
            "local_candidate_set": {
                "definition": "analytical_WT_plus_19_frozen_single_mutants",
                "choice_count": 20,
                "standard_amino_acid_order": STANDARD_AMINO_ACIDS,
            },
            "margin": {
                "definition": "highest_score_minus_second_highest_score",
                "score": "score_mean_logp_mask",
                "primary": "margin_mean_both",
                "epsilon_substitution": False,
            },
            "perturbation_to_margin": {
                "definition": (
                    "position_mean_abs_interaction_divided_by_margin_mean_both"
                ),
                "zero_margin": "null_with_margin_ratio_status_zero_margin",
                "semantics": "descriptive_only_not_probability_or_risk_score",
            },
            "ranking": {
                "direction": "higher_is_better",
                "exact_tie_break": "canonical_standard_amino_acid_order",
                "practical_tie_atol": 1.0e-6,
                "practical_tie_rtol": 1.0e-6,
                "practical_tie_changes_strict_ranking": False,
            },
            "association_matrix": [
                {"id": association_id, "x": x_field, "y": y_field}
                for association_id, x_field, y_field in ASSOCIATION_SPECS
            ],
            "association_semantics": (
                "within_protein_descriptive_pearson_and_spearman_only"
            ),
            "undefined_correlation": "null_with_structured_reason",
            "hypothesis_testing": False,
            "repeat_count": result.repeat_count,
            "repeat_seeds": list(range(result.repeat_count)),
        },
        "protein_order": list(result.protein_order),
        "project_fork": result.project_fork,
        "outputs": outputs,
        "final_counts": {
            "position_mechanism_rows": len(result.position_mechanism),
            "protein_association_rows": len(result.protein_associations),
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
    }


def materialize_fixed_probe_mechanism(
    result: FixedProbeMechanismResult,
    output_root: Path,
    *,
    expected_position_rows: int = EXPECTED_POSITION_COUNT,
    expected_protein_rows: int = EXPECTED_PROTEIN_COUNT,
) -> dict[str, Any]:
    """Validate and immutably render both tables, then the manifest last."""
    validate_mechanism_result(
        result,
        expected_position_rows=expected_position_rows,
        expected_protein_rows=expected_protein_rows,
    )
    _rehash_inputs(result)
    output_root.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "position_mechanism": output_root / "fixed_probe_position_mechanism.parquet",
        "protein_associations": output_root
        / "fixed_probe_protein_mechanism_association.parquet",
    }
    frames = {
        "position_mechanism": result.position_mechanism,
        "protein_associations": result.protein_associations,
    }
    write_status = {
        label: _write_immutable_parquet(output_paths[label], frames[label])
        for label in output_paths
    }
    _rehash_inputs(result)
    manifest = _mechanism_manifest(result, output_paths)
    manifest_path = output_root / "fixed_probe_mechanism_manifest.json"
    write_status["manifest"] = _write_immutable_json(manifest_path, manifest)
    _rehash_inputs(result)
    return {
        "status": "PASS",
        "manifest_path": manifest_path,
        "write_status": write_status,
        "position_mechanism_rows": len(result.position_mechanism),
        "protein_association_rows": len(result.protein_associations),
        "project_fork": result.project_fork["recommendation"],
    }


def run_fixed_probe_mechanism(project_root: Path) -> FixedProbeMechanismResult:
    """Load frozen inputs and build the single canonical mechanism result."""
    return build_fixed_probe_mechanism_result(
        load_frozen_mechanism_inputs(project_root)
    )
