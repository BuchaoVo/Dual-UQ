"""Confirmatory descriptive P/M/SDFI analysis over the A1 measurement layer.

This module consumes the immutable A1 paired response table.  It computes the
repository-defined perturbation (P), margin (M), and SDFI=P/M summaries while
preserving probe, position, protein, and realization nesting.  It never runs a
scorer and does not produce decision, mechanism, or hypothesis-test outputs.
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import pandas as pd

from dual_uq.core.atomic_io import atomic_write_new_bytes
from dual_uq.core.hashing import sha256_file

A1_DIR = Path("experiments/p2_design_baseline/scale1b-v2/structural_response")
OUTPUT_DIR = A1_DIR / "p_m_sdfi"
ANALYSIS_PROTOCOL = "scale1b_v2_p_m_sdfi_characterization_v1"
P_METRIC = "position_mean_abs_interaction"
M_METRIC = "margin_mean_both"
SDFI_METRIC = "sdfi"

PAIRED_REQUIRED = {
    "protein_id",
    "sequence_hash",
    "position",
    "repeat_index",
    "decoding_realization_sha256",
    "wt_aa",
    "mut_aa",
    "pdb_score_mean_logp_mask",
    "afdb_score_mean_logp_mask",
    "pdb_wt_score",
    "afdb_wt_score",
    "candidate_remodeling_d",
}
POSITION_A1_REQUIRED = {
    "protein_id",
    "position",
    "median_abs_c",
    "q90_abs_c",
    "median_abs_d",
    "q90_abs_d",
}
PROTEIN_A1_REQUIRED = {
    "protein_id",
    "median_abs_c",
    "median_abs_d",
    "g_median_abs",
    "position_count",
}
POSITION_KEYS = ["protein_id", "position"]
PROBE_KEYS = ["protein_id", "sequence_hash", "position", "wt_aa", "mut_aa"]
REPEAT_KEYS = ["protein_id", "position", "repeat_index"]


class PMSDFIError(ValueError):
    """Structured input, metric, or output failure."""

    def __init__(self, code: str, message: str, *, outcome: str = "BLOCKED") -> None:
        self.code = code
        self.outcome = outcome
        super().__init__(message)


@dataclass(frozen=True)
class PMSDFIInputs:
    project_root: Path
    a1_manifest_path: Path
    a1_manifest_sha256: str
    paired: pd.DataFrame
    a1_position: pd.DataFrame
    a1_protein: pd.DataFrame
    input_provenance: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class PMSDFIResult:
    project_root: Path
    position_metrics: pd.DataFrame
    protein_metrics: pd.DataFrame
    summary: dict[str, Any]
    input_provenance: dict[str, dict[str, Any]]


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise PMSDFIError("a1_schema_mismatch", f"{label} is missing columns: {missing}")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PMSDFIError("a1_manifest_unreadable", f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise PMSDFIError("a1_manifest_schema_mismatch", f"{label} must be an object")
    return value


def _bound_artifact(root: Path, base: Path, record: Any, label: str) -> tuple[Path, str]:
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise PMSDFIError("a1_manifest_schema_mismatch", f"Missing A1 path for {label}")
    path = (base / record["path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PMSDFIError("a1_manifest_schema_mismatch", f"A1 path escapes project root: {label}") from exc
    if not path.is_file():
        raise PMSDFIError("a1_input_missing", f"A1 artifact is missing: {path}")
    actual = sha256_file(path)
    if record.get("sha256") and actual != record["sha256"]:
        raise PMSDFIError("a1_input_hash_mismatch", f"A1 artifact hash differs: {label}")
    return path, actual


def load_p_m_sdfi_inputs(project_root: Path) -> PMSDFIInputs:
    """Load the four immutable A1 artifacts and verify their manifest bindings."""
    root = project_root.expanduser().resolve()
    base = (root / A1_DIR).resolve()
    manifest_path = base / "manifest.json"
    manifest = _load_json(manifest_path, "A1 structural-response manifest")
    if manifest.get("status") != "COMPLETE":
        raise PMSDFIError("a1_not_complete", "A1 structural-response manifest is not COMPLETE")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise PMSDFIError("a1_manifest_schema_mismatch", "A1 output bindings are absent")
    bound: dict[str, tuple[Path, str]] = {}
    for label in ("paired", "position_summary", "protein_summary"):
        bound[label] = _bound_artifact(root, base, outputs.get(label), label)
    paired = pd.read_parquet(bound["paired"][0])
    position = pd.read_parquet(bound["position_summary"][0])
    protein = pd.read_parquet(bound["protein_summary"][0])
    _validate_a1_tables(paired, position, protein, expected_repeat_count=30)
    provenance = {
        "a1_manifest": {
            "path": manifest_path.relative_to(root).as_posix(),
            "sha256": sha256_file(manifest_path),
        },
        **{
            label: {
                "path": path.relative_to(root).as_posix(),
                "sha256": digest,
                "rows": outputs[label].get("rows"),
            }
            for label, (path, digest) in bound.items()
        },
    }
    return PMSDFIInputs(
        project_root=root,
        a1_manifest_path=manifest_path,
        a1_manifest_sha256=sha256_file(manifest_path),
        paired=paired,
        a1_position=position,
        a1_protein=protein,
        input_provenance=provenance,
    )


def _validate_a1_tables(
    paired: pd.DataFrame,
    a1_position: pd.DataFrame,
    a1_protein: pd.DataFrame,
    *,
    expected_repeat_count: int,
) -> None:
    _require_columns(paired, PAIRED_REQUIRED, "A1 paired response")
    _require_columns(a1_position, POSITION_A1_REQUIRED, "A1 position summary")
    _require_columns(a1_protein, PROTEIN_A1_REQUIRED, "A1 protein summary")
    numeric = [
        "pdb_score_mean_logp_mask",
        "afdb_score_mean_logp_mask",
        "pdb_wt_score",
        "afdb_wt_score",
        "candidate_remodeling_d",
    ]
    if not np.isfinite(paired[numeric].to_numpy(dtype=np.float64)).all():
        raise PMSDFIError("a1_nonfinite", "A1 paired scores contain non-finite values")
    if paired.duplicated(PROBE_KEYS + ["repeat_index"]).any():
        raise PMSDFIError("a1_duplicate_key", "A1 probe-repeat scientific keys repeat")
    if set(paired["repeat_index"].astype(int)) != set(range(expected_repeat_count)):
        raise PMSDFIError("a1_repeat_grid_mismatch", "A1 repeat grid is not 0..29")
    probe_sizes = paired.groupby(PROBE_KEYS, sort=False).size()
    if not (probe_sizes == expected_repeat_count).all():
        raise PMSDFIError("a1_repeat_grid_mismatch", "A1 probe repeat coverage is incomplete")
    position_sizes = paired.groupby(POSITION_KEYS, sort=False)["sequence_hash"].nunique()
    if not (position_sizes == 19).all():
        raise PMSDFIError("a1_probe_grid_mismatch", "Each A1 position must contain 19 probes")
    if a1_position.duplicated(POSITION_KEYS).any() or a1_protein["protein_id"].duplicated().any():
        raise PMSDFIError("a1_duplicate_key", "A1 summary scientific keys repeat")
    if len(a1_position) != paired[POSITION_KEYS].drop_duplicates().shape[0]:
        raise PMSDFIError("a1_position_membership_mismatch", "A1 position summary membership differs")
    if len(a1_protein) != paired["protein_id"].nunique():
        raise PMSDFIError("a1_protein_membership_mismatch", "A1 protein summary membership differs")


def _quantiles(values: pd.Series, prefix: str) -> dict[str, float]:
    array = values.to_numpy(dtype=np.float64)
    return {
        f"{prefix}_q10": float(np.quantile(array, 0.10, method="linear")),
        f"{prefix}_q25": float(np.quantile(array, 0.25, method="linear")),
        f"{prefix}_median": float(np.quantile(array, 0.50, method="linear")),
        f"{prefix}_q75": float(np.quantile(array, 0.75, method="linear")),
        f"{prefix}_q90": float(np.quantile(array, 0.90, method="linear")),
        f"{prefix}_q95": float(np.quantile(array, 0.95, method="linear")),
        f"{prefix}_max": float(array.max()),
    }


def _top_two_margin(values: np.ndarray, wt: np.ndarray) -> np.ndarray:
    matrix = np.column_stack((values, wt))
    top_two = np.partition(matrix, -2, axis=1)[:, -2:]
    return top_two.max(axis=1) - top_two.min(axis=1)


def _build_repeat_margins(work: pd.DataFrame) -> pd.DataFrame:
    keys = ["protein_id", "position", "repeat_index"]
    ordered = work.sort_values(keys + ["mut_aa"], kind="stable")
    counts = ordered.groupby(keys, sort=False).size()
    if not (counts == 19).all():
        raise PMSDFIError("a1_probe_grid_mismatch", "Margin input does not contain 19 probes per position-repeat")
    group_ids = ordered.drop_duplicates(keys, keep="first")[keys].reset_index(drop=True)
    pdb_values = ordered["pdb_score_mean_logp_mask"].to_numpy(dtype=np.float64).reshape(-1, 19)
    afdb_values = ordered["afdb_score_mean_logp_mask"].to_numpy(dtype=np.float64).reshape(-1, 19)
    pdb_wt = ordered.groupby(keys, sort=False)["pdb_wt_score"].first().to_numpy(dtype=np.float64)
    afdb_wt = ordered.groupby(keys, sort=False)["afdb_wt_score"].first().to_numpy(dtype=np.float64)
    if not np.isfinite(np.concatenate((pdb_wt, afdb_wt))).all():
        raise PMSDFIError("a1_nonfinite", "A1 WT scores contain non-finite values")
    result = group_ids.copy()
    result["m_margin_pdb"] = _top_two_margin(pdb_values, pdb_wt)
    result["m_margin_afdb"] = _top_two_margin(afdb_values, afdb_wt)
    result["m_margin_both"] = (result["m_margin_pdb"] + result["m_margin_afdb"]) / 2.0
    if (result[["m_margin_pdb", "m_margin_afdb", "m_margin_both"]] < -1.0e-12).any().any():
        raise PMSDFIError("negative_margin", "A2 margins are materially negative", outcome="FAIL")
    result[["m_margin_pdb", "m_margin_afdb", "m_margin_both"]] = result[
        ["m_margin_pdb", "m_margin_afdb", "m_margin_both"]
    ].clip(lower=0.0)
    return result


def _build_position_metrics(
    paired: pd.DataFrame,
    a1_position: pd.DataFrame,
    *,
    expected_repeat_count: int,
    return_repeat: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    probe = paired.groupby(PROBE_KEYS, sort=False).agg(
        probe_interaction_mean=("candidate_remodeling_d", "mean"),
        probe_interaction_sd=("candidate_remodeling_d", "std"),
        probe_interaction_q25=("candidate_remodeling_d", lambda x: np.quantile(x, 0.25, method="linear")),
        probe_interaction_q75=("candidate_remodeling_d", lambda x: np.quantile(x, 0.75, method="linear")),
    ).reset_index().fillna(0.0)
    probe["probe_abs_interaction_mean"] = probe["probe_interaction_mean"].abs()
    position = probe.groupby(POSITION_KEYS, sort=False).agg(
        p_position_mean_abs_interaction=("probe_abs_interaction_mean", "mean"),
        p_position_rms_interaction=("probe_interaction_mean", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        p_position_max_abs_interaction=("probe_abs_interaction_mean", "max"),
        p_position_signed_mean=("probe_interaction_mean", "mean"),
        p_probe_count=("sequence_hash", "nunique"),
    ).reset_index()
    repeat_p = paired.groupby(REPEAT_KEYS, sort=False)["candidate_remodeling_d"].apply(
        lambda x: float(np.abs(x.to_numpy(dtype=np.float64)).mean())
    ).rename("p_repeat_mean_abs_interaction").reset_index()
    margins = _build_repeat_margins(paired)
    repeat = repeat_p.merge(margins, on=REPEAT_KEYS, how="outer", validate="one_to_one")
    if repeat.isna().any().any() or len(repeat) != len(margins):
        raise PMSDFIError("a2_repeat_join_mismatch", "P/M repeat grids do not reconcile")
    repeat_position = repeat.groupby(POSITION_KEYS, sort=False).agg(
        p_repeat_sd=("p_repeat_mean_abs_interaction", "std"),
        p_repeat_iqr=("p_repeat_mean_abs_interaction", lambda x: np.quantile(x, 0.75, method="linear") - np.quantile(x, 0.25, method="linear")),
        m_margin_mean_pdb=("m_margin_pdb", "mean"),
        m_margin_mean_afdb=("m_margin_afdb", "mean"),
        m_margin_mean_both=("m_margin_both", "mean"),
        m_margin_median_both=("m_margin_both", "median"),
        m_margin_iqr_both=("m_margin_both", lambda x: np.quantile(x, 0.75, method="linear") - np.quantile(x, 0.25, method="linear")),
        m_repeat_sd=("m_margin_both", "std"),
        m_repeat_iqr=("m_margin_both", lambda x: np.quantile(x, 0.75, method="linear") - np.quantile(x, 0.25, method="linear")),
    ).reset_index().fillna(0.0)
    repeat["sdfi_repeat"] = np.where(
        repeat["m_margin_both"] > 0,
        repeat["p_repeat_mean_abs_interaction"] / repeat["m_margin_both"],
        np.nan,
    )
    sdfi_repeat = repeat.groupby(POSITION_KEYS, sort=False)["sdfi_repeat"].agg(
        sdfi_repeat_sd="std",
        sdfi_repeat_iqr=lambda x: np.nanquantile(x, 0.75, method="linear") - np.nanquantile(x, 0.25, method="linear") if x.notna().any() else np.nan,
        sdfi_repeat_defined_fraction=lambda x: float(x.notna().mean()),
    ).reset_index().fillna({"sdfi_repeat_sd": 0.0, "sdfi_repeat_iqr": 0.0})
    position = position.merge(repeat_position, on=POSITION_KEYS, validate="one_to_one")
    position = position.merge(sdfi_repeat, on=POSITION_KEYS, validate="one_to_one")
    position["sdfi"] = np.where(
        position["m_margin_mean_both"] > 0,
        position["p_position_mean_abs_interaction"] / position["m_margin_mean_both"],
        np.nan,
    )
    position["sdfi_status"] = np.where(position["m_margin_mean_both"] > 0, "defined", "zero_margin")
    g_position = paired.groupby(REPEAT_KEYS, sort=False)["wt_baseline_shift_g"].first().groupby(POSITION_KEYS, sort=False).agg(
        g_position_median_abs=(lambda x: float(np.median(np.abs(x.to_numpy(dtype=np.float64)))))
    ).reset_index()
    position = position.merge(g_position, on=POSITION_KEYS, validate="one_to_one")
    position = position.merge(
        a1_position[["protein_id", "position", "median_abs_c", "q90_abs_c", "median_abs_d", "q90_abs_d"]].rename(
            columns={"median_abs_c": "a1_median_abs_c", "q90_abs_c": "a1_q90_abs_c", "median_abs_d": "a1_median_abs_d", "q90_abs_d": "a1_q90_abs_d"}
        ),
        on=POSITION_KEYS,
        validate="one_to_one",
    )
    position["realization_count"] = expected_repeat_count
    position = position.reset_index(drop=True)
    return (position, repeat) if return_repeat else position


def _spearman(x: pd.Series, y: pd.Series) -> float | None:
    valid = np.isfinite(x.to_numpy(dtype=np.float64)) & np.isfinite(y.to_numpy(dtype=np.float64))
    if int(valid.sum()) < 2:
        return None
    xv = x.to_numpy(dtype=np.float64)[valid]
    yv = y.to_numpy(dtype=np.float64)[valid]
    if np.all(xv == xv[0]) or np.all(yv == yv[0]):
        return None
    return float(pd.Series(xv).corr(pd.Series(yv), method="spearman"))


def _quantile_summary(frame: pd.DataFrame, column: str, prefix: str) -> dict[str, float]:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return _quantiles(values, prefix) if not values.empty else {}


def _build_protein_metrics(
    position: pd.DataFrame,
    a1_protein: pd.DataFrame,
    repeat: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metrics = position.groupby("protein_id", sort=False).agg(
        position_count=("position", "nunique"),
        p_position_mean=("p_position_mean_abs_interaction", "mean"),
        p_position_median=("p_position_mean_abs_interaction", "median"),
        p_position_q75=("p_position_mean_abs_interaction", lambda x: np.quantile(x, 0.75, method="linear")),
        p_position_q90=("p_position_mean_abs_interaction", lambda x: np.quantile(x, 0.90, method="linear")),
        p_position_max=("p_position_mean_abs_interaction", "max"),
        m_position_mean=("m_margin_mean_both", "mean"),
        m_position_median=("m_margin_mean_both", "median"),
        m_position_q75=("m_margin_mean_both", lambda x: np.quantile(x, 0.75, method="linear")),
        m_position_q90=("m_margin_mean_both", lambda x: np.quantile(x, 0.90, method="linear")),
        m_position_max=("m_margin_mean_both", "max"),
        sdfi_position_mean=("sdfi", "mean"),
        sdfi_position_median=("sdfi", "median"),
        sdfi_position_q75=("sdfi", lambda x: np.nanquantile(x, 0.75, method="linear")),
        sdfi_position_q90=("sdfi", lambda x: np.nanquantile(x, 0.90, method="linear")),
        sdfi_position_max=("sdfi", "max"),
        sdfi_defined_fraction=("sdfi", lambda x: float(x.notna().mean())),
        p_realization_sd=("p_repeat_sd", "mean"),
        m_realization_sd=("m_repeat_sd", "mean"),
        sdfi_realization_sd=("sdfi_repeat_sd", "mean"),
    ).reset_index()
    # Preserve a deterministic, null-safe protein-level A1 join.
    metrics = metrics.merge(a1_protein[["protein_id", "median_abs_c", "median_abs_d", "g_median_abs"]].rename(
        columns={"median_abs_c": "a1_median_abs_c", "median_abs_d": "a1_median_abs_d", "g_median_abs": "a1_g_median_abs"}
    ), on="protein_id", validate="one_to_one")
    metrics = metrics.replace([np.inf, -np.inf], np.nan)
    repeat_stability: dict[str, Any] = {}
    if repeat is not None:
        repeat_protein = repeat.groupby(["protein_id", "repeat_index"], sort=False).agg(
            p_repeat_median=("p_repeat_mean_abs_interaction", "median"),
            m_repeat_median=("m_margin_both", "median"),
            sdfi_repeat_median=("sdfi_repeat", "median"),
        ).reset_index()
        repeat_summary = repeat_protein.groupby("protein_id", sort=False).agg(
            p_realization_sd_direct=("p_repeat_median", "std"),
            m_realization_sd_direct=("m_repeat_median", "std"),
            sdfi_realization_sd_direct=("sdfi_repeat_median", "std"),
        ).reset_index().fillna(0.0)
        metrics = metrics.drop(columns=["p_realization_sd", "m_realization_sd", "sdfi_realization_sd"])
        metrics = metrics.merge(repeat_summary, on="protein_id", validate="one_to_one").rename(
            columns={
                "p_realization_sd_direct": "p_realization_sd",
                "m_realization_sd_direct": "m_realization_sd",
                "sdfi_realization_sd_direct": "sdfi_realization_sd",
            }
        )
        medians = metrics.set_index("protein_id")
        for metric, repeat_column, full_column in (
            ("P", "p_repeat_median", "p_position_median"),
            ("M", "m_repeat_median", "m_position_median"),
            ("SDFI", "sdfi_repeat_median", "sdfi_position_median"),
        ):
            pivot = repeat_protein.pivot(index="protein_id", columns="repeat_index", values=repeat_column)
            full = medians[full_column].reindex(pivot.index)
            correlations = [_spearman(pivot[column], full) for column in pivot.columns]
            top_count = max(1, int(np.ceil(len(full) * 0.25)))
            full_top = set(full.nlargest(top_count).index)
            overlaps = []
            for column in pivot.columns:
                repeat_top = set(pivot[column].nlargest(top_count).index)
                overlaps.append(len(full_top & repeat_top) / top_count)
            finite_correlations = [value for value in correlations if value is not None]
            repeat_stability[metric] = {
                "repeat_vs_protein_median_spearman_median": float(np.median(finite_correlations)) if finite_correlations else None,
                "repeat_vs_protein_median_spearman_min": float(np.min(finite_correlations)) if finite_correlations else None,
                "repeat_vs_protein_median_spearman_max": float(np.max(finite_correlations)) if finite_correlations else None,
                "top_quartile_overlap_median": float(np.median(overlaps)),
                "repeat_count": len(pivot.columns),
            }
    summary: dict[str, Any] = {
        "protein_count": len(metrics),
        "position_count": len(position),
        "p_probe_count": int(position["p_probe_count"].sum()),
        "realization_count": int(position["realization_count"].iloc[0]),
        "definitions": {
            "P": "position_mean_abs_interaction = mean over 19 probes of abs(mean over 30 repeats of D)",
            "M": "margin_mean_both = mean over 30 repeats and PDB/AFDB of (top score - second score) across WT plus 19 mutants",
            "SDFI": "P / M when M > 0; null with status zero_margin otherwise",
            "D": "A1 candidate_remodeling_d under AFDB-minus-PDB orientation",
            "aggregation": "probe x realization -> probe -> position -> protein -> cohort",
        },
        "protein_metric_distributions": {
            **_quantile_summary(metrics, "p_position_median", "p_protein_median"),
            **_quantile_summary(metrics, "m_position_median", "m_protein_median"),
            **_quantile_summary(metrics, "sdfi_position_median", "sdfi_protein_median"),
        },
        "position_metric_distributions": {
            **_quantile_summary(position, "p_position_mean_abs_interaction", "p_position"),
            **_quantile_summary(position, "m_margin_mean_both", "m_position"),
            **_quantile_summary(position, "sdfi", "sdfi_position"),
        },
        "prevalence": {
            "sdfi_defined_position_count": int(position["sdfi"].notna().sum()),
            "sdfi_zero_margin_position_count": int((position["sdfi_status"] == "zero_margin").sum()),
            "sdfi_defined_position_fraction": float(position["sdfi"].notna().mean()),
            "zero_p_position_count": int((position["p_position_mean_abs_interaction"] == 0).sum()),
        },
        "relation_to_a1": {},
        "relation_to_a1_position": {},
        "realization_stability": repeat_stability,
    }
    for metric_column, metric_name in (
        ("p_position_median", "P"),
        ("m_position_median", "M"),
        ("sdfi_position_median", "SDFI"),
    ):
        for a1_column, a1_name in (
            ("a1_g_median_abs", "g_median_abs"),
            ("a1_median_abs_c", "c_median_abs"),
            ("a1_median_abs_d", "d_median_abs"),
        ):
            summary["relation_to_a1"][f"{metric_name}_vs_{a1_name}_spearman"] = _spearman(metrics[metric_column], metrics[a1_column])
    for metric_column, metric_name in (
        ("p_position_mean_abs_interaction", "P"),
        ("m_margin_mean_both", "M"),
        ("sdfi", "SDFI"),
    ):
        for a1_column, a1_name in (
            ("g_position_median_abs", "g_median_abs"),
            ("a1_median_abs_c", "c_median_abs"),
            ("a1_median_abs_d", "d_median_abs"),
        ):
            summary["relation_to_a1_position"][f"{metric_name}_vs_{a1_name}_spearman"] = _spearman(position[metric_column], position[a1_column])
    return metrics, summary


def compute_p_m_sdfi(
    paired: pd.DataFrame,
    *,
    a1_position: pd.DataFrame | None = None,
    a1_protein: pd.DataFrame | None = None,
    expected_repeat_count: int | None = 30,
) -> PMSDFIResult:
    """Compute P, M, and SDFI from one A1 paired measurement table."""
    _validate_a1_tables(
        paired,
        a1_position if a1_position is not None else paired[POSITION_KEYS].drop_duplicates().assign(
            median_abs_c=0.0, q90_abs_c=0.0, median_abs_d=0.0, q90_abs_d=0.0
        ),
        a1_protein if a1_protein is not None else paired[["protein_id"]].drop_duplicates().assign(
            median_abs_c=0.0, median_abs_d=0.0, g_median_abs=0.0, position_count=1
        ),
        expected_repeat_count=expected_repeat_count or int(paired["repeat_index"].nunique()),
    )
    repeats = expected_repeat_count or int(paired["repeat_index"].nunique())
    if repeats != int(paired["repeat_index"].nunique()):
        raise PMSDFIError("a1_repeat_grid_mismatch", "Observed repeat count differs from requested count")
    a1_position_frame = a1_position if a1_position is not None else paired[POSITION_KEYS].drop_duplicates().assign(
        median_abs_c=0.0, q90_abs_c=0.0, median_abs_d=0.0, q90_abs_d=0.0
    )
    a1_protein_frame = a1_protein if a1_protein is not None else paired[["protein_id"]].drop_duplicates().assign(
        median_abs_c=0.0, median_abs_d=0.0, g_median_abs=0.0, position_count=1
    )
    position, repeat = _build_position_metrics(
        paired, a1_position_frame, expected_repeat_count=repeats, return_repeat=True
    )
    protein, summary = _build_protein_metrics(position, a1_protein_frame, repeat)
    summary["paired_observation_count"] = len(paired)
    summary["fixed_probe_count"] = int(paired[PROBE_KEYS].drop_duplicates().shape[0])
    summary["condition_contrast"] = "AFDB - PDB"
    summary["interpretation_boundary"] = [
        "descriptive P/M/SDFI heterogeneity only",
        "no decision-level sensitivity, mechanism, or H1 inference",
    ]
    return PMSDFIResult(
        project_root=Path("."),
        position_metrics=position,
        protein_metrics=protein,
        summary=summary,
        input_provenance={},
    )


def analyze_p_m_sdfi(inputs: PMSDFIInputs) -> PMSDFIResult:
    """Analyze loaded A1 inputs and retain their provenance for materialization."""
    position, repeat = _build_position_metrics(
        inputs.paired, inputs.a1_position, expected_repeat_count=30, return_repeat=True
    )
    protein, summary = _build_protein_metrics(position, inputs.a1_protein, repeat)
    summary["paired_observation_count"] = len(inputs.paired)
    summary["fixed_probe_count"] = int(inputs.paired[PROBE_KEYS].drop_duplicates().shape[0])
    summary["condition_contrast"] = "AFDB - PDB"
    summary["interpretation_boundary"] = [
        "descriptive P/M/SDFI heterogeneity only",
        "no decision-level sensitivity, mechanism, or H1 inference",
    ]
    return PMSDFIResult(
        project_root=inputs.project_root,
        position_metrics=position,
        protein_metrics=protein,
        summary=summary,
        input_provenance=inputs.input_provenance,
    )


def _immutable_parquet(frame: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        digest = sha256_file(temporary)
        if path.exists():
            if sha256_file(path) != digest:
                raise PMSDFIError("immutable_output_conflict", f"Output differs: {path}")
            return "reused_identical"
        os.link(temporary, path)
        return "created"
    finally:
        temporary.unlink(missing_ok=True)


def _figure_bytes(result: PMSDFIResult, name: str) -> bytes:
    import matplotlib.pyplot as plt

    protein = result.protein_metrics
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    if name == "protein_metric_distributions":
        for axis, column, label in zip(
            axes,
            ("p_position_median", "m_position_median", "sdfi_position_median"),
            ("P", "M", "SDFI"),
            strict=True,
        ):
            axis.boxplot(protein[column].dropna().to_numpy(), widths=0.6, showfliers=False)
            axis.set_title(label)
            axis.set_xticks([])
            axis.set_ylabel("protein-level value")
        figure.suptitle("Protein-level P / M / SDFI distributions")
    elif name == "protein_heterogeneity":
        for column, label in (("p_position_median", "P"), ("m_position_median", "M"), ("sdfi_position_median", "SDFI")):
            ordered = np.sort(protein[column].dropna().to_numpy(dtype=np.float64))
            axes[0].plot(np.linspace(0, 1, len(ordered)), ordered, label=label)
        axes[0].legend()
        axes[0].set_xlabel("protein quantile")
        axes[0].set_ylabel("ordered metric value")
        axes[0].set_title("Between-protein heterogeneity")
        axes[1].axis("off")
        axes[2].axis("off")
    elif name == "realization_stability":
        for axis, value, spread, label in zip(
            axes,
            ("p_position_median", "m_position_median", "sdfi_position_median"),
            ("p_realization_sd", "m_realization_sd", "sdfi_realization_sd"),
            ("P", "M", "SDFI"),
            strict=True,
        ):
            axis.scatter(protein[value], protein[spread], s=14, alpha=0.75)
            axis.set_xlabel(f"{label} median")
            axis.set_ylabel("mean repeat SD")
            axis.set_title(label)
    elif name == "metric_vs_a1_response":
        for axis, column, label in zip(
            axes,
            ("p_position_median", "m_position_median", "sdfi_position_median"),
            ("P", "M", "SDFI"),
            strict=True,
        ):
            axis.scatter(protein["a1_median_abs_d"], protein[column], s=14, alpha=0.75)
            axis.set_xlabel("A1 median |D|")
            axis.set_ylabel(label)
            axis.set_title(f"{label} vs A1 |D|")
    else:
        plt.close(figure)
        raise PMSDFIError("unknown_figure", name)
    output = io.BytesIO()
    figure.savefig(output, format="png", metadata={"Software": "Dual-UQ"}, dpi=140)
    plt.close(figure)
    return output.getvalue()


def materialize_p_m_sdfi(result: PMSDFIResult, output_dir: Path) -> dict[str, Any]:
    """Write position/protein metrics, summary, and four diagnostics immutably."""
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    statuses: dict[str, str] = {}
    outputs: dict[str, Any] = {}
    for label, frame, filename in (
        ("position_metrics", result.position_metrics, "position_metrics.parquet"),
        ("protein_metrics", result.protein_metrics, "protein_metrics.parquet"),
    ):
        path = output_dir / filename
        statuses[label] = _immutable_parquet(frame, path)
        outputs[label] = {"path": filename, "rows": len(frame), "sha256": sha256_file(path)}
    payload = {
        "analysis_protocol": ANALYSIS_PROTOCOL,
        "metric_definitions": result.summary["definitions"],
        "input_provenance": result.input_provenance,
        "summary": result.summary,
        "outputs": outputs,
        "scope_exclusions": ["decision_level_sensitivity", "Top-1", "rank", "regret", "mechanism", "H1"],
    }
    summary_bytes = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    summary_path = output_dir / "metric_summary.json"
    if summary_path.exists() and summary_path.read_bytes() != summary_bytes:
        raise PMSDFIError("immutable_output_conflict", f"Output differs: {summary_path}")
    if not summary_path.exists():
        atomic_write_new_bytes(summary_path, summary_bytes)
        statuses["metric_summary"] = "created"
    else:
        statuses["metric_summary"] = "reused_identical"
    figure_dir = output_dir / "figures"
    for name in ("protein_metric_distributions", "protein_heterogeneity", "realization_stability", "metric_vs_a1_response"):
        path = figure_dir / f"{name}.png"
        data = _figure_bytes(result, name)
        if path.exists() and path.read_bytes() != data:
            raise PMSDFIError("immutable_output_conflict", f"Output differs: {path}")
        if not path.exists():
            atomic_write_new_bytes(path, data)
            statuses[name] = "created"
        else:
            statuses[name] = "reused_identical"
        outputs.setdefault("figures", {})[name] = {"path": f"figures/{path.name}", "sha256": sha256_file(path)}
    return {"output_dir": output_dir, "outputs": outputs, "write_status": statuses}
