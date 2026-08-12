"""Descriptive PDB/AFDB structural-response characterization.

The module consumes the completed formal ProteinMPNN measurement release and
constructs one analysis-ready paired layer. It does not execute ProteinMPNN
and does not implement downstream decision, mechanism, or hypothesis tests.
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

RAW_REQUIRED_COLUMNS = {
    "protein_id",
    "sequence_hash",
    "position",
    "wt_aa",
    "mut_aa",
    "backbone_condition",
    "backbone_sha256",
    "repeat_index",
    "seed",
    "decoding_realization_sha256",
    "score_mean_logp_mask",
    "delta_score_vs_wt",
}
WT_REQUIRED_COLUMNS = {
    "protein_id",
    "backbone_condition",
    "backbone_sha256",
    "repeat_index",
    "seed",
    "decoding_realization_sha256",
    "score_mean_logp_mask",
}
COHORT_REQUIRED_COLUMNS = {"protein_id", "sequence_cluster"}
PROBE_KEY_COLUMNS = {"protein_id", "sequence_hash", "position", "wt_aa", "mut_aa"}
PAIR_KEY_COLUMNS = [
    "protein_id",
    "sequence_hash",
    "position",
    "repeat_index",
    "decoding_realization_sha256",
]
WT_PAIR_KEY_COLUMNS = ["protein_id", "repeat_index", "decoding_realization_sha256"]
ANALYSIS_PROTOCOL_VERSION = "scale1b_v2_structural_response_characterization_v1"
STRUCTURAL_ORIENTATION = "AFDB_minus_PDB"
EPSILON = 1.0e-12


class StructuralResponseError(ValueError):
    """Structured input, pairing, or materialization failure."""

    def __init__(self, code: str, message: str, *, outcome: str = "BLOCKED") -> None:
        self.code = code
        self.outcome = outcome
        super().__init__(message)


@dataclass
class StructuralResponseInputs:
    project_root: Path
    scoring_manifest_path: Path
    scoring_manifest_sha256: str
    scoring_manifest: dict[str, Any]
    wt_scores: pd.DataFrame
    raw_scores: pd.DataFrame
    cohort: pd.DataFrame
    input_provenance: dict[str, dict[str, Any]]
    frozen_probe_keys: pd.DataFrame | None = None


@dataclass
class StructuralResponseSummary:
    candidate_summary: pd.DataFrame
    position_summary: pd.DataFrame
    protein_summary: pd.DataFrame
    cohort_summary: dict[str, Any]


@dataclass
class StructuralResponseResult:
    inputs: StructuralResponseInputs
    paired: pd.DataFrame
    summary: StructuralResponseSummary


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise StructuralResponseError(
            "schema_mismatch", f"{label} is missing required columns: {missing}"
        )


def _require_finite(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    values = frame[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise StructuralResponseError("nonfinite_input", f"{label} contains non-finite scores")


def _validate_input_tables(inputs: StructuralResponseInputs) -> None:
    _require_columns(inputs.raw_scores, RAW_REQUIRED_COLUMNS, "raw scores")
    _require_columns(inputs.wt_scores, WT_REQUIRED_COLUMNS, "WT scores")
    _require_columns(inputs.cohort, COHORT_REQUIRED_COLUMNS, "cohort")
    if inputs.raw_scores.empty or inputs.wt_scores.empty:
        raise StructuralResponseError("empty_input", "score tables must not be empty")
    if set(inputs.raw_scores["backbone_condition"].astype(str)) != {"PDB", "AFDB"}:
        raise StructuralResponseError("condition_grid_mismatch", "raw scores must contain PDB and AFDB")
    if set(inputs.wt_scores["backbone_condition"].astype(str)) != {"PDB", "AFDB"}:
        raise StructuralResponseError("condition_grid_mismatch", "WT scores must contain PDB and AFDB")
    if inputs.raw_scores.duplicated(PAIR_KEY_COLUMNS + ["backbone_condition"]).any():
        raise StructuralResponseError("duplicate_raw_key", "raw scientific keys are duplicated")
    if inputs.wt_scores.duplicated(WT_PAIR_KEY_COLUMNS + ["backbone_condition"]).any():
        raise StructuralResponseError("duplicate_wt_key", "WT scientific keys are duplicated")
    _require_finite(
        inputs.raw_scores,
        ["score_mean_logp_mask", "delta_score_vs_wt"],
        "raw scores",
    )
    _require_finite(inputs.wt_scores, ["score_mean_logp_mask"], "WT scores")
    cohort_ids = set(inputs.cohort["protein_id"].astype(str))
    raw_ids = set(inputs.raw_scores["protein_id"].astype(str))
    wt_ids = set(inputs.wt_scores["protein_id"].astype(str))
    if raw_ids != cohort_ids or wt_ids != cohort_ids:
        raise StructuralResponseError("cohort_identity_mismatch", "cohort and score protein identities differ")
    if inputs.frozen_probe_keys is not None:
        probe_columns = sorted(PROBE_KEY_COLUMNS)
        _require_columns(inputs.frozen_probe_keys, set(probe_columns), "frozen probes")
        observed = inputs.raw_scores[probe_columns].drop_duplicates().sort_values(probe_columns)
        expected = inputs.frozen_probe_keys[probe_columns].drop_duplicates().sort_values(probe_columns)
        observed_values = observed.reset_index(drop=True).to_numpy(dtype=object)
        expected_values = expected.reset_index(drop=True).to_numpy(dtype=object)
        if observed_values.shape != expected_values.shape or not np.array_equal(
            observed_values, expected_values
        ):
            raise StructuralResponseError("probe_identity_mismatch", "raw scores do not match frozen probe keys")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StructuralResponseError("input_manifest_unreadable", f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise StructuralResponseError("input_manifest_schema_mismatch", f"{label} must be a JSON object")
    return value


def _normalize_frozen_probe_keys(probes: pd.DataFrame) -> pd.DataFrame:
    """Adapt the frozen Scale-1B-v2 probe schema to scoring scientific keys.

    The frozen probe release names the position and substituted residue as
    ``canonical_position`` and ``candidate_aa``.  Formal scoring uses the
    corresponding stable scientific-key names ``position`` and ``mut_aa``.
    This is a schema adapter only; values and probe identity are unchanged.
    """
    required_release_columns = {
        "protein_id",
        "sequence_hash",
        "canonical_position",
        "wt_aa",
        "candidate_aa",
    }
    _require_columns(probes, required_release_columns, "frozen probes")
    normalized = probes.rename(
        columns={"canonical_position": "position", "candidate_aa": "mut_aa"}
    ).copy()
    return normalized


def _bound_path(root: Path, record: dict[str, Any], label: str) -> tuple[Path, str]:
    logical = Path(str(record["path"]))
    path = (root / logical).resolve()
    if not path.is_file():
        raise StructuralResponseError("input_missing", f"{label} is missing: {logical}")
    expected = str(record.get("sha256", ""))
    actual = sha256_file(path)
    if expected and actual != expected:
        raise StructuralResponseError("input_hash_mismatch", f"{label} hash differs: {logical}")
    return path, actual


def load_structural_response_inputs(project_root: Path) -> StructuralResponseInputs:
    """Load and validate the frozen formal measurement inputs."""
    root = project_root.expanduser().resolve()
    consolidated = root / "runs/design_baseline/scale1b-v2/formal/consolidated"
    scoring_manifest_path = consolidated / "fixed_probe_scoring_manifest.json"
    scoring_manifest = _load_json(scoring_manifest_path, "scoring manifest")
    if scoring_manifest.get("status") != "COMPLETE":
        raise StructuralResponseError("scoring_not_complete", "formal scoring manifest is not COMPLETE")
    scoring_manifest_sha = sha256_file(scoring_manifest_path)
    outputs = scoring_manifest.get("outputs", {})
    raw_path, raw_sha = _bound_path(consolidated, outputs.get("probe", {}), "probe scores")
    wt_path, wt_sha = _bound_path(consolidated, outputs.get("wt", {}), "WT scores")

    freeze_path = root / "experiments/p2_design_baseline/scale1/scale1b_v2/scale1b_v2_freeze_manifest.json"
    freeze = _load_json(freeze_path, "Scale-1B-v2 freeze manifest")
    freeze_outputs = freeze.get("outputs", {})
    cohort_path, cohort_sha = _bound_path(root, freeze_outputs["primary_cohort"], "primary cohort")
    probes_path, probes_sha = _bound_path(root, freeze_outputs["fixed_probes"], "fixed probes")
    raw = pd.read_parquet(raw_path)
    wt = pd.read_parquet(wt_path)
    cohort = pd.read_parquet(cohort_path)
    probes = _normalize_frozen_probe_keys(pd.read_parquet(probes_path))
    inputs = StructuralResponseInputs(
        project_root=root,
        scoring_manifest_path=scoring_manifest_path,
        scoring_manifest_sha256=scoring_manifest_sha,
        scoring_manifest=scoring_manifest,
        wt_scores=wt,
        raw_scores=raw,
        cohort=cohort,
        frozen_probe_keys=probes,
        input_provenance={
            "scoring_manifest": {"path": scoring_manifest_path.relative_to(root).as_posix(), "sha256": scoring_manifest_sha},
            "raw_scores": {"path": raw_path.relative_to(root).as_posix(), "sha256": raw_sha, "rows": len(raw)},
            "wt_scores": {"path": wt_path.relative_to(root).as_posix(), "sha256": wt_sha, "rows": len(wt)},
            "freeze_manifest": {"path": freeze_path.relative_to(root).as_posix(), "sha256": sha256_file(freeze_path)},
            "primary_cohort": {"path": cohort_path.relative_to(root).as_posix(), "sha256": cohort_sha, "rows": len(cohort)},
            "fixed_probes": {"path": probes_path.relative_to(root).as_posix(), "sha256": probes_sha, "rows": len(probes)},
        },
    )
    _validate_input_tables(inputs)
    return inputs


def _merge_or_fail(
    left: pd.DataFrame,
    right: pd.DataFrame,
    keys: list[str],
    label: str,
    *,
    validate: str = "one_to_one",
) -> pd.DataFrame:
    merged = left.merge(right, on=keys, how="outer", indicator=True, validate=validate)
    if not (merged["_merge"] == "both").all():
        raise StructuralResponseError("missing_structural_pair", f"{label} PDB/AFDB pairing is incomplete", outcome="FAIL")
    return merged.drop(columns="_merge")


def build_paired_structural_response(inputs: StructuralResponseInputs) -> pd.DataFrame:
    """Construct one row per candidate/probe and decoding realization."""
    _validate_input_tables(inputs)
    raw = inputs.raw_scores
    pdb = raw.loc[raw["backbone_condition"] == "PDB"].copy()
    afdb = raw.loc[raw["backbone_condition"] == "AFDB"].copy()
    score_columns = ["score_mean_logp_mask", "delta_score_vs_wt", "backbone_sha256"]
    pdb = pdb[PAIR_KEY_COLUMNS + ["wt_aa", "mut_aa"] + score_columns].rename(
        columns={column: f"pdb_{column}" for column in score_columns}
    )
    afdb = afdb[PAIR_KEY_COLUMNS + ["wt_aa", "mut_aa"] + score_columns].rename(
        columns={column: f"afdb_{column}" for column in score_columns}
    )
    paired = _merge_or_fail(pdb, afdb, PAIR_KEY_COLUMNS, "candidate")
    if not (paired["wt_aa_x"] == paired["wt_aa_y"]).all() or not (paired["mut_aa_x"] == paired["mut_aa_y"]).all():
        raise StructuralResponseError("probe_annotation_mismatch", "PDB/AFDB probe annotations differ", outcome="FAIL")
    paired = paired.rename(columns={"wt_aa_x": "wt_aa", "mut_aa_x": "mut_aa"}).drop(
        columns=["wt_aa_y", "mut_aa_y"]
    )

    wt = inputs.wt_scores
    wt_pdb = wt.loc[wt["backbone_condition"] == "PDB", WT_PAIR_KEY_COLUMNS + ["score_mean_logp_mask", "backbone_sha256"]].rename(
        columns={"score_mean_logp_mask": "pdb_wt_score", "backbone_sha256": "pdb_wt_backbone_sha256"}
    )
    wt_afdb = wt.loc[wt["backbone_condition"] == "AFDB", WT_PAIR_KEY_COLUMNS + ["score_mean_logp_mask", "backbone_sha256"]].rename(
        columns={"score_mean_logp_mask": "afdb_wt_score", "backbone_sha256": "afdb_wt_backbone_sha256"}
    )
    paired = _merge_or_fail(
        paired, wt_pdb, WT_PAIR_KEY_COLUMNS, "WT", validate="many_to_one"
    )
    paired = _merge_or_fail(
        paired, wt_afdb, WT_PAIR_KEY_COLUMNS, "WT", validate="many_to_one"
    )
    paired["wt_baseline_shift_g"] = paired["afdb_wt_score"] - paired["pdb_wt_score"]
    paired["candidate_absolute_shift_c"] = paired["afdb_score_mean_logp_mask"] - paired["pdb_score_mean_logp_mask"]
    paired["candidate_remodeling_d"] = paired["candidate_absolute_shift_c"] - paired["wt_baseline_shift_g"]
    delta_pair = paired["afdb_delta_score_vs_wt"] - paired["pdb_delta_score_vs_wt"]
    if not np.allclose(paired["candidate_remodeling_d"], delta_pair, atol=EPSILON, rtol=EPSILON):
        raise StructuralResponseError("delta_algebra_mismatch", "D does not equal AFDB-minus-PDB delta_score_vs_wt", outcome="FAIL")
    order = {str(value): index for index, value in enumerate(inputs.cohort["protein_id"].astype(str))}
    paired["_protein_order"] = paired["protein_id"].map(order)
    paired = paired.sort_values(
        ["_protein_order", "position", "mut_aa", "repeat_index", "decoding_realization_sha256"],
        kind="mergesort",
    ).drop(columns="_protein_order").reset_index(drop=True)
    return paired


def _quantiles(values: pd.Series, prefix: str) -> dict[str, float]:
    array = values.to_numpy(dtype=np.float64)
    return {
        f"{prefix}_median": float(values.median()),
        f"{prefix}_q75": float(np.quantile(array, 0.75, method="linear")),
        f"{prefix}_q90": float(np.quantile(array, 0.90, method="linear")),
        f"{prefix}_q95": float(np.quantile(array, 0.95, method="linear")),
    }


def _q(values: pd.Series, probability: float) -> float:
    return float(np.quantile(values.to_numpy(dtype=np.float64), probability, method="linear"))


def summarize_structural_response(paired: pd.DataFrame) -> StructuralResponseSummary:
    """Summarize candidate, position, and protein units before cohort aggregation."""
    required = set(PAIR_KEY_COLUMNS + ["wt_aa", "mut_aa", "wt_baseline_shift_g", "candidate_absolute_shift_c", "candidate_remodeling_d"])
    _require_columns(paired, required, "paired response")
    work = paired.copy()
    work["abs_g"] = work["wt_baseline_shift_g"].abs()
    work["abs_c"] = work["candidate_absolute_shift_c"].abs()
    work["abs_d"] = work["candidate_remodeling_d"].abs()
    candidate_keys = ["protein_id", "sequence_hash", "position", "wt_aa", "mut_aa"]
    candidate = work.groupby(candidate_keys, sort=False).agg(
        median_g=("wt_baseline_shift_g", "median"),
        median_abs_c=("abs_c", "median"),
        median_abs_d=("abs_d", "median"),
        q75_abs_c=("abs_c", lambda x: _q(x, 0.75)),
        q75_abs_d=("abs_d", lambda x: _q(x, 0.75)),
        repeat_sd_c=("candidate_absolute_shift_c", "std"),
        repeat_sd_d=("candidate_remodeling_d", "std"),
        repeat_count=("repeat_index", "nunique"),
    ).reset_index()
    candidate[["repeat_sd_c", "repeat_sd_d"]] = candidate[["repeat_sd_c", "repeat_sd_d"]].fillna(0.0)

    position = candidate.groupby(["protein_id", "position"], sort=False).agg(
        median_abs_c=("median_abs_c", "median"),
        q75_abs_c=("median_abs_c", lambda x: _q(x, 0.75)),
        q90_abs_c=("median_abs_c", lambda x: _q(x, 0.90)),
        median_abs_d=("median_abs_d", "median"),
        q75_abs_d=("median_abs_d", lambda x: _q(x, 0.75)),
        q90_abs_d=("median_abs_d", lambda x: _q(x, 0.90)),
        substitution_sd_d=("median_abs_d", "std"),
        substitution_count=("sequence_hash", "nunique"),
    ).reset_index()
    position["substitution_sd_d"] = position["substitution_sd_d"].fillna(0.0)

    g = work[["protein_id", "repeat_index", "wt_baseline_shift_g"]].drop_duplicates()
    g_protein = g.groupby("protein_id", sort=False).agg(
        g_mean=("wt_baseline_shift_g", "mean"),
        g_median=("wt_baseline_shift_g", "median"),
        g_sd=("wt_baseline_shift_g", "std"),
        g_min=("wt_baseline_shift_g", "min"),
        g_max=("wt_baseline_shift_g", "max"),
        g_median_abs=("wt_baseline_shift_g", lambda x: x.abs().median()),
        g_iqr=("wt_baseline_shift_g", lambda x: _q(x, 0.75) - _q(x, 0.25)),
    ).reset_index()
    g_protein["g_sd"] = g_protein["g_sd"].fillna(0.0)
    g_sign = g.groupby("protein_id", sort=False)["wt_baseline_shift_g"].apply(
        lambda x: max(float((x > 0).mean()), float((x < 0).mean()))
    ).rename("g_sign_consistency").reset_index()
    protein = candidate.groupby("protein_id", sort=False).agg(
        candidate_count=("sequence_hash", "nunique"),
        position_count=("position", "nunique"),
        median_abs_c=("median_abs_c", "median"),
        mean_abs_c=("median_abs_c", "mean"),
        q75_abs_c=("median_abs_c", lambda x: _q(x, 0.75)),
        q90_abs_c=("median_abs_c", lambda x: _q(x, 0.90)),
        median_abs_d=("median_abs_d", "median"),
        mean_abs_d=("median_abs_d", "mean"),
        q75_abs_d=("median_abs_d", lambda x: _q(x, 0.75)),
        q90_abs_d=("median_abs_d", lambda x: _q(x, 0.90)),
        substitution_sd_d=("median_abs_d", "std"),
    ).reset_index()
    protein["substitution_sd_d"] = protein["substitution_sd_d"].fillna(0.0)
    position_protein = position.groupby("protein_id", sort=False).agg(
        position_median_abs_d_sd=("median_abs_d", "std"),
        position_median_abs_d_iqr=("median_abs_d", lambda x: _q(x, 0.75) - _q(x, 0.25)),
    ).reset_index()
    position_protein["position_median_abs_d_sd"] = position_protein["position_median_abs_d_sd"].fillna(0.0)
    repeat_c = work.groupby(["protein_id", "repeat_index"], sort=False)["candidate_absolute_shift_c"].apply(lambda x: x.abs().median()).rename("repeat_median_abs_c").reset_index()
    repeat_d = work.groupby(["protein_id", "repeat_index"], sort=False)["candidate_remodeling_d"].apply(lambda x: x.abs().median()).rename("repeat_median_abs_d").reset_index()
    repeat_stats = repeat_c.merge(repeat_d, on=["protein_id", "repeat_index"], validate="one_to_one").groupby("protein_id", sort=False).agg(
        realization_sd_abs_c=("repeat_median_abs_c", "std"),
        realization_iqr_abs_c=("repeat_median_abs_c", lambda x: _q(x, 0.75) - _q(x, 0.25)),
        realization_sd_abs_d=("repeat_median_abs_d", "std"),
        realization_iqr_abs_d=("repeat_median_abs_d", lambda x: _q(x, 0.75) - _q(x, 0.25)),
    ).reset_index().fillna(0.0)
    protein = protein.merge(g_protein, on="protein_id", validate="one_to_one")
    protein = protein.merge(g_sign, on="protein_id", validate="one_to_one")
    protein = protein.merge(position_protein, on="protein_id", validate="one_to_one")
    protein = protein.merge(repeat_stats, on="protein_id", validate="one_to_one")
    cohort_summary: dict[str, Any] = {
        "protein_count": int(protein["protein_id"].nunique()),
        "position_count": int(position[["protein_id", "position"]].drop_duplicates().shape[0]),
        "candidate_count": int(candidate[["protein_id", "sequence_hash"]].drop_duplicates().shape[0]),
        "paired_row_count": len(paired),
        "repeat_count": int(paired["repeat_index"].nunique()),
        "orientation": STRUCTURAL_ORIENTATION,
    }
    cohort_summary.update(_quantiles(protein["g_median_abs"], "protein_g_median_abs"))
    cohort_summary.update(_quantiles(protein["median_abs_c"], "protein_abs_c"))
    cohort_summary.update(_quantiles(protein["median_abs_d"], "protein_abs_d"))
    cohort_summary["protein_abs_c_abs_d_spearman"] = float(protein["median_abs_c"].corr(protein["median_abs_d"], method="spearman"))
    return StructuralResponseSummary(
        candidate_summary=candidate,
        position_summary=position,
        protein_summary=protein,
        cohort_summary=cohort_summary,
    )


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if pd.isna(value) if not isinstance(value, (dict, list, tuple, np.ndarray)) else False:
        return None
    return value


def _immutable_parquet(frame: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        digest = sha256_file(temporary)
        if path.exists():
            if sha256_file(path) != digest:
                raise StructuralResponseError("immutable_output_conflict", f"output differs: {path}")
            return "reused_identical"
        os.link(temporary, path)
        return "created"
    finally:
        temporary.unlink(missing_ok=True)


def _immutable_bytes(data: bytes, path: Path) -> str:
    if path.exists():
        if path.read_bytes() != data:
            raise StructuralResponseError("immutable_output_conflict", f"output differs: {path}")
        return "reused_identical"
    atomic_write_new_bytes(path, data)
    return "created"


def _figure_bytes(result: StructuralResponseResult, name: str) -> bytes:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    if name == "paired_difference_distribution":
        values = result.paired["candidate_remodeling_d"].to_numpy(dtype=np.float64)
        stride = max(1, len(values) // 250_000)
        plotted = values[::stride]
        if np.ptp(plotted) <= EPSILON:
            center = float(plotted[0])
            axis.hist(plotted, bins=1, range=(center - 0.5, center + 0.5), color="#355c7d", alpha=0.9)
        else:
            axis.hist(plotted, bins=80, color="#355c7d", alpha=0.9)
        axis.set_xlabel("D = AFDB−PDB candidate remodeling")
        axis.set_ylabel("Nested paired observations")
        axis.set_title("Candidate-specific structural response")
    elif name == "per_protein_response_summary":
        protein = result.summary.protein_summary.sort_values("median_abs_d")
        axis.scatter(protein["median_abs_c"], protein["median_abs_d"], s=18, alpha=0.75, color="#c06c84")
        axis.set_xlabel("Protein median |C|")
        axis.set_ylabel("Protein median |D|")
        axis.set_title("Global shift versus preference remodeling")
    elif name == "repeat_stability":
        repeat = result.paired.assign(abs_d=result.paired["candidate_remodeling_d"].abs())
        repeat = repeat.groupby(["protein_id", "repeat_index"], sort=False)["abs_d"].median().reset_index()
        grouped = [group["abs_d"].to_numpy(dtype=np.float64) for _, group in repeat.groupby("protein_id", sort=False)]
        axis.boxplot(grouped, showfliers=False, widths=0.7)
        axis.set_xlabel("Proteins ordered by input cohort")
        axis.set_ylabel("Repeat median |D|")
        axis.set_title("Realization robustness of remodeling magnitude")
    else:
        plt.close(figure)
        raise ValueError(f"unknown figure: {name}")
    output = io.BytesIO()
    figure.savefig(output, format="png", metadata={"Software": "Dual-UQ"}, dpi=140)
    plt.close(figure)
    return output.getvalue()


def _markdown_report(result: StructuralResponseResult) -> str:
    summary = result.summary.cohort_summary
    lines = [
        "# Structural Response Characterization",
        "",
        "Status: descriptive G/C/D characterization only; no P/M/SDFI, rank, regret, mechanism, or H1 analysis.",
        "",
        "## Dataset",
        "",
        f"- Proteins: {summary['protein_count']}",
        f"- Positions: {summary['position_count']}",
        f"- Candidate probes: {summary['candidate_count']}",
        f"- Paired candidate-repeat observations: {summary['paired_row_count']}",
        f"- Decoding realizations: {summary['repeat_count']}",
        "- Orientation: AFDB minus PDB",
        "",
        "## Quantitative observations",
        "",
        f"- WT baseline |G| protein-median distribution: median={summary['protein_g_median_abs_median']:.6g}, q90={summary['protein_g_median_abs_q90']:.6g}.",
        f"- Candidate absolute |C| protein-median distribution: median={summary['protein_abs_c_median']:.6g}, q90={summary['protein_abs_c_q90']:.6g}.",
        f"- WT-relative remodeling |D| protein-median distribution: median={summary['protein_abs_d_median']:.6g}, q90={summary['protein_abs_d_q90']:.6g}.",
        f"- Protein-level association between median |C| and median |D|: Spearman={summary['protein_abs_c_abs_d_spearman']:.6g}.",
        "",
        "## Interpretation boundary",
        "",
        "The outputs characterize measurable PDB/AFDB changes in ProteinMPNN compatibility and the component remaining after matched-WT referencing. They do not establish biological conformational uncertainty, decision changes, stochastic dominance, P/M/SDFI relationships, or H1 support.",
        "",
        "## Limitations / open questions",
        "",
        "- Probes and decoding realizations are nested within proteins; cohort summaries use protein-level aggregation.",
        "- No formal stochastic-noise comparison or decision-level analysis is included.",
        "- Continuous response magnitudes are reported without arbitrary sensitivity thresholds.",
    ]
    return "\n".join(lines) + "\n"


def materialize_structural_response(result: StructuralResponseResult, output_dir: Path) -> dict[str, Any]:
    """Write all analysis outputs from one structured result, immutably."""
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    statuses: dict[str, str] = {}
    outputs: dict[str, dict[str, Any]] = {}
    for label, frame, filename in (
        ("paired", result.paired, "paired_structural_response.parquet"),
        ("candidate_summary", result.summary.candidate_summary, "candidate_summary.parquet"),
        ("position_summary", result.summary.position_summary, "position_summary.parquet"),
        ("protein_summary", result.summary.protein_summary, "protein_summary.parquet"),
    ):
        path = output_dir / filename
        statuses[label] = _immutable_parquet(frame, path)
        outputs[label] = {"path": filename, "rows": len(frame), "sha256": sha256_file(path)}
    summary_payload = {
        "analysis_protocol": ANALYSIS_PROTOCOL_VERSION,
        "orientation": STRUCTURAL_ORIENTATION,
        "cohort_summary": _plain(result.summary.cohort_summary),
    }
    summary_bytes = (json.dumps(summary_payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    statuses["summary"] = _immutable_bytes(summary_bytes, output_dir / "summary.json")
    statuses["report"] = _immutable_bytes(_markdown_report(result).encode(), output_dir / "report.md")
    figure_dir = output_dir / "figures"
    for figure_name in ("paired_difference_distribution", "per_protein_response_summary", "repeat_stability"):
        figure_path = figure_dir / f"{figure_name}.png"
        statuses[figure_name] = _immutable_bytes(_figure_bytes(result, figure_name), figure_path)
        outputs.setdefault("figures", {})[figure_name] = {
            "path": f"figures/{figure_path.name}",
            "sha256": sha256_file(figure_path),
            "takeaway": {
                "paired_difference_distribution": "Distribution of nested candidate-specific remodeling observations.",
                "per_protein_response_summary": "Continuous protein-level separation of shared shift and remodeling.",
                "repeat_stability": "Within-protein realization variability of remodeling magnitude.",
            }[figure_name],
        }
    manifest = {
        "status": "COMPLETE",
        "schema_version": "dual-uq.structural-response-manifest.v1",
        "analysis_protocol": ANALYSIS_PROTOCOL_VERSION,
        "orientation": STRUCTURAL_ORIENTATION,
        "input_provenance": _plain(result.inputs.input_provenance),
        "outputs": outputs,
        "counts": _plain(result.summary.cohort_summary),
        "non_goals": ["P/M/SDFI", "Top-1", "rank", "regret", "mechanism", "H1"],
        "write_status": {key: "materialized" for key in statuses},
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    statuses["manifest"] = _immutable_bytes(manifest_bytes, output_dir / "manifest.json")
    return {"manifest_path": output_dir / "manifest.json", "write_status": statuses, "outputs": outputs}
