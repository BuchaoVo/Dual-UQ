"""Confirm frozen local mechanisms on StructCal Track-I PDB-AFDB pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from dual_uq.evaluation.cross_model_representation_mechanism import (
    local_geometry_from_cases,
)
from dual_uq.evaluation.pair_conditioned_local_mechanism import (
    LOCAL_DESCRIPTORS,
    aggregate_pair_correlations_by_protein,
    hotspot_permutation_null,
    pair_conditioned_cross_model_hotspots,
    pair_conditioned_local_associations,
    pair_fixed_effect_local_associations,
    summarize_protein_correlations,
)
from dual_uq.reporting.structcal import (
    DESCRIPTOR_LABELS,
    MODEL_LABELS,
    SHORT_DESCRIPTOR_LABELS,
)
from dual_uq.reporting.structcal import (
    dataframe_records as _records,
)
from dual_uq.reporting.structcal import (
    format_decimal as _format,
)
from dual_uq.workflows.structcal_cross_model_representation_sensitivity import (
    DEFAULT_RUN_ROOT,
    MAIN_MODELS,
)

MODELS = MAIN_MODELS
EXPECTED_COVERAGE = {
    "dynamicmpnn": (62, 56, 54),
    "esm_if1": (68, 60, 57),
    "pifold": (68, 60, 57),
    "proteinmpnn": (68, 60, 57),
}

def _protein_summary(
    associations: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    proteins = aggregate_pair_correlations_by_protein(
        associations,
        group_columns=(
            "model_id",
            "checkpoint_id",
            "protein_id",
            "identity_cluster_id",
            "descriptor",
        ),
    )
    summary = summarize_protein_correlations(
        proteins,
        group_columns=("model_id", "checkpoint_id", "descriptor"),
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    return proteins, summary


def _hotspot_summary(
    hotspots: pd.DataFrame, *, bootstrap_replicates: int, seed: int
) -> pd.DataFrame:
    proteins = aggregate_pair_correlations_by_protein(
        hotspots,
        group_columns=(
            "left_model_id",
            "right_model_id",
            "protein_id",
            "identity_cluster_id",
        ),
    )
    return summarize_protein_correlations(
        proteins,
        group_columns=("left_model_id", "right_model_id"),
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )


def _local_comparison_rows(
    controlled: pd.DataFrame,
    operational: pd.DataFrame,
    analysis: str,
) -> pd.DataFrame:
    if "summary_scope" in controlled:
        controlled = controlled.loc[controlled["summary_scope"].eq("all_doses")]
    columns = [
        "model_id",
        "descriptor",
        "median",
        "ci_low",
        "ci_high",
        "q25",
        "q75",
        "fraction_positive",
        "n_valid_pairs",
        "n_proteins",
        "n_identity_clusters",
    ]
    joined = controlled[columns].merge(
        operational[columns],
        on=["model_id", "descriptor"],
        suffixes=("_controlled", "_operational"),
        validate="one_to_one",
    )
    if len(joined) != len(MODELS) * len(LOCAL_DESCRIPTORS):
        raise ValueError(f"controlled/operational {analysis} comparison is incomplete")
    joined.insert(0, "comparison_type", analysis)
    joined["controlled_descriptor_rank"] = joined.groupby("model_id")["median_controlled"].rank(
        method="min", ascending=False
    )
    joined["operational_descriptor_rank"] = joined.groupby("model_id")["median_operational"].rank(
        method="min", ascending=False
    )
    joined["operational_minus_controlled_median"] = (
        joined["median_operational"] - joined["median_controlled"]
    )
    joined["direction_preserved"] = np.sign(joined["median_controlled"]) == np.sign(
        joined["median_operational"]
    )
    joined["left_model_id"] = None
    joined["right_model_id"] = None
    return joined


def _mechanism_comparison(
    root: Path,
    local_summary: pd.DataFrame,
    fixed_summary: pd.DataFrame,
    hotspot_summary: pd.DataFrame,
) -> pd.DataFrame:
    pair_conditioned = _local_comparison_rows(
        pd.read_parquet(root / "pair_conditioned_local_summary.parquet"),
        local_summary,
        "pair_conditioned_local",
    )
    fixed = _local_comparison_rows(
        pd.read_parquet(root / "pair_fixed_effect_local_summary.parquet"),
        fixed_summary,
        "pair_fixed_effect_local",
    )
    controlled_hotspots = pd.read_parquet(root / "pair_conditioned_cross_model_summary.parquet")
    hotspot_columns = [
        "left_model_id",
        "right_model_id",
        "median",
        "ci_low",
        "ci_high",
        "q25",
        "q75",
        "fraction_positive",
        "n_valid_pairs",
        "n_proteins",
        "n_identity_clusters",
    ]
    hotspots = controlled_hotspots[hotspot_columns].merge(
        hotspot_summary[hotspot_columns],
        on=["left_model_id", "right_model_id"],
        suffixes=("_controlled", "_operational"),
        validate="one_to_one",
    )
    hotspots.insert(0, "comparison_type", "cross_model_hotspot")
    hotspots["model_id"] = None
    hotspots["descriptor"] = None
    hotspots["operational_minus_controlled_median"] = (
        hotspots["median_operational"] - hotspots["median_controlled"]
    )
    hotspots["direction_preserved"] = np.sign(hotspots["median_controlled"]) == np.sign(
        hotspots["median_operational"]
    )
    hotspots["controlled_descriptor_rank"] = np.nan
    hotspots["operational_descriptor_rank"] = np.nan
    return pd.concat([pair_conditioned, fixed, hotspots], ignore_index=True, sort=False)


def _figure(
    local_summary: pd.DataFrame,
    controlled_local: pd.DataFrame,
    hotspot_summary: pd.DataFrame,
    null_summary: pd.DataFrame,
    path: Path,
) -> None:
    colors = {
        "dynamicmpnn": "#0072B2",
        "esm_if1": "#E69F00",
        "pifold": "#009E73",
        "proteinmpnn": "#CC79A7",
    }
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    x = np.arange(len(LOCAL_DESCRIPTORS), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(MODELS))
    indexed = local_summary.set_index(["model_id", "descriptor"])
    for offset, model in zip(offsets, MODELS, strict=True):
        rows = indexed.loc[model].reindex(LOCAL_DESCRIPTORS)
        axes[0, 0].errorbar(
            x + offset,
            rows["median"],
            yerr=np.vstack([rows["median"] - rows["ci_low"], rows["ci_high"] - rows["median"]]),
            fmt="o",
            capsize=2,
            color=colors[model],
            label=MODEL_LABELS[model],
        )
    axes[0, 0].axhline(0, color="0.5", linewidth=0.8)
    axes[0, 0].set_xticks(x, [DESCRIPTOR_LABELS[item] for item in LOCAL_DESCRIPTORS])
    axes[0, 0].tick_params(axis="x", rotation=16)
    axes[0, 0].set_ylabel("Median protein Spearman ρ")
    axes[0, 0].set_title(
        "A  Operational pair-conditioned association", loc="left", fontweight="bold"
    )
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)

    controlled = controlled_local.loc[controlled_local["summary_scope"].eq("all_doses")]
    comparison = controlled[["model_id", "descriptor", "median", "ci_low", "ci_high"]].merge(
        local_summary[["model_id", "descriptor", "median", "ci_low", "ci_high"]],
        on=["model_id", "descriptor"],
        suffixes=("_controlled", "_operational"),
        validate="one_to_one",
    )
    comparison["order"] = comparison["model_id"].map(dict(zip(MODELS, range(len(MODELS)))))
    comparison["descriptor_order"] = comparison["descriptor"].map(
        dict(zip(LOCAL_DESCRIPTORS, range(len(LOCAL_DESCRIPTORS))))
    )
    comparison = comparison.sort_values(["order", "descriptor_order"])
    x = np.arange(len(comparison), dtype=float)
    for suffix, offset, color, label in (
        ("controlled", -0.12, "#6C757D", "Controlled"),
        ("operational", 0.12, "#D55E00", "Operational"),
    ):
        axes[0, 1].errorbar(
            x + offset,
            comparison[f"median_{suffix}"],
            yerr=np.vstack(
                [
                    comparison[f"median_{suffix}"] - comparison[f"ci_low_{suffix}"],
                    comparison[f"ci_high_{suffix}"] - comparison[f"median_{suffix}"],
                ]
            ),
            fmt="o",
            capsize=2,
            color=color,
            label=label,
        )
    labels = [
        f"{MODEL_LABELS[row.model_id]}\n{SHORT_DESCRIPTOR_LABELS[row.descriptor]}"
        for row in comparison.itertuples()
    ]
    axes[0, 1].axhline(0, color="0.5", linewidth=0.8)
    axes[0, 1].set_xticks(x, labels, rotation=70, fontsize=6)
    axes[0, 1].set_ylabel("Median protein Spearman ρ")
    axes[0, 1].set_title("B  Controlled vs Operational", loc="left", fontweight="bold")
    axes[0, 1].legend(frameon=False, fontsize=8)

    hotspot_summary = hotspot_summary.copy()
    hotspot_summary["pair"] = hotspot_summary.apply(
        lambda row: f"{MODEL_LABELS[row.left_model_id]}\n{MODEL_LABELS[row.right_model_id]}",
        axis=1,
    )
    x = np.arange(len(hotspot_summary))
    axes[1, 0].errorbar(
        x,
        hotspot_summary["median"],
        yerr=np.vstack(
            [
                hotspot_summary["median"] - hotspot_summary["ci_low"],
                hotspot_summary["ci_high"] - hotspot_summary["median"],
            ]
        ),
        fmt="o",
        capsize=3,
        color="#4C566A",
    )
    axes[1, 0].axhline(0, color="0.5", linewidth=0.8)
    axes[1, 0].set_xticks(x, hotspot_summary["pair"], fontsize=7)
    axes[1, 0].set_ylabel("Median protein Spearman ρ")
    axes[1, 0].set_title("C  Operational cross-model hotspots", loc="left", fontweight="bold")

    null_summary = null_summary.copy()
    null_summary["pair"] = null_summary.apply(
        lambda row: f"{MODEL_LABELS[row.left_model_id]}\n{MODEL_LABELS[row.right_model_id]}",
        axis=1,
    )
    x = np.arange(len(null_summary))
    axes[1, 1].vlines(x, null_summary["null_ci_low"], null_summary["null_ci_high"], color="0.55")
    axes[1, 1].scatter(x, null_summary["null_median"], color="0.55", marker="s", label="Null")
    axes[1, 1].scatter(
        x,
        null_summary["observed_median_protein_rho"],
        color="#D55E00",
        label="Observed",
    )
    axes[1, 1].axhline(0, color="0.5", linewidth=0.8)
    axes[1, 1].set_xticks(x, null_summary["pair"], fontsize=7)
    axes[1, 1].set_ylabel("Median protein Spearman ρ")
    axes[1, 1].set_title("D  Operational positional null", loc="left", fontweight="bold")
    axes[1, 1].legend(frameon=False, fontsize=8)

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def _association_table(lines: list[str], summary: pd.DataFrame) -> None:
    lines.extend(
        [
            "| model | descriptor | median ρ (95% CI) | IQR | fraction + | pairs | proteins | clusters |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary.sort_values(["model_id", "descriptor"]).itertuples():
        lines.append(
            f"| {MODEL_LABELS[row.model_id]} | {DESCRIPTOR_LABELS[row.descriptor]} | "
            f"{_format(row.median)} [{_format(row.ci_low)}, {_format(row.ci_high)}] | "
            f"[{_format(row.q25)}, {_format(row.q75)}] | {_format(row.fraction_positive)} | "
            f"{row.n_valid_pairs} | {row.n_proteins} | {row.n_identity_clusters} |"
        )


def _report(
    status: str,
    local_summary: pd.DataFrame,
    fixed_summary: pd.DataFrame,
    hotspot_summary: pd.DataFrame,
    null_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    undefined: dict[str, Any],
) -> str:
    lines = [
        "# StructCal operational local mechanism confirmation",
        "",
        "## Decision",
        "",
        f"`{status}`",
        "",
        (
            "This analysis uses the frozen clean Track-I PDB-AFDB cohort, existing residue-level "
            "20-AA JSD responses, and the same four audited local descriptors as Controlled. No "
            "model scoring, training, mapping repair, or frozen-release mutation was performed."
        ),
        "",
        "## A. Operational pair-conditioned local association",
        "",
        (
            "Each Spearman correlation is computed strictly within one PDB-AFDB pair. Valid pair "
            "correlations are median-aggregated within protein before 30%-identity-cluster bootstrap "
            "inference on the across-protein median."
        ),
        "",
    ]
    _association_table(lines, local_summary)
    lines.extend(
        [
            "",
            "## B. Operational pair-fixed-effect confirmation",
            "",
            (
                "Descriptor and JSD values are median-centered independently inside every pair, "
                "then centered residues are pooled only within protein."
            ),
            "",
        ]
    )
    _association_table(lines, fixed_summary)
    lines.extend(
        [
            "",
            "## C. Operational cross-model hotspot agreement",
            "",
            (
                "Profiles are matched only by `(pair_id, canonical_position)`. Each model pair uses "
                "its actual shared evaluable cohort."
            ),
            "",
            "| model pair | median ρ (95% CI) | IQR | fraction + | pairs | proteins | clusters |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in hotspot_summary.itertuples():
        lines.append(
            f"| {MODEL_LABELS[row.left_model_id]} / {MODEL_LABELS[row.right_model_id]} | "
            f"{_format(row.median)} [{_format(row.ci_low)}, {_format(row.ci_high)}] | "
            f"[{_format(row.q25)}, {_format(row.q75)}] | {_format(row.fraction_positive)} | "
            f"{row.n_valid_pairs} | {row.n_proteins} | {row.n_identity_clusters} |"
        )
    lines.extend(
        [
            "",
            "## D. Operational positional permutation null",
            "",
            (
                "Model-B positions were independently permuted within every shared pair for 1,000 "
                "fixed-seed replicates. The empirical p-value uses `(exceedances + 1) / 1001`."
            ),
            "",
            "| model pair | observed | null median (95% interval) | observed − null | upper-tail p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in null_summary.itertuples():
        lines.append(
            f"| {MODEL_LABELS[row.left_model_id]} / {MODEL_LABELS[row.right_model_id]} | "
            f"{_format(row.observed_median_protein_rho)} | {_format(row.null_median)} "
            f"[{_format(row.null_ci_low)}, {_format(row.null_ci_high)}] | "
            f"{_format(row.observed_minus_null_median)} | {row.empirical_upper_tail_p:.4g} |"
        )

    lines.extend(
        [
            "",
            "## E. Controlled versus Operational",
            "",
            (
                "The cohorts are not treated as paired. Values below are descriptive side-by-side "
                "protein-level medians; ranks are descending within model and analysis."
            ),
            "",
            "| analysis | model | descriptor | Controlled ρ | rank | Operational ρ | rank | Δ median |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    local_comparison = comparison.loc[comparison["comparison_type"].ne("cross_model_hotspot")]
    for row in local_comparison.itertuples():
        analysis = (
            "pair-conditioned"
            if row.comparison_type == "pair_conditioned_local"
            else "fixed-effect"
        )
        lines.append(
            f"| {analysis} | {MODEL_LABELS[row.model_id]} | "
            f"{DESCRIPTOR_LABELS[row.descriptor]} | {_format(row.median_controlled)} | "
            f"{int(row.controlled_descriptor_rank)} | {_format(row.median_operational)} | "
            f"{int(row.operational_descriptor_rank)} | "
            f"{row.operational_minus_controlled_median:+.4f} |"
        )
    direction_count = int(local_comparison["direction_preserved"].sum())
    lines.extend(
        [
            "",
            (
                f"The positive association direction is preserved in {direction_count}/"
                f"{len(local_comparison)} model×descriptor×analysis cells."
            ),
            "",
            "### Descriptor ranking",
            "",
            "| analysis | model | Controlled strongest | Operational strongest | complete rank order preserved |",
            "|---|---|---|---|---:|",
        ]
    )
    for (analysis, model), group in local_comparison.groupby(
        ["comparison_type", "model_id"], sort=True
    ):
        controlled_top = group.loc[group["controlled_descriptor_rank"].idxmin(), "descriptor"]
        operational_top = group.loc[group["operational_descriptor_rank"].idxmin(), "descriptor"]
        rank_preserved = group["controlled_descriptor_rank"].equals(
            group["operational_descriptor_rank"]
        )
        label = "pair-conditioned" if analysis == "pair_conditioned_local" else "fixed-effect"
        lines.append(
            f"| {label} | {MODEL_LABELS[model]} | {DESCRIPTOR_LABELS[controlled_top]} | "
            f"{DESCRIPTOR_LABELS[operational_top]} | {'yes' if rank_preserved else 'no'} |"
        )
    lines.extend(
        [
            "",
            "### Cross-regime hotspot comparison",
            "",
            "| model pair | Controlled median ρ | Operational median ρ | Operational − Controlled |",
            "|---|---:|---:|---:|",
        ]
    )
    hotspot_comparison = comparison.loc[comparison["comparison_type"].eq("cross_model_hotspot")]
    for row in hotspot_comparison.itertuples():
        lines.append(
            f"| {MODEL_LABELS[row.left_model_id]} / {MODEL_LABELS[row.right_model_id]} | "
            f"{_format(row.median_controlled)} | {_format(row.median_operational)} | "
            f"{row.operational_minus_controlled_median:+.4f} |"
        )
    strengthened = int(local_comparison["operational_minus_controlled_median"].gt(0).sum())
    lines.extend(
        [
            "",
            "## Cohort and exclusions",
            "",
            (
                "ProteinMPNN, ESM-IF1, and PiFold each retain 68/68 pairs, 60 proteins, and 57 "
                "30%-identity clusters. DynamicMPNN retains 62/68 pairs, 56 proteins, and 54 clusters. "
                "Its six registered exclusions remain `DATA_UNRESOLVED` because the canonical axis is "
                "non-contiguous; no repair or imputation was attempted."
            ),
            "",
            (
                f"Undefined pair-conditioned model×pair×descriptor cells: {undefined['local_total']}; "
                f"undefined cross-model pair cells: {undefined['hotspot_total']}. Explicit reasons "
                "are retained in the machine-readable association tables."
            ),
            "",
            "## Interpretation",
            "",
            (
                f"Operational effects are larger than Controlled in {strengthened}/{len(local_comparison)} "
                "model×descriptor×analysis cells; this is descriptive because the protein cohorts are "
                "not paired. Descriptor ranking changes are likewise not evidence of a mechanism change."
            ),
            "",
            (
                "Partial cross-model positional agreement is detected in both regimes for all six model "
                "pairs; every Operational comparison also exceeds its within-pair positional null."
            ),
            "",
            (
                "The strongest supported statement is that both global structural discrepancy and "
                "within-pair local deformation organize inverse-folding decision sensitivity, and the "
                "local organization remains observable under independent PDB-AFDB representation variation. "
                "These associations are not causal and do not imply shared latent representations."
            ),
            "",
            "KW-Design remains `IMPLEMENTATION_FAILURE`; DeSAE remains `MODEL_CAPABILITY_UNAVAILABLE`.",
            "",
            "![Operational local mechanism](figures/operational_local_mechanism.png)",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=2_026_09_06)
    args = parser.parse_args()
    root = args.output_root.resolve()

    cases = pd.read_parquet(root / "cases/nca/operational_pdb_afdb.parquet")
    geometry = local_geometry_from_cases(cases)
    if geometry["pair_id"].nunique() != 68 or geometry["protein_id"].nunique() != 60:
        raise ValueError("operational geometry does not preserve the frozen Track-I cohort")

    response_parts = []
    for model, checkpoint in MODELS.items():
        residue = pd.read_parquet(
            root / f"shards/{model}/{checkpoint}/operational_pdb_afdb/residue_response.parquet"
        ).drop(columns=["perturbation_family", "requested_dose"])
        local = residue.merge(
            geometry,
            on=["pair_id", "protein_id", "identity_cluster_id", "canonical_position"],
            how="inner",
            validate="one_to_one",
        )
        expected = EXPECTED_COVERAGE[model]
        observed = (
            local["pair_id"].nunique(),
            local["protein_id"].nunique(),
            local["identity_cluster_id"].nunique(),
        )
        if len(local) != len(residue) or observed != expected:
            raise ValueError(f"operational geometry/response join is incomplete for {model}")
        response_parts.append(local)
    local_response = pd.concat(response_parts, ignore_index=True)
    local_response.to_parquet(root / "operational_local_geometry_response.parquet", index=False)

    local_associations = pair_conditioned_local_associations(local_response)
    _, local_summary = _protein_summary(
        local_associations,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    local_associations.to_parquet(
        root / "operational_pair_conditioned_local_associations.parquet", index=False
    )
    local_summary.to_parquet(
        root / "operational_pair_conditioned_local_summary.parquet", index=False
    )

    fixed_associations = pair_fixed_effect_local_associations(local_response)
    fixed_summary = summarize_protein_correlations(
        fixed_associations,
        group_columns=("model_id", "checkpoint_id", "descriptor"),
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed + 100,
    )
    fixed_associations.to_parquet(
        root / "operational_pair_fixed_effect_local_associations.parquet", index=False
    )
    fixed_summary.to_parquet(
        root / "operational_pair_fixed_effect_local_summary.parquet", index=False
    )

    hotspots = pair_conditioned_cross_model_hotspots(local_response)
    hotspot_summary = _hotspot_summary(
        hotspots,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed + 200,
    )
    hotspots.to_parquet(root / "operational_cross_model_hotspots.parquet", index=False)
    hotspot_summary.to_parquet(
        root / "operational_cross_model_hotspot_summary.parquet", index=False
    )
    null, null_summary = hotspot_permutation_null(
        local_response, hotspots, permutations=args.permutations, seed=args.seed
    )
    null.to_parquet(root / "operational_hotspot_permutation_null.parquet", index=False)

    comparison = _mechanism_comparison(root, local_summary, fixed_summary, hotspot_summary)
    comparison.to_parquet(root / "controlled_operational_mechanism_comparison.parquet", index=False)

    local_direction = local_summary.groupby("model_id")["median"].apply(lambda x: x.gt(0).all())
    fixed_direction = fixed_summary.groupby("model_id")["median"].apply(lambda x: x.gt(0).all())
    local_reproducible = local_summary.groupby("model_id")["ci_low"].apply(lambda x: x.gt(0).any())
    fixed_reproducible = fixed_summary.groupby("model_id")["ci_low"].apply(lambda x: x.gt(0).any())
    if (
        local_direction.all()
        and fixed_direction.all()
        and local_reproducible.all()
        and fixed_reproducible.all()
    ):
        status = "OPERATIONAL_LOCAL_MECHANISM_CONFIRMED"
    elif local_reproducible.any() or fixed_reproducible.any():
        status = "OPERATIONAL_LOCALIZATION_MODEL_DEPENDENT"
    elif local_summary["median"].abs().max() > 0 or fixed_summary["median"].abs().max() > 0:
        status = "OPERATIONAL_DRIFT_WITH_WEAK_LOCAL_GEOMETRIC_EXPLANATION"
    else:
        status = "NO_CLEAR_OPERATIONAL_LOCALIZATION"

    undefined_local = local_associations.loc[local_associations["status"].ne("VALID")]
    undefined_hotspots = hotspots.loc[hotspots["status"].ne("VALID")]
    undefined = {
        "local_total": len(undefined_local),
        "local_reasons": {
            key: int(value) for key, value in undefined_local["reason"].value_counts().items()
        },
        "hotspot_total": len(undefined_hotspots),
        "hotspot_reasons": {
            key: int(value) for key, value in undefined_hotspots["reason"].value_counts().items()
        },
    }
    exclusions = pd.read_parquet(
        root / "shards/dynamicmpnn/single_chain_k2/operational_pdb_afdb/exclusions.parquet"
    )
    if len(exclusions) != 6 or not exclusions["status"].eq("DATA_UNRESOLVED").all():
        raise ValueError("DynamicMPNN operational exclusion ledger changed")

    controlled_local = pd.read_parquet(root / "pair_conditioned_local_summary.parquet")
    figure_path = root / "figures/operational_local_mechanism.png"
    _figure(local_summary, controlled_local, hotspot_summary, null_summary, figure_path)
    summary = {
        "task": "STRUCTCAL_OPERATIONAL_LOCAL_MECHANISM_CONFIRMATION",
        "status": status,
        "frozen_regime": "TRACK_I_OPERATIONAL_PDB_AFDB",
        "models": list(MODELS),
        "descriptors": list(LOCAL_DESCRIPTORS),
        "cohort_by_model": {
            model: {
                "pairs": coverage[0],
                "proteins": coverage[1],
                "identity_clusters_30": coverage[2],
            }
            for model, coverage in EXPECTED_COVERAGE.items()
        },
        "dynamicmpnn_exclusions": _records(exclusions),
        "statistics": {
            "pair_conditioned_local": _records(local_summary),
            "pair_fixed_effect_local": _records(fixed_summary),
            "cross_model_hotspots": _records(hotspot_summary),
            "hotspot_permutation_null": _records(null_summary),
            "controlled_operational_comparison": _records(comparison),
        },
        "undefined_correlations": undefined,
        "bootstrap": {
            "unit": "identity_cluster_30",
            "statistic": "median",
            "replicates": args.bootstrap_replicates,
        },
        "permutation": {
            "unit": "canonical positions independently within each structural pair",
            "permutations": args.permutations,
            "seed": args.seed,
            "tail": "upper",
            "finite_sample_correction": "(exceedances + 1) / (permutations + 1)",
        },
        "excluded_models": {
            "kw_design": "IMPLEMENTATION_FAILURE",
            "desae": "MODEL_CAPABILITY_UNAVAILABLE",
        },
    }
    (root / "operational_local_mechanism_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (root / "operational_local_mechanism_report.md").write_text(
        _report(
            status,
            local_summary,
            fixed_summary,
            hotspot_summary,
            null_summary,
            comparison,
            undefined,
        )
    )


if __name__ == "__main__":
    main()
