"""Confirm StructCal local mechanisms without pooling distinct structural pairs."""

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
)
from dual_uq.reporting.structcal import (
    dataframe_records as _records,
)
from dual_uq.reporting.structcal import (
    format_decimal as _format,
)
from dual_uq.workflows.structcal_cross_model_representation_sensitivity import (
    DEFAULT_RUN_ROOT,
    PRIMARY_MODELS,
)

EXPECTED_MODELS = PRIMARY_MODELS


def _summarize_pair_associations(
    associations: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    protein_groups = (
        "model_id",
        "checkpoint_id",
        "protein_id",
        "identity_cluster_id",
        "descriptor",
    )
    proteins = aggregate_pair_correlations_by_protein(associations, group_columns=protein_groups)
    summary = summarize_protein_correlations(
        proteins,
        group_columns=("model_id", "checkpoint_id", "descriptor"),
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    summary.insert(0, "summary_scope", "all_doses")
    summary.insert(4, "requested_dose", np.nan)

    dose_proteins = aggregate_pair_correlations_by_protein(
        associations,
        group_columns=(
            "model_id",
            "checkpoint_id",
            "protein_id",
            "identity_cluster_id",
            "descriptor",
            "requested_dose",
        ),
    )
    dose_summary = summarize_protein_correlations(
        dose_proteins,
        group_columns=(
            "model_id",
            "checkpoint_id",
            "descriptor",
            "requested_dose",
        ),
        bootstrap_replicates=bootstrap_replicates,
        seed=seed + 100,
    )
    dose_summary.insert(0, "summary_scope", "dose_specific")
    return proteins, pd.concat([summary, dose_summary], ignore_index=True)


def _cross_model_summary(
    hotspots: pd.DataFrame, *, bootstrap_replicates: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    proteins = aggregate_pair_correlations_by_protein(
        hotspots,
        group_columns=(
            "left_model_id",
            "right_model_id",
            "protein_id",
            "identity_cluster_id",
        ),
    )
    summary = summarize_protein_correlations(
        proteins,
        group_columns=("left_model_id", "right_model_id"),
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    return proteins, summary


def _figure(
    local_summary: pd.DataFrame,
    fixed_summary: pd.DataFrame,
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
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)

    def descriptor_panel(axis: Any, data: pd.DataFrame, title: str) -> None:
        selected = (
            data.loc[data["summary_scope"].eq("all_doses")] if "summary_scope" in data else data
        )
        x = np.arange(len(LOCAL_DESCRIPTORS), dtype=float)
        offsets = np.linspace(-0.24, 0.24, len(EXPECTED_MODELS))
        for offset, model in zip(offsets, EXPECTED_MODELS, strict=True):
            model_rows = selected.set_index(["model_id", "descriptor"]).loc[model]
            model_rows = model_rows.reindex(LOCAL_DESCRIPTORS)
            axis.errorbar(
                x + offset,
                model_rows["median"],
                yerr=np.vstack(
                    [
                        model_rows["median"] - model_rows["ci_low"],
                        model_rows["ci_high"] - model_rows["median"],
                    ]
                ),
                fmt="o",
                capsize=2,
                color=colors[model],
                label=MODEL_LABELS[model],
            )
        axis.axhline(0, color="0.5", linewidth=0.8)
        axis.set_xticks(x, [DESCRIPTOR_LABELS[item] for item in LOCAL_DESCRIPTORS])
        axis.tick_params(axis="x", rotation=18)
        axis.set_ylabel("Median protein Spearman ρ")
        axis.set_title(title, loc="left", fontweight="bold")

    descriptor_panel(axes[0, 0], local_summary, "A  Pair-conditioned local association")
    descriptor_panel(axes[0, 1], fixed_summary, "B  Pair fixed-effect association")
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)

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
    axes[1, 0].set_title("C  Pair-conditioned cross-model hotspots", loc="left", fontweight="bold")

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
        marker="o",
        label="Observed",
    )
    axes[1, 1].axhline(0, color="0.5", linewidth=0.8)
    axes[1, 1].set_xticks(x, null_summary["pair"], fontsize=7)
    axes[1, 1].set_ylabel("Median protein Spearman ρ")
    axes[1, 1].set_title("D  Pairwise positional permutation null", loc="left", fontweight="bold")
    axes[1, 1].legend(frameon=False, fontsize=8)

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def _report(
    local_summary: pd.DataFrame,
    fixed_summary: pd.DataFrame,
    hotspot_summary: pd.DataFrame,
    null_summary: pd.DataFrame,
    global_summary: pd.DataFrame,
    undefined: dict[str, Any],
    mechanism_status: str,
    shared_hotspots: bool,
) -> str:
    local_primary = local_summary.loc[local_summary["summary_scope"].eq("all_doses")]
    lines = [
        "# StructCal pair-conditioned local mechanism confirmation",
        "",
        (
            "This confirmation uses the frozen residue logits and four frozen geometry descriptors. "
            "No model scoring, training, cohort change, or StructCal v1 mutation was performed."
        ),
        "",
        "## Decision",
        "",
        f"`{mechanism_status}`",
        "",
        (
            "Pair-conditioned and pair-median-centered associations are summarized first within "
            "protein and then across proteins. Shared hotspots are evaluated within each structural "
            "pair and exceed a pairwise positional permutation null."
            if shared_hotspots
            else "Pair-conditioned evidence is not uniformly beyond the positional null."
        ),
        "",
        "## A. Pair-conditioned local association",
        "",
        (
            "Spearman correlations were computed separately within every model/protein/pair. "
            "Valid pair correlations were median-aggregated within protein; the table reports the "
            "across-protein median with a 30%-identity-cluster bootstrap 95% CI, protein IQR, "
            "fraction positive, and valid counts."
        ),
        "",
        "| model | descriptor | median ρ (95% CI) | IQR | fraction + | valid pairs | proteins |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in local_primary.sort_values(["model_id", "descriptor"]).itertuples():
        lines.append(
            f"| {MODEL_LABELS[row.model_id]} | {DESCRIPTOR_LABELS[row.descriptor]} | "
            f"{_format(row.median)} [{_format(row.ci_low)}, {_format(row.ci_high)}] | "
            f"[{_format(row.q25)}, {_format(row.q75)}] | {_format(row.fraction_positive)} | "
            f"{row.n_valid_pairs} | {row.n_proteins} |"
        )
    lines.extend(
        [
            "",
            "### Dose-specific check",
            "",
            "The dose strata are summarized independently; missing high-dose proteins are not imputed or paired.",
            "",
            "| dose (Å) | proteins | valid pairs | minimum median ρ | maximum median ρ | cells with CI > 0 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    dose = local_summary.loc[local_summary["summary_scope"].eq("dose_specific")]
    for requested_dose, group in dose.groupby("requested_dose", sort=True):
        lines.append(
            f"| {requested_dose:.2f} | {int(group['n_proteins'].min())} | "
            f"{int(group['n_valid_pairs'].min())} | {_format(group['median'].min())} | "
            f"{_format(group['median'].max())} | {int(group['ci_low'].gt(0).sum())}/{len(group)} |"
        )
    lines.extend(
        [
            "",
            "## B. Pair fixed-effect residual association",
            "",
            (
                "Descriptor and JSD values were median-centered separately inside each pair before "
                "residues were pooled within protein. This removes pair-level severity offsets while "
                "retaining within-pair positional ordering."
            ),
            "",
            "| model | descriptor | median ρ (95% CI) | IQR | fraction + | valid pairs | proteins |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in fixed_summary.sort_values(["model_id", "descriptor"]).itertuples():
        lines.append(
            f"| {MODEL_LABELS[row.model_id]} | {DESCRIPTOR_LABELS[row.descriptor]} | "
            f"{_format(row.median)} [{_format(row.ci_low)}, {_format(row.ci_high)}] | "
            f"[{_format(row.q25)}, {_format(row.q75)}] | {_format(row.fraction_positive)} | "
            f"{row.n_valid_pairs} | {row.n_proteins} |"
        )
    lines.extend(
        [
            "",
            "## C. Pair-conditioned cross-model hotspots",
            "",
            (
                "Profiles were joined only on `(pair_id, canonical_position)` and correlated within "
                "each shared pair before pair medians and protein-level inference."
            ),
            "",
            "| model pair | median ρ (95% CI) | IQR | fraction + | valid pairs | proteins |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in hotspot_summary.itertuples():
        lines.append(
            f"| {MODEL_LABELS[row.left_model_id]} / {MODEL_LABELS[row.right_model_id]} | "
            f"{_format(row.median)} [{_format(row.ci_low)}, {_format(row.ci_high)}] | "
            f"[{_format(row.q25)}, {_format(row.q75)}] | {_format(row.fraction_positive)} | "
            f"{row.n_valid_pairs} | {row.n_proteins} |"
        )
    lines.extend(
        [
            "",
            "## D. Pairwise positional permutation null",
            "",
            (
                "For 1,000 fixed-seed replicates, model-B rank values were independently permuted "
                "within every pair. This preserves pair/protein membership, sequence length, and each "
                "profile's value distribution while destroying positional correspondence. The p-value "
                "is the finite-sample corrected upper-tail probability."
            ),
            "",
            "| model pair | observed | null median (95% interval) | observed − null | empirical p |",
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
            "## Eligibility and undefined correlations",
            "",
            (
                f"All 567 frozen controlled pairs and 98 proteins per model were eligible. "
                f"Undefined pair-conditioned local cells: {undefined['local_total']}; undefined "
                f"cross-model pair cells: {undefined['hotspot_total']}. No result-driven length or "
                "correlation threshold was applied."
            ),
            "",
            "## Interpretation boundaries",
            "",
            (
                "The result supports spatially localized representation sensitivity after conditioning "
                "on structural pair and after removing pair-level median severity. Positive cross-model "
                "agreement beyond the positional null supports shared hotspot localization. These are "
                "associations, not evidence that a descriptor is causal or that the four architectures "
                "share one mechanism."
            ),
            "",
            (
                "The earlier aligned-Cα-RMSD analysis remains a separate global pair-discrepancy "
                f"analysis ({len(global_summary)} model rows); it is not treated as a fifth local descriptor."
            ),
            "",
            "KW-Design remains `IMPLEMENTATION_FAILURE`; DeSAE remains `MODEL_CAPABILITY_UNAVAILABLE`.",
            "",
            "![Pair-conditioned local mechanism confirmation](figures/pair_conditioned_local_mechanism.png)",
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
    parser.add_argument("--seed", type=int, default=2_026_09_05)
    args = parser.parse_args()
    root = args.output_root.resolve()

    local_response = pd.read_parquet(root / "local_geometry_response.parquet")
    if tuple(sorted(local_response["model_id"].unique())) != EXPECTED_MODELS:
        raise ValueError("local response does not contain the four accepted official models")
    coverage = local_response.groupby("model_id").agg(
        pairs=("pair_id", "nunique"), proteins=("protein_id", "nunique")
    )
    if not coverage["pairs"].eq(567).all() or not coverage["proteins"].eq(98).all():
        raise ValueError("local response does not preserve the frozen 567-pair/98-protein cohort")

    local_associations = pair_conditioned_local_associations(local_response)
    _, local_summary = _summarize_pair_associations(
        local_associations,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    local_associations.to_parquet(root / "pair_conditioned_local_associations.parquet", index=False)
    local_summary.to_parquet(root / "pair_conditioned_local_summary.parquet", index=False)

    fixed_associations = pair_fixed_effect_local_associations(local_response)
    fixed_summary = summarize_protein_correlations(
        fixed_associations,
        group_columns=("model_id", "checkpoint_id", "descriptor"),
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed + 200,
    )
    fixed_associations.to_parquet(
        root / "pair_fixed_effect_local_associations.parquet", index=False
    )
    fixed_summary.to_parquet(root / "pair_fixed_effect_local_summary.parquet", index=False)

    hotspots = pair_conditioned_cross_model_hotspots(local_response)
    _, hotspot_summary = _cross_model_summary(
        hotspots,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed + 300,
    )
    hotspots.to_parquet(root / "pair_conditioned_cross_model_hotspots.parquet", index=False)
    hotspot_summary.to_parquet(root / "pair_conditioned_cross_model_summary.parquet", index=False)

    null, null_summary = hotspot_permutation_null(
        local_response, hotspots, permutations=args.permutations, seed=args.seed
    )
    null.to_parquet(root / "hotspot_permutation_null.parquet", index=False)

    local_primary = local_summary.loc[local_summary["summary_scope"].eq("all_doses")]
    local_all_positive = bool(local_primary["ci_low"].gt(0).all())
    fixed_all_positive = bool(fixed_summary["ci_low"].gt(0).all())
    if local_all_positive and fixed_all_positive:
        mechanism_status = "LOCAL_MECHANISM_CONFIRMED"
    elif local_primary["median"].gt(0).any() or fixed_summary["median"].gt(0).any():
        mechanism_status = "LOCALIZATION_SUPPORTED_BUT_MODEL_DESCRIPTOR_DEPENDENT"
    else:
        mechanism_status = "SEVERITY_DOMINATED_ASSOCIATION"
    shared_hotspots = bool(
        hotspot_summary["ci_low"].gt(0).all()
        and null_summary["observed_minus_null_median"].gt(0).all()
        and null_summary["empirical_upper_tail_p"].le(0.05).all()
    )

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
    global_summary = pd.read_parquet(root / "geometry_response_association_statistics.parquet")
    global_summary = global_summary.loc[global_summary["descriptor"].eq("aligned_ca_rmsd")]

    figure_path = root / "figures/pair_conditioned_local_mechanism.png"
    _figure(local_summary, fixed_summary, hotspot_summary, null_summary, figure_path)
    summary = {
        "task": "STRUCTCAL_PAIR_CONDITIONED_LOCAL_MECHANISM_CONFIRMATION",
        "status": mechanism_status,
        "shared_hotspots_beyond_positional_null": shared_hotspots,
        "frozen_input": "local_geometry_response.parquet",
        "models": list(EXPECTED_MODELS),
        "descriptors": list(LOCAL_DESCRIPTORS),
        "cohort": {"pairs_per_model": 567, "proteins_per_model": 98},
        "statistics": {
            "pair_conditioned_local": _records(local_summary),
            "pair_fixed_effect_local": _records(fixed_summary),
            "pair_conditioned_cross_model": _records(hotspot_summary),
            "hotspot_permutation_null": _records(null_summary),
            "global_pair_discrepancy_reference": _records(global_summary),
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
    (root / "local_mechanism_confirmation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    report = _report(
        local_summary,
        fixed_summary,
        hotspot_summary,
        null_summary,
        global_summary,
        undefined,
        mechanism_status,
        shared_hotspots,
    )
    (root / "local_mechanism_confirmation_report.md").write_text(report)


if __name__ == "__main__":
    main()
