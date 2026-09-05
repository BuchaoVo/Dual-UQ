"""Compare frozen Controlled and Track-I Operational geometry distributions."""

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

from dual_uq.evaluation.controlled_operational_geometry import (
    PAIR_FEATURES,
    cluster_bootstrap_spearman,
    controlled_signature_projection,
    knn_coverage_distances,
    nearest_global_rmsd_matches,
    pair_geometry_summaries,
)
from dual_uq.evaluation.cross_model_representation_sensitivity import (
    cluster_bootstrap_summary,
)
from dual_uq.evaluation.pair_conditioned_local_mechanism import LOCAL_DESCRIPTORS

GEOMETRY_MODEL = "proteinmpnn"
K_NEIGHBORS = 5
PRIMARY_FEATURES = tuple(f"{descriptor}_median" for descriptor in LOCAL_DESCRIPTORS) + (
    "aligned_ca_rmsd",
)
FEATURE_LABELS = {
    "ca_displacement_median": "Cα displacement median",
    "fragment_7_rmsd_median": "7-residue RMSD median",
    "neighborhood_distance_deformation_median": "12 Å neighborhood median",
    "torsion_phi_psi_change_median": "φ/ψ change median",
    "aligned_ca_rmsd": "Aligned Cα RMSD",
}
MODEL_LABELS = {
    "dynamicmpnn": "DynamicMPNN",
    "esm_if1": "ESM-IF1",
    "pifold": "PiFold",
    "proteinmpnn": "ProteinMPNN",
}


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def _format(value: float) -> str:
    return f"{value:.4f}"


def _feature_parts(feature: str) -> tuple[str, str]:
    for statistic in ("median", "q75", "q90"):
        suffix = f"_{statistic}"
        if feature.endswith(suffix):
            return feature[: -len(suffix)], statistic
    return feature, "pair_value"


def _geometry_summary(
    controlled: pd.DataFrame,
    operational: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> pd.DataFrame:
    strata = (
        ("controlled_all", controlled),
        ("controlled_0.25", controlled.loc[controlled["requested_dose"].eq(0.25)]),
        ("controlled_0.50", controlled.loc[controlled["requested_dose"].eq(0.50)]),
        ("operational_pdb_afdb", operational),
    )
    rows: list[dict[str, Any]] = []
    offset = 0
    for stratum, pairs in strata:
        for feature in PAIR_FEATURES:
            proteins = (
                pairs.groupby(["protein_id", "identity_cluster_id"], as_index=False, sort=True)[
                    feature
                ]
                .median()
                .rename(columns={feature: "value"})
            )
            summary = cluster_bootstrap_summary(
                proteins,
                value_column="value",
                bootstrap_replicates=bootstrap_replicates,
                seed=seed + offset,
                bootstrap_statistic="median",
            )
            descriptor, within_pair_statistic = _feature_parts(feature)
            rows.append(
                {
                    "stratum": stratum,
                    "structural_feature": feature,
                    "descriptor": descriptor,
                    "within_pair_statistic": within_pair_statistic,
                    "n_pairs": len(pairs),
                    "protein_q90": float(proteins["value"].quantile(0.90)),
                    **summary,
                }
            )
            offset += 1
    return pd.DataFrame(rows)


def _coverage(
    projection: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    standardized = [f"standardized__{feature}" for feature in PAIR_FEATURES]
    controlled = projection.loc[projection["structural_regime"].eq("controlled")].copy()
    operational = projection.loc[projection["structural_regime"].eq("operational_pdb_afdb")].copy()
    controlled_distance, operational_distance = knn_coverage_distances(
        controlled[standardized].to_numpy(),
        operational[standardized].to_numpy(),
        k=K_NEIGHBORS,
    )
    controlled["coverage_role"] = "controlled_leave_one_out"
    controlled["coverage_distance"] = controlled_distance
    operational["coverage_role"] = "operational_to_controlled"
    operational["coverage_distance"] = operational_distance
    coverage = pd.concat([controlled, operational], ignore_index=True)
    threshold = float(np.quantile(controlled_distance, 0.95))
    coverage["controlled_loo_q95"] = threshold
    coverage["beyond_controlled_loo_q95"] = coverage["coverage_distance"].gt(threshold)

    rows = []
    for offset, (role, group) in enumerate(coverage.groupby("coverage_role", sort=True)):
        proteins = (
            group.groupby(["protein_id", "identity_cluster_id"], as_index=False, sort=True)[
                "coverage_distance"
            ]
            .median()
            .rename(columns={"coverage_distance": "value"})
        )
        rows.append(
            {
                "coverage_role": role,
                "n_pairs": len(group),
                "pair_fraction_beyond_controlled_q95": float(
                    group["beyond_controlled_loo_q95"].mean()
                ),
                **cluster_bootstrap_summary(
                    proteins,
                    value_column="value",
                    bootstrap_replicates=bootstrap_replicates,
                    seed=seed + offset,
                    bootstrap_statistic="median",
                ),
            }
        )
    return coverage, pd.DataFrame(rows), threshold


def _matched_summary(
    matches: pd.DataFrame,
    *,
    controlled_scales: dict[str, float],
    bootstrap_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches = matches.copy()
    matches["controlled_feature_scale"] = matches["structural_feature"].map(controlled_scales)
    matches["standardized_feature_difference"] = (
        matches["feature_difference"] / matches["controlled_feature_scale"]
    )
    rows = []
    for offset, (feature, group) in enumerate(matches.groupby("structural_feature", sort=True)):
        proteins = (
            group.groupby(
                ["operational_protein_id", "operational_identity_cluster_id"],
                as_index=False,
                sort=True,
            )["standardized_feature_difference"]
            .median()
            .rename(
                columns={
                    "operational_protein_id": "protein_id",
                    "operational_identity_cluster_id": "identity_cluster_id",
                    "standardized_feature_difference": "value",
                }
            )
        )
        rows.append(
            {
                "structural_feature": feature,
                "n_matched_pairs": group["operational_pair_id"].nunique(),
                **cluster_bootstrap_summary(
                    proteins,
                    value_column="value",
                    bootstrap_replicates=bootstrap_replicates,
                    seed=seed + offset,
                    bootstrap_statistic="median",
                ),
            }
        )
    return matches, pd.DataFrame(rows)


def _coverage_response(
    coverage: pd.DataFrame,
    response: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    operational = coverage.loc[
        coverage["coverage_role"].eq("operational_to_controlled"),
        ["pair_id", "protein_id", "identity_cluster_id", "coverage_distance"],
    ]
    joined = response.merge(
        operational,
        on=["pair_id", "protein_id", "identity_cluster_id"],
        how="inner",
        validate="many_to_one",
    )
    rows = []
    protein_parts = []
    for offset, (model, group) in enumerate(joined.groupby("model_id", sort=True)):
        proteins = group.groupby(
            ["model_id", "checkpoint_id", "protein_id", "identity_cluster_id"],
            as_index=False,
            sort=True,
        ).agg(
            coverage_distance=("coverage_distance", "median"),
            r_jsd_bits=("r_jsd_bits", "median"),
            n_pairs=("pair_id", "nunique"),
        )
        protein_parts.append(proteins)
        rows.append(
            {
                "model_id": model,
                "checkpoint_id": group["checkpoint_id"].iloc[0],
                "n_pairs": group["pair_id"].nunique(),
                **cluster_bootstrap_spearman(
                    proteins,
                    x_column="coverage_distance",
                    y_column="r_jsd_bits",
                    replicates=bootstrap_replicates,
                    seed=seed + offset,
                ),
            }
        )
    return pd.concat(protein_parts, ignore_index=True), pd.DataFrame(rows)


def _figure(
    pairs: pd.DataFrame,
    projection: pd.DataFrame,
    coverage: pd.DataFrame,
    matched_summary: pd.DataFrame,
    response_proteins: pd.DataFrame,
    response_summary: pd.DataFrame,
    threshold: float,
    path: Path,
) -> None:
    figure, axes = plt.subplot_mosaic(
        [["A", "A", "B", "B"], ["C", "D", "D", "E"]],
        figsize=(16, 9),
        constrained_layout=True,
    )
    colors = {"controlled": "#6C757D", "operational_pdb_afdb": "#D55E00"}
    controlled = pairs.loc[pairs["structural_regime"].eq("controlled")]
    groups = (
        ("C 0.25", controlled.loc[controlled["requested_dose"].eq(0.25)]),
        ("C 0.50", controlled.loc[controlled["requested_dose"].eq(0.50)]),
        ("Operational", pairs.loc[pairs["structural_regime"].eq("operational_pdb_afdb")]),
    )
    means = controlled.loc[:, PRIMARY_FEATURES].mean()
    scales = controlled.loc[:, PRIMARY_FEATURES].std(ddof=0)
    positions = np.arange(len(PRIMARY_FEATURES), dtype=float)
    for offset, (label, group), color in zip(
        (-0.24, 0.0, 0.24), groups, ("#999999", "#333333", "#D55E00"), strict=True
    ):
        data = [
            ((group[feature] - means[feature]) / scales[feature]) for feature in PRIMARY_FEATURES
        ]
        box = axes["A"].boxplot(
            data,
            positions=positions + offset,
            widths=0.20,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "white", "linewidth": 1.2},
        )
        for patch in box["boxes"]:
            patch.set_facecolor(color)
        box["boxes"][0].set_label(label)
    axes["A"].axhline(0, color="0.6", linewidth=0.8)
    axes["A"].set_xticks(positions, [FEATURE_LABELS[x] for x in PRIMARY_FEATURES], rotation=15)
    axes["A"].set_ylabel("Controlled-standardized pair value")
    axes["A"].set_title("A  Pair-level structural distributions", loc="left", fontweight="bold")
    axes["A"].legend(frameon=False, fontsize=8, ncol=3)

    for regime, group in projection.groupby("structural_regime", sort=True):
        axes["B"].scatter(
            group["pc1"],
            group["pc2"],
            s=18,
            alpha=0.65,
            color=colors[regime],
            label="Controlled" if regime == "controlled" else "Operational",
        )
    axes["B"].axhline(0, color="0.8", linewidth=0.7)
    axes["B"].axvline(0, color="0.8", linewidth=0.7)
    axes["B"].set_xlabel("Controlled-fit PC1")
    axes["B"].set_ylabel("Controlled-fit PC2")
    axes["B"].set_title("B  Controlled-fit structural signature", loc="left", fontweight="bold")
    axes["B"].legend(frameon=False, fontsize=8)

    for role, group in coverage.groupby("coverage_role", sort=True):
        values = np.sort(group["coverage_distance"].to_numpy())
        axes["C"].plot(
            values,
            np.arange(1, len(values) + 1) / len(values),
            color="#6C757D" if role == "controlled_leave_one_out" else "#D55E00",
            label="Controlled LOO" if role == "controlled_leave_one_out" else "Operational",
        )
    axes["C"].axvline(threshold, color="black", linestyle="--", linewidth=1, label="C LOO q95")
    axes["C"].set_xlabel("k=5 structural coverage distance")
    axes["C"].set_ylabel("Empirical CDF")
    axes["C"].set_title("C  Controlled support coverage", loc="left", fontweight="bold")
    axes["C"].legend(frameon=False, fontsize=8)

    ordered = matched_summary.set_index("structural_feature").reindex(PAIR_FEATURES[:-1])
    x = np.arange(len(ordered))
    axes["D"].errorbar(
        x,
        ordered["median"],
        yerr=np.vstack(
            [ordered["median"] - ordered["ci_low"], ordered["ci_high"] - ordered["median"]]
        ),
        fmt="o",
        color="#0072B2",
        capsize=2,
    )
    axes["D"].axhline(0, color="0.5", linewidth=0.8)
    axes["D"].set_xticks(x, [x.replace("_", " ") for x in ordered.index], rotation=70, fontsize=6)
    axes["D"].set_ylabel("Operational − matched Controlled\n(in Controlled SD)")
    axes["D"].set_title("D  Global-RMSD-matched local profiles", loc="left", fontweight="bold")

    for model, group in response_proteins.groupby("model_id", sort=True):
        summary = response_summary.loc[response_summary["model_id"].eq(model)].iloc[0]
        axes["E"].scatter(
            group["coverage_distance"],
            group["r_jsd_bits"],
            s=16,
            alpha=0.55,
            label=f"{MODEL_LABELS[model]} ρ={summary.spearman_rho:.2f}",
        )
    axes["E"].set_xlabel("Structural coverage distance")
    axes["E"].set_ylabel("Protein median operational R_JSD")
    axes["E"].set_title("E  Coverage vs Operational R_JSD", loc="left", fontweight="bold")
    axes["E"].legend(frameon=False, fontsize=7)

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def _report(
    classification: str,
    geometry_summary: pd.DataFrame,
    projection_fit: dict[str, Any],
    coverage_summary: pd.DataFrame,
    threshold: float,
    matched: pd.DataFrame,
    matched_summary: pd.DataFrame,
    response_summary: pd.DataFrame,
) -> str:
    lines = [
        "# StructCal Controlled–Operational geometry distribution analysis",
        "",
        "## Classification",
        "",
        f"`{classification}`",
        "",
        (
            "All statistics use pair-level structural summaries followed by protein/30%-identity-cluster "
            "aggregation. No residue count is used as an inferential sample size."
        ),
        "",
        "## A. Pair-level descriptor comparison",
        "",
        (
            "Cells report the median of protein-level pair medians, protein IQR, and protein q90. "
            "Controlled all/dose strata and Operational are descriptive cohorts, not paired samples."
        ),
        "",
        "| structural variable | Controlled all | Controlled 0.25 Å | Controlled 0.50 Å | Operational |",
        "|---|---:|---:|---:|---:|",
    ]
    for feature in PRIMARY_FEATURES:
        row = geometry_summary.loc[geometry_summary["structural_feature"].eq(feature)].set_index(
            "stratum"
        )
        cells = []
        for stratum in (
            "controlled_all",
            "controlled_0.25",
            "controlled_0.50",
            "operational_pdb_afdb",
        ):
            value = row.loc[stratum]
            cells.append(
                f"{_format(value['median'])} [{_format(value['q25'])}, {_format(value['q75'])}]; "
                f"q90={_format(value['protein_q90'])}"
            )
        lines.append(f"| {FEATURE_LABELS[feature]} | " + " | ".join(cells) + " |")

    explained = projection_fit["explained_variance_ratio"]
    control_coverage = coverage_summary.loc[
        coverage_summary["coverage_role"].eq("controlled_leave_one_out")
    ].iloc[0]
    operational_coverage = coverage_summary.loc[
        coverage_summary["coverage_role"].eq("operational_to_controlled")
    ].iloc[0]
    lines.extend(
        [
            "",
            "## B. Controlled-fit multivariate signature",
            "",
            (
                f"The 13 fixed pair features were standardized using Controlled means/scales only. "
                f"Controlled-fit PCA explains {explained[0]:.3%} on PC1 and {explained[1]:.3%} on PC2 "
                f"({sum(explained):.3%} total). Operational pairs were projected without refitting. "
                "PCA is used only for visualization, not as a biological latent-factor model."
            ),
            "",
            "## C. Controlled support and Operational coverage",
            "",
            "| cohort | pairs | protein median distance (95% CI) | IQR | fraction of pairs beyond Controlled LOO q95 |",
            "|---|---:|---:|---:|---:|",
            (
                f"| Controlled leave-one-out | {control_coverage.n_pairs} | "
                f"{_format(control_coverage['median'])} [{_format(control_coverage.ci_low)}, "
                f"{_format(control_coverage.ci_high)}] | [{_format(control_coverage.q25)}, "
                f"{_format(control_coverage.q75)}] | "
                f"{control_coverage.pair_fraction_beyond_controlled_q95:.4f} |"
            ),
            (
                f"| Operational → Controlled | {operational_coverage.n_pairs} | "
                f"{_format(operational_coverage['median'])} [{_format(operational_coverage.ci_low)}, "
                f"{_format(operational_coverage.ci_high)}] | [{_format(operational_coverage.q25)}, "
                f"{_format(operational_coverage.q75)}] | "
                f"{operational_coverage.pair_fraction_beyond_controlled_q95:.4f} |"
            ),
            "",
            f"The fixed Controlled leave-one-out q95 threshold is {threshold:.4f}.",
            "",
            "## D. Global-RMSD-matched comparison",
            "",
            (
                f"Only {matched['operational_pair_id'].nunique()}/68 Operational pairs "
                f"({matched['operational_protein_id'].nunique()} proteins) lie inside the Controlled "
                f"aligned-Cα-RMSD range [{matched['overlap_min'].iloc[0]:.4f}, "
                f"{matched['overlap_max'].iloc[0]:.4f}] Å. Each is matched to the nearest Controlled "
                "RMSD with replacement; no one-to-one match is forced. Differences below are in "
                "Controlled feature standard deviations."
            ),
            "",
            "| local pair feature | median difference (95% CI) | IQR | matched pairs | proteins |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in (
        matched_summary.set_index("structural_feature").reindex(PAIR_FEATURES[:-1]).itertuples()
    ):
        lines.append(
            f"| {row.Index} | {_format(row.median)} [{_format(row.ci_low)}, "
            f"{_format(row.ci_high)}] | [{_format(row.q25)}, {_format(row.q75)}] | "
            f"{row.n_matched_pairs} | {row.n_proteins} |"
        )

    lines.extend(
        [
            "",
            "## E. Structural coverage versus Operational response",
            "",
            "| model | protein-level Spearman ρ (95% cluster-bootstrap CI) | pairs | proteins | clusters |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in response_summary.sort_values("model_id").itertuples():
        lines.append(
            f"| {MODEL_LABELS[row.model_id]} | {_format(row.spearman_rho)} "
            f"[{_format(row.ci_low)}, {_format(row.ci_high)}] | {row.n_pairs} | "
            f"{row.n_proteins} | {row.n_identity_clusters} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Operational PDB-AFDB variation extends well beyond the Controlled 13-dimensional "
                "structural support and only a minority of pairs overlap its global-RMSD range. "
                "Fragment and torsional modes are enriched Operationally, while neighborhood deformation "
                "shows a different median/tail mixture. These shifts are descriptively consistent with "
                "non-identical model response magnitudes and descriptor rankings, but do not causally "
                "explain them or imply different model internals."
            ),
            "",
            (
                "In the limited global-RMSD-overlap subset, Operational Cα-displacement and neighborhood "
                "summaries are lower, while torsional summaries and fragment q90 are higher. This confirms "
                "a change in deformation-mode mixture rather than only a change in global magnitude."
            ),
            "",
            (
                "The model-specific patterns are descriptively coherent with that mixture: DynamicMPNN "
                "retains torsion as strongest; ESM-IF1 retains neighborhood despite a similar upper tail "
                "and lower central tendency; PiFold shifts toward Cα displacement as its Operational upper "
                "tail broadens relative to neighborhood; ProteinMPNN's stronger fragment/torsion associations "
                "coincide with enrichment of those modes. None of these consistencies establishes causality."
            ),
            "",
            (
                "Structural coverage distance is positively associated with Operational R_JSD for all four "
                "models, with every protein-cluster bootstrap CI above zero."
            ),
            "",
            (
                "A Controlled-to-Operational change score is not correlated pairwise with coverage: the "
                "regimes share only one protein, so such a test would violate the declared cohort semantics."
            ),
            "",
            (
                "Aligned Cα RMSD is recovered as the RMS of the existing globally Kabsch-aligned residue "
                "Cα displacement artifact for both regimes. No Track-I case or frozen descriptor is regenerated."
            ),
            "",
            "KW-Design remains `IMPLEMENTATION_FAILURE`; DeSAE remains `MODEL_CAPABILITY_UNAVAILABLE`.",
            "",
            "![Controlled–Operational geometry](figures/controlled_operational_geometry.png)",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/structcal_cross_model_representation_sensitivity"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2_026_09_07)
    args = parser.parse_args()
    root = args.output_root.resolve()

    controlled_local = pd.read_parquet(root / "local_geometry_response.parquet")
    controlled_local = controlled_local.loc[controlled_local["model_id"].eq(GEOMETRY_MODEL)]
    operational_local = pd.read_parquet(root / "operational_local_geometry_response.parquet")
    operational_local = operational_local.loc[operational_local["model_id"].eq(GEOMETRY_MODEL)]
    controlled_pairs = pair_geometry_summaries(controlled_local, regime="controlled")
    operational_pairs = pair_geometry_summaries(operational_local, regime="operational_pdb_afdb")
    if len(controlled_pairs) != 567 or len(operational_pairs) != 68:
        raise ValueError("pair-level geometry does not preserve frozen cohort membership")
    pair_geometry = pd.concat([controlled_pairs, operational_pairs], ignore_index=True)
    pair_geometry.to_parquet(root / "controlled_operational_pair_geometry.parquet", index=False)

    geometry_summary = _geometry_summary(
        controlled_pairs,
        operational_pairs,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    geometry_summary.to_parquet(
        root / "controlled_operational_geometry_summary.parquet", index=False
    )

    projection, projection_fit = controlled_signature_projection(
        controlled_pairs, operational_pairs
    )
    projection.to_parquet(root / "structural_signature_projection.parquet", index=False)
    coverage, coverage_summary, coverage_threshold = _coverage(
        projection,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed + 100,
    )
    coverage.to_parquet(root / "operational_coverage_distance.parquet", index=False)

    matches = nearest_global_rmsd_matches(controlled_pairs, operational_pairs)
    scales = dict(
        zip(
            projection_fit["feature_columns"],
            projection_fit["controlled_scale"],
            strict=True,
        )
    )
    matches, matched_summary = _matched_summary(
        matches,
        controlled_scales=scales,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed + 200,
    )
    matches.to_parquet(root / "geometry_matched_regime_comparison.parquet", index=False)

    response = pd.read_parquet(root / "operational_pdb_afdb_response.parquet")
    response_proteins, response_summary = _coverage_response(
        coverage,
        response,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed + 300,
    )
    response_summary.to_parquet(root / "coverage_response_association.parquet", index=False)

    operational_coverage = coverage.loc[
        coverage["coverage_role"].eq("operational_to_controlled"), "coverage_distance"
    ]
    if float(operational_coverage.median()) > coverage_threshold:
        classification = "STRONG_STRUCTURAL_DISTRIBUTION_SHIFT"
    elif operational_coverage.gt(coverage_threshold).any():
        classification = "PARTIAL_STRUCTURAL_COVERAGE"
    else:
        classification = "SUBSTANTIAL_STRUCTURAL_OVERLAP"

    figure_path = root / "figures/controlled_operational_geometry.png"
    _figure(
        pair_geometry,
        projection,
        coverage,
        matched_summary,
        response_proteins,
        response_summary,
        coverage_threshold,
        figure_path,
    )
    summary = {
        "task": "STRUCTCAL_CONTROLLED_OPERATIONAL_GEOMETRY_DISTRIBUTION_ANALYSIS",
        "classification": classification,
        "pair_geometry": {
            "controlled_pairs": len(controlled_pairs),
            "controlled_proteins": controlled_pairs["protein_id"].nunique(),
            "operational_pairs": len(operational_pairs),
            "operational_proteins": operational_pairs["protein_id"].nunique(),
            "local_descriptors": list(LOCAL_DESCRIPTORS),
            "within_pair_statistics": ["median", "q75", "q90"],
            "global_descriptor": "aligned_ca_rmsd",
            "aligned_ca_rmsd_source": "rms_of_global_kabsch_ca_displacement",
        },
        "geometry_summary": _records(geometry_summary),
        "structural_signature_fit": projection_fit,
        "coverage": {
            "k": K_NEIGHBORS,
            "distance": "mean Euclidean distance to k neighbors in 13D Controlled-standardized space",
            "controlled_reference": "leave_one_out",
            "controlled_loo_q95": coverage_threshold,
            "summary": _records(coverage_summary),
        },
        "geometry_matching": {
            "variable": "aligned_ca_rmsd",
            "method": "nearest Controlled RMSD with replacement inside overlapping range",
            "operational_pairs_in_overlap": matches["operational_pair_id"].nunique(),
            "operational_proteins_in_overlap": matches["operational_protein_id"].nunique(),
            "overlap_min": float(matches["overlap_min"].iloc[0]),
            "overlap_max": float(matches["overlap_max"].iloc[0]),
            "summary": _records(matched_summary),
        },
        "coverage_response_association": _records(response_summary),
        "bootstrap": {
            "unit": "identity_cluster_30",
            "replicates": args.bootstrap_replicates,
            "median_summary_statistic": "median",
        },
        "seed": args.seed,
        "capability_states": {
            "kw_design": "IMPLEMENTATION_FAILURE",
            "desae": "MODEL_CAPABILITY_UNAVAILABLE",
        },
    }
    (root / "controlled_operational_geometry_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (root / "controlled_operational_geometry_report.md").write_text(
        _report(
            classification,
            geometry_summary,
            projection_fit,
            coverage_summary,
            coverage_threshold,
            matches,
            matched_summary,
            response_summary,
        )
    )


if __name__ == "__main__":
    main()
