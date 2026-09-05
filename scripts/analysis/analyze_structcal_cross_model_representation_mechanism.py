"""Analyze local geometry and cross-model response localization after the clean gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dual_uq.evaluation.cross_model_representation_mechanism import (
    cross_model_localization,
    local_geometry_from_cases,
    within_protein_spearman,
)
from dual_uq.evaluation.cross_model_representation_sensitivity import (
    cluster_bootstrap_summary,
)

MAIN_MODELS = {
    "proteinmpnn": "v_48_020",
    "esm_if1": "esm_if1_gvp4_t16_142M_UR50",
    "pifold": "official_checkpoint_pth",
    "dynamicmpnn": "single_chain_k2",
}


def _geometry(cases: pd.DataFrame, pair_metadata: pd.DataFrame) -> pd.DataFrame:
    geometry = local_geometry_from_cases(cases)
    metadata_columns = [
        "pair_id",
        "protein_id",
        "perturbation_family",
        "requested_dose",
    ]
    return geometry.merge(
        pair_metadata[metadata_columns].drop_duplicates(),
        on=["pair_id", "protein_id"],
        how="left",
        validate="many_to_one",
    )


def _association_summary(associations: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouper: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for values, group in associations.groupby(grouper, dropna=False, sort=True):
        selected = group.loc[np.isfinite(group["spearman_rho"])].copy()
        if selected.empty:
            continue
        group_values = values if isinstance(values, tuple) else (values,)
        rows.append(
            {
                **dict(zip(group_columns, group_values, strict=True)),
                **cluster_bootstrap_summary(
                    selected,
                    value_column="spearman_rho",
                    bootstrap_statistic="median",
                ),
            }
        )
    return pd.DataFrame(rows)


def _format(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/structcal_cross_model_representation_sensitivity"),
    )
    parser.add_argument(
        "--release-root", type=Path, default=Path("artifacts/releases/structcal_v1")
    )
    args = parser.parse_args()
    root = args.output_root.resolve()
    release = args.release_root.resolve()

    controlled = pd.read_parquet(root / "controlled_response.parquet")
    pair_metadata = controlled.loc[controlled["model_id"].eq("proteinmpnn")]
    cases = pd.read_parquet(root / "cases/nca/controlled.parquet")
    geometry = _geometry(cases, pair_metadata)

    frozen_pair = pd.read_parquet(release / "annotations/pair_structural_descriptors.parquet")
    pair_geometry = frozen_pair.loc[
        frozen_pair["pair_id"].isin(geometry["pair_id"]),
        ["pair_id", "aligned_ca_rmsd"],
    ].copy()
    if (
        len(pair_geometry) != 567
        or pair_geometry["pair_id"].duplicated().any()
        or set(pair_geometry["pair_id"]) != set(geometry["pair_id"])
    ):
        raise ValueError("frozen controlled pair geometry does not cover the projected cohort")

    response_parts = []
    for model, checkpoint in MAIN_MODELS.items():
        residue = pd.read_parquet(
            root / "shards" / model / checkpoint / "controlled/residue_response.parquet"
        )
        drop = [
            column
            for column in (
                "perturbation_family",
                "requested_dose",
                "identity_cluster_id",
            )
            if column in residue
        ]
        local = residue.drop(columns=drop).merge(
            geometry,
            on=["pair_id", "protein_id", "canonical_position"],
            how="inner",
            validate="one_to_one",
        )
        if len(local) != len(residue):
            raise ValueError(f"local geometry join is incomplete for {model}")
        response_parts.append(local)
    local_response = pd.concat(response_parts, ignore_index=True)
    local_response.to_parquet(root / "local_geometry_response.parquet", index=False)

    local_associations = within_protein_spearman(local_response)
    local_associations.to_parquet(root / "local_geometry_associations.parquet", index=False)
    local_statistics = _association_summary(
        local_associations, ["model_id", "checkpoint_id", "descriptor"]
    )

    global_response = controlled.merge(
        pair_geometry, on="pair_id", how="inner", validate="many_to_one"
    )
    global_response.to_parquet(root / "global_geometry_response.parquet", index=False)
    global_association_input = global_response.rename(columns={"r_jsd_bits": "jsd_bits"})
    global_associations = within_protein_spearman(
        global_association_input, descriptors=("aligned_ca_rmsd",)
    )
    global_associations["association_scope"] = "within_protein_controlled_pairs_global"
    global_statistics = _association_summary(
        global_associations, ["model_id", "checkpoint_id", "descriptor"]
    )
    associations = pd.concat([local_associations, global_associations], ignore_index=True)
    statistics = pd.concat([local_statistics, global_statistics], ignore_index=True)
    associations.to_parquet(root / "geometry_response_associations.parquet", index=False)
    statistics.to_parquet(root / "geometry_response_association_statistics.parquet", index=False)

    localization = cross_model_localization(local_response)
    localization.to_parquet(root / "cross_model_localization.parquet", index=False)
    localization_statistics = _association_summary(
        localization, ["left_model_id", "right_model_id"]
    )
    localization_statistics.to_parquet(
        root / "cross_model_localization_statistics.parquet", index=False
    )

    local_supported = bool(local_statistics.groupby("model_id")["median"].max().gt(0).all())
    summary_path = root / "phenomenon_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["status"] = "PHENOMENON_AND_LOCAL_MECHANISM_COMPLETE"
    summary["local_geometry"] = {
        "descriptor_protocol": {
            "global_alignment": "Kabsch C-alpha over the complete paired axis",
            "fragment_rmsd": "strict centered 7-residue C-alpha fragment",
            "torsion_change": "RMS shortest circular phi/psi change in degrees",
            "neighborhood_deformation": (
                "mean absolute pair-distance change over reference C-alpha neighbors within 12 angstrom"
            ),
        },
        "pair_geometry_source": "frozen_structcal_v1_pair_structural_descriptors",
        "association_statistics": local_statistics.to_dict("records"),
        "global_pair_association_statistics": global_statistics.to_dict("records"),
        "cross_model_localization_statistics": localization_statistics.to_dict("records"),
        "locally_organized_response_supported": local_supported,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    report_path = root / "phenomenon_report.md"
    report = report_path.read_text().split("\n## Local geometric mechanism", maxsplit=1)[0]
    lines = [
        report.rstrip(),
        "",
        "## Local geometric mechanism",
        "",
        (
            "Associations are Spearman correlations computed separately within each protein over "
            "all frozen controlled pair-residue observations. The table reports the across-protein "
            "median and a 30%-identity-cluster bootstrap 95% CI for that median."
        ),
        "",
        "| model | descriptor | median rho | 95% CI | fraction positive | proteins |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in local_statistics.sort_values(["model_id", "descriptor"]).itertuples():
        lines.append(
            f"| {row.model_id} | {row.descriptor} | {_format(row.median)} | "
            f"[{_format(row.ci_low)}, {_format(row.ci_high)}] | "
            f"{_format(row.fraction_positive)} | {row.n_proteins} |"
        )
    lines.extend(
        [
            "",
            "## Global pair discrepancy",
            "",
            (
                "This separate analysis correlates the frozen pair-level aligned C-alpha RMSD "
                "with pair-level R_JSD within each protein; it is not one of the four residue-local descriptors."
            ),
            "",
            "| model | descriptor | median rho | 95% CI | fraction positive | proteins |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in global_statistics.sort_values(["model_id", "descriptor"]).itertuples():
        lines.append(
            f"| {row.model_id} | {row.descriptor} | {_format(row.median)} | "
            f"[{_format(row.ci_low)}, {_format(row.ci_high)}] | "
            f"{_format(row.fraction_positive)} | {row.n_proteins} |"
        )
    lines.extend(
        [
            "",
            "## Cross-model localization",
            "",
            "| model pair | median within-protein response-profile rho | 95% CI | fraction positive |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in localization_statistics.itertuples():
        lines.append(
            f"| {row.left_model_id} / {row.right_model_id} | {_format(row.median)} | "
            f"[{_format(row.ci_low)}, {_format(row.ci_high)}] | "
            f"{_format(row.fraction_positive)} |"
        )
    lines.extend(
        [
            "",
            (
                "These analyses preserve protein as the independent unit. Residue and repeated-pair "
                "observations determine each protein's internal rank association only; they do not "
                "inflate the cross-protein sample size."
            ),
        ]
    )
    report_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
