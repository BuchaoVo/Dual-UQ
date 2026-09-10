"""Materialize cross-model APO/HOLO generative-propagation comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from dual_uq.core.artifacts import json_safe, parquet_bytes, write_immutable_bytes
from dual_uq.core.hashing import sha256_bytes as _sha256
from dual_uq.evaluation.apo_holo_generative_cross_model import (
    ApoHoloGenerativeCrossModelResult,
    build_cross_model_result,
)


def _write_bytes(path: Path, payload: bytes) -> str:
    write_immutable_bytes(path, payload)
    return _sha256(payload)


def _write_table(path: Path, table: pd.DataFrame) -> str:
    return _write_bytes(path, parquet_bytes(table))


def _write_json(path: Path, value: Any) -> str:
    payload = (json.dumps(json_safe(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    return _write_bytes(path, payload)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _find_association(result: ApoHoloGenerativeCrossModelResult, label: str) -> dict[str, Any] | None:
    rows = result.associations.loc[result.associations["association"].eq(label)]
    return None if rows.empty else rows.iloc[0].to_dict()


def _report(result: ApoHoloGenerativeCrossModelResult) -> str:
    summary = result.summary
    models = summary["model_summaries"]
    lines = [
        "# APO/HOLO Generative Propagation — Cross-Model Analysis",
        "",
        f"- Verdict: `{summary['verdict']}`",
        f"- Primary requested cohort: `{summary['primary_requested_protein_count']}` proteins",
        f"- Actual shared generation intersection: `{summary['shared_generation_protein_count']}` proteins",
        "- Statistical unit: protein; generated sequences are not treated as independent biological replicates.",
        "",
        "## Q1 — Does between-state divergence exceed within-state diversity?",
        "",
    ]
    for model, label in (("proteinmpnn", "ProteinMPNN"), ("esm_if1", "ESM-IF1")):
        values = models[model]
        excess = values["d_excess_64"]
        lines.append(
            f"- {label}: positive D_excess in {values['d_excess_positive_count']}/{values['protein_count']} "
            f"proteins; median {excess['median']}, q10 {excess['q10']}, q90 {excess['q90']}."
        )
    lines.extend(
        [
            "- D_excess is independent-sample normalized Hamming divergence minus the mean within-state divergence; it describes marginal sequence separation, not a full joint distribution distance.",
            "",
            "## Q2 — Does local state response propagate to generated position distributions?",
            "",
        ]
    )
    for model, label in (("proteinmpnn", "ProteinMPNN"), ("esm_if1", "ESM-IF1")):
        js = models[model]["generative_js_burden_bits_64"]
        lines.append(f"- {label}: protein-level generative JS burden median {js['median']} bits (q10 {js['q10']}, q90 {js['q90']}).")
    lines.extend(
        [
            "- Position-level JS values are empirical 20-AA distributions from the same 64-sequence ensembles and are summarized at protein level.",
            "",
            "## Q3 — Does local response track generative propagation?",
            "",
        ]
    )
    for label in ("proteinmpnn_local_to_d_excess", "proteinmpnn_local_to_generative_js", "esm_if1_local_to_d_excess", "esm_if1_local_to_generative_js"):
        row = _find_association(result, label)
        if row is not None:
            lines.append(f"- `{label}`: Spearman rho {row['spearman']} (n={row['n']}).")
    lines.extend(
        [
            "- These are descriptive protein-level associations; no p-values or new composite uncertainty metric are introduced.",
            "",
            "## Q4 — Is generative propagation shared across architectures?",
            "",
            f"- D_excess cross-model Spearman rho: `{summary['cross_model_generation_spearman']['d_excess_64']}`.",
            f"- Generative JS cross-model Spearman rho: `{summary['cross_model_generation_spearman']['generative_js_burden']}`.",
            f"- Frozen local-response reference rho: `{summary['local_response_reference_spearman']}`; observed local rho on this shared generation intersection: `{summary['local_response_observed_shared_spearman']}`.",
            "",
            "## Q5 — How heterogeneous are the effects?",
            "",
        ]
    )
    for model, label in (("proteinmpnn", "ProteinMPNN"), ("esm_if1", "ESM-IF1")):
        excess = models[model]["d_excess_64"]
        js = models[model]["generative_js_burden_bits_64"]
        lines.append(f"- {label}: D_excess q10/median/q90 = {excess['q10']} / {excess['median']} / {excess['q90']}; JS q10/median/q90 = {js['q10']} / {js['median']} / {js['q90']} bits.")
    lines.extend(["", "## Convergence", ""])
    for row in result.convergence.itertuples(index=False):
        lines.append(f"- {row.model} n={row.sample_count}: D_excess median {row.d_excess_median}, positive {row.positive_count}/{row.protein_count}, rank rho to final {row.rank_spearman_to_final}.")
    lines.extend(["", "## Structural association", ""])
    for row in result.associations.loc[result.associations["y"].isin({"aligned_ca_rmsd", "median_residue_displacement", "p90_residue_displacement", "median_local_pairwise_distance_change", "contact_turnover_fraction_mean"})].itertuples(index=False):
        lines.append(f"- {row.model} {row.x} vs {row.y}: Spearman rho {row.spearman} (n={row.n}).")
    lines.extend(["", "## Ligand-related generative response", ""])
    if result.ligand_summary.empty:
        lines.append("- Ligand localization summary unavailable in the supplied geometry release.")
    else:
        for row in result.ligand_summary.itertuples(index=False):
            lines.append(f"- {row.model} {row.ligand_status}: n={row.protein_count}, D_excess median {row.d_excess_median}, JS median {row.js_median} bits.")
    lines.extend(
        [
            "- These geometry and ligand summaries are descriptive within the available released intersection; they do not establish causality, allostery, fitness, stability, or function.",
            "",
            "## Interpretation boundary",
            "",
            summary["interpretation_boundary"],
            "",
            "## Next",
            "",
            "`MULTI_STATE_BASELINE_EVALUATION`",
            "",
        ]
    )
    return "\n".join(lines)


def materialize_cross_model_analysis(
    *,
    project_root: Path,
    output_root: Path,
    proteinmpnn_generation_root: Path,
    esm_if1_generation_root: Path,
    proteinmpnn_local_path: Path,
    esm_if1_local_path: Path,
    pair_geometry_path: Path | None,
    residue_geometry_path: Path | None,
    local_reference_spearman: float | None,
) -> dict[str, Any]:
    pnn_generation = pd.read_parquet(proteinmpnn_generation_root / "protein_generation_summary.parquet")
    esm_generation = pd.read_parquet(esm_if1_generation_root / "protein_generation_summary.parquet")
    pnn_local = pd.read_parquet(proteinmpnn_local_path)
    esm_local = pd.read_parquet(esm_if1_local_path)
    pnn_convergence = pd.read_parquet(proteinmpnn_generation_root / "generation_convergence.parquet")
    esm_convergence = pd.read_parquet(esm_if1_generation_root / "generation_convergence.parquet")
    pair_geometry = pd.read_parquet(pair_geometry_path) if pair_geometry_path is not None else None
    residue_geometry = pd.read_parquet(residue_geometry_path) if residue_geometry_path is not None else None
    result = build_cross_model_result(
        proteinmpnn_generation=pnn_generation,
        esm_if1_generation=esm_generation,
        proteinmpnn_local=pnn_local,
        esm_if1_local=esm_local,
        pair_geometry=pair_geometry,
        residue_geometry=residue_geometry,
        proteinmpnn_convergence=pnn_convergence,
        esm_if1_convergence=esm_convergence,
        local_reference_spearman=local_reference_spearman,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        "protein_summary": output_root / "cross_model_protein_summary.parquet",
        "convergence": output_root / "cross_model_convergence.parquet",
        "associations": output_root / "cross_model_associations.parquet",
        "ligand_summary": output_root / "cross_model_ligand_summary.parquet",
    }
    hashes = {name: _write_table(path, table) for name, path, table in (
        ("protein_summary", artifact_paths["protein_summary"], result.protein_summary),
        ("convergence", artifact_paths["convergence"], result.convergence),
        ("associations", artifact_paths["associations"], result.associations),
        ("ligand_summary", artifact_paths["ligand_summary"], result.ligand_summary),
    )}
    hashes["summary"] = _write_json(output_root / "summary.json", result.summary)
    report = _report(result).encode()
    hashes["report"] = _write_bytes(output_root / "report.md", report)
    manifest = {
        "schema_version": "apo_holo_generative_cross_model_manifest_v1",
        "status": result.summary["verdict"],
        "protocol": {
            "sample_count_per_condition": 64,
            "nested_prefixes": [16, 32, 64],
            "distance": "normalized Hamming",
            "position_divergence": "Jensen-Shannon bits over standard 20-AA empirical distributions",
            "statistical_unit": "protein",
        },
        "input_paths": {
            "proteinmpnn_generation_root": _relative(proteinmpnn_generation_root, project_root),
            "esm_if1_generation_root": _relative(esm_if1_generation_root, project_root),
            "proteinmpnn_local": _relative(proteinmpnn_local_path, project_root),
            "esm_if1_local": _relative(esm_if1_local_path, project_root),
            "pair_geometry": _relative(pair_geometry_path, project_root) if pair_geometry_path else None,
            "residue_geometry": _relative(residue_geometry_path, project_root) if residue_geometry_path else None,
        },
        "summary": result.summary,
        "artifact_sha256": hashes,
    }
    hashes["manifest"] = _write_json(output_root / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--proteinmpnn-generation-root", type=Path, required=True)
    parser.add_argument("--esm-if1-generation-root", type=Path, required=True)
    parser.add_argument("--proteinmpnn-local", type=Path, required=True)
    parser.add_argument("--esm-if1-local", type=Path, required=True)
    parser.add_argument("--pair-geometry", type=Path)
    parser.add_argument("--residue-geometry", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--local-reference-spearman", type=float, default=0.783822)
    args = parser.parse_args(argv)
    try:
        manifest = materialize_cross_model_analysis(
            project_root=args.project_root,
            output_root=args.output_root,
            proteinmpnn_generation_root=args.proteinmpnn_generation_root,
            esm_if1_generation_root=args.esm_if1_generation_root,
            proteinmpnn_local_path=args.proteinmpnn_local,
            esm_if1_local_path=args.esm_if1_local,
            pair_geometry_path=args.pair_geometry,
            residue_geometry_path=args.residue_geometry,
            local_reference_spearman=args.local_reference_spearman,
        )
        print(
            json.dumps(
                json_safe({"status": "COMPLETE", "manifest": manifest}),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "BLOCKED", "message": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
