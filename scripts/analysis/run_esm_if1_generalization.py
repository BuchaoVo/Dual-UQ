"""Run the official ESM-IF1 cross-model generalization analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from dual_uq.core.atomic_io import atomic_write_new_bytes
from dual_uq.evaluation.esm_if1_generalization import (
    ESM_IF1_ANALYSIS_PROTOCOL,
    TRUE_GAP_SENSITIVITY_PROTOCOL,
    build_frozen_common_cases,
    build_gap_preserving_cases,
    find_true_gap_proteins,
    generate_esm_if1_ensemble,
    summarize_generation_propagation,
    summarize_local_response,
    teacher_forced_local_response,
)
from dual_uq.models.esm_if1 import ESMIF1ContractError, load_esm_if1


def _immutable_table(path: Path, table: pd.DataFrame) -> str:
    buffer = BytesIO()
    table.to_parquet(buffer, index=False)
    payload = buffer.getvalue()
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"immutable artifact conflict: {path}")
    else:
        atomic_write_new_bytes(path, payload)
    return hashlib.sha256(payload).hexdigest()


def _immutable_json(path: Path, payload: object) -> str:
    data = (json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"immutable artifact conflict: {path}")
    else:
        atomic_write_new_bytes(path, data)
    return hashlib.sha256(data).hexdigest()


def _paths(project_root: Path) -> dict[str, Path]:
    return {
        "pairs": project_root / "experiments/dataset/analysis/pair_validity/pair_validity.parquet",
        "masks": project_root / "experiments/dataset/releases/confirmatory/primary_common_masks.parquet",
        "remodeling": project_root / "experiments/dataset/analysis/inverse_folding_remodeling/protein_remodeling.parquet",
        "generation": project_root / "experiments/dataset/analysis/generative_propagation/protein_generation_summary.parquet",
        "cross_structure": project_root / "experiments/dataset/analysis/cross_structure_compatibility/protein_cross_structure_summary.parquet",
        "structural_asymmetry": project_root / "experiments/dataset/analysis/independent_structure_validation/asymmetry/baseline_adjusted_protein_effects.parquet",
    }


def _report(
    summary: dict[str, object],
    local: pd.DataFrame,
    generation: pd.DataFrame | None,
    exclusions: pd.DataFrame | None,
) -> str:
    lines = [
        "# ESM-IF1 Cross-Model Generalization",
        "",
        f"- Verdict: `{summary['verdict']}`",
        f"- Cohort: `{summary['cohort_label']}`",
        f"- Protein-level unit: `{summary['protein_unit']}`",
        f"- Model: `{summary['model_name']}`",
        f"- Official source revision: `{summary['implementation_revision']}`",
        f"- Checkpoint SHA256: `{summary['checkpoint_sha256']}`",
        "",
        "## Local response",
        "",
        f"- Comparable proteins: {summary['protein_count']}",
        f"- Common positions: {summary['position_count']}",
        f"- Median protein local JS burden (bits): {local['esm_if1_local_burden'].median():.6f}",
        "- The primary local-response quantity is native 20-AA teacher-forced JS divergence; ProteinMPNN P/M/SDFI are not reused.",
        "",
    ]
    if generation is not None:
        lines.extend(
            [
                "## Generative propagation",
                "",
                f"- Generated records: {summary['generated_record_count']}",
                f"- Median D_excess: {generation['d_excess'].median():.6f}",
                f"- Proteins with positive D_excess: {int((generation['d_excess'] > 0).sum())}/{len(generation)}",
                f"- Median position-level JS burden (bits): {generation['js_burden'].median():.6f}",
                "- D_PP, D_AA, D_PA and D_excess are normalized Hamming descriptors; they describe marginal sequence divergence, not a full joint-distribution distance.",
                "",
            ]
        )
        if summary.get("convergence"):
            lines.append("- Nested convergence prefixes of the same 128-sample ensemble:")
            for sample_count, values in summary["convergence"].items():
                if values.get("status"):
                    lines.append(f"  - {sample_count}: `{values['status']}`.")
                else:
                    lines.append(
                        f"  - {sample_count}: median D_excess={values['median_d_excess']:.6f}; "
                        f"positive={values['positive_d_excess_proteins']}/{summary['protein_count']}"
                    )
            lines.append("")
    elif summary.get("generation_status", "").startswith("unavailable"):
        lines.extend(
            [
                "## Generative propagation",
                "",
                f"- Status: `{summary['generation_status']}`.",
                f"- The official sampler failure was retained as a structured outcome: `{summary.get('generation_failure')}`.",
                "- No gap-preserving generation artifact was materialized; this sensitivity result is local-response only.",
                "",
            ]
        )
    lines.extend(
        [
            "## Cross-model comparison",
            "",
            f"- ESM-IF1 local JS vs frozen ProteinMPNN magnitude rho: {summary.get('local_proteinmpnn_spearman', {}).get('magnitude_median')}",
            f"- ESM-IF1 local JS vs frozen ProteinMPNN rank-displacement rho: {summary.get('local_proteinmpnn_spearman', {}).get('reordering_rank_displacement_median')}",
            f"- ESM-IF1 D_excess vs frozen ProteinMPNN D_PA excess rho: {summary.get('generation_proteinmpnn_spearman', {}).get('d_pa_excess_independent')}",
            f"- ESM-IF1 JS burden vs frozen ProteinMPNN JS burden rho: {summary.get('generation_proteinmpnn_spearman', {}).get('js_burden_mean')}",
            f"- ESM-IF1 D_excess vs frozen ProteinMPNN Delta_cross rho: {summary.get('generation_proteinmpnn_spearman', {}).get('delta_cross')}",
            "ProteinMPNN descriptors are read-only frozen inputs. Associations are descriptive protein-level comparisons and do not establish causality or biological fitness.",
            "",
        ]
    )
    if exclusions is not None:
        values = summary["exclusion_comparison"]
        lines.extend(
            [
                "## Excluded-protein characterization",
                "",
                f"- True-gap proteins excluded from the primary cohort: {summary['excluded_protein_count']}.",
                f"- Median canonical length, included/excluded: {values['canonical_length_median_included']:.3f} / {values['canonical_length_median_excluded']:.3f}.",
                f"- Median common-mask fraction, included/excluded: {values['common_mask_fraction_median_included']:.6f} / {values['common_mask_fraction_median_excluded']:.6f}.",
                f"- Median ProteinMPNN remodeling magnitude, included/excluded: {values['magnitude_median_median_included']:.6g} / {values['magnitude_median_median_excluded']:.6g}.",
                f"- Median frozen generative JS burden, included/excluded: {values['js_burden_median_included']:.6g} / {values['js_burden_median_excluded']:.6g}.",
                f"- Median frozen D_PA excess, included/excluded: {values['d_pa_excess_median_included']:.6g} / {values['d_pa_excess_median_excluded']:.6g}.",
                f"- Median frozen Delta_cross, included/excluded: {values['delta_cross_median_included']:.6g} / {values['delta_cross_median_excluded']:.6g}.",
                f"- Median ESMFold AFDB-generated baseline-adjusted effect, included/excluded: {values['esmfold_effect_median_included']:.6g} / {values['esmfold_effect_median_excluded']:.6g}.",
                "- These comparisons use frozen pre-ESM-IF1 descriptors and do not define cohort membership from ESM-IF1 outcomes.",
                f"- Scope consequence: `{summary.get('generalization_to_full_68', 'primary result is scoped to the evaluated cohort')}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "This analysis concerns model-level structure-conditioned amino-acid preferences and generated sequence distributions only. It does not establish biological fitness, stability, functional failure, or an experimental outcome.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_exclusion_table(paths: dict[str, Path], gap_ids: tuple[str, ...]) -> tuple[pd.DataFrame, dict[str, object]]:
    pairs = pd.read_parquet(paths["pairs"])
    remodeling = pd.read_parquet(paths["remodeling"])
    generation = pd.read_parquet(paths["generation"])
    cross_structure = pd.read_parquet(paths["cross_structure"])
    structural = pd.read_parquet(paths["structural_asymmetry"])
    table = pairs.loc[pairs["high_comparability_eligible"].eq(True), [
        "protein_id", "canonical_sequence_length", "common_mask_fraction"
    ]].merge(remodeling[["protein_id", "magnitude_median"]], on="protein_id", how="left", validate="one_to_one")
    table = table.merge(
        generation[["protein_id", "js_burden_mean", "d_pa_excess_independent"]],
        on="protein_id", how="left", validate="one_to_one"
    )
    table = table.merge(cross_structure[["protein_id", "delta_cross"]], on="protein_id", how="left", validate="one_to_one")
    table = table.merge(
        structural[["protein_id", "afdb_generated_baseline_adjusted"]],
        on="protein_id", how="left", validate="one_to_one"
    )
    table["cohort_status"] = np.where(
        table["protein_id"].isin(gap_ids), "excluded_true_gap", "included_gap_free"
    )
    included = table.loc[table["cohort_status"] == "included_gap_free"]
    excluded = table.loc[table["cohort_status"] == "excluded_true_gap"]
    if len(included) != 62 or len(excluded) != 6:
        raise RuntimeError(f"unexpected gap exclusion counts: {len(included)}/{len(excluded)}")
    values: dict[str, object] = {}
    for label, column in (
        ("canonical_length", "canonical_sequence_length"),
        ("common_mask_fraction", "common_mask_fraction"),
        ("magnitude_median", "magnitude_median"),
        ("js_burden", "js_burden_mean"),
        ("d_pa_excess", "d_pa_excess_independent"),
        ("delta_cross", "delta_cross"),
        ("esmfold_effect", "afdb_generated_baseline_adjusted"),
    ):
        values[f"{label}_median_included"] = float(included[column].median())
        values[f"{label}_median_excluded"] = float(excluded[column].median())
    return table, values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--subset", type=int, default=None, help="deterministic prefix for validation")
    parser.add_argument("--cohort", choices=("primary", "gaps"), default="primary")
    parser.add_argument("--phase", choices=("local", "all"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    config = yaml.safe_load((project_root / "configs/models/esm_if1.yaml").read_text())
    source_root = project_root / str(config["source_root"])
    checkpoint_value = str(args.checkpoint) if args.checkpoint is not None else os.environ.get("ESM_IF1_CHECKPOINT_PATH")
    if not checkpoint_value:
        parser.error("--checkpoint or ESM_IF1_CHECKPOINT_PATH is required")
    checkpoint = Path(checkpoint_value)
    if args.cohort == "gaps" and args.subset is not None:
        parser.error("--subset is only supported for the primary cohort")
    output_root = args.output_root or (
        project_root
        / "experiments/interventions/pdb_afdb/models/esm_if1/cross_model_resume"
        / ("primary_gap_free" if args.cohort == "primary" else "gap_preserving_sensitivity")
        if args.subset is None
        else project_root / "experiments/interventions/pdb_afdb/models/esm_if1/validation_subset"
    )
    paths = _paths(project_root)
    pairs_for_gap_ids = pd.read_parquet(paths["pairs"])
    clean_gap_ids = pairs_for_gap_ids.loc[
        pairs_for_gap_ids["high_comparability_eligible"].eq(True), "protein_id"
    ]
    gap_ids = find_true_gap_proteins(pd.read_parquet(paths["masks"]), clean_gap_ids)
    if args.cohort == "primary":
        cases = build_frozen_common_cases(
            project_root,
            subset=args.subset,
            exclude_true_gap_proteins=True,
        )
        allow_missing_coordinates = False
        comparable_only = False
        expected_count = 62 if args.subset is None else args.subset
    else:
        cases = build_gap_preserving_cases(project_root, paths["pairs"], paths["masks"])
        allow_missing_coordinates = True
        comparable_only = True
        expected_count = 6
    adapter = load_esm_if1(
        source_root,
        checkpoint,
        expected_revision=str(config["implementation_revision"]),
        expected_checkpoint_sha256=str(config["checkpoint_sha256"]),
        device=args.device,
    )
    local_position = teacher_forced_local_response(
        adapter,
        cases,
        allow_missing_coordinates=allow_missing_coordinates,
        comparable_only=comparable_only,
    )
    local_protein = summarize_local_response(local_position)
    if len(local_protein) != expected_count:
        raise RuntimeError(f"unexpected ESM-IF1 cohort count: {len(local_protein)} != {expected_count}")
    output_root.mkdir(parents=True, exist_ok=True)
    hashes = {
        "local_response_positions.parquet": _immutable_table(
            output_root / "local_response_positions.parquet", local_position
        ),
        "local_response_proteins.parquet": _immutable_table(
            output_root / "local_response_proteins.parquet", local_protein
        ),
    }
    generation_protein: pd.DataFrame | None = None
    generation_position: pd.DataFrame | None = None
    if args.phase == "all":
        try:
            generated = generate_esm_if1_ensemble(
                adapter,
                cases,
                n_samples=args.n_samples,
                temperature=float(config["temperature"]),
                seed_namespace="esm_if1_independent_v1",
                allow_missing_coordinates=allow_missing_coordinates,
            )
            generation_protein, generation_position, generation_summary = summarize_generation_propagation(generated)
            convergence: dict[str, dict[str, object]] = {}
            for prefix in (32, 64, 128):
                prefix_records = generated.loc[generated["sample_index"] < prefix].copy()
                prefix_protein, _, prefix_summary = summarize_generation_propagation(prefix_records)
                convergence[str(prefix)] = {
                    "median_d_excess": float(prefix_protein["d_excess"].median()),
                    "positive_d_excess_proteins": int((prefix_protein["d_excess"] > 0).sum()),
                    "generated_record_count": int(prefix_summary["generated_record_count"]),
                }
            hashes.update(
                {
                    "generated_sequences.parquet": _immutable_table(output_root / "generated_sequences.parquet", generated),
                    "generation_proteins.parquet": _immutable_table(output_root / "generation_proteins.parquet", generation_protein),
                    "generation_position_js.parquet": _immutable_table(output_root / "generation_position_js.parquet", generation_position),
                }
            )
            generation_status = "complete"
        except (ESMIF1ContractError, ValueError) as exc:
            if args.cohort != "gaps":
                raise
            generation_summary = {}
            convergence = {}
            generation_status = "unavailable_model_native_missing_coordinate_sampling"
            generation_failure = str(exc)
    else:
        generation_summary = {}
        convergence = {}
        generation_status = "not_requested"
    if args.phase == "all" and generation_protein is None and args.cohort == "gaps":
        generation_failure = locals().get("generation_failure", "generation was not materialized")
    remodeling = pd.read_parquet(paths["remodeling"])
    local_comparison = local_protein.merge(
        remodeling, on="protein_id", how="inner", validate="one_to_one"
    )
    local_associations: dict[str, float | None] = {}
    for column in ("magnitude_median", "breadth_median", "reordering_rank_displacement_median"):
        valid = local_comparison[["esm_if1_local_burden", column]].dropna()
        local_associations[column] = (
            float(spearmanr(valid.iloc[:, 0], valid.iloc[:, 1]).statistic)
            if len(valid) >= 3
            else None
        )
    generation_associations: dict[str, float | None] = {}
    if generation_protein is not None:
        frozen_generation = pd.read_parquet(paths["generation"])
        frozen_cross = pd.read_parquet(paths["cross_structure"])
        generation_comparison = generation_protein.merge(
            frozen_generation, on="protein_id", how="inner", validate="one_to_one"
        )
        generation_comparison = generation_comparison.merge(
            frozen_cross[["protein_id", "delta_cross"]],
            on="protein_id",
            how="inner",
            validate="one_to_one",
        )
        for left, right in (
            ("d_excess", "d_pa_excess_independent"),
            ("js_burden", "js_burden_mean"),
            ("d_excess", "delta_cross"),
        ):
            valid = generation_comparison[[left, right]].dropna()
            generation_associations[right] = (
                float(spearmanr(valid[left], valid[right]).statistic)
                if len(valid) >= 3
                else None
            )
    exclusions: pd.DataFrame | None = None
    exclusion_summary: dict[str, object] = {}
    if args.cohort == "primary" and args.subset is None:
        exclusions, exclusion_summary = _build_exclusion_table(paths, gap_ids)
        hashes["exclusion_comparison.parquet"] = _immutable_table(
            output_root / "exclusion_comparison.parquet", exclusions
        )
    protocol = ESM_IF1_ANALYSIS_PROTOCOL if args.cohort == "primary" else TRUE_GAP_SENSITIVITY_PROTOCOL
    summary: dict[str, object] = {
        "analysis_protocol": protocol,
        "verdict": (
            "VALIDATION_SUBSET"
            if args.subset is not None
            else "PASS"
            if args.cohort == "primary" and args.phase == "all"
            else "LIMITED"
        ),
        "cohort_label": "62 gap-free clean proteins" if args.cohort == "primary" else "six gap-preserving true-gap proteins",
        "protein_unit": "protein",
        "protein_count": int(local_protein["protein_id"].nunique()),
        "requested_protein_count": expected_count,
        "excluded_protein_count": len(gap_ids) if args.cohort == "primary" else 0,
        "excluded_protein_ids": list(gap_ids) if args.cohort == "primary" else [],
        "position_count": int(local_position[["protein_id", "position"]].drop_duplicates().shape[0]),
        "model_name": adapter.binding().get("model_name"),
        "implementation_revision": adapter.binding().get("implementation_revision"),
        "checkpoint_sha256": adapter.binding().get("checkpoint_sha256"),
        "generated_record_count": int(generation_summary.get("generated_record_count", 0)),
        "generation": generation_summary,
        "generation_status": generation_status,
        "generation_failure": generation_failure if generation_status.startswith("unavailable") else None,
        "convergence": convergence,
        "local_proteinmpnn_spearman": local_associations,
        "generation_proteinmpnn_spearman": generation_associations,
        "exclusion_comparison": exclusion_summary,
        "artifacts": hashes,
    }
    summary_sha = _immutable_json(output_root / "summary.json", summary)
    manifest = {
        "analysis_protocol": protocol,
        "model_binding": adapter.binding(),
        "cohort": {"protein_count": summary["protein_count"], "subset": args.subset, "label": summary["cohort_label"]},
        "input_paths": {name: path.relative_to(project_root).as_posix() for name, path in paths.items()},
        "artifacts": hashes,
        "summary_sha256": summary_sha,
        "generation_temperature": float(config["temperature"]),
        "independent_samples_per_condition": args.n_samples,
        "seed_namespace": "esm_if1_independent_v1",
        "allow_missing_coordinates": allow_missing_coordinates,
        "excluded_true_gap_protein_ids": list(gap_ids),
        "generation_status": generation_status,
    }
    _immutable_json(output_root / "manifest.json", manifest)
    report_bytes = (_report(summary, local_protein, generation_protein, exclusions) + "\n").encode()
    report_path = output_root / "report.md"
    if report_path.exists():
        if report_path.read_bytes() != report_bytes:
            raise RuntimeError(f"immutable artifact conflict: {report_path}")
    else:
        atomic_write_new_bytes(report_path, report_bytes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
