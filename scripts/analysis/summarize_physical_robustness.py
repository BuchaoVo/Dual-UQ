"""Materialize compact EvoEF2 physical-compatibility analysis artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from dual_uq.evaluation.physical_robustness import (
    ESMFOLD_SAMPLE_INDICES,
    paired_sequence_effects,
    summarize_protein_effects,
)


def _rank_correlation(left: pd.Series, right: pd.Series) -> float | None:
    joined = pd.concat([left, right], axis=1).dropna()
    if len(joined) < 3:
        return None
    value = joined.iloc[:, 0].corr(joined.iloc[:, 1], method="spearman")
    return None if pd.isna(value) else float(value)


def _portable_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_evaluability_comparison(project_root: Path, screen_path: Path, output_root: Path) -> tuple[Path, dict[str, object]]:
    """Join frozen upstream descriptors to the protein-level evaluability status."""
    status = pd.read_csv(screen_path, sep="\t", usecols=["protein_id", "evoef2_status"])
    tables = {
        "pair_validity": (
            project_root / "experiments/p2_design_baseline/scale1b-v2/pair_validity/pair_validity.parquet",
            ["protein_id", "canonical_sequence_length", "common_mask_count_x", "common_mask_fraction", "paired_sequence_identity", "afdb_global_plddt_median", "afdb_global_pae_median"],
        ),
        "remodeling": (
            project_root / "experiments/dataset/analysis/inverse_folding_remodeling/protein_remodeling.parquet",
            ["protein_id", "magnitude_median", "position_magnitude_upper_tail_excess", "breadth_median", "reordering_rank_displacement_median"],
        ),
        "generation": (
            project_root / "experiments/dataset/analysis/generative_propagation/protein_generation_summary.parquet",
            ["protein_id", "js_burden_mean", "d_pa_excess_independent"],
        ),
        "cross_structure": (
            project_root / "experiments/dataset/analysis/cross_structure_compatibility/protein_cross_structure_summary.parquet",
            ["protein_id", "delta_cross"],
        ),
        "esmfold": (
            project_root / "experiments/dataset/analysis/independent_structure_validation/asymmetry/baseline_adjusted_protein_effects.parquet",
            ["protein_id", "pdb_generated_baseline_adjusted", "afdb_generated_baseline_adjusted"],
        ),
    }
    merged = status.copy()
    source_paths: dict[str, str] = {}
    for name, (path, columns) in tables.items():
        if not path.is_file():
            continue
        table = pd.read_parquet(path, columns=columns).drop_duplicates("protein_id")
        merged = merged.merge(table, on="protein_id", how="left", validate="one_to_one")
        source_paths[name] = _portable_path(path, project_root)
    output_path = output_root / "evaluable_vs_excluded.parquet"
    merged.to_parquet(output_path, index=False)
    numeric = [c for c in merged.columns if c not in {"protein_id", "evoef2_status"}]
    group_summary: dict[str, object] = {}
    for status_value, group in merged.groupby("evoef2_status", sort=True):
        group_summary[str(status_value)] = {
            "proteins": int(len(group)),
            "medians": {
                column: (float(group[column].median()) if group[column].notna().any() else None)
                for column in numeric
            },
        }
    included = int((status["evoef2_status"] == "evaluable").sum())
    excluded = int((status["evoef2_status"] != "evaluable").sum())
    group_summary["screen"] = {
        "screened_proteins": int(len(status)),
        "evaluable_proteins": included,
        "excluded_proteins": excluded,
        "excluded_status": "unavailable_build_incompatibility",
        "exclusion_policy": "exclude the entire protein when any required WT or generated case fails exact BuildMutant/RepairStructure screening",
    }
    return output_path, {"source_paths": source_paths, "group_summary": group_summary}


def _coverage(
    records: pd.DataFrame,
    expected_sample_indices: tuple[int, ...] = ESMFOLD_SAMPLE_INDICES,
) -> tuple[list[str], dict[str, dict[str, int]]]:
    expected = {
        (condition, sample, evaluated)
        for condition in ("PDB", "AFDB")
        for sample in expected_sample_indices
        for evaluated in ("PDB", "AFDB")
    }
    complete: list[str] = []
    details: dict[str, dict[str, int]] = {}
    for protein_id, group in records.groupby("protein_id", sort=True):
        success = group[group["status"] == "success"]
        generated = success[success["generated_condition"] != "WT"]
        keys = set(generated[["generated_condition", "sample_index", "evaluated_condition"]].itertuples(index=False, name=None))
        wt_keys = set(success.loc[success["generated_condition"] == "WT", "evaluated_condition"].astype(str))
        details[str(protein_id)] = {
            "successful_generated_keys": len(keys),
            "successful_wt_conditions": len(wt_keys & {"PDB", "AFDB"}),
            "failed_rows": int((group["status"] != "success").sum()),
        }
        if keys == expected and wt_keys == {"PDB", "AFDB"}:
            complete.append(str(protein_id))
    return complete, details


def build_outputs(
    records: pd.DataFrame,
    project_root: Path,
    output_root: Path,
    *,
    cohort: str = "clean_68",
    expected_sample_indices: tuple[int, ...] = ESMFOLD_SAMPLE_INDICES,
    input_manifest: Path | None = None,
    expected_proteins: int | None = None,
    evaluability_screen: Path | None = None,
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    records = records.copy()
    if "status" not in records.columns:
        records["status"] = "success"
    records.to_parquet(output_root / "scores.parquet", index=False)
    complete_proteins, coverage = _coverage(records, expected_sample_indices)
    effects = summarize_protein_effects(records)
    sequence_preferences = paired_sequence_effects(records)

    generated = records[records["generated_condition"] != "WT"]
    failed = records[records["status"] != "success"]
    success = records[records["status"] == "success"]
    effects.to_parquet(output_root / "protein_effects.parquet", index=False)
    sequence_preferences.to_parquet(output_root / "sequence_preferences.parquet", index=False)

    wt = effects["wt_preference"].dropna()
    comparisons: dict[str, object] = {}
    sources = {
        "esmfold": project_root / "experiments/dataset/analysis/independent_structure_validation/asymmetry/baseline_adjusted_protein_effects.parquet",
        "generative": project_root / "experiments/dataset/analysis/generative_propagation/protein_generation_summary.parquet",
        "cross_structure": project_root / "experiments/dataset/analysis/cross_structure_compatibility/protein_cross_structure_summary.parquet",
        "remodeling": project_root / "experiments/dataset/analysis/inverse_folding_remodeling/protein_remodeling.parquet",
    }
    evoef_effect = effects.set_index("protein_id")["baseline_adjusted_condition_difference"]
    for name, path in sources.items():
        if not path.is_file():
            comparisons[name] = {"status": "unavailable", "path": _portable_path(path, project_root)}
            continue
        table = pd.read_parquet(path).set_index("protein_id")
        candidate_columns = {
            "esmfold": "afdb_generated_baseline_adjusted",
            "generative": "d_pa_excess_independent",
            "cross_structure": "delta_cross",
            "remodeling": "position_magnitude_upper_tail_excess",
        }
        column = candidate_columns[name]
        if column not in table:
            comparisons[name] = {"status": "unavailable", "path": _portable_path(path, project_root)}
            continue
        comparisons[name] = {
            "status": "available",
            "path": _portable_path(path, project_root),
            "spearman_with_evoef2_condition_difference": _rank_correlation(
                evoef_effect, table[column]
            ),
        }
        if name == "generative" and "js_burden_mean" in table:
            comparisons["generative_js"] = {
                "status": "available",
                "path": _portable_path(path, project_root),
                "spearman_with_evoef2_condition_difference": _rank_correlation(
                    evoef_effect, table["js_burden_mean"]
                ),
            }

    protein_count = int(records["protein_id"].nunique())
    expected_count = expected_proteins if expected_proteins is not None else (4 if cohort == "pilot" else 68)
    analysis_complete = protein_count == expected_count and len(complete_proteins) == expected_count
    analysis_status = (
        "COMPLETE_EVALUABLE_SUBSET" if analysis_complete and cohort.startswith("evaluable_")
        else "COMPLETE" if analysis_complete
        else "BLOCKED_INCOMPLETE_PAIRED_COHORT"
    )
    comparison_path = None
    evaluability_comparison = None
    if evaluability_screen is not None and evaluability_screen.is_file():
        comparison_path, evaluability_comparison = _build_evaluability_comparison(
            project_root, evaluability_screen, output_root,
        )
    if evaluability_comparison is not None:
        screen_summary = evaluability_comparison["group_summary"].get("screen", {})
        evaluated = int(screen_summary.get("evaluable_proteins", len(complete_proteins)))
        screened = int(screen_summary.get("screened_proteins", protein_count))
        scientific_scope = (
            "LIMITED_SUPPLEMENTARY_ONLY"
            if evaluated < screened
            else "SUPPLEMENTARY_PHYSICAL_VALIDATION"
        )
    else:
        scientific_scope = "UNSCREENED_DIAGNOSTIC_ONLY"
    reproducibility_path = output_root / "reproducibility.json"
    reproducibility = None
    if reproducibility_path.is_file():
        reproducibility = json.loads(reproducibility_path.read_text(encoding="utf-8"))
    summary: dict[str, object] = {
        "schema": "evoef2_physical_compatibility_summary_v1",
        "cohort": cohort,
        "analysis_status": analysis_status,
        "scientific_scope_verdict": scientific_scope,
        "effect_ready_proteins": len(complete_proteins),
        "blocked_proteins": sorted(set(records["protein_id"].astype(str)) - set(complete_proteins)),
        "coverage": coverage,
        "stop_reason": None if analysis_complete else "Required BuildMutant, repair, mapping, or energy cases failed; no complete-cohort conclusion is released.",
        "proteins": protein_count,
        "wt_controls": int((records["generated_condition"] == "WT").groupby(records["protein_id"]).any().sum()),
        "generated_records": len(generated),
        "generated_sequences": int(generated["sequence_hash"].nunique()),
        "generated_sequences_by_condition": {
            str(key): {
                "records": len(group),
                "unique_sequence_hashes": int(group["sequence_hash"].nunique()),
            }
            for key, group in generated.groupby("generated_condition")
        },
        "evaluations": {
            "successful": len(success),
            "failed": len(failed),
            "nonfinite_total_energy": int((~pd.to_numeric(success["total_energy"], errors="coerce").notna()).sum()),
            "failure_types": {str(key): int(value) for key, value in failed["failure_type"].value_counts(dropna=False).items()},
        },
        "wt_baseline": {
            "preference_definition": "E_A_minus_E_P",
            "median": float(wt.median()) if not wt.empty else None,
            "q10": float(wt.quantile(0.10)) if not wt.empty else None,
            "q90": float(wt.quantile(0.90)) if not wt.empty else None,
            "positive_fraction": float((wt > 0).mean()) if not wt.empty else None,
            "negative_fraction": float((wt < 0).mean()) if not wt.empty else None,
        },
        "generated_effects": {
            "status": "complete" if analysis_complete else "diagnostic_partial",
            "protein_scope": "successful_complete_proteins_only",
            "proteins": int(effects["protein_id"].nunique()),
            "pdb_conditioned_median": float(effects["pdb_baseline_adjusted_preference_median"].median()),
            "afdb_conditioned_median": float(effects["afdb_baseline_adjusted_preference_median"].median()),
            "condition_difference_median": float(effects["baseline_adjusted_condition_difference"].median()),
            "pdb_positive_fraction": float((effects["pdb_baseline_adjusted_preference_median"] > 0).mean()),
            "afdb_positive_fraction": float((effects["afdb_baseline_adjusted_preference_median"] > 0).mean()),
        },
        "comparisons": comparisons,
        "reproducibility": reproducibility,
        "evaluable_vs_excluded": evaluability_comparison,
        "evaluator_scope": {
            "included_proteins": protein_count,
            "screened_proteins": 68 if evaluability_screen is not None else protein_count,
            "scope_statement": "EvoEF2 conclusions apply only to proteins passing exact all-case evaluability screening.",
        },
        "limitations": [
            "EvoEF2 energies are model-level compatibility readouts, not experimental stability.",
            "Protein is the primary inference unit; sequences are repeated measurements.",
            "No arbitrary energetic-validity threshold was applied.",
            "Partial effect summaries are diagnostic only when the 68-protein paired cohort is incomplete.",
            "The evaluability screen is outcome-blind and excludes a protein wholesale when any required case is not exactly representable.",
        ],
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = _render_report(summary, failed)
    (output_root / "report.md").write_text(report, encoding="utf-8")
    input_manifest = input_manifest or project_root / "runs/physical_validation/evoef2/cohort_inputs/input_manifest.json"
    executable = project_root / "third_party/EvoEF2/EvoEF2"
    release_manifest = {
        "schema": "evoef2_physical_compatibility_manifest_v1",
        "analysis_status": analysis_status,
        "scientific_scope_verdict": scientific_scope,
        "evaluator": "EvoEF2",
        "evaluator_scope": summary["evaluator_scope"],
        "source_url": "https://github.com/tommyhuangthu/EvoEF2.git",
        "source_revision": "38df01d305ed728ef067c3e0d22072058f33e255",
        "compiler": "g++ 11.4.0",
        "build_command": "g++ -O3 --fast-math -o EvoEF2 src/*.cpp",
        "validated_commands": ["RepairStructure", "BuildMutant", "ComputeStability"],
        "executable": _portable_path(executable, project_root),
        "executable_sha256": _sha256(executable) if executable.is_file() else None,
        "input_manifest": _portable_path(input_manifest, project_root),
        "input_manifest_sha256": _sha256(input_manifest) if input_manifest.is_file() else None,
        "evaluability_screen": _portable_path(evaluability_screen, project_root) if evaluability_screen else None,
        "evaluability_screen_sha256": _sha256(evaluability_screen) if evaluability_screen and evaluability_screen.is_file() else None,
        "cohort": cohort,
        "requested_proteins": protein_count,
        "effect_ready_proteins": len(complete_proteins),
        "score_rows": len(records),
        "successful_rows": len(success),
        "failed_rows": len(failed),
        "energy_preference_definition": "E_A_minus_E_P",
        "wt_baseline_adjustment": "generated_preference_median_minus_WT_preference",
        "artifacts": {
            name: _sha256(output_root / name)
            for name in (
                "scores.parquet", "sequence_preferences.parquet",
                "protein_effects.parquet", "summary.json", "report.md",
            )
        },
    }
    if comparison_path is not None:
        release_manifest["artifacts"]["evaluable_vs_excluded.parquet"] = _sha256(comparison_path)
    if reproducibility_path.is_file():
        release_manifest["artifacts"]["reproducibility.json"] = _sha256(reproducibility_path)
    for name in (
        "evaluability_screen.parquet",
        "evaluability_screen.tsv",
        "evaluability_screen_summary.json",
    ):
        path = output_root / name
        if path.is_file():
            release_manifest["artifacts"][name] = _sha256(path)
    (output_root / "manifest.json").write_text(
        json.dumps(release_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _render_report(summary: dict[str, object], failed: pd.DataFrame) -> str:
    wt = summary["wt_baseline"]
    generated = summary["generated_effects"]
    audit = summary.get("evaluable_vs_excluded") or {}
    audit_groups = audit.get("group_summary", {}) if isinstance(audit, dict) else {}
    included = audit_groups.get("evaluable", {})
    excluded = audit_groups.get("unavailable_build_incompatibility", {})
    included_medians = included.get("medians", {})
    excluded_medians = excluded.get("medians", {})
    audit_lines = [
        f"- Screened proteins: {audit_groups.get('screen', {}).get('screened_proteins', summary['proteins'])}; evaluable: {audit_groups.get('screen', {}).get('evaluable_proteins', summary['effect_ready_proteins'])}; excluded: {audit_groups.get('screen', {}).get('excluded_proteins', 0)}.",
        "- Exclusion is protein-level: any required WT or generated case failing exact BuildMutant/RepairStructure screening excludes the entire protein; no failed sequence was retained selectively.",
    ]
    for key, label in (
        ("canonical_sequence_length", "canonical length"),
        ("common_mask_count_x", "common-mask count"),
        ("common_mask_fraction", "common-mask fraction"),
        ("magnitude_median", "local remodeling magnitude"),
        ("js_burden_mean", "generative JS burden"),
        ("d_pa_excess_independent", "D_PA excess"),
        ("delta_cross", "Delta_cross"),
        ("pdb_generated_baseline_adjusted", "ESMFold PDB baseline-adjusted effect"),
        ("afdb_generated_baseline_adjusted", "ESMFold AFDB baseline-adjusted effect"),
    ):
        if key in included_medians or key in excluded_medians:
            audit_lines.append(
                f"- {label} median, included/excluded: {included_medians.get(key)} / {excluded_medians.get(key)}."
            )
    if included_medians and excluded_medians:
        audit_lines.append(
            "- Interpretation: the evaluable fraction is below the full 68-protein cohort and the included/excluded length and coverage distributions differ; this is a limited supplementary result, not a full-cohort physical validation."
        )
    lines = [
        "# EvoEF2 independent physical compatibility evaluation",
        "",
        "## VERDICT",
        f"**Status: {summary['analysis_status']}**",
        (f"Stop reason: {summary['stop_reason']}" if summary["stop_reason"] else ""),
        "",
        "EvoEF2 is used as an independent model-level sequence–structure energy readout.",
        "Absolute energy is not interpreted as experimental stability or biological fitness.",
        "",
        "## INPUT COVERAGE",
        f"- Proteins: {summary['proteins']}",
        f"- WT controls: {summary['wt_controls']}",
        f"- Generated evaluation records: {summary['generated_records']}",
        f"- Unique generated sequence hashes: {summary['generated_sequences']}",
        f"- Complete paired proteins: {summary['effect_ready_proteins']}/{summary['proteins']}",
        f"- EvoEF2 scope: {summary['evaluator_scope']['scope_statement']}",
        f"- Scientific scope verdict: **{summary['scientific_scope_verdict']}**",
        "",
        "## EVOEF2 EXECUTION",
        f"- Successful rows: {summary['evaluations']['successful']}; failed rows: {summary['evaluations']['failed']}",
        "- Generated cases use BuildMutant, then RepairStructure and ComputeStability; component terms retained.",
        f"- Reproducibility probe: {summary.get('reproducibility', 'not materialized')}",
        "",
        "## WT BASELINE",
        f"- Preference is `E_A - E_P`; median={wt['median']}, q10={wt['q10']}, q90={wt['q90']}",
        f"- Positive fraction={wt['positive_fraction']}; negative fraction={wt['negative_fraction']}",
        "",
        "## GENERATED EFFECTS",
        f"- PDB-conditioned median={generated['pdb_conditioned_median']}",
        f"- AFDB-conditioned median={generated['afdb_conditioned_median']}",
        f"- AFDB minus PDB condition-difference median={generated['condition_difference_median']}",
        "- These effects are WT-baseline-adjusted `E_A - E_P` compatibility contrasts.",
        "",
        "## QUESTIONS",
        f"- Q1: PDB-generated adjusted preference is positive in {generated['pdb_positive_fraction']:.3f} of included proteins; median={generated['pdb_conditioned_median']:.6g}.",
        f"- Q2: AFDB-generated adjusted preference median={generated['afdb_conditioned_median']:.6g}; positive fraction={generated['afdb_positive_fraction']:.3f}.",
        "- Q3: ESMFold concordance is descriptive only and is scoped to the included proteins.",
        f"- Q4: Between-protein heterogeneity is reported by q10/q90 and positive fractions above; included proteins={summary['effect_ready_proteins']}.",
        "- Q5: Associations with existing ESMFold, JS, D_PA, Delta_cross, and remodeling descriptors are descriptive and do not establish causality.",
        "",
        "## ESMFOLD COMPARISON",
        f"- Descriptive association: {summary['comparisons']['esmfold']}",
        "",
        "## PROTEINMPNN / GENERATIVE COMPARISON",
        f"- Generative comparison: {summary['comparisons']['generative']}",
        f"- Generative JS-burden comparison: {summary['comparisons'].get('generative_js', {'status': 'unavailable'})}",
        f"- Cross-structure comparison: {summary['comparisons']['cross_structure']}",
        f"- Local-remodeling comparison: {summary['comparisons'].get('remodeling', {'status': 'unavailable'})}",
        "Protein-level Spearman associations are descriptive and do not establish independent evidence.",
        "",
        "## EVALUABILITY BIAS AUDIT",
        *audit_lines,
        "",
        "## HETEROGENEITY",
        f"- Complete paired proteins: {summary['effect_ready_proteins']}/{summary['proteins']}; proteins are the inference units.",
        "- Absolute energies are not compared across unrelated proteins.",
        f"- Evaluability comparison: {'included/excluded descriptor audit materialized' if summary.get('evaluable_vs_excluded') else 'not requested'}.",
        "- Included versus excluded descriptor medians are recorded in `evaluable_vs_excluded.parquet` and the summary manifest; they are not used to reclassify proteins.",
        "- Any length/common-mask or response-descriptor imbalance limits generalization beyond the evaluable subset.",
        "",
        "## FAILURES / LIMITATIONS",
        f"- Failed evaluations retained in the score table: {len(failed)}",
        f"- Blocked proteins: {', '.join(summary['blocked_proteins']) or 'none'}",
        "- Cohort interpretation is stopped when any required BuildMutant, repair, mapping, or energy case fails.",
        "- No biological fitness, stability, or functional claim is made.",
        "",
        "## OUTPUTS",
        "- scores.parquet; sequence_preferences.parquet; protein_effects.parquet; summary.json; manifest.json; report.md",
        "",
        "## VALIDATION",
        "- Failed rows retain structured BuildMutant/repair/energy messages; successful outputs contain Total and component terms.",
        "",
        "## NEXT EXPERIMENT",
        "- Do not generalize this result to the full 68-protein cohort; the evaluable subset is the declared scope.",
        "- Any future coverage change requires an explicit EvoEF2 protocol decision; no fallback was applied here.",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("experiments/interventions/pdb_afdb/external_evaluation/evoef2"))
    parser.add_argument("--cohort", default="clean_68")
    parser.add_argument("--expected-proteins", type=int, default=None)
    parser.add_argument("--sample-index", action="append", type=int, default=None)
    parser.add_argument("--input-manifest", type=Path, default=None)
    parser.add_argument("--evaluability-screen", type=Path, default=None)
    args = parser.parse_args()
    files = sorted(args.records_dir.glob("*/energy_records.parquet"))
    direct = args.records_dir / "energy_records.parquet"
    if direct.is_file():
        files = [direct]
    if not files:
        raise SystemExit(f"no energy records found under {args.records_dir}")
    records = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    summary = build_outputs(
        records, args.project_root.resolve(), args.output,
        cohort=args.cohort,
        expected_sample_indices=tuple(args.sample_index) if args.sample_index else ESMFOLD_SAMPLE_INDICES,
        input_manifest=args.input_manifest.resolve() if args.input_manifest else None,
        expected_proteins=args.expected_proteins,
        evaluability_screen=args.evaluability_screen.resolve() if args.evaluability_screen else None,
    )
    print(json.dumps({"proteins": summary["proteins"], "evaluations": summary["evaluations"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
