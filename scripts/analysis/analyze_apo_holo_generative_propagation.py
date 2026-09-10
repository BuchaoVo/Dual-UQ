"""Materialize canonical APO/HOLO generative-propagation summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dual_uq.core.artifacts import parquet_bytes, write_immutable_bytes
from dual_uq.core.hashing import sha256_bytes as _sha256_bytes
from dual_uq.evaluation.apo_holo_generative_propagation import (
    GenerativePropagationResult,
    summarize_generation_propagation,
)
from dual_uq.inference.apo_holo_generation import load_apo_holo_generation_records


def _write_table(path: Path, table: pd.DataFrame) -> str:
    payload = parquet_bytes(table)
    write_immutable_bytes(path, payload)
    return _sha256_bytes(payload)


def _write_json(path: Path, payload: object) -> str:
    rendered = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    write_immutable_bytes(path, rendered)
    return _sha256_bytes(rendered)


def _quantiles(values: pd.Series) -> dict[str, float]:
    return {
        "median": float(values.median()),
        "q10": float(values.quantile(0.1, interpolation="linear")),
        "q90": float(values.quantile(0.9, interpolation="linear")),
    }


def _summary(model: str, result: GenerativePropagationResult, record_count: int) -> dict[str, object]:
    proteins = result.protein_summary
    final = result.convergence.loc[result.convergence["sample_count"].eq(64)]
    js = proteins["js_bits_mean_64"]
    excess = final["d_excess"]
    d_ah = final["d_ah"]
    return {
        "schema_version": "apo_holo_generative_propagation_summary_v1",
        "model": model,
        "temperature": 0.1,
        "sample_count_per_condition": 64,
        "nested_prefixes": [16, 32, 64],
        "protein_count": len(proteins),
        "generated_record_count": record_count,
        "position_row_count": len(result.position_shift),
        "d_ah_64": _quantiles(d_ah),
        "d_excess_64": _quantiles(excess),
        "d_excess_positive_proteins": int((excess > 0).sum()),
        "d_excess_positive_fraction": float((excess > 0).mean()),
        "generative_js_burden_bits_64": _quantiles(js),
        "interpretation": "Hamming distances and position JS describe generated-distribution separation within protein; they are not biological fitness measures.",
    }


def _report(summary: dict[str, object]) -> str:
    excess = summary["d_excess_64"]
    js = summary["generative_js_burden_bits_64"]
    return "\n".join(
        [
            "# APO/HOLO Generative Propagation",
            "",
            f"- Model: `{summary['model']}`",
            f"- Protein-level cohort: `{summary['protein_count']}`",
            f"- Generated records: `{summary['generated_record_count']}`",
            "- Conditions: independent APO and HOLO ensembles; 64 sequences per condition at temperature 0.1.",
            "- Nested checkpoints reuse the same 64-sequence ensembles: 16, 32, and 64.",
            "",
            "## Q1 — Between-structure divergence",
            "",
            f"- D_excess (D_AH − mean(D_APO, D_HOLO)) median `{excess['median']:.6f}`, q10 `{excess['q10']:.6f}`, q90 `{excess['q90']:.6f}`.",
            f"- Positive D_excess: `{summary['d_excess_positive_proteins']}/{summary['protein_count']}` proteins.",
            "- This is a within-protein Hamming-based divergence comparison, not a full joint-sequence distribution distance.",
            "",
            "## Q2 — Position-level generative shift",
            "",
            f"- Mean position-level Jensen–Shannon burden at 64 samples: median `{js['median']:.6f}` bits, q10 `{js['q10']:.6f}`, q90 `{js['q90']:.6f}`.",
            "- Position-level empirical distributions are reported in the canonical position table.",
            "",
            "## Q3 — Upstream descriptor propagation",
            "",
            "- Protein-level association with frozen local-response and structural descriptors is a downstream join; no new composite uncertainty metric is introduced.",
            "",
            "## Q4 — Cross-structure compatibility",
            "",
            "- Cross-structure compatibility scoring is a separate analysis layer and is not recomputed here.",
            "",
            "## Q5 — Heterogeneity",
            "",
            f"- Protein-level D_excess spans q10–q90 `{excess['q10']:.6f}`–`{excess['q90']:.6f}`; positive-fraction reporting preserves protein-level heterogeneity.",
            "",
            "Interpretation is limited to ProteinMPNN/ESM-IF1 generated-distribution separation under the frozen apo/holo protocol; no claims about fitness, stability, function, or causality are made.",
            "",
        ]
    )


def materialize_model_result(
    *,
    model: str,
    generation_root: Path,
    output_root: Path,
    project_root: Path,
) -> dict[str, object]:
    records = load_apo_holo_generation_records(generation_root)
    result = summarize_generation_propagation(records)
    output_root.mkdir(parents=True, exist_ok=True)
    hashes = {
        "protein_summary": _write_table(output_root / "protein_generation_summary.parquet", result.protein_summary),
        "position_shift": _write_table(output_root / "position_generation_shift.parquet", result.position_shift),
        "convergence": _write_table(output_root / "generation_convergence.parquet", result.convergence),
    }
    summary = _summary(model, result, len(records))
    hashes["summary"] = _write_json(output_root / "summary.json", summary)
    report = _report(summary).encode("utf-8")
    report_path = output_root / "report.md"
    write_immutable_bytes(report_path, report)
    hashes["report"] = _sha256_bytes(report)
    manifest = {
        "schema_version": "apo_holo_generative_propagation_manifest_v1",
        "model": model,
        "generation_root": str(generation_root.resolve().relative_to(project_root.resolve())),
        "output_root": str(output_root.resolve().relative_to(project_root.resolve())),
        "protocol": {
            "temperature": 0.1,
            "sample_count_per_condition": 64,
            "nested_prefixes": [16, 32, 64],
            "distance": "normalized Hamming",
            "position_divergence": "Jensen-Shannon bits over standard 20-AA empirical distributions",
            "statistical_unit": "protein",
        },
        "counts": {
            "protein_count": len(result.protein_summary),
            "position_row_count": len(result.position_shift),
            "record_count": len(records),
        },
        "artifact_sha256": hashes,
    }
    _write_json(output_root / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model", choices=("ProteinMPNN", "ESM-IF1"), required=True)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = materialize_model_result(
            model=args.model,
            generation_root=args.generation_root,
            output_root=args.output_root,
            project_root=args.project_root,
        )
        print(json.dumps({"status": "COMPLETE", "manifest": manifest}, indent=2, sort_keys=True))
        return 0
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "BLOCKED", "message": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
