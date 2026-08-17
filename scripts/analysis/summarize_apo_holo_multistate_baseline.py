"""Render the canonical A--G multi-state APO/HOLO baseline summary."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import pandas as pd

from dual_uq.core.atomic_io import atomic_write_new_bytes
from dual_uq.core.hashing import sha256_file
from dual_uq.evaluation.multi_state_baseline_summary import build_summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--single-proteinmpnn-scores", type=Path, required=True)
    parser.add_argument("--single-esm-if1-scores", type=Path, required=True)
    parser.add_argument("--multi-proteinmpnn-scores", type=Path, required=True)
    parser.add_argument("--multi-esm-if1-scores", type=Path, required=True)
    parser.add_argument("--proteinmpnn-generation-root", type=Path, required=True)
    parser.add_argument("--esm-if1-generation-root", type=Path, required=True)
    parser.add_argument("--multi-generation-root", type=Path, required=True)
    parser.add_argument("--upstream-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def _path(project_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else project_root / value


def _load_single_ensembles(root: Path) -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = {}
    for path in sorted((root / "shards").glob("*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        condition = payload.get("condition", {})
        protein = str(condition["protein_id"])
        state = str(condition["state"])
        if state not in {"APO", "HOLO"}:
            raise ValueError(f"unexpected single-state condition: {state}")
        sequences = [str(record["sequence"]) for record in payload["records"]]
        if len(sequences) != 64:
            raise ValueError(f"{path} has {len(sequences)} sequences, expected 64")
        state_map = result.setdefault(protein, {})
        if state in state_map:
            raise ValueError(f"duplicate {state} shard for {protein}")
        state_map[state] = sequences
    return result


def _load_multi_ensembles(root: Path) -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = {}
    for path in sorted((root / "shards").glob("*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        protein = str(payload["protein_id"])
        sequences = [str(record["sequence"]) for record in payload["records"]]
        expected = int(payload.get("n_samples", 64))
        if len(sequences) != expected or expected != 64:
            raise ValueError(f"{path} has invalid multi-state sample count")
        if protein in result:
            raise ValueError(f"duplicate MULTI shard for {protein}")
        result[protein] = {"MULTI": sequences}
    return result


def _merge_ensembles(
    pnn: dict[str, dict[str, list[str]]],
    esm: dict[str, dict[str, list[str]]],
    multi: dict[str, dict[str, list[str]]],
) -> dict[str, dict[str, list[str]]]:
    proteins = set(pnn) & set(esm) & set(multi)
    if not proteins:
        raise ValueError("no shared generated proteins")
    if not set(multi) <= set(pnn) or not set(multi) <= set(esm):
        raise ValueError("multi-state cohort is not contained in both single-state cohorts")
    return {
        protein: {**pnn[protein], **multi[protein]}
        for protein in sorted(proteins)
    }


def _render_report(summary: dict[str, Any], table: pd.DataFrame) -> str:
    excess = table["divergence_apo_holo"] - (
        table["diversity_apo"] + table["diversity_holo"]
    ) / 2.0
    lines_by_model: list[str] = []
    for prefix, label in (("proteinmpnn", "ProteinMPNN"), ("esm_if1", "ESM-IF1")):
        delta = table[f"{prefix}_multi_minus_best_single_worst"]
        lines_by_model.append(
            f"- **{label}.** MULTI median worst compatibility = "
            f"{table[f'{prefix}_multi_worst_compat'].median():.4f}; "
            f"MULTI−best-single median = {delta.median():.4f} "
            f"(positive in {(delta > 0).mean():.3f} of proteins); "
            f"MULTI state-gap median = {table[f'{prefix}_multi_state_gap'].median():.4f}."
        )
    association_lines = []
    for evaluator in summary["evaluators"]:
        rows = [row for row in summary["associations"] if row["evaluator"] == evaluator]
        if rows:
            best = max(rows, key=lambda row: abs(row["spearman_rho"]))
            association_lines.append(
                f"- **{evaluator}.** strongest recorded descriptor association: "
                f"{best['descriptor']} (Spearman rho = {best['spearman_rho']:.4f}, "
                f"n = {best['n_proteins']})."
            )
    lines = [
        "# Multi-state APO/HOLO baseline summary",
        "",
        f"Protein cohort: **{summary['protein_count']}** shared proteins.",
        "",
        "The protein is the inference unit. Sequence diversity is a marginal "
        "Hamming summary and is not a joint-distribution or biological replicate claim.",
        "",
        "## A–G result inventory",
        "",
        "- **A/E.** All model-specific rows are WT-normalized APO/HOLO compatibility "
        "endpoints with a MULTI equal-weight joint-decoder condition.",
        "- **B/D.** MULTI is compared with the separate APO and HOLO ensembles; "
        "the protein table retains median, q10 and q90 endpoint summaries.",
        "- **F.** `worst_compat` is the primary condition-comparison endpoint; "
        "mean compatibility and state gap are secondary.",
        "- **G.** Associations are Spearman summaries against frozen upstream "
        "generative/local-response descriptors; no sequence rows are treated as "
        "independent proteins.",
        "",
        "## Numeric answers",
        "",
        f"- **Q1.** APO-vs-HOLO marginal between-condition divergence minus the "
        f"within-condition mean has median **{excess.median():.4f}** and is positive "
        f"for **{(excess > 0).mean():.3f}** of proteins. This is a marginal "
        "Hamming result, not a full joint-sequence distribution claim.",
        "- **Q2/Q3.** The MULTI condition is retained as the equal-weight joint "
        "decoder; upstream local-response/generative descriptors are joined without "
        "redefining them. Descriptor associations are reported below.",
        *lines_by_model,
        *association_lines,
        "- **Q4.** The state-gap and worst-compatibility columns retain both "
        "cross-state compatibility behavior and its heterogeneity; interpretation "
        "is restricted to model-level compatibility.",
        "- **Q5.** Heterogeneity is represented by protein-level q10/q90 columns "
        "for every endpoint and by the sign fractions above; no sequence-level "
        "independence assumption is used.",
        "",
        "## Materialized columns",
        "",
        f"- Protein summary rows: **{len(table)}**",
        f"- Protein summary columns: **{len(table.columns)}**",
        "- Full interpretation remains limited to the frozen shared APO/HOLO cohort "
        "and the ProteinMPNN/ESM-IF1 model semantics.",
        "",
        "Absolute scores are model-level compatibility quantities; no claim about "
        "fitness, stability, function, or experimental outcome is made.",
        "",
    ]
    return "\n".join(lines)


def _write_immutable(path: Path, data: bytes) -> None:
    try:
        atomic_write_new_bytes(path, data)
    except FileExistsError:
        if path.read_bytes() != data:
            raise ValueError(f"immutable output conflict: {path}") from None


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    stream = io.BytesIO()
    frame.to_parquet(stream, index=False)
    return stream.getvalue()


def run(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(args.project_root).resolve()
    input_paths = {
        key: _path(project_root, value).resolve()
        for key, value in vars(args).items()
        if key.endswith("scores") or key.endswith("summary")
    }
    multi_generation_manifest = _path(project_root, args.multi_generation_root).resolve() / "manifest.json"
    if multi_generation_manifest.is_file():
        input_paths["multi_generation_manifest"] = multi_generation_manifest
    single_pnn = pd.read_parquet(input_paths["single_proteinmpnn_scores"])
    single_esm = pd.read_parquet(input_paths["single_esm_if1_scores"])
    multi_pnn = pd.read_parquet(input_paths["multi_proteinmpnn_scores"])
    multi_esm = pd.read_parquet(input_paths["multi_esm_if1_scores"])
    upstream = pd.read_parquet(input_paths["upstream_summary"])
    ensembles = _merge_ensembles(
        _load_single_ensembles(_path(project_root, args.proteinmpnn_generation_root).resolve()),
        _load_single_ensembles(_path(project_root, args.esm_if1_generation_root).resolve()),
        _load_multi_ensembles(_path(project_root, args.multi_generation_root).resolve()),
    )
    result = build_summary(
        single_tables={"ProteinMPNN": single_pnn, "ESM-IF1": single_esm},
        multi_tables={"ProteinMPNN": multi_pnn, "ESM-IF1": multi_esm},
        sequence_ensembles=ensembles,
        upstream=upstream,
    )
    output_root = _path(project_root, args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    table_path = output_root / "protein_summary.parquet"
    association_path = output_root / "associations.parquet"
    _write_immutable(table_path, _parquet_bytes(result.protein_summary))
    _write_immutable(association_path, _parquet_bytes(result.associations))
    summary = {
        **result.summary,
        "input_protein_count": {
            "ProteinMPNN_single": int(single_pnn.protein_id.nunique()),
            "ESM_IF1_single": int(single_esm.protein_id.nunique()),
            "ProteinMPNN_multi": int(multi_pnn.protein_id.nunique()),
            "ESM_IF1_multi": int(multi_esm.protein_id.nunique()),
        },
        "output_rows": int(len(result.protein_summary)),
        "output_columns": int(len(result.protein_summary.columns)),
        "q1_divergence_excess_median": float(
            (
                result.protein_summary["divergence_apo_holo"]
                - (result.protein_summary["diversity_apo"] + result.protein_summary["diversity_holo"]) / 2.0
            ).median()
        ),
        "q1_divergence_excess_positive_fraction": float(
            (
                result.protein_summary["divergence_apo_holo"]
                > (result.protein_summary["diversity_apo"] + result.protein_summary["diversity_holo"]) / 2.0
            ).mean()
        ),
        "model_level_endpoint_medians": {
            prefix: {
                "multi_worst_compat": float(result.protein_summary[f"{prefix}_multi_worst_compat"].median()),
                "multi_minus_best_single_worst": float(result.protein_summary[f"{prefix}_multi_minus_best_single_worst"].median()),
                "multi_state_gap": float(result.protein_summary[f"{prefix}_multi_state_gap"].median()),
            }
            for prefix in ("proteinmpnn", "esm_if1")
        },
    }
    summary_bytes = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode()
    manifest = {
        "schema_version": "multi_state_baseline_summary_manifest_v1",
        "inputs": {
            str(path.relative_to(project_root)): {"sha256": sha256_file(path), "size": path.stat().st_size}
            for path in input_paths.values()
        },
        "outputs": {
            "protein_summary.parquet": {"sha256": sha256_file(table_path), "rows": len(result.protein_summary)},
            "associations.parquet": {"sha256": sha256_file(association_path), "rows": len(result.associations)},
        },
        "primary_unit": "protein",
        "model_semantics": ["ProteinMPNN", "ESM-IF1"],
        "no_biological_interpretation": True,
    }
    _write_immutable(output_root / "summary.json", summary_bytes)
    _write_immutable(output_root / "manifest.json", (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    _write_immutable(output_root / "report.md", _render_report(summary, result.protein_summary).encode())
    return summary


if __name__ == "__main__":
    run(parse_args())
