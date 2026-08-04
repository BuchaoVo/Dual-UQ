"""Deterministic report projections from one structured derivation result."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from dual_uq.core.atomic_io import atomic_write_text

from ..models import DerivationRunResult


def _json_cell(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, frozenset):
        return sorted(_plain(item) for item in value)
    return value


def build_report(result: DerivationRunResult) -> dict[str, Any]:
    """Build the single compatibility mapping consumed by every renderer."""
    summary = _plain(result.summary)
    candidate_count_field = str(
        result.report_metadata.get("candidate_count_field", "candidate_count")
    )
    if candidate_count_field != "candidate_count":
        summary[candidate_count_field] = summary.pop("candidate_count")
    summary.update(_plain(result.report_metadata.get("summary_fields", {})))
    return {
        "schema_version": result.schema_version,
        "scope": _plain(result.scope),
        "preflight": _plain(result.preflight),
        "selection_policy": _plain(result.selection_policy),
        "summary": summary,
        "attrition_bias_probe": _plain(result.attrition_bias_probe),
        "candidates": [_plain(candidate.report_record) for candidate in result.candidates],
    }


def write_report_mapping(
    report: Mapping[str, Any],
    output_dir: Path,
    *,
    output_basename: str = "derive_pilot_v1",
    markdown_title: str = "Dataset-A DERIVE-PILOT v1",
    markdown_status: str = (
        "derivation/protocol validation only; not candidate admission."
    ),
) -> None:
    """Write deterministic JSON, candidate-level TSV and audit Markdown."""
    if re.fullmatch(r"[A-Za-z0-9_.-]+", output_basename) is None:
        raise ValueError("output_basename must be one portable filename stem")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{output_basename}.json"
    tsv_path = output_dir / f"{output_basename}.tsv"
    md_path = output_dir / f"{output_basename}.md"
    atomic_write_text(
        json_path,
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    candidates = sorted(
        report.get("candidates", []), key=lambda row: int(row["candidate_index"])
    )
    fieldnames = sorted(
        key for row in candidates for key in row if key != "candidate_index"
    ) + ["candidate_index"]
    stream = StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    for row in candidates:
        writer.writerow({key: _json_cell(row.get(key)) for key in fieldnames})
    atomic_write_text(tsv_path, stream.getvalue())
    summary = dict(report.get("summary", {}))
    lines = [
        f"# {markdown_title}",
        "",
        f"Status: {markdown_status}",
        "",
        "## Summary",
        "",
    ]
    lines.extend(
        f"- `{key}`: `{_json_cell(value)}`" for key, value in sorted(summary.items())
    )
    stage_fields = [
        "raw_complete",
        "canonical_sequence_complete",
        "pair_qc_complete",
        "mapping_complete",
        "fragment_resolved",
        "pae_bound",
        "confidence_bound",
        "mechanism_observable",
    ]
    lines.extend(["", "## Stage outcomes", ""])
    lines.extend(
        f"- `{field}`: `{sum(bool(row.get(field)) for row in candidates)}/{len(candidates)}`"
        for field in stage_fields
    )
    lines.extend(["", "## Candidate audit", ""])
    for row in candidates:
        lines.append(
            f"- Index {row['candidate_index']} `{row.get('pair_id')}`: "
            f"roles={_json_cell(row.get('selection_roles', []))}; "
            f"pair-QC={row.get('pair_qc_status', 'unknown')}; "
            f"mechanism={row.get('sampling_stratum_prior_recomputed', 'unknown')}; "
            f"primary failure={row.get('primary_failure_code') or 'none'}"
        )
    attrition = report.get("attrition_bias_probe")
    if isinstance(attrition, Mapping):
        lines.extend(["", "## Descriptive attrition probe", ""])
        lines.append(f"- Scope: {attrition.get('scope')}")
        for name, group in sorted(dict(attrition.get("groups", {})).items()):
            lines.append(f"- `{name}`: `{_json_cell(group)}`")
    atomic_write_text(md_path, "\n".join(lines) + "\n")


def write_reports(result: DerivationRunResult, output_dir: Path) -> None:
    """Render all formats from exactly one canonical report mapping."""
    write_report_mapping(
        build_report(result),
        output_dir,
        output_basename=str(
            result.report_metadata.get("output_basename", "derive_pilot_v1")
        ),
        markdown_title=str(
            result.report_metadata.get(
                "markdown_title", "Dataset-A DERIVE-PILOT v1"
            )
        ),
        markdown_status=str(
            result.report_metadata.get(
                "markdown_status",
                "derivation/protocol validation only; not candidate admission.",
            )
        ),
    )
