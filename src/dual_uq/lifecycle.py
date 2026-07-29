from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

TERMINAL_SEGMENT_STATUSES = {"complete", "successful_no_segments"}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _legacy_failure(status: str, stage: str) -> str | None:
    normalized = status.replace(":", "_")
    if normalized == "unsupported_afdb_fragment":
        return normalized
    if normalized in {f"failed_{stage}", f"running_{stage}"}:
        return normalized
    return None


def _artifact_stage_status(
    *,
    artifact_exists: bool,
    stage: str,
    legacy_status: str,
    preflight_status: str,
) -> str:
    if artifact_exists:
        return "complete"
    failure = _legacy_failure(legacy_status, stage)
    if failure is not None:
        return failure
    if legacy_status == "skipped_preflight" or preflight_status == "fail_preflight":
        return "skipped_preflight"
    return "not_started"


def _segment_status(
    pair_dir: Path,
    *,
    robust_status: str,
    legacy_status: str,
    preflight_status: str,
) -> str:
    json_path = pair_dir / "segment_context.json"
    csv_path = pair_dir / "segment_context.csv"
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        return "complete" if payload else "successful_no_segments"
    if csv_path.exists():
        if not csv_path.read_text(encoding="utf-8").strip():
            return "successful_no_segments"
        table = pd.read_csv(csv_path)
        return "complete" if not table.empty else "successful_no_segments"

    segments_path = pair_dir / "disagreement_segments.csv"
    if robust_status == "complete" and segments_path.exists():
        if not segments_path.read_text(encoding="utf-8").strip():
            return "successful_no_segments"
        segments = pd.read_csv(segments_path)
        if segments.empty:
            return "successful_no_segments"

    return _artifact_stage_status(
        artifact_exists=False,
        stage="segment_context",
        legacy_status=legacy_status,
        preflight_status=preflight_status,
    )


def _counts(table: pd.DataFrame, column: str) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in table[column].fillna("missing").value_counts().sort_index().items()
    }


def build_candidate_lifecycle(
    *,
    pool: pd.DataFrame,
    preflight: pd.DataFrame,
    status: pd.DataFrame,
    pair_root: str | Path,
    candidate_summary: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join candidate-level QC and independently verified stage artifacts."""
    pair_root = Path(pair_root)
    base = pool.copy()
    base["screening_index"] = base["screening_index"].astype(int)

    preflight_columns = [
        "screening_index",
        "preflight_status",
        "preflight_reason",
        "full_length_mapping_coverage",
        "entity_mapping_coverage",
        "sequence_identity",
        "observed_ca_fraction_of_mapped",
    ]
    available_preflight = [column for column in preflight_columns if column in preflight.columns]
    preflight_join = preflight[available_preflight].copy()
    preflight_join["screening_index"] = preflight_join["screening_index"].astype(int)
    preflight_join = preflight_join.drop_duplicates("screening_index", keep="last")
    base = base.merge(preflight_join, on="screening_index", how="left")

    status_lookup: dict[int, str] = {}
    if not status.empty and {"screening_index", "status"}.issubset(status.columns):
        latest = status.copy()
        latest["screening_index"] = latest["screening_index"].astype(int)
        latest = latest.drop_duplicates("screening_index", keep="last")
        status_lookup = dict(zip(latest["screening_index"], latest["status"].astype(str)))

    summary_lookup: dict[str, dict[str, Any]] = {}
    if candidate_summary is not None and not candidate_summary.empty:
        summary_lookup = {
            str(row["pair_name"]): row
            for row in candidate_summary.to_dict("records")
        }

    rows: list[dict[str, Any]] = []
    for record in base.to_dict("records"):
        index = int(record["screening_index"])
        pdb_id = str(record["pdb_id"]).lower()
        chain_id = str(record["chain_id"])
        uniprot_id = str(record["uniprot_id"]).upper()
        pair_name = f"{pdb_id}_{chain_id}__{uniprot_id}"
        pair_dir = pair_root / pair_name
        preflight_status = str(record.get("preflight_status") or "not_run")
        legacy_status = status_lookup.get(index, "")

        pair_status = _artifact_stage_status(
            artifact_exists=(pair_dir / "pair_qc.json").exists(),
            stage="pair",
            legacy_status=legacy_status,
            preflight_status=preflight_status,
        )
        geometry_status = _artifact_stage_status(
            artifact_exists=(pair_dir / "pair_geometry_qc.json").exists(),
            stage="geometry",
            legacy_status=legacy_status,
            preflight_status=preflight_status,
        )
        robust_status = _artifact_stage_status(
            artifact_exists=(pair_dir / "robust_pair_diagnostics.json").exists(),
            stage="robust",
            legacy_status=legacy_status,
            preflight_status=preflight_status,
        )
        segment_context_status = _segment_status(
            pair_dir,
            robust_status=robust_status,
            legacy_status=legacy_status,
            preflight_status=preflight_status,
        )

        pair_qc = _read_json(pair_dir / "pair_qc.json")
        geometry_qc = _read_json(pair_dir / "pair_geometry_qc.json")
        complete_diagnostics = (
            pair_status == "complete"
            and geometry_status == "complete"
            and robust_status == "complete"
            and segment_context_status in TERMINAL_SEGMENT_STATUSES
        )
        quality_pass = bool(
            complete_diagnostics
            and preflight_status == "pass_full_length"
            and float(record.get("full_length_mapping_coverage") or 0.0) >= 0.90
            and float(record.get("sequence_identity") or 0.0) >= 0.95
            and float(record.get("observed_ca_fraction_of_mapped") or 0.0) >= 0.90
            and float(pair_qc.get("mapping_coverage") or 0.0) >= 0.90
            and float(pair_qc.get("sequence_identity") or 0.0) >= 0.95
            and pair_qc.get("quality_flag") != "fail"
            and float(geometry_qc.get("pdb_afdb_aa_match_fraction") or 0.0) >= 0.95
            and float(geometry_qc.get("mapped_ca_coverage") or 0.0) >= 0.90
        )

        summary = summary_lookup.get(pair_name, {})
        exclusion_reason = ""
        if preflight_status != "pass_full_length":
            exclusion_reason = str(record.get("preflight_reason") or preflight_status)
        elif not complete_diagnostics:
            incomplete = [
                f"{stage}={value}"
                for stage, value in (
                    ("pair", pair_status),
                    ("geometry", geometry_status),
                    ("robust", robust_status),
                    ("segment_context", segment_context_status),
                )
                if value not in {"complete", "successful_no_segments"}
            ]
            exclusion_reason = ";".join(incomplete)
        elif not quality_pass:
            exclusion_reason = "quality_threshold_failure"

        rows.append(
            {
                **record,
                "pair_name": pair_name,
                "pair_status": pair_status,
                "geometry_status": geometry_status,
                "robust_status": robust_status,
                "segment_context_status": segment_context_status,
                "complete_diagnostics": complete_diagnostics,
                "quality_pass": quality_pass,
                "candidate_tags": str(summary.get("candidate_tags") or ""),
                "primary_category": str(summary.get("primary_category") or "unclassified"),
                "exclusion_reason": exclusion_reason,
            }
        )

    lifecycle = pd.DataFrame(rows).sort_values("screening_index").reset_index(drop=True)
    audit = {
        "total_candidates": len(lifecycle),
        "preflight_status_counts": _counts(lifecycle, "preflight_status"),
        "pair_status_counts": _counts(lifecycle, "pair_status"),
        "geometry_status_counts": _counts(lifecycle, "geometry_status"),
        "robust_status_counts": _counts(lifecycle, "robust_status"),
        "segment_context_status_counts": _counts(lifecycle, "segment_context_status"),
        "complete_diagnostics": int(lifecycle["complete_diagnostics"].sum()),
        "quality_pass": int(lifecycle["quality_pass"].sum()),
        "primary_category_counts": _counts(lifecycle, "primary_category"),
    }
    return lifecycle, audit
