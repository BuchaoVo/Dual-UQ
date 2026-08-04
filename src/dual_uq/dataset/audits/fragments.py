"""Offline V5 empirical probe for AFDB fragmentation evidence.

This TASK-C analysis reads only already-local round-1, pair-QC, selected AFDB
metadata, and preflight evidence.  It does not query AFDB, select a different
model, infer offsets, stitch fragments, or change Dataset A eligibility.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Any

from dual_uq.core.atomic_io import atomic_write_text
from dual_uq.core.paths import ProjectPaths
from dual_uq.dataset.paths import DatasetPaths

PROJECT_PATHS = ProjectPaths.discover(anchor=Path(__file__))
PROJECT_ROOT = PROJECT_PATHS.repository_root
DATASET_PATHS = DatasetPaths.from_project(PROJECT_PATHS)
ROUND1_REPORT_PATH = DATASET_PATHS.reports / "census/round1_report.json"
PREFLIGHT_PATH = PROJECT_ROOT / "data/manifests/screening_pool_preflight.tsv"
PAIRS_ROOT = PROJECT_ROOT / "data/processed/pairs"
OUTPUT_PATH = DATASET_PATHS.reports / "census/v5_fragment_empirical_probe.json"

SCHEMA_VERSION = "dataset-a.v5-fragment-empirical-probe.v1"
UNRESOLVED_CONCLUSION = (
    "Current 26-protein cohort does not empirically identify the AFDB fragmentation "
    "threshold; PDR-01 §2.3 must retain the threshold as unresolved."
)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _prediction_interval(metadata: dict[str, Any]) -> tuple[int, int] | None:
    start = _optional_int(metadata.get("uniprotStart", metadata.get("sequenceStart")))
    end = _optional_int(metadata.get("uniprotEnd", metadata.get("sequenceEnd")))
    if start is None or end is None or start < 1 or end < start:
        return None
    return start, end


def _fragment_number(model_entity_id: str | None) -> int | None:
    if not model_entity_id:
        return None
    match = re.search(r"-F([1-9][0-9]*)$", model_entity_id)
    return int(match.group(1)) if match else None


def _parse_fragment_intervals(value: Any) -> list[list[int]]:
    if value is None or value == "":
        return []
    payload = value
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(payload, list):
        return []

    intervals: set[tuple[int, int]] = set()
    for item in payload:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        start = _optional_int(item[0])
        end = _optional_int(item[1])
        if start is not None and end is not None and start >= 1 and end >= start:
            intervals.add((start, end))
    return [list(interval) for interval in sorted(intervals)]


def build_fragment_observation(
    record: dict[str, Any],
    pair_qc: dict[str, Any],
    metadata: dict[str, Any] | None,
    *,
    preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one observation using only explicit, already-local evidence."""
    preflight = preflight or {}
    canonical_length = _optional_int(preflight.get("canonical_uniprot_length"))
    if canonical_length is not None:
        full_length = canonical_length
        full_length_source = "preflight.canonical_uniprot_length"
    else:
        full_length = _optional_int(record.get("uniprot_full_length"))
        full_length_source = (
            "round1_report.uniprot_full_length" if full_length is not None else None
        )

    base = {
        "protein_id": record.get("protein_id"),
        "screening_index": _optional_int(record.get("screening_index")),
        "pair_id": record.get("pair_id"),
        "uniprot_full_length": full_length,
        "uniprot_full_length_source": full_length_source,
        "selected_model_entity_id": None,
        "fragment_number": None,
        "fragment_start": None,
        "fragment_end": None,
        "fragment_length": None,
        "fragmentation_status": "unknown",
        "is_fragmented_protein": None,
        "fragment_count": None,
        "fragment_intervals": [],
        "mapped_uniprot_start": _optional_int(preflight.get("mapped_uniprot_start")),
        "mapped_uniprot_end": _optional_int(preflight.get("mapped_uniprot_end")),
        "afdb_coverage_status": preflight.get("afdb_coverage_status") or None,
        "evidence_note": "missing selected AFDB metadata",
    }
    if metadata is None:
        return base

    model_entity_id = str(
        metadata.get("modelEntityId")
        or metadata.get("entryId")
        or pair_qc.get("afdb_model_entity_id")
        or ""
    ) or None
    interval = _prediction_interval(metadata)
    explicit_intervals = _parse_fragment_intervals(preflight.get("afdb_fragment_intervals"))

    base["selected_model_entity_id"] = model_entity_id
    base["fragment_number"] = _fragment_number(model_entity_id)
    base["fragment_intervals"] = explicit_intervals
    if interval is not None:
        start, end = interval
        base["fragment_start"] = start
        base["fragment_end"] = end
        base["fragment_length"] = end - start + 1

    if len(explicit_intervals) > 1:
        base["fragmentation_status"] = "multi_fragment"
        base["is_fragmented_protein"] = True
        base["fragment_count"] = len(explicit_intervals)
        base["evidence_note"] = "multiple explicit preflight fragment intervals"
    elif (
        interval is not None
        and full_length is not None
        and interval == (1, full_length)
    ):
        base["fragmentation_status"] = "single_fragment"
        base["is_fragmented_protein"] = False
        base["evidence_note"] = "selected metadata interval explicitly covers full UniProt length"
    elif interval is None:
        base["evidence_note"] = "selected AFDB metadata lacks a valid explicit interval"
    else:
        base["evidence_note"] = (
            "selected interval is partial but total fragment inventory is not explicit"
        )
    return base


def _uniform_fragment_geometry(observations: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    lengths: list[int] = []
    steps: list[int] = []
    for observation in observations:
        if observation.get("fragmentation_status") != "multi_fragment":
            continue
        intervals = [tuple(item) for item in observation.get("fragment_intervals", [])]
        intervals = sorted((int(start), int(end)) for start, end in intervals)
        lengths.extend(end - start + 1 for start, end in intervals)
        steps.extend(current[0] - previous[0] for previous, current in pairwise(intervals))
    observed_length = lengths[0] if lengths and len(set(lengths)) == 1 else None
    observed_step = steps[0] if steps and len(set(steps)) == 1 else None
    return observed_length, observed_step


def summarize_observations(observations: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(item.get("fragmentation_status", "unknown") for item in observations)
    single_lengths = sorted(
        int(item["uniprot_full_length"])
        for item in observations
        if item.get("fragmentation_status") == "single_fragment"
        and item.get("uniprot_full_length") is not None
    )
    multi_lengths = sorted(
        int(item["uniprot_full_length"])
        for item in observations
        if item.get("fragmentation_status") == "multi_fragment"
        and item.get("uniprot_full_length") is not None
    )
    max_single = max(single_lengths, default=None)
    min_multi = min(multi_lengths, default=None)
    transition = None
    if max_single is not None and min_multi is not None and max_single < min_multi:
        transition = {
            "lower_exclusive": max_single,
            "upper_inclusive": min_multi,
            "notation": f"({max_single}, {min_multi}]",
        }

    enough_boundary_evidence = len(single_lengths) >= 2 and len(multi_lengths) >= 2
    threshold_identifiable = transition is not None and enough_boundary_evidence
    if not multi_lengths:
        conclusion = "No explicit multi-fragment protein is present; threshold is not identifiable."
    elif not single_lengths:
        conclusion = "No explicit single-fragment protein is present; threshold is not identifiable."
    elif transition is None:
        conclusion = (
            "Observed single- and multi-fragment protein lengths overlap; threshold is not identifiable."
        )
    elif not enough_boundary_evidence:
        conclusion = (
            f"An empirical interval {transition['notation']} is observed, but evidence near the "
            "boundary is insufficient because fewer than two proteins support one or both classes."
        )
    else:
        conclusion = (
            f"The current cohort constrains an empirical transition interval to "
            f"{transition['notation']}; this is not an AFDB official global threshold."
        )

    observed_length, observed_step = _uniform_fragment_geometry(observations)
    all_intervals = sorted(
        {
            (int(start), int(end))
            for item in observations
            for start, end in item.get("fragment_intervals", [])
        }
    )
    length_values = sorted({end - start + 1 for start, end in all_intervals})
    number_counts = Counter(
        "unknown" if item.get("fragment_number") is None else str(item["fragment_number"])
        for item in observations
    )
    return {
        "single_fragment_count": counts.get("single_fragment", 0),
        "multi_fragment_count": counts.get("multi_fragment", 0),
        "unknown_fragmentation_count": counts.get("unknown", 0),
        "max_full_length_single_fragment": max_single,
        "min_full_length_multi_fragment": min_multi,
        "threshold_identifiable": threshold_identifiable,
        "fragmentation_threshold": None,
        "empirical_transition_interval": transition,
        "fragment_length_values": length_values,
        "fragment_start_end_patterns": [list(interval) for interval in all_intervals],
        "fragment_number_distribution": dict(sorted(number_counts.items())),
        "observed_fragment_length": observed_length,
        "observed_step": observed_step,
        "fragment_geometry_note": (
            "uniform across explicit multi-fragment intervals"
            if observed_length is not None and observed_step is not None
            else "not identifiable from current cohort"
        ),
        "conclusion": conclusion,
    }


def _index9_regression(observations: list[dict[str, Any]]) -> dict[str, Any]:
    index9 = next((item for item in observations if item.get("screening_index") == 9), None)
    if index9 is None:
        return {
            "screening_index": 9,
            "mapped_interval": None,
            "selected_fragment_interval": None,
            "afdb_coverage_status": None,
            "regression_pass": False,
        }
    mapped = [index9.get("mapped_uniprot_start"), index9.get("mapped_uniprot_end")]
    selected = [index9.get("fragment_start"), index9.get("fragment_end")]
    status = index9.get("afdb_coverage_status")
    return {
        "screening_index": 9,
        "mapped_interval": mapped,
        "selected_fragment_interval": selected,
        "afdb_coverage_status": status,
        "regression_pass": (
            mapped == [1024, 1192]
            and selected == [1368, 1493]
            and status == "unsupported_afdb_fragment"
        ),
    }


def build_report(observations: list[dict[str, Any]], *, head: str) -> dict[str, Any]:
    ordered = sorted(
        observations,
        key=lambda item: (
            item.get("screening_index") is None,
            item.get("screening_index") or 0,
            str(item.get("protein_id") or ""),
        ),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {"head": head, "cohort_size": len(ordered)},
        "proteins": ordered,
        "summary": summarize_observations(ordered),
        "index9_regression": _index9_regression(ordered),
    }


def _load_preflight(path: Path) -> dict[int, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        return {
            int(row["screening_index"]): row
            for row in rows
            if _optional_int(row.get("screening_index")) is not None
        }


def load_local_observations(
    round1_report_path: Path,
    preflight_path: Path,
    pairs_root: Path,
) -> list[dict[str, Any]]:
    report = json.loads(round1_report_path.read_text(encoding="utf-8"))
    preflight = _load_preflight(preflight_path)
    observations: list[dict[str, Any]] = []
    for record in report["records"]:
        if record.get("outcome") == "skipped":
            continue
        pair_qc_path = pairs_root / str(record["pair_id"]) / "pair_qc.json"
        pair_qc = json.loads(pair_qc_path.read_text(encoding="utf-8"))
        model_path = Path(pair_qc["afdb_model_path"])
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
        metadata_path = model_path.with_name("metadata.json")
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.is_file()
            else None
        )
        observations.append(
            build_fragment_observation(
                record,
                pair_qc,
                metadata,
                preflight=preflight.get(int(record["screening_index"])),
            )
        )
    return observations


def _git_head(project_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    atomic_write_text(path, serialized)


def main() -> None:
    observations = load_local_observations(ROUND1_REPORT_PATH, PREFLIGHT_PATH, PAIRS_ROOT)
    payload = build_report(observations, head=_git_head(PROJECT_ROOT))
    if len(payload["proteins"]) != 26:
        raise RuntimeError(f"Expected 26 evaluated proteins, found {len(payload['proteins'])}")
    if not payload["index9_regression"]["regression_pass"]:
        raise RuntimeError("Index 9 frozen unsupported_afdb_fragment regression failed")
    if not payload["summary"]["threshold_identifiable"]:
        payload["summary"]["conclusion"] = UNRESOLVED_CONCLUSION
    _write_json_atomic(payload, OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH}")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
