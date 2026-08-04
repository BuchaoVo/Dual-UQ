"""Complete the 15 H6-authorized AFDB assets resolved by ACQ-4."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from dual_uq.core.atomic_io import atomic_write_text
from dual_uq.core.paths import ProjectPaths

from . import acquisition as _BATCH1

ACQ1_LEDGER_PATH = Path(
    "artifacts/dataset/reports/acquisition/batch1_acquisition_run_v1.json"
)
ACQ4_REPORT_PATH = Path(
    "artifacts/dataset/audits/acquisition/afdb_metadata_record_resolution_v1.json"
)
OUTPUT_DIR = Path("artifacts/dataset/reports/acquisition")

TARGETS = {
    7: ("2wfi_A__Q13427", "Q13427"),
    24: ("7a5m_A__Q8N8S7", "Q8N8S7"),
    38: ("3ui4_A__Q9Y237", "Q9Y237"),
    103: ("5emb_A__Q8K4V4", "Q8K4V4"),
    208: ("6fmc_A__O14786", "O14786"),
}
DEPENDENT_ASSET_TYPES = ("afdb_structure", "afdb_pae", "afdb_confidence")
ASSET_STATUS_ORDER = {asset: position for position, asset in enumerate(DEPENDENT_ASSET_TYPES)}


class ACQ5Error(RuntimeError):
    """The ACQ-5 manifest/exact-record gate is not satisfied."""


def _rows_by_index(document: Mapping[str, Any], label: str) -> dict[int, dict[str, Any]]:
    rows = [dict(row) for row in document.get("records", [])]
    output = {int(row["candidate_index"]): row for row in rows}
    if len(output) != len(rows):
        raise ACQ5Error(f"{label} contains duplicate candidate indices")
    return output


def _validate_target_row(row: Mapping[str, Any], index: int) -> None:
    expected_pair, expected_accession = TARGETS[index]
    if str(row.get("pair_id")) != expected_pair:
        raise ACQ5Error(f"Candidate {index} pair identity changed")
    if str(row.get("UniProt")) != expected_accession:
        raise ACQ5Error(f"Candidate {index} accession identity changed")
    required = set(row.get("required_external_assets", []))
    if not set(DEPENDENT_ASSET_TYPES).issubset(required):
        raise ACQ5Error(f"Candidate {index} dependent authorization changed")


def build_acq5_specs(
    plan: Mapping[str, Any], resolution: Mapping[str, Any], project_root: Path
) -> list[Any]:
    """Build only the 15 dependent specs from unique exact metadata records."""
    plan_rows = _rows_by_index(plan, "Batch-1 plan")
    if len(plan_rows) != 48:
        raise ACQ5Error("Authorized Batch-1 candidate count is not 48")
    resolution_rows = _rows_by_index(resolution, "ACQ-4 resolution report")
    root = project_root.resolve()
    specs: list[Any] = []
    for index, (expected_pair, expected_accession) in TARGETS.items():
        if index not in plan_rows or index not in resolution_rows:
            raise ACQ5Error(f"Target candidate {index} is absent from a bound input")
        plan_row = plan_rows[index]
        audit_row = resolution_rows[index]
        _validate_target_row(plan_row, index)
        if (
            str(audit_row.get("pair_id")) != expected_pair
            or str(audit_row.get("candidate_accession")) != expected_accession
        ):
            raise ACQ5Error(f"Candidate {index} identity differs in ACQ-4 evidence")
        if (
            int(audit_row.get("exact_accession_record_count", 0)) != 1
            or audit_row.get("resolution_status") != "exact_record_resolved"
        ):
            raise ACQ5Error(f"Candidate {index} does not have one resolved exact record")
        payload_path = root / str(audit_row["payload_path"])
        try:
            payload = payload_path.read_bytes()
        except OSError as exc:
            raise ACQ5Error(f"Candidate {index} exact metadata payload is unavailable") from exc
        if _BATCH1.sha256_bytes(payload) != str(audit_row.get("payload_SHA256")):
            raise ACQ5Error(f"Candidate {index} exact metadata payload SHA256 changed")
        try:
            metadata = _BATCH1.validate_payload(
                "afdb_metadata", payload, expected_accession, None
            )
        except _BATCH1.AcquisitionError as exc:
            raise ACQ5Error(
                f"Candidate {index} exact metadata record no longer validates"
            ) from exc
        if metadata is None or metadata.get("uniprotAccession") != expected_accession:
            raise ACQ5Error(f"Candidate {index} exact metadata identity changed")
        if int(metadata.get("_exact_accession_record_count", 0)) != 1:
            raise ACQ5Error(f"Candidate {index} does not have one resolved exact record")
        for asset_type in DEPENDENT_ASSET_TYPES:
            try:
                spec = _BATCH1._dependent_spec(root, plan_row, asset_type, metadata)
            except _BATCH1.AcquisitionError as exc:
                raise ACQ5Error(
                    f"Candidate {index} exact record lacks authorized {asset_type}"
                ) from exc
            specs.append(spec)
    if len(specs) != 15:
        raise ACQ5Error("ACQ-5 dependent asset scope is not exactly 15")
    return specs


def verify_acq5_gate(project_root: Path) -> tuple[dict[str, Any], dict[str, Any], list[Any]]:
    """Validate every authorized binding before any network request is possible."""
    root = project_root.resolve()
    try:
        plan = _BATCH1.verify_repository_bindings(root)
    except (OSError, json.JSONDecodeError, _BATCH1.AcquisitionError) as exc:
        raise ACQ5Error("Authorized Batch-1 manifest binding failed") from exc
    try:
        resolution = json.loads((root / ACQ4_REPORT_PATH).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ACQ5Error("ACQ-4 resolution report is unavailable") from exc
    summary = resolution.get("summary", {})
    if (
        int(summary.get("payload_count", 0)) != 38
        or summary.get("exact_accession_record_count_distribution") != {"1": 38}
        or int(summary.get("true_identity_mismatch_count", -1)) != 0
        or int(summary.get("ambiguous_exact_record_set_count", -1)) != 0
    ):
        raise ACQ5Error("ACQ-4 collection-level identity invariants changed")
    source_head = resolution.get("source_bindings", {}).get("repository_HEAD")
    if source_head != _BATCH1.EXPECTED_COMMIT:
        raise ACQ5Error("ACQ-4 repository binding changed")
    specs = build_acq5_specs(plan, resolution, root)
    return plan, resolution, specs


def acquire_dependents(
    specs: list[Any],
    *,
    transport: Callable[[str, float], Any] = _BATCH1.default_transport,
    sleeper: Callable[[float], None] = _BATCH1.time.sleep,
    clock: Callable[[], str] = _BATCH1.utc_now,
    max_attempts: int = 3,
) -> list[dict[str, Any]]:
    """Acquire or reuse the already resolved dependent assets only."""
    records = []
    for spec in specs:
        record = _BATCH1.acquire_asset(
            spec,
            transport=transport,
            sleeper=sleeper,
            clock=clock,
            max_attempts=max_attempts,
        )
        metadata = spec.metadata_record or {}
        record.update(
            candidate_accession=metadata.get("uniprotAccession"),
            exact_record_model_identity=spec.expected_identity,
            exact_accession_record_count=metadata.get(
                "_exact_accession_record_count"
            ),
            metadata_identity_resolution=metadata.get(
                "_identity_resolution_status"
            ),
            nonselected_sibling_record_count=metadata.get(
                "_nonexact_record_count"
            ),
        )
        records.append(record)
    records.sort(
        key=lambda row: (
            int(row["candidate_index"]),
            ASSET_STATUS_ORDER[str(row["asset_type"])],
        )
    )
    return records


def _successful(record: Mapping[str, Any] | None) -> bool:
    return record is not None and record.get("status") in {
        "reused_valid",
        "downloaded_new",
    }


def build_current_state(
    plan: Mapping[str, Any],
    acq1: Mapping[str, Any],
    resolution: Mapping[str, Any],
    acq5_records: list[dict[str, Any]],
    *,
    source_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Build current accounting while retaining the historical ACQ-1 failure."""
    plan_rows = _rows_by_index(plan, "Batch-1 plan")
    if len(plan_rows) != 48:
        raise ACQ5Error("Current-state accounting requires exactly 48 candidates")
    resolution_rows = _rows_by_index(resolution, "ACQ-4 resolution report")
    historical_completeness = {
        int(row["candidate_index"]): row.get("completeness")
        for row in acq1.get("candidate_completeness", [])
    }
    historical_records = {
        (int(row["candidate_index"]), str(row["asset_type"])): row
        for row in acq1.get("records", [])
    }
    dependent_records = {
        (int(row["candidate_index"]), str(row["asset_type"])): row
        for row in acq5_records
    }
    if len(dependent_records) != len(acq5_records):
        raise ACQ5Error("ACQ-5 dependent records contain duplicates")

    candidate_states: list[dict[str, Any]] = []
    supersession: list[dict[str, Any]] = []
    for index in sorted(plan_rows):
        row = plan_rows[index]
        accession = str(row["UniProt"])
        historical_complete = (
            historical_completeness.get(index) == "fully_raw_complete"
        )
        if index in TARGETS:
            audit = resolution_rows.get(index, {})
            identity_resolved = (
                audit.get("resolution_status") == "exact_record_resolved"
                and int(audit.get("exact_accession_record_count", 0)) == 1
            )
            asset_complete = {
                asset: _successful(dependent_records.get((index, asset)))
                for asset in DEPENDENT_ASSET_TYPES
            }
            other_required_complete = all(
                _successful(historical_records.get((index, asset)))
                for asset in row.get("required_external_assets", [])
                if asset not in {"afdb_metadata", *DEPENDENT_ASSET_TYPES}
            )
            raw_complete = (
                identity_resolved
                and other_required_complete
                and all(asset_complete.values())
            )
            supersession.append(
                {
                    "candidate_index": index,
                    "pair_id": row["pair_id"],
                    "historical_failure": "container_level_identity_validation",
                    "historical_failure_code": "identity_mismatch",
                    "historical_dependent_asset_missing": 3,
                    "superseded_by": "ACQ-4",
                    "current_interpretation": "metadata_collection_semantics_artifact",
                    "current_resolution": "exact_record_resolved",
                    "candidate_identity_failure_current": False,
                }
            )
        else:
            identity_resolved = historical_complete
            asset_complete = {asset: historical_complete for asset in DEPENDENT_ASSET_TYPES}
            raw_complete = historical_complete
        candidate_states.append(
            {
                "candidate_index": index,
                "polymer_entity_id": row["polymer_entity_id"],
                "pair_id": row["pair_id"],
                "candidate_accession": accession,
                "metadata_transport_complete": identity_resolved,
                "metadata_parse_complete": identity_resolved,
                "record_level_identity_resolved": identity_resolved,
                "structure_complete": asset_complete["afdb_structure"],
                "PAE_complete": asset_complete["afdb_pae"],
                "confidence_complete": asset_complete["afdb_confidence"],
                "candidate_raw_complete": raw_complete,
                "candidate_identity_failure_current": not identity_resolved,
            }
        )

    ordered_records = sorted(
        (dict(row) for row in acq5_records),
        key=lambda row: (
            int(row["candidate_index"]),
            ASSET_STATUS_ORDER[str(row["asset_type"])],
        ),
    )
    summary = {
        "candidate_count": len(candidate_states),
        "metadata_transport_complete": sum(
            row["metadata_transport_complete"] for row in candidate_states
        ),
        "metadata_parse_complete": sum(
            row["metadata_parse_complete"] for row in candidate_states
        ),
        "record_level_identity_resolved": sum(
            row["record_level_identity_resolved"] for row in candidate_states
        ),
        "structure_complete": sum(row["structure_complete"] for row in candidate_states),
        "PAE_complete": sum(row["PAE_complete"] for row in candidate_states),
        "confidence_complete": sum(
            row["confidence_complete"] for row in candidate_states
        ),
        "candidate_raw_complete": sum(
            row["candidate_raw_complete"] for row in candidate_states
        ),
        "current_candidate_identity_failures": sum(
            row["candidate_identity_failure_current"] for row in candidate_states
        ),
        "planned_dependent_assets": len(ordered_records),
        "reused_dependent_assets": sum(
            row.get("status") == "reused_valid" for row in ordered_records
        ),
        "downloaded_dependent_assets": sum(
            row.get("status") == "downloaded_new" for row in ordered_records
        ),
        "failed_dependent_assets": sum(
            row.get("status") == "failed" for row in ordered_records
        ),
        "historical_container_identity_failures": len(supersession),
        "historical_dependent_asset_missing": sum(
            int(row["historical_dependent_asset_missing"])
            for row in supersession
        ),
    }
    return {
        "schema_version": "dataset-a.batch1-acquisition-current-state.v2",
        "acquisition_plan_commit": _BATCH1.EXPECTED_COMMIT,
        "source_bindings": dict(source_bindings),
        "scope": {
            "target_candidate_indices": list(TARGETS),
            "dependent_asset_types": list(DEPENDENT_ASSET_TYPES),
            "authorized_dependent_asset_count": 15,
            "alternate_accession_policy": "forbidden",
            "metadata_url_rediscovery": "not_performed",
            "fragment_resolution": "not_performed",
        },
        "summary": summary,
        "candidate_states": candidate_states,
        "acq5_records": ordered_records,
        "supersession": supersession,
        "missingness_interpretation": {
            "status": "superseded_by_acq4_schema_artifact_finding",
            "historical_former_failure_median_known_length": 646.5,
            "historical_former_failure_median_global_plddt": 78.62,
            "historical_raw_complete_median_known_length": 251,
            "historical_raw_complete_median_global_plddt": 94.12,
            "interpretation": (
                "The concentration cannot support candidate-level acquisition-missingness "
                "inference because ACQ-4 identified false attrition from container-level "
                "metadata validation."
            ),
            "biological_inference": "not_performed",
        },
    }


def write_outputs(report: Mapping[str, Any], output_dir: Path) -> None:
    """Write deterministic current-state JSON, TSV, and Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "batch1_acquisition_current_state_v2.json"
    tsv_path = output_dir / "batch1_acquisition_current_state_v2.tsv"
    md_path = output_dir / "batch1_acquisition_current_state_v2.md"
    atomic_write_text(json_path, json.dumps(report, indent=2, sort_keys=True) + "\n")

    fields = [
        "candidate_index",
        "polymer_entity_id",
        "pair_id",
        "candidate_accession",
        "metadata_transport_complete",
        "metadata_parse_complete",
        "record_level_identity_resolved",
        "structure_complete",
        "PAE_complete",
        "confidence_complete",
        "candidate_raw_complete",
        "candidate_identity_failure_current",
    ]
    stream = StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=fields,
        delimiter="\t",
        lineterminator="\n",
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(report["candidate_states"])
    atomic_write_text(tsv_path, stream.getvalue())

    summary = report["summary"]
    lines = [
        "# Dataset-A Batch-1 Acquisition Current State v2",
        "",
        "Historical ACQ-1 evidence is preserved; this report records its current interpretation.",
        "",
        "## Current accounting",
        "",
        f"- Candidates: {summary['candidate_count']}",
        f"- Metadata transport complete: {summary.get('metadata_transport_complete', 0)}",
        f"- Metadata parse complete: {summary.get('metadata_parse_complete', 0)}",
        f"- Record-level identity resolved: {summary.get('record_level_identity_resolved', 0)}",
        f"- AFDB structure complete: {summary.get('structure_complete', 0)}",
        f"- AFDB PAE complete: {summary.get('PAE_complete', 0)}",
        f"- AFDB confidence complete: {summary.get('confidence_complete', 0)}",
        f"- Candidate raw-complete: {summary['candidate_raw_complete']}",
        f"- Current candidate identity failures: {summary.get('current_candidate_identity_failures', 0)}",
        "",
        "## ACQ-5 dependent assets",
        "",
        f"- Planned: {summary.get('planned_dependent_assets', 0)}",
        f"- Reused valid: {summary.get('reused_dependent_assets', 0)}",
        f"- Downloaded new: {summary.get('downloaded_dependent_assets', 0)}",
        f"- Failed: {summary.get('failed_dependent_assets', 0)}",
        "",
        "## Historical supersession",
        "",
        (
            "The five ACQ-1 container-level identity failures remain historical facts. "
            "ACQ-4 supersedes their scientific interpretation as metadata "
            "collection-semantics artifacts."
        ),
        "",
        "## Missingness interpretation",
        "",
        report.get("missingness_interpretation", {}).get(
            "interpretation", "No missingness interpretation supplied."
        ),
        "No new biological inference was performed.",
    ]
    atomic_write_text(md_path, "\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=ProjectPaths.discover(anchor=Path(__file__)).repository_root,
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    plan, resolution, specs = verify_acq5_gate(root)
    acq1_path = root / ACQ1_LEDGER_PATH
    acq4_path = root / ACQ4_REPORT_PATH
    historical_hashes = {
        ACQ1_LEDGER_PATH.as_posix(): _BATCH1.sha256_file(acq1_path),
        ACQ4_REPORT_PATH.as_posix(): _BATCH1.sha256_file(acq4_path),
    }
    records = acquire_dependents(specs, max_attempts=args.max_attempts)
    current_hashes = {
        ACQ1_LEDGER_PATH.as_posix(): _BATCH1.sha256_file(acq1_path),
        ACQ4_REPORT_PATH.as_posix(): _BATCH1.sha256_file(acq4_path),
    }
    if current_hashes != historical_hashes:
        raise ACQ5Error("Historical ACQ-1/ACQ-4 evidence changed during acquisition")
    acq1 = json.loads(acq1_path.read_text(encoding="utf-8"))
    source_bindings = {
        **dict(plan.get("input_bindings", {})),
        "acq1_ledger_path": ACQ1_LEDGER_PATH.as_posix(),
        "acq1_ledger_SHA256": historical_hashes[ACQ1_LEDGER_PATH.as_posix()],
        "acq4_report_path": ACQ4_REPORT_PATH.as_posix(),
        "acq4_report_SHA256": historical_hashes[ACQ4_REPORT_PATH.as_posix()],
    }
    report = build_current_state(
        plan,
        acq1,
        resolution,
        records,
        source_bindings=source_bindings,
    )
    write_outputs(report, root / OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
