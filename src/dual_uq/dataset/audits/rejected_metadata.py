"""Re-fetch the five ACQ-1 rejected AFDB metadata responses for forensics."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import subprocess
from collections.abc import Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from dual_uq.core.atomic_io import atomic_write_text
from dual_uq.core.hashing import sha256_bytes, sha256_file

from ..stages import acquisition as _BATCH1

AcquisitionError = _BATCH1.AcquisitionError
AssetSpec = _BATCH1.AssetSpec
TransportResponse = _BATCH1.TransportResponse

FAILED_IDENTITIES = (
    (24, "Q8N8S7"),
    (7, "Q13427"),
    (38, "Q9Y237"),
    (103, "Q8K4V4"),
    (208, "O14786"),
)
AFDB_PREDICTION_API = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"
LEDGER_PATH = Path(
    "artifacts/dataset/reports/acquisition/batch1_acquisition_run_v1.json"
)
ACQ2_PATH = Path(
    "artifacts/dataset/audits/acquisition/acq1_identity_mismatch_audit_v1.json"
)
OUTPUT_DIR = Path("artifacts/dataset/audits/acquisition")
REJECTED_DIR = Path("artifacts/dataset/audits/acquisition_rejected")


class ForensicScopeError(RuntimeError):
    """The historical failure set or source binding differs from ACQ-1."""


def select_historical_failures(ledger: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Select only the five ordered ACQ-1 metadata identity failures."""
    by_index = {
        int(row["candidate_index"]): dict(row)
        for row in ledger.get("records", [])
        if row.get("asset_type") == "afdb_metadata"
        and row.get("failure_code") == "identity_mismatch"
    }
    if set(by_index) != {index for index, _ in FAILED_IDENTITIES}:
        raise ForensicScopeError("Historical identity-failure candidate scope differs")
    selected = []
    for index, accession in FAILED_IDENTITIES:
        row = by_index[index]
        expected_url = AFDB_PREDICTION_API.format(accession=accession)
        pair_accession = str(row.get("pair_id", "")).split("__")[-1]
        if pair_accession != accession:
            raise ForensicScopeError(f"Candidate identity differs for Index {index}")
        if row.get("source_url") != expected_url:
            raise ForensicScopeError(f"Historical URL differs for Index {index}")
        selected.append(row)
    return selected


def classify_returned_identity(
    expected_accession: str, returned_accessions: list[str] | None
) -> str:
    """Classify only explicit returned accession strings, without rescue inference."""
    if not returned_accessions:
        return "insufficient_evidence"
    observed = set(returned_accessions)
    if observed == {expected_accession}:
        return "exact_identity_mismatch"
    if any(
        value.startswith(f"{expected_accession}-")
        or expected_accession.startswith(f"{value}-")
        for value in observed
    ):
        return "possible_isoform_relation"
    if expected_accession in observed and len(observed) > 1:
        return "metadata_schema_or_record_anomaly"
    return "cross_source_accession_conflict"


def _display_path(path: str | Path | None, project_root: Path) -> str | None:
    if path is None:
        return None
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def run_forensic_refetch(
    historical_ledger: Mapping[str, Any],
    *,
    project_root: Path,
    run_dir: Path,
    rejected_dir: Path,
    transport: Any = _BATCH1.default_transport,
    sleeper: Any = _BATCH1.time.sleep,
    clock: Any = _BATCH1.utc_now,
    max_attempts: int = 3,
) -> dict[str, Any]:
    """Re-request only the five historical metadata URLs and preserve provenance."""
    records: list[dict[str, Any]] = []
    for historical in select_historical_failures(historical_ledger):
        index = int(historical["candidate_index"])
        expected = str(historical["pair_id"]).split("__")[-1]
        observation_path = (
            run_dir / "outputs" / "valid_metadata_observations" / f"{index}_{expected}.json"
        )
        spec = AssetSpec(
            candidate_index=index,
            polymer_entity_id=str(historical["polymer_entity_id"]),
            pair_id=str(historical["pair_id"]),
            asset_type="afdb_metadata",
            source_url=str(historical["source_url"]),
            local_path=observation_path,
            expected_identity=expected,
            display_path=observation_path.as_posix(),
        )
        current = _BATCH1.acquire_asset(
            spec,
            transport=transport,
            sleeper=sleeper,
            clock=clock,
            max_attempts=max_attempts,
            rejected_evidence_dir=rejected_dir,
        )
        if current["status"] in {"downloaded_new", "reused_valid"}:
            current_validation = "exact_match"
            parsed = _BATCH1.inspect_afdb_metadata_evidence(observation_path.read_bytes())
        elif current.get("failure_code") in {
            "identity_mismatch",
            "exact_identity_mismatch",
            "ambiguous_exact_record_set",
        }:
            current_validation = "identity_mismatch"
            parsed = {
                "parsed_returned_accessions": current.get("parsed_returned_accessions"),
                "parsed_sequence_field_presence": current.get(
                    "parsed_sequence_field_presence"
                ),
                "parsed_model_identifiers": current.get("parsed_model_identifiers"),
            }
        else:
            current_validation = "transport_or_http_failure"
            parsed = {
                "parsed_returned_accessions": None,
                "parsed_sequence_field_presence": None,
                "parsed_model_identifiers": None,
            }
        current_sha = current.get("SHA256")
        if current_sha is None:
            interpretation = {
                "sha_relation": "unavailable",
                "outcome": "transport_or_http_failure",
                "historical_failure_recovered": False,
            }
        else:
            interpretation = _BATCH1.interpret_refetch_outcome(
                historical_sha=str(historical["SHA256"]),
                current_sha=str(current_sha),
                current_validation=current_validation,
            )
        classification = (
            interpretation["outcome"]
            if current_validation == "exact_match"
            else classify_returned_identity(
                expected, parsed["parsed_returned_accessions"]
            )
            if current_validation == "identity_mismatch"
            else "insufficient_evidence"
        )
        records.append(
            {
                "candidate_index": index,
                "polymer_entity_id": historical["polymer_entity_id"],
                "pair_id": historical["pair_id"],
                "expected_accession": expected,
                "source_url": historical["source_url"],
                "historical_HTTP_status": historical.get("HTTP_status"),
                "historical_byte_count": historical.get("byte_count"),
                "historical_SHA256": historical.get("SHA256"),
                "historical_timestamp": historical.get("timestamp"),
                "current_HTTP_status": current.get("HTTP_status"),
                "current_byte_count": current.get("byte_count"),
                "current_SHA256": current_sha,
                "current_timestamp": current.get("timestamp"),
                "current_response_headers": current.get("response_headers"),
                "current_validation": current_validation,
                "current_failure_code": current.get("failure_code"),
                "returned_accessions": parsed["parsed_returned_accessions"],
                "sequence_field_presence": parsed[
                    "parsed_sequence_field_presence"
                ],
                "model_identifiers": parsed["parsed_model_identifiers"],
                "raw_rejected_payload_path": _display_path(
                    current.get("rejected_payload_path"), project_root
                ),
                "rejected_evidence_status": current.get(
                    "rejected_evidence_status"
                ),
                "current_payload_path": (
                    _display_path(observation_path, project_root)
                    if observation_path.exists()
                    else None
                ),
                "failure_classification": classification,
                **interpretation,
            }
        )
    return {
        "schema_version": "dataset-a.acq1-rejected-metadata-forensics.v1",
        "scope": {
            "candidate_indices": [index for index, _ in FAILED_IDENTITIES],
            "metadata_request_count": len(records),
            "dependent_asset_request_count": 0,
            "alternate_source_or_accession_used": False,
        },
        "records": records,
    }


def _encode_tsv(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def write_reports(audit: Mapping[str, Any], output_dir: Path) -> None:
    """Write deterministic curated reports without altering historical evidence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "acq1_rejected_metadata_forensics_v1.json"
    tsv_path = output_dir / "acq1_rejected_metadata_forensics_v1.tsv"
    md_path = output_dir / "acq1_rejected_metadata_forensics_v1.md"
    atomic_write_text(json_path, json.dumps(audit, indent=2, sort_keys=True) + "\n")

    fields = [
        "candidate_index",
        "polymer_entity_id",
        "pair_id",
        "expected_accession",
        "source_url",
        "historical_HTTP_status",
        "historical_byte_count",
        "historical_SHA256",
        "historical_timestamp",
        "current_HTTP_status",
        "current_byte_count",
        "current_SHA256",
        "current_timestamp",
        "current_response_headers",
        "current_validation",
        "current_failure_code",
        "returned_accessions",
        "sequence_field_presence",
        "model_identifiers",
        "raw_rejected_payload_path",
        "rejected_evidence_status",
        "current_payload_path",
        "failure_classification",
        "sha_relation",
        "historical_failure_recovered",
        "outcome",
    ]
    stream = StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    for row in audit["records"]:
        writer.writerow({field: _encode_tsv(row.get(field)) for field in fields})
    atomic_write_text(tsv_path, stream.getvalue())

    lines = [
        "# ACQ-1 Rejected AFDB Metadata Forensics v1",
        "",
        "Status: manifest-bound forensic re-fetch only; historical failures are not converted to success.",
        "",
        "| Index | Pair | Historical SHA | Current SHA | SHA relation | Returned accession(s) | Classification | Outcome |",
        "| ---: | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in audit["records"]:
        returned = ",".join(row.get("returned_accessions") or []) or "unknown"
        lines.append(
            f"| {row['candidate_index']} | `{row['pair_id']}` | `{row['historical_SHA256']}` | "
            f"`{row.get('current_SHA256') or 'unavailable'}` | `{row['sha_relation']}` | "
            f"`{returned}` | `{row['failure_classification']}` | `{row['outcome']}` |"
        )
    lines.extend(
        [
            "",
            "## Contract",
            "",
            "Rejected HTTP 200 payloads are stored immutably by SHA256 outside `data/raw`. Valid current observations, if any, remain run-local and do not recover the historical failure. No alternate accession, source, dependent asset, or fallback is used.",
            "",
            "## Summary",
            "",
            f"- Metadata requests: {audit['scope']['metadata_request_count']}",
            f"- Dependent asset requests: {audit['scope']['dependent_asset_request_count']}",
            f"- Source/identity fallback used: {audit['scope']['alternate_source_or_accession_used']}",
        ]
    )
    atomic_write_text(md_path, "\n".join(lines) + "\n")


def verify_forensic_bindings(root: Path) -> dict[str, Any]:
    """Revalidate ACQ-1 bindings and the ACQ-2 historical ledger digest."""
    _BATCH1.verify_repository_bindings(root)
    ledger_path = root / LEDGER_PATH
    acq2_path = root / ACQ2_PATH
    ledger = json.loads(ledger_path.read_text())
    acq2 = json.loads(acq2_path.read_text())
    expected_ledger_sha = acq2["source_bindings"]["acq1_ledger"]["SHA256"]
    if sha256_file(ledger_path) != expected_ledger_sha:
        raise ForensicScopeError("ACQ-1 ledger SHA differs from the ACQ-2 binding")
    select_historical_failures(ledger)
    return ledger


def _run_id(root: Path, timestamp: str, rows: list[dict[str, Any]]) -> str:
    safe_time = re.sub(r"[^0-9TZ]", "", timestamp)
    head = subprocess.run(
        ["git", "rev-parse", "--short=8", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    config = json.dumps(
        [(row["candidate_index"], row["source_url"]) for row in rows],
        separators=(",", ":"),
    ).encode()
    config_sha = sha256_bytes(config)[:8]
    return f"{safe_time}_batch1_{head}_{config_sha}"


def create_run_directory(
    root: Path, timestamp: str, historical_rows: list[dict[str, Any]]
) -> Path:
    """Create one non-overwriting ACQ-3 run record under the canonical runs tree."""
    run_dir = root / "runs/dataset/acquisition" / _run_id(
        root, timestamp, historical_rows
    )
    if run_dir.exists():
        raise ForensicScopeError(f"Run directory already exists: {run_dir}")
    (run_dir / "outputs").mkdir(parents=True)
    config = {
        "task": "ACQ-3",
        "candidate_indices": [row["candidate_index"] for row in historical_rows],
        "metadata_urls": [row["source_url"] for row in historical_rows],
        "dependent_assets": False,
        "fallback": False,
    }
    atomic_write_text(
        run_dir / "resolved_config.yaml", json.dumps(config, indent=2) + "\n"
    )
    atomic_write_text(
        run_dir / "command.txt",
        "python scripts/dataset/validate.py rejected-metadata --project-root .\n",
    )
    atomic_write_text(
        run_dir / "versions.json",
        json.dumps(
            {"python": platform.python_version(), "platform": platform.platform()},
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    atomic_write_text(
        run_dir / "git_state.json",
        json.dumps(
            {
                "HEAD": head,
                "branch": branch,
                "dirty": bool(status),
                "dirty_paths": [line[3:] for line in status],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    manifest_stream = StringIO(newline="")
    writer = csv.DictWriter(
        manifest_stream,
        fieldnames=["candidate_index", "polymer_entity_id", "pair_id", "source_url"],
        delimiter="\t",
        lineterminator="\n",
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(historical_rows)
    atomic_write_text(run_dir / "input_manifest.tsv", manifest_stream.getvalue())
    atomic_write_text(
        run_dir / "input_digests.json",
        json.dumps(
            {
                "acq1_ledger": sha256_file(root / LEDGER_PATH),
                "acq2_audit": sha256_file(root / ACQ2_PATH),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    atomic_write_text(run_dir / "stdout.log", "")
    atomic_write_text(run_dir / "stderr.log", "")
    atomic_write_text(
        run_dir / "status.json",
        json.dumps(
            {"stage": "metadata_refetch", "started_at": timestamp, "state": "running"},
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    ledger = verify_forensic_bindings(root)
    historical_rows = select_historical_failures(ledger)
    timestamp = _BATCH1.utc_now()
    run_dir = create_run_directory(root, timestamp, historical_rows)
    try:
        audit = run_forensic_refetch(
            ledger,
            project_root=root,
            run_dir=run_dir,
            rejected_dir=root / REJECTED_DIR,
            max_attempts=args.max_attempts,
        )
    except AcquisitionError as exc:
        failure = {
            "stage": "metadata_refetch",
            "exception_type": type(exc).__name__,
            "failure_code": exc.code,
            "message": str(exc),
            "last_validated_checkpoint": "historical_bindings",
        }
        atomic_write_text(
            run_dir / "failure.json", json.dumps(failure, indent=2, sort_keys=True) + "\n"
        )
        atomic_write_text(run_dir / "FAILED", "")
        raise
    audit["source_bindings"] = {
        "acq1_ledger_path": LEDGER_PATH.as_posix(),
        "acq1_ledger_SHA256": sha256_file(root / LEDGER_PATH),
        "acq2_audit_path": ACQ2_PATH.as_posix(),
        "acq2_audit_SHA256": sha256_file(root / ACQ2_PATH),
        "acquisition_plan_commit": _BATCH1.EXPECTED_COMMIT,
    }
    audit["run_directory"] = run_dir.relative_to(root).as_posix()
    write_reports(audit, root / OUTPUT_DIR)
    atomic_write_text(
        run_dir / "metrics.json",
        json.dumps(
            {
                "metadata_requests": len(audit["records"]),
                "sha_relations": dict(
                    sorted(
                        __import__("collections").Counter(
                            row["sha_relation"] for row in audit["records"]
                        ).items()
                    )
                ),
                "classifications": dict(
                    sorted(
                        __import__("collections").Counter(
                            row["failure_classification"] for row in audit["records"]
                        ).items()
                    )
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    atomic_write_text(
        run_dir / "status.json",
        json.dumps(
            {"stage": "metadata_refetch", "finished_at": _BATCH1.utc_now(), "state": "complete"},
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    atomic_write_text(run_dir / "SUCCESS", "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
