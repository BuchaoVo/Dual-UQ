"""Audit ACQ-1 AFDB identity failures from retained local evidence only."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import statistics
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from dual_uq.core.atomic_io import atomic_write_text
from dual_uq.core.hashing import sha256_file

FAILED_INDICES = (24, 7, 38, 103, 208)
LEDGER_PATH = Path(
    "artifacts/dataset/reports/acquisition/batch1_acquisition_run_v1.json"
)
INVENTORY_PATH = Path(
    "artifacts/dataset/reports/census/candidate_inventory_v1.tsv"
)
DISCOVERY_PATH = Path("data/processed/discovery/discovered_candidates.parquet")
LIFECYCLE_PATH = Path("reports/candidate_lifecycle.csv")
OUTPUT_DIR = Path("artifacts/dataset/audits/acquisition")


class IdentityContractError(RuntimeError):
    """A successful record violates the exact AFDB identity contract."""


def _records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list) and payload and all(isinstance(row, dict) for row in payload):
        return [dict(row) for row in payload]
    raise IdentityContractError("AFDB metadata payload has no auditable prediction records")


def audit_success_metadata(data: bytes, expected_accession: str) -> dict[str, Any]:
    """Require every prediction record to use the exact manifest accession."""
    try:
        records = _records(json.loads(data))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise IdentityContractError("Successful metadata is not valid JSON") from exc
    exact = 0
    normalized = 0
    aliases = 0
    models: list[str] = []
    sequence_present = True
    for record in records:
        observed = str(record.get("uniprotAccession", ""))
        if observed == expected_accession:
            exact += 1
        elif observed.strip().upper() == expected_accession.strip().upper():
            normalized += 1
        else:
            aliases += 1
        sequence = record.get("uniprotSequence") or record.get("sequence")
        sequence_present = sequence_present and isinstance(sequence, str) and bool(sequence.strip())
        model = record.get("modelEntityId") or record.get("entryId")
        if isinstance(model, str) and model.strip():
            models.append(model.strip())
    if normalized or aliases or exact != len(records):
        raise IdentityContractError(
            f"Successful AFDB metadata used a non-exact accession for {expected_accession}"
        )
    return {
        "prediction_record_count": len(records),
        "exact_accession_match_count": exact,
        "normalized_but_nonexact_match_count": normalized,
        "unexpected_alias_count": aliases,
        "sequence_field_present_for_all": sequence_present,
        "model_identifiers": models,
    }


def classify_unretained_identity_failure(row: Mapping[str, Any]) -> dict[str, Any]:
    """Represent only what the ACQ-1 failure ledger can establish."""
    return {
        "requested_afdb_metadata_identity": str(row["pair_id"]).split("__")[-1],
        "returned_uniprot_accession": None,
        "returned_sequence_field_presence": "unknown_not_retained",
        "returned_model_identifiers": None,
        "raw_response_body_retained": False,
        "exact_identity_predicate_result": "failed_for_at_least_one_prediction_record",
        "failure_taxonomy": "insufficient_local_evidence",
        "taxonomy_note": (
            "ACQ-1 retained the response URL, HTTP status, byte count and SHA256 but did "
            "not retain validation-failed response bytes; secondary, obsolete, isoform and "
            "schema-anomaly explanations therefore cannot be distinguished locally."
        ),
        "recovery_requirement": "additional_evidence_only",
        "recovery_status": (
            "blocked_until_exact_returned_metadata_provenance_is_available; no accession "
            "fallback or candidate identity change is authorized"
        ),
    }


def summarize_failure_accounting(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    records = list(records)
    primary = [
        row
        for row in records
        if row.get("asset_type") == "afdb_metadata"
        and row.get("failure_code") == "identity_mismatch"
    ]
    dependent = [
        row for row in records if row.get("failure_code") == "metadata_asset_missing"
    ]
    failed = [row for row in records if row.get("status") == "failed"]
    return {
        "candidate_count_with_primary_metadata_failure": len(
            {int(row["candidate_index"]) for row in primary}
        ),
        "primary_metadata_identity_failure_assets": len(primary),
        "downstream_metadata_asset_missing_assets": len(dependent),
        "total_failed_assets": len(failed),
    }


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _listify(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    return [str(item).strip() for item in values if item not in {None, "", ".", "?"}]


def _pdb_uniprot_accessions(path: Path, chain: str) -> list[str]:
    parsed = MMCIF2Dict(str(path))
    accessions = _listify(parsed.get("_struct_ref_seq.pdbx_db_accession", []))
    chains = _listify(parsed.get("_struct_ref_seq.pdbx_strand_id", []))
    if len(accessions) == len(chains):
        selected = [acc for acc, strands in zip(accessions, chains, strict=True) if chain in strands.split(",")]
        if selected:
            return sorted(set(selected))
    refs = _listify(parsed.get("_struct_ref.pdbx_db_accession", []))
    names = _listify(parsed.get("_struct_ref.db_name", []))
    if len(refs) == len(names):
        return sorted({acc for acc, name in zip(refs, names, strict=True) if name == "UNP"})
    return []


def _sifts_uniprot_accessions(path: Path, chain: str) -> list[str]:
    root = ET.fromstring(gzip.decompress(path.read_bytes()))
    accessions: set[str] = set()
    for residue in root.iter():
        if not residue.tag.endswith("residue"):
            continue
        refs = [child.attrib for child in residue if child.tag.endswith("crossRefDb")]
        pdb_refs = [ref for ref in refs if ref.get("dbSource") == "PDB"]
        if not any(ref.get("dbChainId") == chain for ref in pdb_refs):
            continue
        accessions.update(
            str(ref["dbAccessionId"])
            for ref in refs
            if ref.get("dbSource") == "UniProt" and ref.get("dbAccessionId")
        )
    return sorted(accessions)


def _none(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value


def _float(value: Any) -> float | None:
    value = _none(value)
    if value in {None, ""}:
        return None
    return float(value)


def _bool(value: Any) -> bool | None:
    value = _none(value)
    if value in {None, ""}:
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _numeric_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return {
        "available_count": len(values),
        "missing_count": len(rows) - len(values),
        "min": min(values) if values else None,
        "median": statistics.median(values) if values else None,
        "max": max(values) if values else None,
    }


def _categorical_summary(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field) or "unknown") for row in rows).items()))


def _comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "candidate_count": len(rows),
        "canonical_uniprot_length": _numeric_summary(rows, "canonical_uniprot_length"),
        "mapped_length": _numeric_summary(rows, "mapped_length"),
        "global_plddt": _numeric_summary(rows, "global_plddt"),
        "historical_screening_status": _categorical_summary(rows, "historical_screening_status"),
        "local_data_readiness": _categorical_summary(rows, "local_data_readiness"),
        "afdb_fragment_status": _categorical_summary(rows, "afdb_fragment_status"),
        "sequence_cluster_known_count": sum(row.get("sequence_cluster") is not None for row in rows),
        "sequence_cluster_unique_known_count": len(
            {row["sequence_cluster"] for row in rows if row.get("sequence_cluster") is not None}
        ),
        "pdb_side_available_count": sum(bool(row.get("pdb_side_available")) for row in rows),
    }


def _historical_statuses(root: Path) -> dict[int, str]:
    path = root / LIFECYCLE_PATH
    if not path.exists():
        return {}
    rows = _read_tsv(path) if path.suffix == ".tsv" else list(
        csv.DictReader(path.open(newline="", encoding="utf-8"))
    )
    output: dict[int, str] = {}
    for row in rows:
        try:
            index = int(float(row["screening_index"]))
        except (KeyError, TypeError, ValueError):
            continue
        output[index] = row.get("preflight_status") or row.get("classification_status") or "unknown"
    return output


def _inventory_rows(root: Path) -> dict[int, dict[str, Any]]:
    historical = _historical_statuses(root)
    output: dict[int, dict[str, Any]] = {}
    for row in _read_tsv(root / INVENTORY_PATH):
        index = int(row["candidate_index"])
        historical_index = _float(row.get("historical_screening_index"))
        fragment_count = _float(row.get("fragment_count"))
        if fragment_count is None:
            fragment_status = "unknown"
        elif fragment_count == 1:
            fragment_status = "single"
        elif fragment_count > 1:
            fragment_status = "multi"
        else:
            fragment_status = "unknown"
        pdb_path = root / f"data/raw/pdb/{row['PDB'].lower()}.cif"
        output[index] = {
            "candidate_index": index,
            "canonical_source_row": int(row["canonical_source_row"]),
            "polymer_entity_id": row["polymer_entity_id"],
            "pair_id": row["pair_id"],
            "PDB": row["PDB"],
            "chain": row["chain"],
            "candidate_uniprot_accession": row["UniProt"],
            "historical_screening_index": int(historical_index) if historical_index is not None else None,
            "historical_screening_status": historical.get(int(historical_index), "unknown") if historical_index is not None else "not_applicable",
            "canonical_uniprot_length": _float(row.get("canonical_uniprot_length")),
            "mapped_length": _float(row.get("mapped_length")),
            "local_data_readiness": row.get("estimated_P0_readiness") or "unknown",
            "global_plddt": _float(row.get("global_pLDDT_proxy")),
            "fragment_count": int(fragment_count) if fragment_count is not None else None,
            "afdb_fragment_status": fragment_status,
            "sequence_cluster": _none(row.get("sequence_cluster")),
            "pdb_side_available": pdb_path.exists(),
        }
    return output


def build_audit(root: Path) -> dict[str, Any]:
    ledger = json.loads((root / LEDGER_PATH).read_text())
    inventory = _inventory_rows(root)
    discovery = pd.read_parquet(root / DISCOVERY_PATH)
    discovery_by_entity = {
        str(row.polymer_entity_id): str(row.uniprot_id)
        for row in discovery.itertuples(index=False)
    }
    metadata_failures = {
        int(row["candidate_index"]): row
        for row in ledger["records"]
        if row.get("asset_type") == "afdb_metadata"
        and row.get("failure_code") == "identity_mismatch"
    }
    if set(metadata_failures) != set(FAILED_INDICES):
        raise IdentityContractError("ACQ-1 identity-failure candidate set changed")

    failures: list[dict[str, Any]] = []
    for index in FAILED_INDICES:
        base = dict(inventory[index])
        row = metadata_failures[index]
        pdb_id = str(base["PDB"]).lower()
        pdb_path = root / f"data/raw/pdb/{pdb_id}.cif"
        sifts_path = root / f"data/raw/mappings/{pdb_id}.xml.gz"
        base.update(
            {
                "discovery_uniprot_accession": discovery_by_entity.get(base["polymer_entity_id"]),
                "requested_afdb_metadata_identity": base["candidate_uniprot_accession"],
                "returned_uniprot_accession": None,
                "returned_sequence_field_presence": "unknown_not_retained",
                "returned_model_identifiers": None,
                "source_url": row.get("source_url"),
                "HTTP_status": row.get("HTTP_status"),
                "response_byte_count": row.get("byte_count"),
                "response_SHA256": row.get("SHA256"),
                "acq1_failure_code": row.get("failure_code"),
                "pdb_chain_uniprot_accessions": _pdb_uniprot_accessions(pdb_path, str(base["chain"])) if pdb_path.exists() else [],
                "sifts_chain_uniprot_accessions": _sifts_uniprot_accessions(sifts_path, str(base["chain"])) if sifts_path.exists() else [],
            }
        )
        base.update(classify_unretained_identity_failure(row))
        failures.append(base)

    controls: list[dict[str, Any]] = []
    for row in ledger["records"]:
        if row.get("asset_type") != "afdb_metadata" or row.get("status") == "failed":
            continue
        expected = str(row["pair_id"]).split("__")[-1]
        detail = audit_success_metadata((root / row["local_path"]).read_bytes(), expected)
        controls.append(
            {
                "candidate_index": int(row["candidate_index"]),
                "pair_id": row["pair_id"],
                "expected_accession": expected,
                "source_url": row["source_url"],
                **detail,
            }
        )
    exact = sum(row["exact_accession_match_count"] for row in controls)
    normalized = sum(row["normalized_but_nonexact_match_count"] for row in controls)
    aliases = sum(row["unexpected_alias_count"] for row in controls)
    if len(controls) != 33 or normalized or aliases:
        raise IdentityContractError("Successful-control identity contract is inconsistent")

    complete_indices = {
        int(row["candidate_index"])
        for row in ledger["candidate_completeness"]
        if row["completeness"] == "fully_raw_complete"
    }
    complete_rows = [inventory[index] for index in sorted(complete_indices)]
    return {
        "schema_version": "dataset-a.acq1-identity-mismatch-audit.v1",
        "audit_scope": {
            "failed_candidate_indices": list(FAILED_INDICES),
            "successful_control_candidate_count": len(controls),
            "network_access": False,
            "candidate_identity_changes": False,
        },
        "source_bindings": {
            "acq1_ledger": {"path": LEDGER_PATH.as_posix(), "SHA256": sha256_file(root / LEDGER_PATH)},
            "candidate_inventory": {"path": INVENTORY_PATH.as_posix(), "SHA256": sha256_file(root / INVENTORY_PATH)},
            "discovery": {"path": DISCOVERY_PATH.as_posix(), "SHA256": sha256_file(root / DISCOVERY_PATH)},
            "acquisition_plan_commit": ledger["acquisition_plan_commit"],
        },
        "failure_semantics": {
            "validator_contract": "every returned prediction record uniprotAccession must exactly equal the manifest accession",
            "retained_evidence_limit": "validation-failed response body was not retained at the canonical raw path",
            "interpretation": "the exact predicate failed, but the returned accession, sequence-field presence, model IDs and relation type are not reconstructable locally",
        },
        "failed_candidates": failures,
        "successful_control_audit": {
            "metadata_asset_count": len(controls),
            "prediction_record_count": sum(row["prediction_record_count"] for row in controls),
            "exact_accession_matches": exact,
            "normalized_but_nonexact_matches": normalized,
            "unexpected_aliases": aliases,
            "all_sequence_fields_present": all(row["sequence_field_present_for_all"] for row in controls),
            "records": controls,
        },
        "failure_accounting": summarize_failure_accounting(ledger["records"]),
        "missingness_comparison": {
            "analysis_type": "descriptive_only_no_biological_inference",
            "identity_failure_candidates": _comparison(failures),
            "raw_complete_candidates": _comparison(complete_rows),
            "length_ge_600": {
                "identity_failure_count": sum((row.get("canonical_uniprot_length") or -1) >= 600 for row in failures),
                "identity_failure_denominator": len(failures),
                "raw_complete_count": sum((row.get("canonical_uniprot_length") or -1) >= 600 for row in complete_rows),
                "raw_complete_denominator": len(complete_rows),
                "interpretation": "descriptive only; small cells do not support a biological or inferential claim",
            },
        },
        "protocol_assessment": {
            "protocol_wide_identity_concern": False,
            "reason": "all 33 successful metadata assets passed the same exact-accession contract with no normalization or alias fallback",
            "failed_case_resolution": "blocked_pending_additional_exact_response_provenance",
            "candidate_identity_change_authorized": False,
            "fallback_authorized": False,
        },
    }


def _tsv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if value is None:
        return ""
    return value


def write_outputs(audit: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "acq1_identity_mismatch_audit_v1.json"
    tsv_path = output_dir / "acq1_identity_mismatch_audit_v1.tsv"
    md_path = output_dir / "acq1_identity_mismatch_audit_v1.md"
    atomic_write_text(json_path, json.dumps(audit, indent=2, sort_keys=True) + "\n")

    rows = list(audit["failed_candidates"])
    fields = [
        "candidate_index", "polymer_entity_id", "pair_id", "candidate_uniprot_accession",
        "discovery_uniprot_accession", "requested_afdb_metadata_identity",
        "returned_uniprot_accession", "returned_sequence_field_presence",
        "returned_model_identifiers", "source_url", "HTTP_status", "response_byte_count",
        "response_SHA256", "pdb_chain_uniprot_accessions", "sifts_chain_uniprot_accessions",
        "canonical_uniprot_length", "mapped_length", "historical_screening_index",
        "historical_screening_status", "local_data_readiness", "global_plddt",
        "afdb_fragment_status", "fragment_count", "sequence_cluster", "pdb_side_available",
        "exact_identity_predicate_result", "failure_taxonomy", "recovery_requirement",
        "recovery_status",
    ]
    stream = StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _tsv_value(row.get(field)) for field in fields})
    atomic_write_text(tsv_path, stream.getvalue())

    controls = audit["successful_control_audit"]
    accounting = audit["failure_accounting"]
    failed_comp = audit["missingness_comparison"]["identity_failure_candidates"]
    complete_comp = audit["missingness_comparison"]["raw_complete_candidates"]
    lines = [
        "# ACQ-1 AFDB Identity-Mismatch Audit v1",
        "",
        "Status: offline diagnostic only; no accession fallback or candidate identity change.",
        "",
        "## Five failed candidates",
        "",
        "| Index | Pair | Requested | Returned | PDB-chain accessions | SIFTS-chain accessions | Taxonomy | Recovery |",
        "| ---: | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate_index']} | `{row['pair_id']}` | `{row['requested_afdb_metadata_identity']}` | "
            f"unknown—not retained | `{','.join(row['pdb_chain_uniprot_accessions'])}` | "
            f"`{','.join(row['sifts_chain_uniprot_accessions'])}` | `{row['failure_taxonomy']}` | "
            f"`{row['recovery_requirement']}` |"
        )
    lines.extend(
        [
            "",
            "The ACQ-1 validator established that at least one returned prediction record was not an exact accession match. The failed response bodies were hashed but not retained, so exact returned accessions, sequence-field presence, model identifiers and relation types cannot be reconstructed locally.",
            "",
            "## Successful-control audit",
            "",
            f"- Metadata assets: {controls['metadata_asset_count']}",
            f"- Prediction records: {controls['prediction_record_count']}",
            f"- Exact accession matches: {controls['exact_accession_matches']}",
            f"- Normalized-but-nonexact matches: {controls['normalized_but_nonexact_matches']}",
            f"- Unexpected aliases: {controls['unexpected_aliases']}",
            "",
            "## Causal failure accounting",
            "",
            f"- Candidates with primary metadata identity failure: {accounting['candidate_count_with_primary_metadata_failure']}",
            f"- Primary metadata identity-failure assets: {accounting['primary_metadata_identity_failure_assets']}",
            f"- Downstream metadata-asset-missing dependencies: {accounting['downstream_metadata_asset_missing_assets']}",
            f"- Total failed assets: {accounting['total_failed_assets']}",
            "",
            "## Descriptive missingness comparison",
            "",
            f"- Identity-failure canonical length: `{json.dumps(failed_comp['canonical_uniprot_length'], sort_keys=True)}`",
            f"- Raw-complete canonical length: `{json.dumps(complete_comp['canonical_uniprot_length'], sort_keys=True)}`",
            f"- Identity-failure readiness: `{json.dumps(failed_comp['local_data_readiness'], sort_keys=True)}`",
            f"- Raw-complete readiness: `{json.dumps(complete_comp['local_data_readiness'], sort_keys=True)}`",
            f"- Identity-failure fragment status: `{json.dumps(failed_comp['afdb_fragment_status'], sort_keys=True)}`",
            f"- Raw-complete fragment status: `{json.dumps(complete_comp['afdb_fragment_status'], sort_keys=True)}`",
            "",
            "This comparison is descriptive only. In particular, the length >=600 cell is small and cannot support a biological claim.",
            "",
            "## Protocol assessment",
            "",
            "No protocol-wide hidden normalization or alias fallback was found among the 33 successful controls. The five failures remain blocked pending additional exact response provenance; no recovery was performed.",
        ]
    )
    atomic_write_text(md_path, "\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    if subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip() != "bf81c19e085982aa90f28a4de98586586d0b2b4c":
        raise IdentityContractError("Repository HEAD differs from the ACQ-1 binding")
    audit = build_audit(root)
    write_outputs(audit, root / OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
