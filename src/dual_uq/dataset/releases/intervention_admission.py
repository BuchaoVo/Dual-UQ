"""Materialize frozen Stage-0 intervention admission decisions without recomputation."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from dual_uq.core.atomic_io import atomic_write_new_bytes

from .intervention_panel import EXPECTED_STAGE0_PANEL, PANEL_VERSION

ORIGINAL_PANEL_SHA256 = (
    "62d6e8576ed21663947c9bef893bcb9e74f7e6fba7b347910a4c0479b4e07e0d"
)
ADMISSION_VERSION = "stage0_intervention_admission_v1"
ADMITTED_SUBSET_VERSION = "stage0_intervention_admitted_v1"
PDR01_PROTOCOL_PATH = "docs/protocols/PDR-01_D1-D2_条款草案_v0.2.md"
STAGE0_1A_AUDIT_ID = "P2P3-STAGE0-1A"

_PANEL_PATH = (
    "experiments/p2_design_baseline/stage0/stage0_intervention_panel_v1.jsonl"
)
_ADMISSION_PATH = (
    "experiments/p2_design_baseline/stage0/"
    "stage0_intervention_admission_v1.jsonl"
)
_ADMITTED_PATH = (
    "experiments/p2_design_baseline/stage0/"
    "stage0_intervention_admitted_v1.jsonl"
)

_AUDITED: dict[str, dict[str, Any]] = {
    "6jgj_A__P42212": {
        "status": "IDENTITY_CONTRACT_FAIL",
        "mismatch_count": 5,
        "sequence_identity_paired": 0.9779735683,
        "observed_variants": ["Q80R", "F99S", "M153T", "V163A", "E222Q"],
        "reason_code": "pdr01_numerical_screen_failed",
        "classification": "IDENTITY_CONTRACT_FAIL",
    },
    "2ykz_A__P00138": {
        "status": "PENDING_HUMAN_VARIANT_REVIEW",
        "mismatch_count": 1,
        "sequence_identity_paired": 0.9920634921,
        "observed_variants": ["A39V"],
        "reason_code": "pending_exact_variant_authorization",
        "classification": "REVIEWABLE_SEQUENCE_VARIANT",
    },
    "1ix9_A__P00448": {
        "status": "PENDING_HUMAN_VARIANT_REVIEW",
        "mismatch_count": 1,
        "sequence_identity_paired": 0.9951219512,
        "observed_variants": ["Y175F"],
        "reason_code": "pending_exact_variant_authorization",
        "classification": "REVIEWABLE_SEQUENCE_VARIANT",
    },
}


class InterventionAdmissionError(ValueError):
    """A structured materialization blocker."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def _portable_path(value: str, field: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value:
        raise InterventionAdmissionError(
            "nonportable_release_path", f"{field} is not repository-relative: {value}"
        )
    return value


def _panel_identity(records: Sequence[Mapping[str, Any]]) -> tuple[tuple[int, str], ...]:
    try:
        return tuple(
            sorted(
                (
                    (int(record["candidate_index"]), str(record["protein_id"]))
                    for record in records
                ),
                key=lambda item: item[0],
            )
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise InterventionAdmissionError(
            "invalid_panel_record", "Stage-0 panel record lacks exact identity"
        ) from exc


def _audit_provenance(panel_path: str, panel_sha256: str) -> dict[str, Any]:
    return {
        "audit_id": STAGE0_1A_AUDIT_ID,
        "evidence_mode": "frozen_stage0_1a_values_no_recomputation",
        "original_panel_path": _portable_path(panel_path, "panel_path"),
        "original_panel_sha256": panel_sha256,
        "pdr01_protocol_path": PDR01_PROTOCOL_PATH,
    }


def build_admission_records(
    panel_records: Sequence[Mapping[str, Any]],
    *,
    panel_path: str,
    panel_sha256: str,
) -> list[dict[str, Any]]:
    """Materialize the frozen 11-way decision without deriving scientific metrics."""
    if panel_sha256 != ORIGINAL_PANEL_SHA256:
        raise InterventionAdmissionError(
            "original_panel_hash_mismatch", "Frozen Stage-0 panel SHA256 differs"
        )
    if _panel_identity(panel_records) != EXPECTED_STAGE0_PANEL:
        raise InterventionAdmissionError(
            "original_panel_identity_mismatch", "Frozen Stage-0 membership differs"
        )
    provenance = _audit_provenance(panel_path, panel_sha256)
    records: list[dict[str, Any]] = []
    for candidate_index, protein_id in EXPECTED_STAGE0_PANEL:
        audited = _AUDITED.get(protein_id)
        if audited is None:
            status = "ADMITTED"
            mismatch_count = None
            identity = None
            metric_status = "not_serialized_in_stage0_1a"
            variants = None
            reason_code = "frozen_panel_version_admitted"
            classification = "ADMITTED_BY_FROZEN_PANEL_VERSION_DECISION"
        else:
            status = str(audited["status"])
            mismatch_count = int(audited["mismatch_count"])
            identity = float(audited["sequence_identity_paired"])
            metric_status = "frozen_stage0_1a"
            variants = list(audited["observed_variants"])
            reason_code = str(audited["reason_code"])
            classification = str(audited["classification"])
        records.append(
            {
                "candidate_index": candidate_index,
                "protein_id": protein_id,
                "declared_panel_member": True,
                "intervention_admission_status": status,
                "pdr01_mismatch_count": mismatch_count,
                "pdr01_sequence_identity_paired": identity,
                "metric_resolution_status": metric_status,
                "observed_sequence_variants": variants,
                "authorized_sequence_variants": None,
                "admission_reason_code": reason_code,
                "admission_basis": {
                    "classification": classification,
                    "decision_state": "materialized_existing_human_panel_version_state",
                    "pdr01_mismatch_budget": 3,
                    "pdr01_sequence_identity_threshold": 0.99,
                    "scientific_admission_recomputed": False,
                    "variant_authorization_assigned": False,
                },
                "audit_provenance": dict(provenance),
                "panel_version": PANEL_VERSION,
                "admission_version": ADMISSION_VERSION,
            }
        )
    return records


def render_admission_jsonl(records: Sequence[Mapping[str, Any]]) -> bytes:
    """Render deterministic, portable JSONL sorted by candidate index."""
    ordered = sorted((dict(record) for record in records), key=lambda row: row["candidate_index"])
    lines = [
        json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        for record in ordered
    ]
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    if b"/home/" in payload or b"/mnt/" in payload:
        raise InterventionAdmissionError(
            "nonportable_release_path", "Admission output contains an absolute path"
        )
    return payload


def build_admitted_subset(
    admission_records: Sequence[Mapping[str, Any]],
    *,
    admission_path: str,
    admission_sha256: str,
) -> list[dict[str, Any]]:
    """Mechanically filter only ADMITTED rows and attach immutable source provenance."""
    source_path = _portable_path(admission_path, "admission_path")
    if _panel_identity(admission_records) != EXPECTED_STAGE0_PANEL:
        raise InterventionAdmissionError(
            "admission_ledger_identity_mismatch",
            "Admission ledger membership differs from the frozen declaration",
        )
    allowed_statuses = {
        "ADMITTED",
        "PENDING_HUMAN_VARIANT_REVIEW",
        "IDENTITY_CONTRACT_FAIL",
    }
    if any(
        record.get("intervention_admission_status") not in allowed_statuses
        for record in admission_records
    ):
        raise InterventionAdmissionError(
            "invalid_admission_status", "Admission ledger contains an unknown status"
        )
    subset = []
    for source in admission_records:
        if source.get("intervention_admission_status") != "ADMITTED":
            continue
        record = dict(source)
        record["subset_provenance"] = {
            "derivation_predicate": "intervention_admission_status == ADMITTED",
            "source_admission_path": source_path,
            "source_admission_sha256": admission_sha256,
            "admission_version": ADMISSION_VERSION,
            "subset_version": ADMITTED_SUBSET_VERSION,
        }
        subset.append(record)
    subset.sort(key=lambda row: int(row["candidate_index"]))
    if len(subset) != 8:
        raise InterventionAdmissionError(
            "admitted_subset_count_mismatch",
            "Mechanical admitted subset does not contain eight records",
        )
    return subset


def render_variant_review_packets(
    admission_records: Sequence[Mapping[str, Any]],
) -> dict[str, bytes]:
    """Render two decision-free evidence packets from frozen ledger values."""
    by_id = {str(record["protein_id"]): record for record in admission_records}
    specifications = {
        "variant_review_2ykz_A39V.md": {
            "protein_id": "2ykz_A__P00138",
            "candidate_index": 77,
            "uniprot": "P00138",
            "variant": "A39V",
            "interval": "1–127",
            "limitations": (
                "- U1: canonical Q maps to coordinate-bearing noncanonical `PCA`.\n"
                "- U127: K lacks a complete supported P1 backbone.\n"
            ),
            "sources": (
                "`data/raw/mappings/2ykz.xml.gz`, `data/raw/pdb/2ykz.cif`, "
                "and `data/raw/afdb/P00138/metadata.json`"
            ),
        },
        "variant_review_1ix9_Y175F.md": {
            "protein_id": "1ix9_A__P00448",
            "candidate_index": 179,
            "uniprot": "P00448",
            "variant": "Y175F",
            "interval": "2–206",
            "limitations": "- None recorded by the frozen Stage-0-1A audit.\n",
            "sources": (
                "`data/raw/mappings/1ix9.xml.gz`, `data/raw/pdb/1ix9.cif`, "
                "and `data/raw/afdb/P00448/metadata.json`"
            ),
        },
    }
    packets: dict[str, bytes] = {}
    for filename, spec in specifications.items():
        record = by_id[spec["protein_id"]]
        text = (
            f"# Human variant evidence packet: {spec['variant']}\n\n"
            "Status: `PENDING_HUMAN_VARIANT_REVIEW`\n\n"
            f"- Protein ID: `{spec['protein_id']}`\n"
            f"- Candidate index: `{spec['candidate_index']}`\n"
            f"- UniProt: `{spec['uniprot']}`\n"
            f"- Observed substitution: `{spec['variant']}`\n"
            f"- Mapped interval: UniProt `{spec['interval']}`\n"
            f"- Mismatch count: `{record['pdr01_mismatch_count']}`\n"
            "- Paired sequence identity: "
            f"`{record['pdr01_sequence_identity_paired']:.10f}`\n\n"
            "## Non-mismatch common-mask limitations\n\n"
            f"{spec['limitations']}\n"
            "## Evidence state\n\n"
            "The numerical PDR-01 review-eligibility gates pass. No exact human "
            "variant authorization currently exists. This packet records evidence "
            "only and contains no human decision.\n\n"
            f"Protocol: [`{PDR01_PROTOCOL_PATH}`](../../../{PDR01_PROTOCOL_PATH})\n\n"
            "Residue evidence provenance: frozen `P2P3-STAGE0-1A` audit using "
            f"{spec['sources']}. No value was recomputed for this packet.\n"
        )
        packets[filename] = text.encode("utf-8")
    return packets


def validate_stage0_admission_bindings(path: Path) -> dict[str, str]:
    """Resolve the three Stage-0 semantic inputs and forbid census aliasing."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    inputs = payload.get("inputs") if isinstance(payload, dict) else None
    if not isinstance(inputs, dict) or "canonical_census" in inputs:
        raise InterventionAdmissionError(
            "invalid_stage0_binding", "Stage-0 inputs are missing or misuse canonical_census"
        )
    expected = {
        "stage0_intervention_panel": _PANEL_PATH,
        "stage0_intervention_admission": _ADMISSION_PATH,
        "stage0_intervention_admitted_subset": _ADMITTED_PATH,
    }
    expected_versions = {
        "stage0_intervention_panel": {"panel_version": PANEL_VERSION},
        "stage0_intervention_admission": {"admission_version": ADMISSION_VERSION},
        "stage0_intervention_admitted_subset": {
            "admission_version": ADMISSION_VERSION,
            "subset_version": ADMITTED_SUBSET_VERSION,
        },
    }
    resolved: dict[str, str] = {}
    for key, expected_path in expected.items():
        binding = inputs.get(key)
        if not isinstance(binding, dict) or binding.get("path") != expected_path:
            raise InterventionAdmissionError(
                "invalid_stage0_binding", f"Stage-0 binding differs for {key}"
            )
        for version_key, expected_version in expected_versions[key].items():
            if binding.get(version_key) != expected_version:
                raise InterventionAdmissionError(
                    "invalid_stage0_binding",
                    f"Stage-0 binding version differs for {key}.{version_key}",
                )
        resolved[key] = _portable_path(str(binding["path"]), key)
    return resolved


def build_release_metadata(
    *,
    panel_sha256: str,
    admission_sha256: str,
    admitted_subset_sha256: str,
    pdr01_protocol_sha256: str,
    admission_records: Sequence[Mapping[str, Any]],
    admitted_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build deterministic release metadata without a runtime timestamp."""
    counts = Counter(
        str(record["intervention_admission_status"])
        for record in admission_records
    )
    return {
        "admission_version": ADMISSION_VERSION,
        "admitted_subset_version": ADMITTED_SUBSET_VERSION,
        "source_panel_path": _PANEL_PATH,
        "source_panel_sha256": panel_sha256,
        "admission_ledger_path": _ADMISSION_PATH,
        "admission_ledger_sha256": admission_sha256,
        "admitted_subset_path": _ADMITTED_PATH,
        "admitted_subset_sha256": admitted_subset_sha256,
        "pdr01_protocol": {
            "path": PDR01_PROTOCOL_PATH,
            "sha256": pdr01_protocol_sha256,
            "mismatch_budget": 3,
            "sequence_identity_paired_threshold": 0.99,
        },
        "stage0_1a_audit_provenance": {
            "audit_id": STAGE0_1A_AUDIT_ID,
            "evidence_mode": "frozen_values_no_recomputation",
        },
        "record_counts": {
            "historical_declaration": len(EXPECTED_STAGE0_PANEL),
            "admission_ledger": len(admission_records),
            "executable_subset": len(admitted_records),
        },
        "admission_status_counts": dict(sorted(counts.items())),
        "semantic_distinction": {
            "stage0_intervention_panel_v1": (
                "historical frozen 11-protein experimental declaration"
            ),
            "stage0_intervention_admission_v1": (
                "current formal admission state of all 11 declared proteins"
            ),
            "stage0_intervention_admitted_v1": (
                "current executable subset under this admission version"
            ),
        },
        "scientific_admission_recomputed": False,
        "human_variant_authorization_assigned": False,
    }


def write_immutable_release(path: Path, payload: bytes) -> str:
    """Create one versioned output or verify exact existing bytes."""
    if path.exists():
        if path.read_bytes() != payload:
            raise InterventionAdmissionError(
                "immutable_admission_conflict", f"Immutable output differs: {path}"
            )
        return "reused_identical"
    atomic_write_new_bytes(path, payload)
    return "created"
