"""Generic deterministic batch/subset orchestration for dataset derivation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from dual_uq.core.paths import ProjectPaths

from .models import (
    CandidateContext,
    CandidateDerivationResult,
    DerivationConfig,
    DerivationRunResult,
)


def derive_candidate(
    context: CandidateContext,
    config: DerivationConfig,
    paths: ProjectPaths,
) -> CandidateDerivationResult:
    """Late-bound candidate executor keeps pipeline orchestration capability-only."""
    from .derivation import derive_candidate as execute_candidate

    return execute_candidate(context, config, paths)


def _known_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _descriptive_attrition(
    candidates: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    fields = [
        "canonical_sequence_length",
        "mapped_residue_count",
        "mapping_coverage",
        "gap_count",
        "segment_count",
        "global_pLDDT_proxy",
        "metadata_record_count",
        "fragment_count",
    ]
    groups: dict[str, Any] = {}
    for name, rows in (
        (
            "mechanism_observable",
            [row for row in candidates if row["mechanism_observable"]],
        ),
        (
            "not_mechanism_observable",
            [row for row in candidates if not row["mechanism_observable"]],
        ),
    ):
        medians: dict[str, float | None] = {}
        for field in fields:
            values = [
                float(row[field])
                for row in rows
                if _known_number(row.get(field)) is not None
            ]
            medians[field] = float(np.median(values)) if values else None
        groups[name] = {
            "count": len(rows),
            "medians": medians,
            "pae_available_count": sum(bool(row.get("pae_bound")) for row in rows),
        }
    return {
        "scope": (
            f"descriptive stress-test only; N={len(candidates)}; "
            "no significance or prevalence inference"
        ),
        "groups": groups,
        "requested_proxies_not_available": [],
    }


def _summary(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    attrition_classes = {
        str(row["attrition_class"])
        for row in candidates
        if row.get("attrition_class")
    }
    if "implementation_or_schema_issue" in attrition_classes:
        verdict = "IMPLEMENTATION_BUG"
    elif "protocol_eligibility_issue" in attrition_classes:
        verdict = "PROTOCOL_CONFLICT"
    elif all(row["mechanism_observable"] for row in candidates) and not any(
        row.get("scientific_attrition_flags") for row in candidates
    ):
        verdict = "DERIVATION_PIPELINE_PASS"
    else:
        verdict = "DERIVATION_PIPELINE_PASS_WITH_DATA_FAILURES"
    return {
        "candidate_count": len(candidates),
        "canonical_sequence_complete_count": sum(
            bool(row["canonical_sequence_complete"]) for row in candidates
        ),
        "pair_qc_complete_count": sum(
            bool(row["pair_qc_complete"]) for row in candidates
        ),
        "pair_qc_pass_count": sum(
            row.get("pair_qc_status") == "pair_qc_pass" for row in candidates
        ),
        "mapping_complete_count": sum(
            bool(row["mapping_complete"]) for row in candidates
        ),
        "fragment_resolved_count": sum(
            bool(row["fragment_resolved"]) for row in candidates
        ),
        "pae_bound_count": sum(bool(row["pae_bound"]) for row in candidates),
        "confidence_bound_count": sum(
            bool(row["confidence_bound"]) for row in candidates
        ),
        "mechanism_observable_count": sum(
            bool(row["mechanism_observable"]) for row in candidates
        ),
        "primary_failure_count": sum(
            row.get("primary_failure_stage") is not None for row in candidates
        ),
        "pair_qc_fail_count": sum(
            row.get("pair_qc_status") == "pair_qc_fail" for row in candidates
        ),
        "protocol_eligibility_issue_count": sum(
            "protocol_eligibility_issue"
            in row.get("scientific_attrition_classes", [])
            for row in candidates
        ),
        "pipeline_verdict": verdict,
    }


def run_derivation(
    panel: Sequence[CandidateContext],
    config: DerivationConfig,
    paths: ProjectPaths,
) -> DerivationRunResult:
    """Run one generic scientific implementation over any registered subset."""
    if not panel:
        raise ValueError("derivation panel must not be empty")
    indices = [context.candidate_index for context in panel]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate candidate_index in derivation panel")
    identities = [context.identity.pair_id for context in panel]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate biological identity in derivation panel")
    biological_keys = [
        (
            context.identity.pdb_id,
            context.identity.chain_id,
            context.identity.uniprot_accession,
            context.identity.polymer_entity_id,
        )
        for context in panel
    ]
    if len(biological_keys) != len(set(biological_keys)):
        raise ValueError("duplicate biological identity in derivation panel")
    if any(context.protocol_binding != config.protocol_binding for context in panel):
        raise ValueError("candidate protocol binding does not match derivation config")

    candidates = tuple(
        derive_candidate(context, config, paths)
        for context in sorted(panel, key=lambda item: item.candidate_index)
    )
    records = [dict(candidate.report_record) for candidate in candidates]
    return DerivationRunResult(
        schema_version=config.schema_version,
        preflight=config.preflight,
        candidates=candidates,
        summary=_summary(records),
        attrition_bias_probe=_descriptive_attrition(records),
        scope=config.scope,
        selection_policy=config.selection_policy,
        report_metadata=config.report_metadata,
    )
