"""One reusable, strict raw-to-observable candidate derivation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dual_uq.confidence import load_plddt
from dual_uq.core.errors import PAEMappingError
from dual_uq.core.hashing import sha256_file
from dual_uq.core.paths import ProjectPaths
from dual_uq.schema import AmbiguousLegacyResidueIdentifier
from dual_uq.structure_io import load_chain_ca_table, residue_name_to_one_letter

from ..audits.observability import mechanism_evidence
from ..models import (
    CandidateContext,
    CandidateDerivationResult,
    DerivationConfig,
    DerivationError,
    LogicalAssetRef,
    StageResult,
    skipped_dependency_results,
)
from ..policies.fragments import (
    resolve_exact_fragment,
    validate_asset_model_binding,
    validate_bound_arrays,
    validate_frozen_model_artifacts,
)
from ..policies.identity import (
    extract_canonical_sequence,
    extract_prediction_record_sequence,
    metadata_records,
)
from ..services.afdb import load_pae_json
from ..services.mapping import (
    afdb_ca_by_uniprot,
    audit_sequence_discrepancies,
    compute_pair_quality,
    mapping_with_provenance,
    pair_qc_attrition,
    parse_sifts_mapping_with_explicit_labels,
    residue_mapping_audit_records,
)

# Preserve the architecture-checkpoint monkeypatch seam while routing the
# implementation through the explicit-label shared adapter.
parse_sifts_residue_mapping = parse_sifts_mapping_with_explicit_labels

_COMPATIBILITY_STAGES = (
    "raw",
    "canonical_sequence",
    "pair_qc",
    "mapping",
    "fragment",
    "pae",
    "confidence",
    "mechanism",
)
_STRUCTURED_STAGES = (
    "raw",
    "identity",
    "pair_qc",
    "mapping",
    "fragment",
    "pae",
    "confidence",
    "observability",
)
_COMPATIBILITY_TO_STRUCTURED = dict(
    zip(_COMPATIBILITY_STAGES, _STRUCTURED_STAGES, strict=True)
)


def verify_bound_hash(path: Path, expected: str) -> str:
    """Verify one immutable input without ever rebasing its declared digest."""
    if not path.is_file():
        raise DerivationError("missing_raw_asset", f"Missing raw asset: {path}")
    digest = sha256_file(path)
    if digest != expected:
        raise DerivationError(
            "raw_hash_drift",
            f"raw hash drift for {path}: expected {expected}, found {digest}",
        )
    return digest


def _verify_asset(asset: LogicalAssetRef, paths: ProjectPaths) -> tuple[Path, str]:
    path = paths.resolve_logical(asset.logical_path)
    if not path.is_file():
        raise DerivationError("missing_raw_asset", f"Missing raw asset: {path}")
    digest = (
        verify_bound_hash(path, asset.sha256)
        if asset.sha256 is not None
        else sha256_file(path)
    )
    return path, digest


def _candidate_base(context: CandidateContext) -> dict[str, Any]:
    evidence = context.evidence()
    identity = context.identity
    return {
        "candidate_index": context.candidate_index,
        "pair_id": identity.pair_id,
        "polymer_entity_id": identity.polymer_entity_id,
        "PDB": identity.pdb_id,
        "chain": identity.chain_id,
        "UniProt": identity.uniprot_accession,
        "selection_roles": list(context.selection_roles),
        "selection_reason": context.selection_reason,
        "prederivation_canonical_length": evidence.get(
            "prederivation_canonical_length"
        ),
        "global_pLDDT_proxy": evidence.get("global_pLDDT_proxy"),
        "prederivation_metadata_record_count": evidence.get(
            "prederivation_metadata_record_count"
        ),
        "pdb_entity_length": evidence.get("pdb_entity_length"),
        "raw_complete": False,
        "canonical_sequence_complete": False,
        "pair_qc_complete": False,
        "mapping_complete": False,
        "fragment_resolved": False,
        "pae_bound": False,
        "confidence_bound": False,
        "mechanism_observable": False,
        "primary_failure_stage": None,
        "primary_failure_code": None,
        "dependent_unavailable_stages": [],
        "attrition_class": None,
        "sampling_stratum_prior_recomputed": "uncertain_or_unclassified",
    }


def _stage_flag(result: Mapping[str, Any], stage: str) -> bool:
    suffix = {
        "raw": "raw_complete",
        "canonical_sequence": "canonical_sequence_complete",
        "pair_qc": "pair_qc_complete",
        "mapping": "mapping_complete",
        "fragment": "fragment_resolved",
        "pae": "pae_bound",
        "confidence": "confidence_bound",
        "mechanism": "mechanism_observable",
    }[stage]
    return bool(result.get(suffix))


def _current_failure_stage(result: Mapping[str, Any]) -> str:
    return next(
        stage for stage in _COMPATIBILITY_STAGES if not _stage_flag(result, stage)
    )


def _attrition_class(stage: str) -> str:
    return {
        "raw": "raw_availability_issue",
        "canonical_sequence": "identity_or_provenance_issue",
        "pair_qc": "mapping_issue",
        "mapping": "mapping_issue",
        "fragment": "fragment_or_model_issue",
        "pae": "fragment_or_model_issue",
        "confidence": "fragment_or_model_issue",
        "mechanism": "mechanism_unobservable",
    }[stage]


def account_stage_failure(
    stages: Mapping[str, Any],
    *,
    failure_stage: str,
    failure_code: str,
    attrition_class: str,
) -> dict[str, Any]:
    suffix = _COMPATIBILITY_STAGES[
        _COMPATIBILITY_STAGES.index(failure_stage) + 1 :
    ]
    dependent = [name for name in suffix if not _stage_flag(stages, name)]
    return {
        "primary_failure_stage": failure_stage,
        "primary_failure_code": failure_code,
        "attrition_class": attrition_class,
        "independent_failure_count": 1,
        "dependent_unavailable_stages": dependent,
    }


def _failed_record(
    result: dict[str, Any],
    *,
    stage: str,
    code: str,
    attrition_class: str,
    message: str | None = None,
) -> dict[str, Any]:
    result.update(
        account_stage_failure(
            result,
            failure_stage=stage,
            failure_code=code,
            attrition_class=attrition_class,
        )
    )
    result["failure_message"] = message
    if stage == "pair_qc":
        result["pair_qc_status"] = "pair_qc_unobservable"
        result.update(pair_qc_attrition("pair_qc_unobservable"))
    result["mechanism_observability_status"] = "still_unobservable"
    result["missing_evidence"] = result["dependent_unavailable_stages"]
    return result


def _structured_failure_stages(
    completed: Sequence[StageResult], *, compatibility_stage: str, code: str
) -> tuple[StageResult, ...]:
    failed_stage = _COMPATIBILITY_TO_STRUCTURED[compatibility_stage]
    completed_names = {stage.stage for stage in completed}
    prefix = tuple(completed)
    if failed_stage in completed_names:
        prefix = tuple(stage for stage in prefix if stage.stage != failed_stage)
    failed = StageResult(
        stage=failed_stage,
        status="unobservable" if compatibility_stage == "pair_qc" else "failed",
        primary_failure_code=code,
    )
    remaining = _STRUCTURED_STAGES[_STRUCTURED_STAGES.index(failed_stage) + 1 :]
    return prefix + (failed,) + skipped_dependency_results(
        upstream_stage=failed_stage,
        stages=remaining,
    )


def _result(
    context: CandidateContext,
    stages: Sequence[StageResult],
    record: Mapping[str, Any],
    evidence: Mapping[str, Any] | None = None,
) -> CandidateDerivationResult:
    return CandidateDerivationResult(
        context=context,
        stages=tuple(stages),
        evidence={} if evidence is None else evidence,
        report_record=record,
    )


def _failed_result(
    context: CandidateContext,
    completed: Sequence[StageResult],
    record: dict[str, Any],
    *,
    stage: str,
    code: str,
    attrition_class: str,
    message: str | None = None,
) -> CandidateDerivationResult:
    failed = _failed_record(
        record,
        stage=stage,
        code=code,
        attrition_class=attrition_class,
        message=message,
    )
    return _result(
        context,
        _structured_failure_stages(completed, compatibility_stage=stage, code=code),
        failed,
    )


def derive_candidate(
    context: CandidateContext,
    config: DerivationConfig,
    paths: ProjectPaths,
) -> CandidateDerivationResult:
    """Derive one candidate using strict identity, mapping and fragment contracts."""
    result = _candidate_base(context)
    completed: list[StageResult] = []
    raw_hashes: dict[str, str] = {}
    try:
        resolved_assets: dict[str, tuple[LogicalAssetRef, Path]] = {}
        for asset in context.assets:
            path, digest = _verify_asset(asset, paths)
            resolved_assets[asset.asset_type] = (asset, path)
            raw_hashes[asset.asset_type] = digest
        required = {
            "afdb_metadata",
            "pdb_mmcif",
            "sifts",
            "afdb_structure",
            "afdb_pae",
            "afdb_confidence",
        }
        missing = sorted(required - set(resolved_assets))
        if missing:
            raise DerivationError(
                "missing_raw_asset", f"Missing logical raw assets: {missing}"
            )
        result["raw_complete"] = True
        completed.append(
            StageResult(stage="raw", status="complete", artifacts=context.assets)
        )

        metadata_asset, metadata_path = resolved_assets["afdb_metadata"]
        metadata_bytes = metadata_path.read_bytes()
        prediction = extract_prediction_record_sequence(
            metadata_bytes, context.exact_afdb_accession
        )
        result.update(
            {
                "prediction_record_sequence_source_field": prediction[
                    "prediction_sequence_source_field"
                ],
                "prediction_record_sequence_length": prediction[
                    "prediction_sequence_length"
                ],
                "prediction_record_sequence_sha256": prediction[
                    "prediction_sequence_sha256"
                ],
                "prediction_record_interval": prediction["prediction_interval"],
                "metadata_path": metadata_asset.logical_path,
                "metadata_sha256": raw_hashes["afdb_metadata"],
                "metadata_provenance": metadata_asset.provenance,
                "metadata_record_count": prediction["metadata_record_count"],
                "exact_accession_prediction_record_count": prediction[
                    "prediction_record_count"
                ],
                "exact_model_identity": prediction["model_entity_id"],
                "nonselected_sibling_record_count": prediction[
                    "nonselected_sibling_record_count"
                ],
            }
        )
        if prediction["model_entity_id"] != context.expected_afdb_model_identity:
            raise DerivationError(
                "metadata_model_identity_mismatch",
                "Exact metadata model identity does not match candidate binding",
            )
        sequence = extract_canonical_sequence(
            metadata_bytes, context.exact_afdb_accession
        )
        records = metadata_records(metadata_bytes)
        result.update(
            {
                "canonical_sequence_complete": True,
                "canonical_sequence_source_field": sequence[
                    "sequence_source_field"
                ],
                "canonical_sequence_length": sequence["sequence_length"],
                "canonical_sequence_sha256": sequence["sequence_sha256"],
                "canonical_sequence_provenance": sequence[
                    "canonical_sequence_provenance"
                ],
            }
        )
        completed.append(
            StageResult(
                stage="identity",
                status="complete",
                metrics={
                    "sequence_length": sequence["sequence_length"],
                    "metadata_record_count": sequence["metadata_record_count"],
                },
                artifacts=(metadata_asset,),
            )
        )

        pdb_asset, pdb_path = resolved_assets["pdb_mmcif"]
        sifts_asset, sifts_path = resolved_assets["sifts"]
        result["raw_sha256"] = {
            "pdb_mmcif": raw_hashes["pdb_mmcif"],
            "sifts": raw_hashes["sifts"],
        }
        result["pdb_path"] = pdb_asset.logical_path
        result["sifts_path"] = sifts_asset.logical_path
        mapping = parse_sifts_residue_mapping(
            sifts_path,
            pdb_path,
            chain_id=context.identity.chain_id,
            uniprot_id=context.identity.uniprot_accession,
        )
        pdb_ca = load_chain_ca_table(pdb_path, context.identity.chain_id)
        pdb_entity_length = result.get("pdb_entity_length")
        if not isinstance(pdb_entity_length, int) or pdb_entity_length <= 0:
            raise DerivationError(
                "missing_pdb_entity_length",
                "Frozen discovery evidence lacks a positive PDB entity length",
            )
        pair = compute_pair_quality(
            mapping,
            pdb_ca,
            canonical_length=int(sequence["sequence_length"]),
            pdb_entity_length=pdb_entity_length,
            thresholds=config.preflight_thresholds,
        )
        result.update(pair)
        result.update(pair_qc_attrition(str(pair["pair_qc_status"])))
        result["pair_qc_complete"] = True
        completed.append(
            StageResult(
                stage="pair_qc",
                status="complete",
                warnings=(
                    ("pair_qc_threshold_not_met",)
                    if pair["pair_qc_status"] == "pair_qc_fail"
                    else ()
                ),
                metrics=pair,
                artifacts=(pdb_asset, sifts_asset),
            )
        )

        mapping, gaps, join_diagnostics = mapping_with_provenance(mapping, pdb_ca)
        result.update(
            {
                "mapping_complete": True,
                "mapped_residue_count": len(mapping),
                "observed_ca_count": int(mapping["observed_ca"].sum()),
                "observed_ca_fraction": float(mapping["observed_ca"].mean()),
                "gap_count": gaps["gap_count"],
                "segment_count": gaps["segment_count"],
                "largest_uniprot_gap": gaps["largest_uniprot_gap"],
                "gap_boundaries": gaps["gap_boundaries"],
                "nearest_gap_metadata_available": gaps[
                    "nearest_gap_metadata_available"
                ],
                "residue_provenance_fields": [
                    "auth_asym_id",
                    "auth_seq_id",
                    "insertion_code",
                    "label_asym_id",
                    "label_seq_id",
                ],
                "residue_join_diagnostics": join_diagnostics,
                "residue_mapping_records": residue_mapping_audit_records(mapping),
            }
        )
        completed.append(
            StageResult(
                stage="mapping",
                status="complete",
                metrics={
                    "mapped_residue_count": len(mapping),
                    "gap_count": gaps["gap_count"],
                    "segment_count": gaps["segment_count"],
                },
            )
        )

        interval = tuple(int(value) for value in pair["mapped_interval"])
        fragment_result = resolve_exact_fragment(
            records, context.exact_afdb_accession, interval
        )
        result.update(
            {
                key: value
                for key, value in fragment_result.items()
                if key != "selected_fragment"
            }
        )
        result["fragment_count"] = len(fragment_result["fragment_candidates"])
        fragment = fragment_result["selected_fragment"]
        if fragment is None:
            return _failed_result(
                context,
                completed,
                result,
                stage="fragment",
                code=str(fragment_result["fragment_resolution_status"]),
                attrition_class="fragment_or_model_issue",
            )
        model_id = fragment.model_entity_id
        version = sequence["exact_record"].get("latestVersion")
        if not isinstance(version, int):
            raise DerivationError(
                "invalid_model_version",
                "Exact prediction record lacks integer model version",
            )
        if model_id != context.expected_afdb_model_identity:
            raise DerivationError(
                "fragment_model_identity_mismatch",
                "Selected fragment model does not match candidate binding",
            )
        model_asset, model_path = resolved_assets["afdb_structure"]
        pae_asset, pae_path = resolved_assets["afdb_pae"]
        confidence_asset, confidence_path = resolved_assets["afdb_confidence"]
        dependent_assets = {
            "afdb_structure": model_asset,
            "afdb_pae": pae_asset,
            "afdb_confidence": confidence_asset,
        }
        result["asset_model_binding_mode"] = validate_asset_model_binding(
            selected_model_id=model_id,
            asset_records={
                name: (
                    {"exact_record_model_identity": asset.expected_model_identity}
                    if asset.expected_model_identity is not None
                    else None
                )
                for name, asset in dependent_assets.items()
            },
        )
        validate_frozen_model_artifacts(
            sequence["exact_record"],
            model_id=model_id,
            version=version,
            model_path=model_path,
            expected_length=fragment.model_residue_count,
        )
        result["fragment_resolved"] = True
        completed.append(
            StageResult(
                stage="fragment",
                status="complete",
                metrics={
                    "model_id": model_id,
                    "uniprot_start": fragment.uniprot_start,
                    "uniprot_end": fragment.uniprot_end,
                },
                artifacts=(model_asset,),
            )
        )
        for asset_type, asset in dependent_assets.items():
            result[f"{asset_type}_provenance"] = asset.provenance
            result["raw_sha256"][asset_type] = raw_hashes[asset_type]
        pae_matrix = load_pae_json(pae_path, fragment).values
        result["pae_bound"] = True
        completed.append(
            StageResult(stage="pae", status="complete", artifacts=(pae_asset,))
        )

        confidence = load_plddt(
            confidence_path, expected_length=fragment.model_residue_count
        )
        pae_matrix, confidence = validate_bound_arrays(
            fragment, pae_matrix, confidence
        )
        result["confidence_bound"] = True
        completed.append(
            StageResult(
                stage="confidence",
                status="complete",
                artifacts=(confidence_asset,),
            )
        )
        result["selected_exact_model"] = model_id
        result["selected_model_version"] = version
        result["selected_model_path"] = model_asset.logical_path
        result["pae_path"] = pae_asset.logical_path
        result["confidence_path"] = confidence_asset.logical_path

        afdb_sequence, afdb_ca = afdb_ca_by_uniprot(model_path, fragment)
        mapping = mapping.copy()
        mapping["mapping_aa"] = mapping["uniprot_residue_name"].map(
            residue_name_to_one_letter
        )
        pdb_sequence = dict(
            zip(
                mapping["uniprot_position"].astype(int),
                mapping["pdb_residue_name"].map(residue_name_to_one_letter),
                strict=True,
            )
        )
        discrepancies = audit_sequence_discrepancies(
            mapping[["uniprot_position", "mapping_aa"]],
            pdb_sequence,
            afdb_sequence,
        )
        result["sequence_discrepancies"] = discrepancies
        result["mapping_vs_pdb_mismatch_count"] = len(
            discrepancies["mapping_vs_pdb"]
        )
        result["mapping_vs_afdb_mismatch_count"] = len(
            discrepancies["mapping_vs_afdb"]
        )
        result["pdb_vs_afdb_mismatch_count"] = len(
            discrepancies["pdb_vs_afdb"]
        )
        evidence = mechanism_evidence(
            mapping,
            pdb_ca,
            afdb_ca,
            pae_matrix,
            confidence,
            fragment,
            {"thresholds": dict(config.observability_thresholds)},
            quality_pass=result.get("preflight_status") == "pass_full_length",
        )
        result.update(evidence)
        result["mechanism_observable"] = True
        completed.append(
            StageResult(stage="observability", status="complete", metrics=evidence)
        )
        return _result(context, completed, result, evidence)
    except PAEMappingError as exc:
        stage = "pae" if result["fragment_resolved"] else "fragment"
        return _failed_result(
            context,
            completed,
            result,
            stage=stage,
            code=exc.code,
            attrition_class="fragment_or_model_issue",
        )
    except AmbiguousLegacyResidueIdentifier as exc:
        return _failed_result(
            context,
            completed,
            result,
            stage="pair_qc",
            code="ambiguous_sifts_author_residue_identifier",
            attrition_class="mapping_issue",
            message=str(exc),
        )
    except LookupError as exc:
        return _failed_result(
            context,
            completed,
            result,
            stage="pair_qc",
            code="no_sifts_mapping",
            attrition_class="mapping_issue",
            message=str(exc),
        )
    except DerivationError as exc:
        stage = (
            "raw"
            if exc.code in {"missing_raw_asset", "raw_hash_drift"}
            and not completed
            else _current_failure_stage(result)
        )
        if stage == "raw":
            result["raw_complete"] = False
        return _failed_result(
            context,
            completed,
            result,
            stage=stage,
            code=exc.code,
            attrition_class=_attrition_class(stage),
        )
    except Exception as exc:  # noqa: BLE001 - preserve per-candidate isolation
        stage = _current_failure_stage(result)
        return _failed_result(
            context,
            completed,
            result,
            stage=stage,
            code=f"unexpected_{type(exc).__name__}",
            attrition_class="implementation_or_schema_issue",
            message=str(exc),
        )
