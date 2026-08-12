"""Freeze an outcome-blind confirmatory cohort, probe space, and score plan."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from io import BytesIO
from numbers import Integral
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from dual_uq.core.hashing import sha256_bytes, sha256_canonical, sha256_file
from dual_uq.core.paths import ProjectPaths
from dual_uq.dataset.fixed_probe_scoring import (
    SCORING_PROTOCOL_VERSION,
    FixedProbeScoringError,
    VerifiedModelIdentity,
    load_stage0_scoring_config,
    verify_authorized_model,
)
from dual_uq.dataset.fixed_probes import build_fixed_probe_candidates
from dual_uq.dataset.redundancy_diversity import (
    RedundancyDiversityError,
    assign_redundancy_roles,
)
from dual_uq.dataset.services.proteinmpnn_scoring import (
    DECODING_REALIZATION_ALGORITHM,
    ProteinMPNNScoringError,
    ScoringBackboneProjection,
    _coordinates,
    implementation_worktree_is_clean,
    make_decoding_realization,
    normalized_probe_score_record,
    normalized_wt_score_record,
    validate_paired_projection,
)
from dual_uq.dataset.storage.proteinmpnn import (
    STANDARD_AMINO_ACIDS,
    sequence_sha256,
)
from dual_uq.inference.formal import (
    FormalRequestDefinition,
    HistoricalReuseDecision,
    HistoricalReuseStatus,
    derive_orchestration_id,
)
from dual_uq.models.proteinmpnn import ProteinMPNNStructureInput
from dual_uq.models.scoring import (
    CandidateCollection,
    ScorerBinding,
    ScoreRecord,
    ScoreRequest,
    ScoringVariant,
    VariantKind,
)
from dual_uq.structure import StructuralIntervention, StructureCondition

BLOCKED_INPUT_INTEGRITY = "BLOCKED_INPUT_INTEGRITY"
BLOCKED_REDUNDANCY_REGRESSION = "BLOCKED_REDUNDANCY_REGRESSION"
BLOCKED_REPRESENTATIVE_SELECTION = "BLOCKED_REPRESENTATIVE_SELECTION"
BLOCKED_COMMON_MASK_INTEGRITY = "BLOCKED_COMMON_MASK_INTEGRITY"
BLOCKED_PROBE_INTEGRITY = "BLOCKED_PROBE_INTEGRITY"
BLOCKED_SCORER_INTEGRITY = "BLOCKED_SCORER_INTEGRITY"
BLOCKED_FINAL_MANIFEST = "BLOCKED_FINAL_MANIFEST"
BLOCKED_REUSE_POLICY = "BLOCKED_REUSE_POLICY"
FINAL_PROTOCOL_FROZEN = "SCALE1B_V2_FINAL_CONFIRMATORY_PROTOCOL_FROZEN"

_ORIGINAL = "ORIGINAL_213_FRAME"
_WAVE1 = "EXPANSION_WAVE1"
_WAVE2 = "EXPANSION_WAVE2"
_FINAL_CLUSTER_N = 127
_REPEATS = tuple(range(30))

_V2_ROOT = Path("experiments/p2_design_baseline/scale1/scale1b_v2")
_V2_FROZEN_BINDINGS = {
    "primary_cohort": (
        "scale1b_v2_primary_cohort.parquet",
        "105d7cab10ea530055b011f600021516aa991edaf4d38c44a49eb5f911b7f793",
    ),
    "primary_common_masks": (
        "scale1b_v2_primary_common_masks.parquet",
        "3d3417b2db651bdba666b1b02c9d5378ee3e5a05b8d1ad36af56cac756dfe36b",
    ),
    "fixed_probes": (
        "scale1b_v2_fixed_probes.parquet",
        "92494b48a5ffb3fc0707b110150d78189edb984f8dcccf5d5db8ba80ab0fe56f",
    ),
    "scoring_plan": (
        "scale1b_v2_proteinmpnn_scoring_plan.parquet",
        "5739996a7a0e03c07158d7afc50744042fc438d16010c94729d34ccc4364e69c",
    ),
    "scoring_protocol": (
        "scale1b_v2_scoring_protocol.json",
        "1ce49ca2dce3e03b41d93ed4658674c289600e014e0e0818fbbd875bdff443aa",
    ),
    "freeze_manifest": (
        "scale1b_v2_freeze_manifest.json",
        "9e41cb2a1116f782564be0559cd24f5335c31b21dcae87d268e9cb02ff685579",
    ),
}
_V2_AGGREGATE_FINGERPRINT = (
    "378c15043e5b46e743c799a42e574524189348de474ff052f387de48ebde49f9"
)
_STAGE0_ROOT = Path("experiments/p2_design_baseline/stage0")

FROZEN_PLAN_SCIENTIFIC_FIELDS = (
    "protein_id",
    "structure_condition",
    "structure_sha256",
    "common_mask_binding",
    "protein_probe_identity_sha256",
    "repeat",
    "seed",
    "explicit_realization_id",
    "realization_algorithm",
    "proteinmpnn_source_commit",
    "checkpoint_sha256",
    "score_contract_id",
)


class FinalConfirmatoryProtocolError(RuntimeError):
    """One structured final-protocol freeze blocker."""

    def __init__(
        self, status: str, code: str, message: str, **details: Any
    ) -> None:
        self.status = status
        self.code = code
        self.details = details
        super().__init__(message)


@dataclass(frozen=True)
class FrozenPlanStructureView:
    """Read-only structural interpretation of one frozen scoring-plan row."""

    condition: StructureCondition
    intervention: StructuralIntervention
    common_mask_binding: str
    scientific_identity: Mapping[str, str | int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scientific_identity",
            MappingProxyType(dict(self.scientific_identity)),
        )


@dataclass(frozen=True)
class FrozenPlanScoreRequestView:
    """Compatibility view that keeps historical plan fields out of models."""

    request: ScoreRequest
    scorer_binding: ScorerBinding
    intervention: StructuralIntervention
    historical_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "historical_metadata",
            MappingProxyType(dict(self.historical_metadata)),
        )


class HistoricalReusePolicyStatus(str, Enum):
    """Frozen protocol decision, resolved before inspecting source results."""

    AUTHORIZED = "HISTORICAL_REUSE_AUTHORIZED"
    NOT_AUTHORIZED = "HISTORICAL_REUSE_NOT_AUTHORIZED"


@dataclass(frozen=True, slots=True)
class HistoricalReusePolicyDecision:
    """Authoritative workflow-level reuse authorization evidence."""

    status: HistoricalReusePolicyStatus
    policy_identity: str
    evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FinalConfirmatoryFormalBundle:
    """Validated frozen requests plus derived historical compatibility decisions."""

    definitions: tuple[FormalRequestDefinition, ...]
    reuse_policy: HistoricalReusePolicyDecision
    historical_reuse_candidates_reconstructed: int
    historical_reuse_accepted: int
    historical_reuse_rejected: int
    frozen_request_aggregate_sha256: str
    frozen_artifacts: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "frozen_artifacts",
            tuple(MappingProxyType(dict(record)) for record in self.frozen_artifacts),
        )


def frozen_plan_scientific_view(row: Mapping[str, Any]) -> dict[str, str | int]:
    """Return the measurement identity of one frozen logical scoring row."""
    missing = set(FROZEN_PLAN_SCIENTIFIC_FIELDS) - set(row)
    if missing:
        raise FinalConfirmatoryProtocolError(
            BLOCKED_FINAL_MANIFEST,
            "logical_scoring_plan_schema_mismatch",
            "Frozen scoring-plan row lacks scientific identity fields",
            missing_fields=sorted(missing),
        )
    view: dict[str, str | int] = {}
    for field in FROZEN_PLAN_SCIENTIFIC_FIELDS:
        value = row[field]
        if field in {"repeat", "seed"}:
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise FinalConfirmatoryProtocolError(
                    BLOCKED_FINAL_MANIFEST,
                    "logical_scoring_plan_schema_mismatch",
                    f"Frozen scoring-plan {field} must be an integer",
                )
            view[field] = int(value)
        elif not isinstance(value, str) or not value:
            raise FinalConfirmatoryProtocolError(
                BLOCKED_FINAL_MANIFEST,
                "logical_scoring_plan_schema_mismatch",
                f"Frozen scoring-plan {field} must be a non-empty string",
            )
        else:
            view[field] = value
    return view


def frozen_plan_scientific_fingerprint(row: Mapping[str, Any]) -> str:
    """Hash only fields that change the frozen scientific measurement."""
    return sha256_canonical(frozen_plan_scientific_view(row))


def interpret_frozen_scoring_plan(
    plan: pd.DataFrame,
) -> tuple[FrozenPlanStructureView, ...]:
    """Interpret, but never regenerate, the frozen PDB/AFDB scoring plan."""
    required = {
        *FROZEN_PLAN_SCIENTIFIC_FIELDS,
        "structure_artifact_reference",
    }
    missing = sorted(required - set(plan.columns))
    if missing:
        raise FinalConfirmatoryProtocolError(
            BLOCKED_FINAL_MANIFEST,
            "logical_scoring_plan_schema_mismatch",
            "Frozen scoring plan lacks structural interpretation fields",
            missing_fields=missing,
        )

    records = plan.to_dict("records")
    conditions: dict[tuple[str, str], StructureCondition] = {}
    masks_by_protein: dict[str, set[str]] = {}
    for row in records:
        scientific = frozen_plan_scientific_view(row)
        protein_id = str(scientific["protein_id"])
        condition_id = str(scientific["structure_condition"])
        source = {"PDB": "PDB", "AFDB": "AlphaFoldDB"}.get(condition_id)
        if source is None:
            raise FinalConfirmatoryProtocolError(
                BLOCKED_FINAL_MANIFEST,
                "unsupported_frozen_structure_condition",
                "Frozen Scale-1B-v2 condition must be PDB or AFDB",
                condition_id=condition_id,
            )
        condition = StructureCondition(
            protein_id=protein_id,
            condition_id=condition_id,
            source=source,
            structure_sha256=str(scientific["structure_sha256"]),
            structure_locator=str(row["structure_artifact_reference"]),
        )
        key = (protein_id, condition_id)
        existing = conditions.setdefault(key, condition)
        if existing.scientific_identity != condition.scientific_identity:
            raise FinalConfirmatoryProtocolError(
                BLOCKED_FINAL_MANIFEST,
                "frozen_structure_condition_drift",
                "Frozen condition identity changes across logical scoring rows",
                protein_id=protein_id,
                condition_id=condition_id,
            )
        masks_by_protein.setdefault(protein_id, set()).add(
            str(scientific["common_mask_binding"])
        )

    interventions: dict[str, StructuralIntervention] = {}
    for protein_id, masks in masks_by_protein.items():
        if len(masks) != 1 or (protein_id, "PDB") not in conditions or (
            protein_id,
            "AFDB",
        ) not in conditions:
            raise FinalConfirmatoryProtocolError(
                BLOCKED_FINAL_MANIFEST,
                "frozen_structural_intervention_mismatch",
                "Frozen PDB/AFDB pair or external common-mask binding is inconsistent",
                protein_id=protein_id,
            )
        interventions[protein_id] = StructuralIntervention(
            intervention_id=f"{protein_id}::PDB__AFDB",
            condition_a=conditions[(protein_id, "PDB")],
            condition_b=conditions[(protein_id, "AFDB")],
        )

    views = []
    for row in records:
        protein_id = str(row["protein_id"])
        condition = conditions[(protein_id, str(row["structure_condition"]))]
        interpreted_identity = frozen_plan_scientific_view(row)
        interpreted_identity.update(
            {
                "protein_id": condition.protein_id,
                "structure_condition": condition.condition_id,
                "structure_sha256": condition.structure_sha256,
            }
        )
        views.append(
            FrozenPlanStructureView(
                condition=condition,
                intervention=interventions[protein_id],
                common_mask_binding=str(row["common_mask_binding"]),
                scientific_identity=interpreted_identity,
            )
        )
    return tuple(views)


def frozen_score_request_scientific_view(
    view: FrozenPlanScoreRequestView,
) -> dict[str, str | int]:
    """Project an R3 request plus external scorer binding to the R0 identity."""
    request = view.request
    binding = view.scorer_binding
    if binding.score_contract_id != request.score_contract_id:
        raise FinalConfirmatoryProtocolError(
            BLOCKED_FINAL_MANIFEST,
            "frozen_scorer_contract_mismatch",
            "Frozen scorer and request score contracts differ",
        )
    return {
        "protein_id": request.protein_id,
        "structure_condition": request.condition.condition_id,
        "structure_sha256": request.condition.structure_sha256,
        "common_mask_binding": request.scoring_domain_id,
        "protein_probe_identity_sha256": request.candidate_collection.collection_id,
        "repeat": request.repeat_index,
        "seed": request.seed,
        "explicit_realization_id": request.realization_id,
        "realization_algorithm": request.realization_algorithm,
        "proteinmpnn_source_commit": binding.implementation_id,
        "checkpoint_sha256": str(binding.checkpoint_id),
        "score_contract_id": request.score_contract_id,
    }


def adapt_frozen_scoring_plan(
    *,
    plan: pd.DataFrame,
    primary_cohort: pd.DataFrame,
    common_masks: pd.DataFrame,
    fixed_probes: pd.DataFrame,
) -> tuple[FrozenPlanScoreRequestView, ...]:
    """Interpret frozen Scale-1B-v2 rows as model-independent requests.

    This is an integrity-preserving compatibility adapter. It consumes frozen
    masks and probes and never invokes their scientific generators.
    """
    cohort_required = {
        "protein_id",
        "canonical_sequence",
        "canonical_sequence_sha256",
    }
    mask_required = {"protein_id", "canonical_position", "common_mask"}
    probe_required = {
        "protein_id",
        "canonical_position",
        "wt_aa",
        "candidate_aa",
        "sequence_hash",
        "full_sequence",
    }
    missing = {
        "primary_cohort": sorted(cohort_required - set(primary_cohort.columns)),
        "common_masks": sorted(mask_required - set(common_masks.columns)),
        "fixed_probes": sorted(probe_required - set(fixed_probes.columns)),
    }
    missing = {name: fields for name, fields in missing.items() if fields}
    if missing:
        raise FinalConfirmatoryProtocolError(
            BLOCKED_FINAL_MANIFEST,
            "frozen_score_request_input_schema_mismatch",
            "Frozen request adapter inputs lack required fields",
            missing_fields=missing,
        )

    structure_views = interpret_frozen_scoring_plan(plan)
    selected_ids = tuple(dict.fromkeys(plan["protein_id"].astype(str)))
    cohort = primary_cohort.loc[
        primary_cohort["protein_id"].astype(str).isin(selected_ids)
    ]
    if cohort["protein_id"].astype(str).duplicated().any() or set(
        cohort["protein_id"].astype(str)
    ) != set(selected_ids):
        raise FinalConfirmatoryProtocolError(
            BLOCKED_FINAL_MANIFEST,
            "frozen_score_request_cohort_mismatch",
            "Frozen request proteins require one canonical cohort row",
        )
    cohort_by_id = {
        str(row["protein_id"]): row for row in cohort.to_dict("records")
    }
    mask_by_id = {
        str(protein_id): tuple(
            group.loc[group["common_mask"].eq(True), "canonical_position"]
            .astype(int)
            .tolist()
        )
        for protein_id, group in common_masks.loc[
            common_masks["protein_id"].astype(str).isin(selected_ids)
        ].groupby("protein_id", sort=False)
    }
    probe_by_id = {
        str(protein_id): group
        for protein_id, group in fixed_probes.loc[
            fixed_probes["protein_id"].astype(str).isin(selected_ids)
        ].groupby("protein_id", sort=False)
    }

    collections: dict[str, CandidateCollection] = {}
    mask_bindings: dict[str, str] = {}
    for protein_id in selected_ids:
        protein = cohort_by_id[protein_id]
        canonical_sequence = str(protein["canonical_sequence"])
        canonical_hash = str(protein["canonical_sequence_sha256"])
        if sequence_sha256(canonical_sequence) != canonical_hash:
            raise FinalConfirmatoryProtocolError(
                BLOCKED_FINAL_MANIFEST,
                "frozen_canonical_sequence_hash_mismatch",
                "Frozen canonical sequence does not match its content hash",
                protein_id=protein_id,
            )
        positions = mask_by_id.get(protein_id, ())
        probes = probe_by_id.get(protein_id)
        if not positions or probes is None or probes.empty:
            raise FinalConfirmatoryProtocolError(
                BLOCKED_FINAL_MANIFEST,
                "frozen_measurement_payload_missing",
                "Frozen request lacks a common mask or probe collection",
                protein_id=protein_id,
            )
        mask_bindings[protein_id] = sha256_canonical(
            {
                "protein_id": protein_id,
                "canonical_positions": list(positions),
                "canonical_sequence_sha256": canonical_hash,
            }
        )
        probe_rows = probes.to_dict("records")
        probe_binding = sha256_canonical(
            {"sequence_hashes": [str(row["sequence_hash"]) for row in probe_rows]}
        )
        variants = []
        for row in probe_rows:
            position = int(row["canonical_position"])
            wt_aa = str(row["wt_aa"])
            mut_aa = str(row["candidate_aa"])
            sequence = str(row["full_sequence"])
            if (
                position > len(canonical_sequence)
                or canonical_sequence[position - 1] != wt_aa
                or len(sequence) != len(canonical_sequence)
                or sequence[: position - 1] + wt_aa + sequence[position:]
                != canonical_sequence
            ):
                raise FinalConfirmatoryProtocolError(
                    BLOCKED_FINAL_MANIFEST,
                    "frozen_probe_sequence_mismatch",
                    "Frozen probe is not the declared canonical substitution",
                    protein_id=protein_id,
                    canonical_position=position,
                )
            digest = str(row["sequence_hash"])
            variants.append(
                ScoringVariant(
                    variant_kind=VariantKind.PROBE,
                    variant_id=digest,
                    sequence_hash=digest,
                    sequence=sequence,
                    position=position,
                    wt_aa=wt_aa,
                    mut_aa=mut_aa,
                )
            )
        collections[protein_id] = CandidateCollection(
            collection_id=probe_binding,
            wt=ScoringVariant(
                variant_kind=VariantKind.WT,
                variant_id=None,
                sequence_hash=canonical_hash,
                sequence=canonical_sequence,
                position=None,
                wt_aa=None,
                mut_aa=None,
            ),
            probes=tuple(variants),
        )

    results = []
    for row, structure_view in zip(
        plan.to_dict("records"), structure_views, strict=True
    ):
        scientific = frozen_plan_scientific_view(row)
        protein_id = str(scientific["protein_id"])
        if (
            str(scientific["common_mask_binding"]) != mask_bindings[protein_id]
            or str(scientific["protein_probe_identity_sha256"])
            != collections[protein_id].collection_id
        ):
            raise FinalConfirmatoryProtocolError(
                BLOCKED_FINAL_MANIFEST,
                "frozen_measurement_binding_mismatch",
                "Frozen plan does not bind the supplied mask/probe payload",
                protein_id=protein_id,
            )
        scorer_binding = ScorerBinding(
            scorer_id="ProteinMPNN",
            implementation_id=str(scientific["proteinmpnn_source_commit"]),
            checkpoint_id=str(scientific["checkpoint_sha256"]),
            score_contract_id=str(scientific["score_contract_id"]),
        )
        request = ScoreRequest(
            condition=structure_view.condition,
            scoring_domain_id=mask_bindings[protein_id],
            canonical_positions=mask_by_id[protein_id],
            candidate_collection=collections[protein_id],
            repeat_index=int(scientific["repeat"]),
            seed=int(scientific["seed"]),
            realization_id=str(scientific["explicit_realization_id"]),
            realization_algorithm=str(scientific["realization_algorithm"]),
            score_contract_id=str(scientific["score_contract_id"]),
        )
        results.append(
            FrozenPlanScoreRequestView(
                request=request,
                scorer_binding=scorer_binding,
                intervention=structure_view.intervention,
                historical_metadata={
                    key: value
                    for key, value in row.items()
                    if key not in FROZEN_PLAN_SCIENTIFIC_FIELDS
                }
                | {
                    "cluster_id_30": row.get("cluster_id_30"),
                    "source_stratum": row.get("source_stratum"),
                },
            )
        )
    return tuple(results)


def resolve_historical_reuse_policy(
    scoring_protocol: Mapping[str, Any], freeze_manifest: Mapping[str, Any]
) -> HistoricalReusePolicyDecision:
    """Resolve reuse authorization only from the frozen target protocol."""
    contract = scoring_protocol.get("historical_result_reuse_contract")
    rule_frozen = freeze_manifest.get("HISTORICAL_SCORE_RESULT_REUSE_RULE_FROZEN")
    unexecuted_not_result = freeze_manifest.get(
        "UNEXECUTED_V1_PLAN_NOT_COUNTED_AS_SCORE_RESULT"
    )
    if (
        isinstance(contract, Mapping)
        and contract.get("exact_compatibility_required") is True
        and rule_frozen is True
        and unexecuted_not_result is True
    ):
        evidence = (
            "historical_result_reuse_contract",
            "HISTORICAL_SCORE_RESULT_REUSE_RULE_FROZEN",
            "UNEXECUTED_V1_PLAN_NOT_COUNTED_AS_SCORE_RESULT",
        )
        return HistoricalReusePolicyDecision(
            status=HistoricalReusePolicyStatus.AUTHORIZED,
            policy_identity=sha256_canonical(
                {
                    "contract": dict(contract),
                    "manifest_flags": {
                        evidence[1]: True,
                        evidence[2]: True,
                    },
                }
            ),
            evidence=evidence,
        )
    if (
        isinstance(contract, Mapping)
        and contract.get("reuse_authorized") is False
        and rule_frozen is True
    ):
        return HistoricalReusePolicyDecision(
            status=HistoricalReusePolicyStatus.NOT_AUTHORIZED,
            policy_identity=sha256_canonical(
                {"contract": dict(contract), "rule_frozen": True}
            ),
            evidence=(
                "historical_result_reuse_contract.reuse_authorized=false",
                "HISTORICAL_SCORE_RESULT_REUSE_RULE_FROZEN",
            ),
        )
    raise FinalConfirmatoryProtocolError(
        BLOCKED_REUSE_POLICY,
        "historical_reuse_policy_ambiguous",
        "Frozen Scale-1B-v2 artifacts do not resolve historical-result reuse",
    )


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FinalConfirmatoryProtocolError(
            BLOCKED_INPUT_INTEGRITY,
            "invalid_frozen_json",
            f"Unable to load frozen {label}",
            path=path.as_posix(),
        ) from exc
    if not isinstance(value, dict):
        raise FinalConfirmatoryProtocolError(
            BLOCKED_INPUT_INTEGRITY,
            "invalid_frozen_json",
            f"Frozen {label} must contain an object",
            path=path.as_posix(),
        )
    return value


def _load_v2_formal_inputs(
    project_root: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
    tuple[Mapping[str, Any], ...],
]:
    root = project_root.resolve()
    paths = {}
    bindings = []
    for label, (filename, expected_sha) in _V2_FROZEN_BINDINGS.items():
        path = root / _V2_ROOT / filename
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise FinalConfirmatoryProtocolError(
                BLOCKED_INPUT_INTEGRITY,
                "frozen_scale1b_v2_hash_mismatch",
                f"Frozen Scale-1B-v2 {label} differs",
                path=(_V2_ROOT / filename).as_posix(),
            )
        paths[label] = path
        bindings.append(
            MappingProxyType(
                {
                    "label": label,
                    "path": (_V2_ROOT / filename).as_posix(),
                    "sha256": expected_sha,
                }
            )
        )
    try:
        cohort = pd.read_parquet(paths["primary_cohort"])
        masks = pd.read_parquet(paths["primary_common_masks"])
        probes = pd.read_parquet(paths["fixed_probes"])
        plan = pd.read_parquet(paths["scoring_plan"])
    except (OSError, ValueError) as exc:
        raise FinalConfirmatoryProtocolError(
            BLOCKED_INPUT_INTEGRITY,
            "invalid_frozen_parquet",
            "Unable to load frozen Scale-1B-v2 formal inputs",
        ) from exc
    protocol = _load_json_object(paths["scoring_protocol"], "scoring protocol")
    manifest = _load_json_object(paths["freeze_manifest"], "freeze manifest")
    if (
        len(cohort) != 127
        or cohort["protein_id"].nunique() != 127
        or cohort["cluster_id_30"].nunique() != 127
        or len(masks) != 27_291
        or len(probes) != 518_529
        or len(plan) != 7_620
    ):
        raise FinalConfirmatoryProtocolError(
            BLOCKED_INPUT_INTEGRITY,
            "frozen_scale1b_v2_cardinality_mismatch",
            "Frozen Scale-1B-v2 cardinalities differ",
        )
    fingerprints = [
        frozen_plan_scientific_fingerprint(row) for row in plan.to_dict("records")
    ]
    if (
        len(set(fingerprints)) != 7_620
        or sha256_canonical({"fingerprints": fingerprints})
        != _V2_AGGREGATE_FINGERPRINT
    ):
        raise FinalConfirmatoryProtocolError(
            BLOCKED_INPUT_INTEGRITY,
            "frozen_scale1b_v2_fingerprint_mismatch",
            "Frozen Scale-1B-v2 aggregate scientific fingerprint differs",
        )
    return cohort, masks, probes, plan, protocol, manifest, tuple(bindings)


def _formal_definitions(
    *,
    plan: pd.DataFrame,
    cohort: pd.DataFrame,
    masks: pd.DataFrame,
    probes: pd.DataFrame,
) -> tuple[FormalRequestDefinition, ...]:
    views = adapt_frozen_scoring_plan(
        plan=plan,
        primary_cohort=cohort,
        common_masks=masks,
        fixed_probes=probes,
    )
    definitions = []
    for row, view in zip(plan.to_dict("records"), views, strict=True):
        scientific = frozen_score_request_scientific_view(view)
        fingerprint = frozen_plan_scientific_fingerprint(scientific)
        definitions.append(
            FormalRequestDefinition(
                scientific_fingerprint=fingerprint,
                orchestration_id=derive_orchestration_id(fingerprint),
                workflow_request_id=str(row["logical_shard_id"]),
                request=view.request,
                scorer_binding=view.scorer_binding,
                workflow_metadata={
                    "cluster_id_30": row.get("cluster_id_30"),
                    "source_stratum": row.get("source_stratum"),
                    "structure_artifact_reference": row.get(
                        "structure_artifact_reference"
                    ),
                },
            )
        )
    return tuple(definitions)


def _stage0_historical_reuse_decisions(
    *,
    project_root: Path,
    definitions: tuple[FormalRequestDefinition, ...],
    policy: HistoricalReusePolicyDecision,
) -> dict[str, HistoricalReuseDecision]:
    if policy.status is not HistoricalReusePolicyStatus.AUTHORIZED:
        return {}
    root = project_root.resolve()
    stage_root = root / _STAGE0_ROOT
    manifest_path = stage_root / "fixed_probe_scoring_manifest.json"
    protein_manifest_path = stage_root / "protein_manifest.json"
    probes_path = stage_root / "fixed_probe_candidates.parquet"
    manifest = _load_json_object(manifest_path, "Stage0 scoring manifest")
    protein_manifest = _load_json_object(
        protein_manifest_path, "Stage0 protein manifest"
    )
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise FinalConfirmatoryProtocolError(
            BLOCKED_INPUT_INTEGRITY,
            "invalid_historical_scoring_manifest",
            "Stage0 scoring manifest lacks output bindings",
        )
    source_paths = {}
    for key in ("fixed_probe_scores", "wt_scores"):
        record = outputs.get(key)
        if not isinstance(record, Mapping):
            raise FinalConfirmatoryProtocolError(
                BLOCKED_INPUT_INTEGRITY,
                "invalid_historical_scoring_manifest",
                f"Stage0 scoring manifest lacks {key}",
            )
        path = root / str(record.get("path", ""))
        expected = str(record.get("sha256", ""))
        if not path.is_file() or sha256_file(path) != expected:
            raise FinalConfirmatoryProtocolError(
                BLOCKED_INPUT_INTEGRITY,
                "historical_score_artifact_hash_mismatch",
                f"Historical Stage0 {key} differs",
            )
        source_paths[key] = path
    proteins = protein_manifest.get("proteins")
    if not isinstance(proteins, list):
        raise FinalConfirmatoryProtocolError(
            BLOCKED_INPUT_INTEGRITY,
            "invalid_historical_protein_manifest",
            "Stage0 protein manifest lacks proteins",
        )
    protein_by_id = {str(row["protein_id"]): row for row in proteins}
    target_ids = {definition.request.protein_id for definition in definitions}
    candidate_ids = set(protein_by_id).intersection(target_ids)
    try:
        historical_probes = pd.read_parquet(probes_path).loc[
            lambda frame: frame["protein_id"].astype(str).isin(candidate_ids)
        ]
        filters = [("protein_id", "in", sorted(candidate_ids))]
        raw = pd.read_parquet(
            source_paths["fixed_probe_scores"], filters=filters
        )
        wt = pd.read_parquet(source_paths["wt_scores"], filters=filters)
    except (OSError, ValueError) as exc:
        raise FinalConfirmatoryProtocolError(
            BLOCKED_INPUT_INTEGRITY,
            "invalid_historical_score_artifact",
            "Unable to load historical Stage0 score artifacts",
        ) from exc
    source_execution_identity = sha256_file(manifest_path)
    source_artifact_reference = (
        "experiments/p2_design_baseline/stage0/fixed_probe_scoring_manifest.json"
        "#outputs(fixed_probe_scores,wt_scores)"
    )
    source_artifact_sha256 = sha256_canonical(
        {
            key: {
                "path": str(outputs[key]["path"]),
                "sha256": str(outputs[key]["sha256"]),
            }
            for key in ("fixed_probe_scores", "wt_scores")
        }
    )
    model = manifest.get("model")
    scoring = manifest.get("scoring_protocol")
    if not isinstance(model, Mapping) or not isinstance(scoring, Mapping):
        raise FinalConfirmatoryProtocolError(
            BLOCKED_INPUT_INTEGRITY,
            "invalid_historical_scoring_manifest",
            "Stage0 scoring model/protocol binding is absent",
        )
    realization_records = manifest.get("decoding_realizations", {}).get("records")
    if not isinstance(realization_records, list):
        raise FinalConfirmatoryProtocolError(
            BLOCKED_INPUT_INTEGRITY,
            "invalid_historical_scoring_manifest",
            "Stage0 realization provenance is absent",
        )
    realization_by_key = {
        (str(row["protein_id"]), int(row["repeat_index"])): row
        for row in realization_records
    }
    source_candidates_by_protein = {
        str(protein_id): tuple(
            (
                str(row.sequence_hash),
                int(row.position),
                str(row.wt_aa),
                str(row.mut_aa),
                str(row.full_sequence),
            )
            for row in group.itertuples(index=False)
        )
        for protein_id, group in historical_probes.groupby("protein_id", sort=False)
    }
    wt_by_request = {
        (str(protein_id), str(condition), int(repeat)): group
        for (protein_id, condition, repeat), group in wt.groupby(
            ["protein_id", "backbone_condition", "repeat_index"], sort=False
        )
    }
    raw_by_request = {
        (str(protein_id), str(condition), int(repeat)): group
        for (protein_id, condition, repeat), group in raw.groupby(
            ["protein_id", "backbone_condition", "repeat_index"], sort=False
        )
    }
    decisions = {}
    for definition in definitions:
        request = definition.request
        protein_id = request.protein_id
        if protein_id not in candidate_ids:
            continue
        reasons = []
        source_protein = protein_by_id[protein_id]
        expected_candidates = tuple(
            (
                probe.sequence_hash,
                probe.position,
                probe.wt_aa,
                probe.mut_aa,
                probe.sequence,
            )
            for probe in request.candidate_collection.probes
        )
        source_candidates = source_candidates_by_protein.get(protein_id, ())
        if str(source_protein.get("canonical_wt_sequence")) != (
            request.candidate_collection.wt.sequence
        ):
            reasons.append("canonical_sequence_mismatch")
        if tuple(source_protein.get("mask_positions", ())) != (
            request.canonical_positions
        ):
            reasons.append("common_mask_mismatch")
        if source_candidates != expected_candidates:
            reasons.append("candidate_set_or_order_mismatch")
        condition = request.condition.condition_id
        expected_structure = source_protein.get(
            "pdb_backbone_sha256" if condition == "PDB" else "afdb_backbone_sha256"
        )
        if expected_structure != request.condition.structure_sha256:
            reasons.append("structure_sha_mismatch")
        if (
            model.get("implementation_commit")
            != definition.scorer_binding.implementation_id
            or model.get("checkpoint_sha256")
            != definition.scorer_binding.checkpoint_id
            or scoring.get("identity") != request.score_contract_id
        ):
            reasons.append("scorer_identity_mismatch")
        realization = realization_by_key.get((protein_id, request.repeat_index))
        if (
            not isinstance(realization, Mapping)
            or realization.get("seed") != request.seed
            or realization.get("decoding_realization_sha256")
            != request.realization_id
            or realization.get("algorithm") != request.realization_algorithm
        ):
            reasons.append("realization_mismatch")
        request_key = (protein_id, condition, request.repeat_index)
        wt_rows = wt_by_request.get(request_key, pd.DataFrame())
        raw_rows = raw_by_request.get(request_key, pd.DataFrame())
        if len(wt_rows) != 1:
            reasons.append("wt_result_cardinality_mismatch")
        if len(raw_rows) != len(expected_candidates):
            reasons.append("probe_result_cardinality_mismatch")
        else:
            observed = tuple(
                (
                    str(row.sequence_hash),
                    int(row.position),
                    str(row.wt_aa),
                    str(row.mut_aa),
                )
                for row in raw_rows.itertuples(index=False)
            )
            expected = tuple(item[:4] for item in expected_candidates)
            if observed != expected:
                reasons.append("probe_result_membership_mismatch")
        for frame, name in ((wt_rows, "wt"), (raw_rows, "probe")):
            if not frame.empty and (
                not frame["seed"].eq(request.seed).all()
                or not frame["decoding_realization_sha256"]
                .astype(str)
                .eq(request.realization_id)
                .all()
                or not frame["backbone_sha256"]
                .astype(str)
                .eq(request.condition.structure_sha256)
                .all()
                or not frame["model_checkpoint_sha256"]
                .astype(str)
                .eq(str(definition.scorer_binding.checkpoint_id))
                .all()
                or not frame["scoring_protocol"]
                .astype(str)
                .eq(request.score_contract_id)
                .all()
            ):
                reasons.append(f"{name}_result_binding_mismatch")
        validation_identity = sha256_canonical(
            {
                "target_scientific_fingerprint": definition.scientific_fingerprint,
                "source_execution_identity": source_execution_identity,
                "source_artifact_sha256": source_artifact_sha256,
                "reuse_contract_identity": policy.policy_identity,
                "failed_predicates": sorted(set(reasons)),
            }
        )
        decisions[definition.scientific_fingerprint] = HistoricalReuseDecision(
            status=(
                HistoricalReuseStatus.ACCEPTED
                if not reasons
                else HistoricalReuseStatus.REJECTED
            ),
            reuse_contract_identity=policy.policy_identity,
            source_execution_identity=source_execution_identity,
            source_artifact_reference=source_artifact_reference,
            source_artifact_sha256=source_artifact_sha256,
            compatibility_validation_identity=validation_identity,
            rejection_reason=(
                None if not reasons else ";".join(sorted(set(reasons)))
            ),
        )
    return decisions


def load_final_confirmatory_formal_bundle(
    project_root: Path,
) -> FinalConfirmatoryFormalBundle:
    """Load frozen v2 requests and derive exact historical reuse decisions."""
    cohort, masks, probes, plan, protocol, manifest, artifacts = (
        _load_v2_formal_inputs(project_root)
    )
    policy = resolve_historical_reuse_policy(protocol, manifest)
    definitions = _formal_definitions(
        plan=plan, cohort=cohort, masks=masks, probes=probes
    )
    decisions = _stage0_historical_reuse_decisions(
        project_root=project_root,
        definitions=definitions,
        policy=policy,
    )
    enriched = tuple(
        replace(
            definition,
            historical_reuse=decisions.get(
                definition.scientific_fingerprint, HistoricalReuseDecision()
            ),
        )
        for definition in definitions
    )
    accepted = sum(
        decision.status is HistoricalReuseStatus.ACCEPTED
        for decision in decisions.values()
    )
    rejected = sum(
        decision.status is HistoricalReuseStatus.REJECTED
        for decision in decisions.values()
    )
    return FinalConfirmatoryFormalBundle(
        definitions=enriched,
        reuse_policy=policy,
        historical_reuse_candidates_reconstructed=len(decisions),
        historical_reuse_accepted=accepted,
        historical_reuse_rejected=rejected,
        frozen_request_aggregate_sha256=_V2_AGGREGATE_FINGERPRINT,
        frozen_artifacts=artifacts,
    )


def iter_final_confirmatory_historical_score_records(
    bundle: FinalConfirmatoryFormalBundle,
    project_root: Path,
) -> Iterator[tuple[FormalRequestDefinition, tuple[ScoreRecord, ...]]]:
    """Stream exact Stage0 records for already accepted v2 reuse decisions."""
    root = project_root.resolve()
    stage_root = root / _STAGE0_ROOT
    manifest = _load_json_object(
        stage_root / "fixed_probe_scoring_manifest.json",
        "Stage0 scoring manifest",
    )
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise FinalConfirmatoryProtocolError(
            BLOCKED_INPUT_INTEGRITY,
            "invalid_historical_scoring_manifest",
            "Stage0 scoring manifest lacks output bindings",
        )
    source_paths = {}
    for key in ("fixed_probe_scores", "wt_scores"):
        record = outputs.get(key)
        if not isinstance(record, Mapping):
            raise FinalConfirmatoryProtocolError(
                BLOCKED_INPUT_INTEGRITY,
                "invalid_historical_scoring_manifest",
                f"Stage0 scoring manifest lacks {key}",
            )
        path = root / str(record.get("path", ""))
        if not path.is_file() or sha256_file(path) != str(record.get("sha256", "")):
            raise FinalConfirmatoryProtocolError(
                BLOCKED_INPUT_INTEGRITY,
                "historical_score_artifact_hash_mismatch",
                f"Historical Stage0 {key} differs",
            )
        source_paths[key] = path
    accepted = tuple(
        definition
        for definition in bundle.definitions
        if definition.historical_reuse.status is HistoricalReuseStatus.ACCEPTED
    )
    accepted_ids = {definition.request.protein_id for definition in accepted}
    try:
        filters = [("protein_id", "in", sorted(accepted_ids))]
        raw = pd.read_parquet(
            source_paths["fixed_probe_scores"], filters=filters
        )
        wt = pd.read_parquet(source_paths["wt_scores"], filters=filters)
    except (OSError, ValueError) as exc:
        raise FinalConfirmatoryProtocolError(
            BLOCKED_INPUT_INTEGRITY,
            "invalid_historical_score_artifact",
            "Unable to load historical Stage0 score artifacts",
        ) from exc
    raw_by_request = {
        (str(protein_id), str(condition), int(repeat)): group
        for (protein_id, condition, repeat), group in raw.groupby(
            ["protein_id", "backbone_condition", "repeat_index"], sort=False
        )
    }
    wt_by_request = {
        (str(protein_id), str(condition), int(repeat)): group
        for (protein_id, condition, repeat), group in wt.groupby(
            ["protein_id", "backbone_condition", "repeat_index"], sort=False
        )
    }
    for definition in accepted:
        request = definition.request
        key = (
            request.protein_id,
            request.condition.condition_id,
            request.repeat_index,
        )
        raw_rows = raw_by_request.get(key)
        wt_rows = wt_by_request.get(key)
        if raw_rows is None or wt_rows is None or len(wt_rows) != 1:
            raise FinalConfirmatoryProtocolError(
                BLOCKED_INPUT_INTEGRITY,
                "historical_score_result_missing",
                f"Accepted historical result is absent: {definition.workflow_request_id}",
            )
        records = (
            normalized_wt_score_record(wt_rows.iloc[0].to_dict()),
            *(
                normalized_probe_score_record(row._asdict())
                for row in raw_rows.itertuples(index=False)
            ),
        )
        yield definition, records


def build_final_confirmatory_projection_resolver(
    project_root: Path,
) -> Callable[[ScoreRequest], ProteinMPNNStructureInput]:
    """Build a lazy resolver from frozen v2 masks and structure bindings.

    The resolver reuses the frozen P1 atom-selection and paired-residue
    implementation.  It does not recompute SIFTS mappings or invoke a model.
    """
    from dual_uq.dataset.policies.fragments import resolve_exact_fragment
    from dual_uq.dataset.policies.identity import metadata_records
    from dual_uq.dataset.stages.derivation import (
        P1ValidationError,
        _load_atom_records,
        _pair_residues,
    )

    root = project_root.resolve()
    cohort, masks, _probes, _plan, _protocol, manifest, _artifacts = (
        _load_v2_formal_inputs(root)
    )
    upstream = manifest.get("upstream_artifacts")
    if not isinstance(upstream, list):
        raise FinalConfirmatoryProtocolError(
            BLOCKED_INPUT_INTEGRITY,
            "invalid_scale1b_v2_upstream_bindings",
            "Frozen Scale-1B-v2 manifest lacks upstream bindings",
        )
    upstream_by_label = {
        str(record.get("label")): record
        for record in upstream
        if isinstance(record, Mapping)
    }

    def load_admission(label: str) -> pd.DataFrame:
        record = upstream_by_label.get(label)
        if not isinstance(record, Mapping):
            raise FinalConfirmatoryProtocolError(
                BLOCKED_INPUT_INTEGRITY,
                "missing_projection_admission_binding",
                f"Frozen upstream admission binding is absent: {label}",
            )
        relative = Path(str(record.get("path", "")))
        expected = str(record.get("sha256", ""))
        path = root / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not path.is_file()
            or sha256_file(path) != expected
        ):
            raise FinalConfirmatoryProtocolError(
                BLOCKED_INPUT_INTEGRITY,
                "projection_admission_hash_mismatch",
                f"Frozen upstream admission differs: {label}",
            )
        try:
            return pd.read_parquet(path)
        except (OSError, ValueError) as exc:
            raise FinalConfirmatoryProtocolError(
                BLOCKED_INPUT_INTEGRITY,
                "invalid_projection_admission",
                f"Unable to load frozen upstream admission: {label}",
            ) from exc

    admission_by_stratum = {
        _ORIGINAL: load_admission("a1_admission"),
        _WAVE1: load_admission("Wave-1 manifest admission"),
        _WAVE2: load_admission("Wave-2 manifest admission"),
    }
    acquisition_ledger = load_admission("a2_ledger")
    cohort_by_id = {
        str(row.protein_id): row._asdict()
        for row in cohort.itertuples(index=False)
    }
    masks_by_id = {
        str(protein_id): group.loc[group["common_mask"].eq(True)]
        .sort_values("canonical_position", kind="mergesort")
        .reset_index(drop=True)
        for protein_id, group in masks.groupby("protein_id", sort=False)
    }
    admissions = {}
    fragment_by_model = {}
    for stratum, frame in admission_by_stratum.items():
        for row in frame.itertuples(index=False):
            record = row._asdict()
            protein_id = str(record["pair_id"])
            if pd.notna(record.get("selected_fragment_start")) and pd.notna(
                record.get("selected_fragment_end")
            ):
                fragment_key = (
                    str(record["canonical_accession"]),
                    str(record["selected_model_entity_id"]),
                )
                fragment_interval = (
                    int(record["selected_fragment_start"]),
                    int(record["selected_fragment_end"]),
                )
                existing_interval = fragment_by_model.setdefault(
                    fragment_key, fragment_interval
                )
                if existing_interval != fragment_interval:
                    raise FinalConfirmatoryProtocolError(
                        BLOCKED_INPUT_INTEGRITY,
                        "conflicting_projection_fragment_provenance",
                        f"Fragment provenance conflicts: {fragment_key[1]}",
                    )
            if protein_id in cohort_by_id and str(
                cohort_by_id[protein_id]["source_stratum"]
            ) == stratum:
                if protein_id in admissions:
                    raise FinalConfirmatoryProtocolError(
                        BLOCKED_INPUT_INTEGRITY,
                        "duplicate_projection_admission",
                        f"Multiple projection admissions bind {protein_id}",
                    )
                if (
                    str(record["canonical_accession"])
                    != str(cohort_by_id[protein_id]["canonical_accession"])
                    or str(record["selected_model_entity_id"])
                    != str(cohort_by_id[protein_id]["afdb_model_id"])
                ):
                    raise FinalConfirmatoryProtocolError(
                        BLOCKED_INPUT_INTEGRITY,
                        "projection_admission_identity_mismatch",
                        f"Admission identity differs: {protein_id}",
                    )
                admissions[protein_id] = record
    ledger_metadata = {
        str(row.candidate_id): row._asdict()
        for row in acquisition_ledger.loc[
            acquisition_ledger["asset_type"].eq("afdb_metadata_collection")
            & acquisition_ledger["validation_status"].eq("VALID")
        ].itertuples(index=False)
    }
    for protein_id, protein in cohort_by_id.items():
        if protein_id in admissions:
            continue
        accession = str(protein["canonical_accession"])
        model_id = str(protein["afdb_model_id"])
        interval = fragment_by_model.get((accession, model_id))
        if interval is None:
            ledger_record = ledger_metadata.get(protein_id)
            if ledger_record is None:
                raise FinalConfirmatoryProtocolError(
                    BLOCKED_INPUT_INTEGRITY,
                    "missing_projection_fragment_provenance",
                    f"Frozen fragment provenance is absent: {protein_id}",
                )
            metadata_path = root / str(ledger_record["local_path_relative"])
            if (
                Path(str(ledger_record["local_path_relative"])).is_absolute()
                or ".."
                in Path(str(ledger_record["local_path_relative"])).parts
                or not metadata_path.is_file()
                or sha256_file(metadata_path) != ledger_record["file_sha256"]
            ):
                raise FinalConfirmatoryProtocolError(
                    BLOCKED_INPUT_INTEGRITY,
                    "projection_metadata_hash_mismatch",
                    f"Frozen AFDB metadata differs: {protein_id}",
                )
            mask = masks_by_id[protein_id]
            positions = mask["canonical_position"].astype(int)
            resolved = resolve_exact_fragment(
                metadata_records(metadata_path.read_bytes()),
                accession,
                (int(positions.min()), int(positions.max())),
            )
            fragment = resolved["selected_fragment"]
            if fragment is None or fragment.model_entity_id != model_id:
                raise FinalConfirmatoryProtocolError(
                    BLOCKED_INPUT_INTEGRITY,
                    "projection_fragment_mismatch",
                    f"Frozen AFDB fragment differs: {protein_id}",
                )
            interval = (fragment.uniprot_start, fragment.uniprot_end)
        admissions[protein_id] = {
            "selected_fragment_start": interval[0],
            "selected_fragment_end": interval[1],
        }
    if set(admissions) != set(cohort_by_id):
        raise FinalConfirmatoryProtocolError(
            BLOCKED_INPUT_INTEGRITY,
            "missing_projection_admission",
            "Frozen cohort and source admissions do not bind one-to-one",
        )

    cache: dict[tuple[str, str], ProteinMPNNStructureInput] = {}

    def resolver(request: ScoreRequest) -> ProteinMPNNStructureInput:
        key = (request.protein_id, request.condition.condition_id)
        if key in cache:
            return cache[key]
        protein = cohort_by_id.get(request.protein_id)
        mask = masks_by_id.get(request.protein_id)
        admission = admissions.get(request.protein_id)
        if protein is None or mask is None or admission is None:
            raise FinalConfirmatoryProtocolError(
                BLOCKED_INPUT_INTEGRITY,
                "missing_projection_binding",
                f"Frozen projection binding is absent: {request.protein_id}",
            )
        positions = tuple(mask["canonical_position"].astype(int))
        if positions != request.canonical_positions:
            raise FinalConfirmatoryProtocolError(
                BLOCKED_INPUT_INTEGRITY,
                "projection_mask_mismatch",
                f"Frozen mask differs for {request.protein_id}",
            )
        mapping = mask.rename(
            columns={"canonical_position": "uniprot_residue_number"}
        ).copy()
        mapping["output_position"] = np.arange(1, len(mapping) + 1)
        pdb_path = root / str(protein["pdb_structure_ref"])
        afdb_path = root / str(protein["afdb_structure_ref"])
        expected_hashes = {
            "PDB": str(protein["pdb_structure_sha256"]),
            "AFDB": str(protein["afdb_structure_sha256"]),
        }
        for condition, path in (("PDB", pdb_path), ("AFDB", afdb_path)):
            if not path.is_file() or sha256_file(path) != expected_hashes[condition]:
                raise FinalConfirmatoryProtocolError(
                    BLOCKED_INPUT_INTEGRITY,
                    "projection_structure_hash_mismatch",
                    f"Frozen {condition} structure differs: {request.protein_id}",
                )
        try:
            residues = _pair_residues(
                mapping=mapping,
                pdb_records=_load_atom_records(pdb_path, request.protein_id),
                afdb_records=_load_atom_records(
                    afdb_path, str(protein["afdb_model_id"])
                ),
                pair_id=request.protein_id,
                model_entity_id=str(protein["afdb_model_id"]),
                fragment_start=int(admission["selected_fragment_start"]),
                fragment_end=int(admission["selected_fragment_end"]),
            )
        except P1ValidationError as exc:
            raise FinalConfirmatoryProtocolError(
                BLOCKED_INPUT_INTEGRITY,
                exc.code,
                str(exc),
                protein_id=request.protein_id,
            ) from exc
        sequence_projection = "".join(row.canonical_aa for row in residues)
        paired = validate_paired_projection(
            ScoringBackboneProjection(
                protein_id=request.protein_id,
                backbone_condition="PDB",
                uniprot_positions=positions,
                wt_sequence_projection=sequence_projection,
                coordinates=np.stack(
                    [_coordinates(row.pdb_backbone) for row in residues]
                ),
                structure_sha256=expected_hashes["PDB"],
            ),
            ScoringBackboneProjection(
                protein_id=request.protein_id,
                backbone_condition="AFDB",
                uniprot_positions=positions,
                wt_sequence_projection=sequence_projection,
                coordinates=np.stack(
                    [_coordinates(row.afdb_backbone) for row in residues]
                ),
                structure_sha256=expected_hashes["AFDB"],
            ),
            expected_positions=positions,
        )
        cache[(request.protein_id, "PDB")] = paired.pdb
        cache[(request.protein_id, "AFDB")] = paired.afdb
        return cache[key]

    return resolver


@dataclass(frozen=True)
class FinalConfirmatoryProtocolConfig:
    """Portable immutable bindings for a final confirmatory protocol."""

    a0_audit_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a0/"
        "scale1_sampling_frame_audit.parquet"
    )
    expected_a0_audit_sha256: str = (
        "695f3587dae24b5945f975a7611c677fca9a5e89f728d2f3802f77b5a31a2d4d"
    )
    a0_manifest_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a0/"
        "scale1_sampling_frame_manifest.json"
    )
    expected_a0_manifest_sha256: str = (
        "de4baae228b15adda0ad41bba674d6a45470a728be92b374f20aa30b72ef4fca"
    )
    a1_admission_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a1/"
        "scale1_formal_admission_census.parquet"
    )
    expected_a1_admission_sha256: str = (
        "ca01e17f55637d0f4584aa7bbb8940e50ea4aaf25b9c7ee7f741bb51a4095a7f"
    )
    a1_manifest_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a1/"
        "scale1_formal_admission_manifest.json"
    )
    expected_a1_manifest_sha256: str = (
        "54227ee11e67fdb6da12cbf9001b603875b1fd3c28d603608d384f0c50988223"
    )
    a2_census_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a2/"
        "scale1_full_frame_census.parquet"
    )
    expected_a2_census_sha256: str = (
        "fc69e13d3d94def1a751b5528230fa5f15dd1ae741d8361b4187236167120793"
    )
    a2_masks_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a2/"
        "scale1_full_frame_common_masks.parquet"
    )
    expected_a2_masks_sha256: str = (
        "4a6d89a9368c3349f0598616e3dda670b17164fe6dff7fd37e131a71e4acaae0"
    )
    a2_ledger_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a2/"
        "scale1a2_acquisition_ledger.parquet"
    )
    expected_a2_ledger_sha256: str = (
        "a6310505eb8ba397b835ccaa9fb5fece2ed6309792d6af07368bfa517959869f"
    )
    a2_summary_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a2/"
        "scale1_full_frame_attrition_summary.json"
    )
    expected_a2_summary_sha256: str = (
        "36ee351614ed3e0f9ba57161bbdc2a4410acc3ca5cb5ad8d1b2de54427ae7002"
    )
    a3_census_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a3/"
        "scale1a3_admitted_redundancy_census.parquet"
    )
    expected_a3_census_sha256: str = (
        "71ac431d6debe223aa0206fa29504b17f5bae6f2cf78fa1354f6b6a86cd9b4a1"
    )
    a3_capacity_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a3/"
        "scale1a3_nonredundant_capacity.parquet"
    )
    expected_a3_capacity_sha256: str = (
        "203125ef114368c015ad86edf7f3802c05fa33085fde280befe955df537e91d9"
    )
    a3_manifest_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a3/scale1a3_manifest.json"
    )
    expected_a3_manifest_sha256: str = (
        "eac4e19e6645d3aa6935e17bf8f25d1116d32a67dd8d3a4c78e68bc93d680d56"
    )
    e0_manifest_ref: str = (
        "experiments/p2_design_baseline/scale1/expansion_design/"
        "scale1_expansion_design_manifest.json"
    )
    expected_e0_manifest_sha256: str = (
        "ead5184664f908c531e6ae316db923f9615bf3794f42734e24685a2032c23e99"
    )
    e0a_amendment_ref: str = (
        "experiments/p2_design_baseline/scale1/expansion_wave1/"
        "scale1e0a_wave1_amendment.json"
    )
    expected_e0a_amendment_sha256: str = (
        "db2ea53d5ad6cffa72b4d9c5cb253aab6cf3d3f1c8f59d8fdecce0a26b4ea576"
    )
    wave1_root_ref: str = (
        "experiments/p2_design_baseline/scale1/expansion_wave1"
    )
    expected_wave1_manifest_sha256: str = (
        "cb1ea3e1cd42a1cdc79fd972e2305570a76afef65f9ec18b6530f017286f3ce2"
    )
    wave2_root_ref: str = (
        "experiments/p2_design_baseline/scale1/expansion_wave2"
    )
    expected_wave2_manifest_sha256: str = (
        "d9fc9f2b2a26f8b9cc1e81a45b25d17cc03f0b16405f8497f402dc62d45e8526"
    )
    scale1b_v1_cohort_manifest_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1b_freeze/"
        "scale1b_cohort_freeze_manifest.json"
    )
    expected_scale1b_v1_cohort_manifest_sha256: str = (
        "275f7373f4f5de9b0a1f12c2b5585bc7ef91e88d11dc2b61b387326798e6e730"
    )
    scale1b_v1_panel_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1b_freeze/"
        "scale1b_scoring_panel.parquet"
    )
    expected_scale1b_v1_panel_sha256: str = (
        "6fdc074d8dd56bb502592b51830efdcafcb34dcb2b530e8fe26e95d657684072"
    )
    scale1b_v1_protocol_manifest_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1b_protocol/"
        "scale1b_protocol_freeze_manifest.json"
    )
    expected_scale1b_v1_protocol_manifest_sha256: str = (
        "485a22a89d0fc532cdee72c09ed7faa36c5fdc54079aa2a971479d41c53af918"
    )
    scale1b_v1_probes_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1b_protocol/"
        "scale1b_fixed_probe_candidates.parquet"
    )
    expected_scale1b_v1_probes_sha256: str = (
        "af405b7dafae84d27e92d3654d173eef29f268b68ad241c9934c6920fcd99126"
    )
    scale1b_v1_probe_manifest_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1b_protocol/"
        "scale1b_probe_manifest.json"
    )
    expected_scale1b_v1_probe_manifest_sha256: str = (
        "5701462be2ebe30b713723bd09bae2a70a980f8cc47eb6d0772884b0bea1fa13"
    )
    scale1b_v1_scoring_protocol_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1b_protocol/"
        "scale1b_scoring_protocol.json"
    )
    expected_scale1b_v1_scoring_protocol_sha256: str = (
        "e5ede85840192ae2191ba2c4523a00c1464f11759492a87e77ae2c2a7595039a"
    )
    h1_manifest_ref: str = (
        "experiments/p2_design_baseline/stage0/h1_confirmatory_audit_manifest.json"
    )
    expected_h1_manifest_sha256: str = (
        "d74252a4393f9e0ebf0b58d7025c7bca0569701921c592bacb425bd6bfc07fbf"
    )
    stage0_config_ref: str = "configs/experiments/design_baseline/stage0.yaml"
    expected_stage0_config_sha256: str = (
        "9908b63e3f556b2148837f6ade55000c0391e026d4cb49b68c719ef7842aeb5c"
    )
    output_root_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1b_v2"
    )


@dataclass(frozen=True)
class FinalConfirmatoryInputs:
    original_admitted: pd.DataFrame
    original_redundancy: pd.DataFrame
    wave1_admitted: pd.DataFrame
    wave2_admitted: pd.DataFrame
    common_masks: pd.DataFrame
    v1_panel: pd.DataFrame
    v1_probes: pd.DataFrame
    v1_probe_manifest: dict[str, Any]
    v1_scoring_protocol: dict[str, Any]
    wave1_manifest: dict[str, Any]
    wave2_manifest: dict[str, Any]
    wave2_summary: dict[str, Any]
    h1_manifest: dict[str, Any]
    model_identity: VerifiedModelIdentity
    input_artifacts: tuple[dict[str, Any], ...]
    output_root_ref: str
    proteinmpnn_forward_executions: int = 0


@dataclass(frozen=True)
class FinalConfirmatoryResult:
    status: str
    primary_cohort: pd.DataFrame
    secondary_pool: pd.DataFrame
    cluster_reconciliation: pd.DataFrame
    primary_common_masks: pd.DataFrame
    fixed_probes: pd.DataFrame
    probe_reuse_audit: pd.DataFrame
    scoring_plan: pd.DataFrame
    probe_manifest: dict[str, Any]
    scoring_protocol: dict[str, Any]
    summary: dict[str, Any]
    manifest: dict[str, Any]
    input_artifacts: tuple[dict[str, Any], ...]


def _require_hash(
    paths: ProjectPaths, ref: str, expected: str, label: str
) -> tuple[Path, dict[str, Any]]:
    path = paths.resolve_logical(ref)
    if not path.is_file() or sha256_file(path) != expected:
        raise FinalConfirmatoryProtocolError(
            BLOCKED_INPUT_INTEGRITY,
            "upstream_sha256_mismatch",
            f"Frozen input differs: {label}",
            path=ref,
        )
    return path, {"path": ref, "sha256": expected, "label": label}


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FinalConfirmatoryProtocolError(
            BLOCKED_INPUT_INTEGRITY,
            "unreadable_upstream_input",
            f"Unable to read {label}",
        ) from exc
    if not isinstance(value, dict):
        raise FinalConfirmatoryProtocolError(
            BLOCKED_INPUT_INTEGRITY,
            "invalid_upstream_schema",
            f"{label} must be a JSON object",
        )
    return value


def _parquet(path: Path, label: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise FinalConfirmatoryProtocolError(
            BLOCKED_INPUT_INTEGRITY,
            "unreadable_upstream_input",
            f"Unable to read {label}",
        ) from exc


def _manifest_bound_files(
    paths: ProjectPaths,
    manifest: dict[str, Any],
    *,
    names: tuple[str, ...],
    label: str,
) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    resolved: dict[str, Path] = {}
    records: list[dict[str, Any]] = []
    for name in names:
        record = manifest.get("outputs", {}).get(name)
        if not isinstance(record, dict):
            raise FinalConfirmatoryProtocolError(
                BLOCKED_INPUT_INTEGRITY,
                "manifest_output_binding_missing",
                f"{label} does not bind {name}",
            )
        ref = str(record.get("path", ""))
        expected = str(record.get("sha256", ""))
        path, artifact = _require_hash(
            paths, ref, expected, f"{label} {name}"
        )
        resolved[name] = path
        records.append(artifact)
    return resolved, records


def _canonical_sequence(mask_rows: pd.DataFrame, protein_id: str) -> str:
    ordered = mask_rows.sort_values("canonical_position", kind="stable")
    positions = ordered["canonical_position"].astype(int).tolist()
    if positions != list(range(1, len(positions) + 1)):
        raise FinalConfirmatoryProtocolError(
            BLOCKED_COMMON_MASK_INTEGRITY,
            "canonical_position_discontinuity",
            f"Canonical positions are not complete for {protein_id}",
        )
    sequence = "".join(ordered["canonical_aa"].astype(str))
    if not sequence:
        raise FinalConfirmatoryProtocolError(
            BLOCKED_COMMON_MASK_INTEGRITY,
            "canonical_sequence_missing",
            f"Canonical sequence is absent for {protein_id}",
        )
    return sequence


def _raw_binding(
    paths: ProjectPaths,
    *,
    candidate_id: str,
    asset_type: str,
    a1: pd.DataFrame,
    a0: pd.DataFrame,
    a0_artifacts: dict[str, str],
    ledger: pd.DataFrame,
) -> tuple[str, str]:
    a1_rows = a1.loc[a1["candidate_id"].eq(candidate_id)]
    if not a1_rows.empty:
        row = a1_rows.iloc[0]
        prefix = "pdb" if asset_type == "pdb_mmcif" else "afdb"
        return str(row[f"{prefix}_file_path_relative"]), str(
            row[f"{prefix}_file_sha256"]
        )
    acquired = ledger.loc[
        ledger["candidate_id"].eq(candidate_id)
        & ledger["asset_type"].eq(asset_type)
        & ledger["validation_status"].eq("VALID")
    ]
    if len(acquired) == 1:
        row = acquired.iloc[0]
        return str(row["local_path_relative"]), str(row["file_sha256"])
    if len(acquired) > 1:
        raise FinalConfirmatoryProtocolError(
            BLOCKED_INPUT_INTEGRITY,
            "ambiguous_structure_binding",
            f"Multiple {asset_type} bindings for {candidate_id}",
        )
    source = a0.loc[a0["candidate_id"].eq(candidate_id)]
    if len(source) != 1:
        raise FinalConfirmatoryProtocolError(
            BLOCKED_INPUT_INTEGRITY,
            "structure_binding_missing",
            f"No frozen structure binding for {candidate_id}",
        )
    column = (
        "pdb_file_path_relative"
        if asset_type == "pdb_mmcif"
        else "afdb_file_path_relative"
    )
    ref = str(source.iloc[0][column])
    expected = a0_artifacts.get(ref)
    if not ref or expected is None:
        raise FinalConfirmatoryProtocolError(
            BLOCKED_INPUT_INTEGRITY,
            "structure_provenance_missing",
            f"Frozen source does not bind {asset_type} for {candidate_id}",
        )
    path = paths.resolve_logical(ref)
    if not path.is_file() or sha256_file(path) != expected:
        raise FinalConfirmatoryProtocolError(
            BLOCKED_INPUT_INTEGRITY,
            "structure_sha256_mismatch",
            f"Frozen structure differs for {candidate_id}",
        )
    return ref, expected


def _normalized_original(
    paths: ProjectPaths,
    *,
    a3: pd.DataFrame,
    a2: pd.DataFrame,
    masks: pd.DataFrame,
    a1: pd.DataFrame,
    a0: pd.DataFrame,
    a0_manifest: dict[str, Any],
    ledger: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    a2_by_id = a2.set_index("candidate_id", drop=False)
    a0_artifacts = {
        str(record.get("path")): str(record.get("sha256"))
        for record in a0_manifest.get("input_artifacts", [])
        if isinstance(record, dict) and record.get("path") and record.get("sha256")
    }
    rows: list[dict[str, Any]] = []
    structure_records: list[dict[str, Any]] = []
    for source in a3.to_dict("records"):
        candidate_id = str(source["candidate_id"])
        if candidate_id not in a2_by_id.index:
            raise FinalConfirmatoryProtocolError(
                BLOCKED_INPUT_INTEGRITY,
                "canonical_identity_missing",
                f"Scale-1A2 lacks {candidate_id}",
            )
        census = a2_by_id.loc[candidate_id]
        protein_masks = masks.loc[masks["candidate_id"].eq(candidate_id)]
        if protein_masks.empty or protein_masks["pair_id"].nunique() != 1:
            raise FinalConfirmatoryProtocolError(
                BLOCKED_COMMON_MASK_INTEGRITY,
                "common_mask_identity_missing",
                f"Common-mask source is missing for {candidate_id}",
            )
        protein_id = str(protein_masks["pair_id"].iloc[0])
        a0_rows = a0.loc[a0["candidate_id"].eq(candidate_id)]
        if (
            len(a0_rows) != 1
            or protein_id != str(source["pair_id"])
            or str(census["canonical_accession"])
            != str(source["canonical_accession"])
            or str(census["pdb_id"]).lower() != str(source["pdb_id"]).lower()
            or str(census["pdb_chain"]) != str(source["pdb_chain"])
        ):
            raise FinalConfirmatoryProtocolError(
                BLOCKED_INPUT_INTEGRITY,
                "canonical_identity_binding_conflict",
                f"Frozen identity bindings disagree for {candidate_id}",
            )
        sequence = _canonical_sequence(protein_masks, protein_id)
        pdb_ref, pdb_sha = _raw_binding(
            paths,
            candidate_id=candidate_id,
            asset_type="pdb_mmcif",
            a1=a1,
            a0=a0,
            a0_artifacts=a0_artifacts,
            ledger=ledger,
        )
        afdb_ref, afdb_sha = _raw_binding(
            paths,
            candidate_id=candidate_id,
            asset_type="afdb_structure",
            a1=a1,
            a0=a0,
            a0_artifacts=a0_artifacts,
            ledger=ledger,
        )
        a1_rows = a1.loc[a1["candidate_id"].eq(candidate_id)]
        acquired_afdb = ledger.loc[
            ledger["candidate_id"].eq(candidate_id)
            & ledger["asset_type"].eq("afdb_structure")
            & ledger["validation_status"].eq("VALID")
        ]
        expected_model_id = str(census["afdb_id"])
        model_identity_matches = (
            len(a1_rows) == 1
            and str(a1_rows.iloc[0]["selected_model_entity_id"])
            == expected_model_id
        ) or (
            len(a1_rows) == 0
            and len(acquired_afdb) == 1
            and str(acquired_afdb.iloc[0]["canonical_identifier"])
            == expected_model_id
        ) or (
            len(a1_rows) == 0
            and acquired_afdb.empty
            and str(a0_rows.iloc[0]["afdb_id"]) == expected_model_id
        )
        if not model_identity_matches:
            raise FinalConfirmatoryProtocolError(
                BLOCKED_INPUT_INTEGRITY,
                "afdb_structure_identity_mismatch",
                f"AFDB structure binding differs from the selected model for {protein_id}",
            )
        for ref, digest, label in (
            (pdb_ref, pdb_sha, "frozen PDB structure"),
            (afdb_ref, afdb_sha, "frozen AFDB structure"),
        ):
            path = paths.resolve_logical(ref)
            if not path.is_file() or sha256_file(path) != digest:
                raise FinalConfirmatoryProtocolError(
                    BLOCKED_INPUT_INTEGRITY,
                    "structure_sha256_mismatch",
                    f"{label} differs for {protein_id}",
                )
            structure_records.append(
                {"path": ref, "sha256": digest, "label": label}
            )
        common_count = int(protein_masks["common_mask"].fillna(False).sum())
        if common_count != int(source["common_mask_count"]):
            raise FinalConfirmatoryProtocolError(
                BLOCKED_COMMON_MASK_INTEGRITY,
                "common_mask_count_mismatch",
                f"Common-mask count differs for {protein_id}",
            )
        rows.append(
            {
                "candidate_id": candidate_id,
                "pair_id": protein_id,
                "protein_id": protein_id,
                "polymer_entity_id": str(protein_masks["polymer_entity_id"].iloc[0]),
                "canonical_accession": str(source["canonical_accession"]),
                "pdb_id": str(source["pdb_id"]),
                "pdb_chain": str(source["pdb_chain"]),
                "afdb_model_id": str(census["afdb_id"]),
                "canonical_sequence": sequence,
                "canonical_sequence_sha256": sequence_sha256(sequence),
                "canonical_sequence_length": len(sequence),
                "sequence_cluster": str(source["redundancy_cluster_id"]),
                "redundancy_metadata_resolved": bool(
                    source["redundancy_metadata_resolved"]
                ),
                "common_mask_count": common_count,
                "common_mask_fraction": float(source["common_mask_fraction"]),
                "sampling_frame_index": int(source["sampling_frame_index"]),
                "source_order": int(source["sampling_frame_index"]),
                "source_stratum": _ORIGINAL,
                "stage0_overlap": bool(a0_rows.iloc[0]["stage0_admitted_member"]),
                "scale1b_v1_overlap": bool(source["scale1b_v1_member"]),
                "scale1b_v1_role": None,
                "scale1a3_representative_overlap": bool(
                    source["is_cluster_representative"]
                ),
                "wave1_declaration_rank": None,
                "wave2_declaration_rank": None,
                "pdb_structure_ref": pdb_ref,
                "pdb_structure_sha256": pdb_sha,
                "afdb_structure_ref": afdb_ref,
                "afdb_structure_sha256": afdb_sha,
                "mapping_provenance_ref": "scale1a2_full_frame_common_masks",
            }
        )
    return pd.DataFrame(rows), structure_records


def _normalized_wave(
    paths: ProjectPaths,
    *,
    admission: pd.DataFrame,
    capacity: pd.DataFrame,
    masks: pd.DataFrame,
    source_stratum: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    admitted = admission.loc[admission["admission_status"].eq("FORMALLY_ADMITTED")]
    joined = admitted.merge(
        capacity,
        on="pair_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    rank_column = (
        "wave1_candidate_index"
        if source_stratum == _WAVE1
        else "wave2_candidate_index"
    )
    if joined["sequence_cluster"].isna().any() or joined[rank_column].isna().any():
        raise FinalConfirmatoryProtocolError(
            BLOCKED_REDUNDANCY_REGRESSION,
            "expansion_cluster_binding_missing",
            f"{source_stratum} lacks a frozen cluster/order binding",
        )
    rows: list[dict[str, Any]] = []
    structures: list[dict[str, Any]] = []
    for source in joined.to_dict("records"):
        protein_id = str(source["pair_id"])
        protein_masks = masks.loc[masks["pair_id"].eq(protein_id)]
        if (
            protein_masks.empty
            or protein_masks["canonical_accession"].astype(str).nunique() != 1
            or str(protein_masks["canonical_accession"].iloc[0])
            != str(source["canonical_accession"])
        ):
            raise FinalConfirmatoryProtocolError(
                BLOCKED_INPUT_INTEGRITY,
                "expansion_identity_binding_conflict",
                f"Frozen expansion identity disagrees for {protein_id}",
            )
        sequence = _canonical_sequence(protein_masks, protein_id)
        common_count = int(protein_masks["common_mask"].fillna(False).sum())
        if (
            common_count != int(source["common_mask_count"])
            or sequence_sha256(sequence) != str(source["canonical_sequence_sha256"])
        ):
            raise FinalConfirmatoryProtocolError(
                BLOCKED_COMMON_MASK_INTEGRITY,
                "expansion_mask_sequence_mismatch",
                f"Frozen expansion mask differs for {protein_id}",
            )
        for ref_field, sha_field, label in (
            ("pdb_file_path_relative", "pdb_file_sha256", "frozen PDB structure"),
            ("afdb_file_path_relative", "afdb_file_sha256", "frozen AFDB structure"),
        ):
            ref = str(source[ref_field])
            digest = str(source[sha_field])
            path = paths.resolve_logical(ref)
            if not path.is_file() or sha256_file(path) != digest:
                raise FinalConfirmatoryProtocolError(
                    BLOCKED_INPUT_INTEGRITY,
                    "structure_sha256_mismatch",
                    f"{label} differs for {protein_id}",
                )
            structures.append(
                {
                    "path": ref,
                    "sha256": digest,
                    "label": label,
                }
            )
        source_order = int(source[rank_column])
        offset = 1_000_000 if source_stratum == _WAVE1 else 2_000_000
        rows.append(
            {
                "candidate_id": str(source["candidate_id"]),
                "pair_id": protein_id,
                "protein_id": protein_id,
                "polymer_entity_id": str(source["polymer_entity_id"]),
                "canonical_accession": str(source["canonical_accession"]),
                "pdb_id": str(source["pdb_id"]),
                "pdb_chain": str(source["pdb_chain"]),
                "afdb_model_id": str(source["selected_model_entity_id"]),
                "canonical_sequence": sequence,
                "canonical_sequence_sha256": str(source["canonical_sequence_sha256"]),
                "canonical_sequence_length": len(sequence),
                "sequence_cluster": str(source["sequence_cluster"]),
                "redundancy_metadata_resolved": True,
                "common_mask_count": common_count,
                "common_mask_fraction": float(
                    source["common_mask_fraction_of_mapped"]
                ),
                "sampling_frame_index": offset + source_order,
                "source_order": source_order,
                "source_stratum": source_stratum,
                "stage0_overlap": False,
                "scale1b_v1_overlap": False,
                "scale1b_v1_role": None,
                "scale1a3_representative_overlap": False,
                "wave1_declaration_rank": (
                    source_order if source_stratum == _WAVE1 else None
                ),
                "wave2_declaration_rank": (
                    source_order if source_stratum == _WAVE2 else None
                ),
                "pdb_structure_ref": str(source["pdb_file_path_relative"]),
                "pdb_structure_sha256": str(source["pdb_file_sha256"]),
                "afdb_structure_ref": str(source["afdb_file_path_relative"]),
                "afdb_structure_sha256": str(source["afdb_file_sha256"]),
                "mapping_provenance_ref": (
                    "scale1_expansion_wave1_common_masks"
                    if source_stratum == _WAVE1
                    else "scale1_expansion_wave2_common_masks"
                ),
            }
        )
    return pd.DataFrame(rows), structures


def validate_final_confirmatory_inputs(
    paths: ProjectPaths, config: FinalConfirmatoryProtocolConfig
) -> FinalConfirmatoryInputs:
    """Validate all frozen cohort, protocol, structure, and H1 trust roots."""
    simple = (
        ("a0_audit", config.a0_audit_ref, config.expected_a0_audit_sha256),
        ("a0_manifest", config.a0_manifest_ref, config.expected_a0_manifest_sha256),
        ("a1_admission", config.a1_admission_ref, config.expected_a1_admission_sha256),
        ("a1_manifest", config.a1_manifest_ref, config.expected_a1_manifest_sha256),
        ("a2_census", config.a2_census_ref, config.expected_a2_census_sha256),
        ("a2_masks", config.a2_masks_ref, config.expected_a2_masks_sha256),
        ("a2_ledger", config.a2_ledger_ref, config.expected_a2_ledger_sha256),
        ("a2_summary", config.a2_summary_ref, config.expected_a2_summary_sha256),
        ("a3_census", config.a3_census_ref, config.expected_a3_census_sha256),
        ("a3_capacity", config.a3_capacity_ref, config.expected_a3_capacity_sha256),
        ("a3_manifest", config.a3_manifest_ref, config.expected_a3_manifest_sha256),
        ("e0_manifest", config.e0_manifest_ref, config.expected_e0_manifest_sha256),
        ("e0a_amendment", config.e0a_amendment_ref, config.expected_e0a_amendment_sha256),
        ("v1_cohort_manifest", config.scale1b_v1_cohort_manifest_ref, config.expected_scale1b_v1_cohort_manifest_sha256),
        ("v1_panel", config.scale1b_v1_panel_ref, config.expected_scale1b_v1_panel_sha256),
        ("v1_protocol_manifest", config.scale1b_v1_protocol_manifest_ref, config.expected_scale1b_v1_protocol_manifest_sha256),
        ("v1_probes", config.scale1b_v1_probes_ref, config.expected_scale1b_v1_probes_sha256),
        ("v1_probe_manifest", config.scale1b_v1_probe_manifest_ref, config.expected_scale1b_v1_probe_manifest_sha256),
        ("v1_scoring_protocol", config.scale1b_v1_scoring_protocol_ref, config.expected_scale1b_v1_scoring_protocol_sha256),
        ("h1_manifest", config.h1_manifest_ref, config.expected_h1_manifest_sha256),
        ("stage0_config", config.stage0_config_ref, config.expected_stage0_config_sha256),
    )
    resolved: dict[str, Path] = {}
    artifacts: list[dict[str, Any]] = []
    for name, ref, expected in simple:
        path, record = _require_hash(paths, ref, expected, name)
        resolved[name] = path
        artifacts.append(record)

    wave1_manifest_ref = f"{config.wave1_root_ref}/scale1_expansion_wave1_manifest.json"
    wave1_manifest_path, record = _require_hash(
        paths,
        wave1_manifest_ref,
        config.expected_wave1_manifest_sha256,
        "Wave-1 manifest",
    )
    artifacts.append(record)
    wave2_manifest_ref = f"{config.wave2_root_ref}/scale1_expansion_wave2_manifest.json"
    wave2_manifest_path, record = _require_hash(
        paths,
        wave2_manifest_ref,
        config.expected_wave2_manifest_sha256,
        "Wave-2 manifest",
    )
    artifacts.append(record)
    wave1_manifest = _json(wave1_manifest_path, "Wave-1 manifest")
    wave2_manifest = _json(wave2_manifest_path, "Wave-2 manifest")
    wave1_paths, wave1_records = _manifest_bound_files(
        paths,
        wave1_manifest,
        names=("admission", "common_masks", "cluster_capacity", "declaration", "summary"),
        label="Wave-1 manifest",
    )
    wave2_paths, wave2_records = _manifest_bound_files(
        paths,
        wave2_manifest,
        names=("admission", "common_masks", "cluster_capacity", "declaration", "summary"),
        label="Wave-2 manifest",
    )
    artifacts.extend(wave1_records)
    artifacts.extend(wave2_records)

    a0 = _parquet(resolved["a0_audit"], "Scale-1A0 audit")
    a0_manifest = _json(resolved["a0_manifest"], "Scale-1A0 manifest")
    a1 = _parquet(resolved["a1_admission"], "Scale-1A1 admission")
    a2 = _parquet(resolved["a2_census"], "Scale-1A2 census")
    a2_masks = _parquet(resolved["a2_masks"], "Scale-1A2 masks")
    ledger = _parquet(resolved["a2_ledger"], "Scale-1A2 ledger")
    a3 = _parquet(resolved["a3_census"], "Scale-1A3 census")
    a3_capacity = _parquet(resolved["a3_capacity"], "Scale-1A3 capacity")
    wave1_admission = _parquet(wave1_paths["admission"], "Wave-1 admission")
    wave1_masks = _parquet(wave1_paths["common_masks"], "Wave-1 masks")
    wave1_capacity = _parquet(wave1_paths["cluster_capacity"], "Wave-1 capacity")
    wave2_admission = _parquet(wave2_paths["admission"], "Wave-2 admission")
    wave2_masks = _parquet(wave2_paths["common_masks"], "Wave-2 masks")
    wave2_capacity = _parquet(wave2_paths["cluster_capacity"], "Wave-2 capacity")
    wave2_summary = _json(wave2_paths["summary"], "Wave-2 summary")

    original = a3.loc[a3["formal_admission_status"].eq("FORMALLY_ADMITTED")]
    if (
        len(original) != 135
        or original["pair_id"].duplicated().any()
        or len(a3_capacity) != 63
        or a3_capacity["redundancy_cluster_id"].nunique() != 63
    ):
        raise FinalConfirmatoryProtocolError(
            BLOCKED_INPUT_INTEGRITY,
            "original_admitted_pool_mismatch",
            "Original admitted pool is not 135 unique proteins",
        )
    original_normalized, original_structures = _normalized_original(
        paths,
        a3=original,
        a2=a2,
        masks=a2_masks,
        a1=a1,
        a0=a0,
        a0_manifest=a0_manifest,
        ledger=ledger,
    )
    wave1_normalized, wave1_structures = _normalized_wave(
        paths,
        admission=wave1_admission,
        capacity=wave1_capacity,
        masks=wave1_masks,
        source_stratum=_WAVE1,
    )
    wave2_normalized, wave2_structures = _normalized_wave(
        paths,
        admission=wave2_admission,
        capacity=wave2_capacity,
        masks=wave2_masks,
        source_stratum=_WAVE2,
    )
    try:
        wave1_declared_n = sum(
            1 for line in wave1_paths["declaration"].read_text().splitlines() if line
        )
        wave2_declared_n = sum(
            1 for line in wave2_paths["declaration"].read_text().splitlines() if line
        )
    except (OSError, UnicodeError) as exc:
        raise FinalConfirmatoryProtocolError(
            BLOCKED_INPUT_INTEGRITY,
            "unreadable_expansion_declaration",
            "Expansion declarations cannot be read",
        ) from exc
    if (
        len(wave1_normalized) != 49
        or len(wave2_normalized) != 15
        or wave1_declared_n != 100
        or wave2_declared_n != 24
    ):
        raise FinalConfirmatoryProtocolError(
            BLOCKED_INPUT_INTEGRITY,
            "expansion_admitted_pool_mismatch",
            "Expansion admitted counts differ from 49 and 15",
        )
    union = pd.concat(
        [original_normalized, wave1_normalized, wave2_normalized],
        ignore_index=True,
    )
    if len(union) != 199 or union["protein_id"].duplicated().any():
        duplicates = union.loc[
            union["protein_id"].duplicated(keep=False), "protein_id"
        ].astype(str).tolist()
        raise FinalConfirmatoryProtocolError(
            BLOCKED_INPUT_INTEGRITY,
            "canonical_admission_identity_conflict",
            "Final admitted identities do not reconcile to 199 unique proteins",
            duplicate_identity_bindings=duplicates,
        )
    if union["sequence_cluster"].nunique() != _FINAL_CLUSTER_N:
        raise FinalConfirmatoryProtocolError(
            BLOCKED_REDUNDANCY_REGRESSION,
            "final_cluster_count_mismatch",
            "Final admitted pool does not contain exactly 127 clusters",
        )
    if (
        wave2_manifest.get("capacity_status")
        != "PRIMARY_CONFIRMATORY_CAPACITY_REACHED"
        or wave2_summary.get("final_N_NR") != 127
        or wave2_summary.get("new_admitted_clusters") != 15
        or wave2_summary.get("ready") != 23
        or wave2_summary.get("scientifically_evaluated") != 23
    ):
        raise FinalConfirmatoryProtocolError(
            BLOCKED_INPUT_INTEGRITY,
            "wave2_capacity_contract_mismatch",
            "Wave-2 capacity gate differs from the frozen contract",
        )

    all_masks = pd.concat(
        [
            a2_masks.assign(source_stratum=_ORIGINAL),
            wave1_masks.assign(source_stratum=_WAVE1),
            wave2_masks.assign(source_stratum=_WAVE2),
        ],
        ignore_index=True,
    )
    v1_panel = _parquet(resolved["v1_panel"], "Scale-1B-v1 panel")
    v1_probes = _parquet(resolved["v1_probes"], "Scale-1B-v1 probes")
    v1_probe_manifest = _json(
        resolved["v1_probe_manifest"], "Scale-1B-v1 probe manifest"
    )
    v1_protocol = _json(
        resolved["v1_scoring_protocol"], "Scale-1B-v1 scoring protocol"
    )
    h1_manifest = _json(resolved["h1_manifest"], "H1 manifest")
    if (
        v1_protocol.get("score_semantics", {}).get("identity")
        != SCORING_PROTOCOL_VERSION
        or h1_manifest.get("analysis_protocol", {}).get("identity")
        != "stage0_h1_confirmatory_audit_v1"
        or v1_probe_manifest.get("identity_invariants", {}).get(
            "backbone_independent"
        )
        is not True
        or v1_probe_manifest.get("alphabet") != "".join(STANDARD_AMINO_ACIDS)
    ):
        raise FinalConfirmatoryProtocolError(
            BLOCKED_INPUT_INTEGRITY,
            "frozen_protocol_identity_mismatch",
            "Scale-1B-v1 score or H1 endpoint contract differs",
        )
    try:
        scoring_config = load_stage0_scoring_config(
            resolved["stage0_config"], paths.repository_root
        )
        model_identity = verify_authorized_model(scoring_config)
    except FixedProbeScoringError as exc:
        raise FinalConfirmatoryProtocolError(
            BLOCKED_SCORER_INTEGRITY,
            exc.code,
            str(exc),
        ) from exc

    v1_roles = v1_panel.set_index("pair_id")["cohort_role"].to_dict()
    original_normalized["scale1b_v1_role"] = original_normalized["pair_id"].map(
        v1_roles
    )
    structure_records = original_structures + wave1_structures + wave2_structures
    unique_records: dict[tuple[str, str], dict[str, Any]] = {
        (str(record["path"]), str(record["sha256"])): record
        for record in artifacts + structure_records
    }
    checkpoint_ref = paths.logical_ref(model_identity.checkpoint_path)
    unique_records[(checkpoint_ref, model_identity.checkpoint_sha256)] = {
        "path": checkpoint_ref,
        "sha256": model_identity.checkpoint_sha256,
        "label": "authorized ProteinMPNN checkpoint",
        "kind": "model_checkpoint",
    }
    implementation_ref = paths.logical_ref(model_identity.implementation_path)
    unique_records[(implementation_ref, model_identity.implementation_commit)] = {
        "path": implementation_ref,
        "commit": model_identity.implementation_commit,
        "label": "authorized ProteinMPNN implementation",
        "kind": "git_repository",
        "sha256": model_identity.implementation_commit,
    }
    return FinalConfirmatoryInputs(
        original_admitted=original_normalized.reset_index(drop=True),
        original_redundancy=original.reset_index(drop=True),
        wave1_admitted=wave1_normalized.reset_index(drop=True),
        wave2_admitted=wave2_normalized.reset_index(drop=True),
        common_masks=all_masks.reset_index(drop=True),
        v1_panel=v1_panel,
        v1_probes=v1_probes,
        v1_probe_manifest=v1_probe_manifest,
        v1_scoring_protocol=v1_protocol,
        wave1_manifest=wave1_manifest,
        wave2_manifest=wave2_manifest,
        wave2_summary=wave2_summary,
        h1_manifest=h1_manifest,
        model_identity=model_identity,
        input_artifacts=tuple(unique_records.values()),
        output_root_ref=config.output_root_ref,
    )


def _freeze_membership(
    inputs: FinalConfirmatoryInputs,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    union = pd.concat(
        [inputs.original_admitted, inputs.wave1_admitted, inputs.wave2_admitted],
        ignore_index=True,
    )
    try:
        assigned = assign_redundancy_roles(union)
    except RedundancyDiversityError as exc:
        raise FinalConfirmatoryProtocolError(
            BLOCKED_REPRESENTATIVE_SELECTION, exc.code, str(exc)
        ) from exc
    assigned = assigned.rename(columns={"redundancy_cluster_id": "cluster_id_30"})
    assigned["cohort_role"] = assigned["is_primary_representative"].map(
        {
            True: "SCALE1B_V2_PRIMARY",
            False: "SCALE1B_V2_SECONDARY_WITHIN_CLUSTER",
        }
    )
    original_observed = set(
        assigned.loc[
            assigned["source_stratum"].eq(_ORIGINAL)
            & assigned["is_primary_representative"],
            "protein_id",
        ]
    )
    original_expected = set(
        inputs.original_redundancy.loc[
            inputs.original_redundancy["is_cluster_representative"], "pair_id"
        ]
    )
    per_cluster = assigned.groupby("cluster_id_30")["is_primary_representative"].sum()
    if (
        assigned["cluster_id_30"].nunique() != _FINAL_CLUSTER_N
        or not per_cluster.eq(1).all()
        or original_observed != original_expected
    ):
        raise FinalConfirmatoryProtocolError(
            BLOCKED_REPRESENTATIVE_SELECTION,
            "representative_selector_regression",
            "Canonical representative selector does not reproduce frozen semantics",
        )
    primary = assigned.loc[assigned["is_primary_representative"]].copy()
    primary = primary.sort_values("cluster_id_30", kind="stable").reset_index(drop=True)
    primary.insert(0, "primary_index", range(1, len(primary) + 1))
    secondary = assigned.loc[~assigned["is_primary_representative"]].copy()
    secondary = secondary.sort_values(
        ["cluster_id_30", "representative_rank_within_cluster"], kind="stable"
    ).reset_index(drop=True)
    secondary.insert(0, "secondary_index", range(1, len(secondary) + 1))
    cluster_rows = []
    for cluster_id, group in assigned.groupby("cluster_id_30", sort=True):
        chosen = group.loc[group["is_primary_representative"]].iloc[0]
        composition = group["source_stratum"].value_counts().sort_index().to_dict()
        cluster_rows.append(
            {
                "cluster_id_30": cluster_id,
                "cluster_size_among_admitted": len(group),
                "primary_protein_id": str(chosen["protein_id"]),
                "secondary_count": len(group) - 1,
                "representative_selection_provenance": (
                    "frozen_scale1a3_scale1b_v1_selector"
                ),
                "source_stratum_composition": json.dumps(
                    {key: int(value) for key, value in composition.items()},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    clusters = pd.DataFrame(cluster_rows)
    if len(primary) != 127 or len(secondary) != 72 or len(clusters) != 127:
        raise FinalConfirmatoryProtocolError(
            BLOCKED_REPRESENTATIVE_SELECTION,
            "final_membership_cardinality_mismatch",
            "Final primary/secondary membership cardinality differs",
        )
    return primary, secondary, clusters


def _primary_masks(
    inputs: FinalConfirmatoryInputs, primary: pd.DataFrame
) -> pd.DataFrame:
    metadata = primary.set_index("protein_id")
    selected = inputs.common_masks.loc[
        inputs.common_masks["pair_id"].isin(metadata.index)
        & inputs.common_masks["common_mask"].fillna(False).astype(bool)
    ].copy()
    selected = selected.rename(columns={"pair_id": "protein_id"})
    selected["cluster_id_30"] = selected["protein_id"].map(
        metadata["cluster_id_30"]
    )
    selected["source_stratum"] = selected["protein_id"].map(
        metadata["source_stratum"]
    )
    selected["primary_index"] = selected["protein_id"].map(
        metadata["primary_index"]
    )
    selected = selected.sort_values(
        ["primary_index", "canonical_position"], kind="stable"
    ).reset_index(drop=True)
    expected = primary.set_index("protein_id")["common_mask_count"].astype(int)
    observed = selected.groupby("protein_id").size()
    if (
        selected["protein_id"].nunique() != 127
        or observed.to_dict() != expected.to_dict()
        or selected.duplicated(["protein_id", "canonical_position"]).any()
    ):
        raise FinalConfirmatoryProtocolError(
            BLOCKED_COMMON_MASK_INTEGRITY,
            "primary_common_mask_mismatch",
            "Primary common-mask view does not exactly match the selected cohort",
        )
    return selected


def _fixed_probes(primary: pd.DataFrame, masks: pd.DataFrame) -> pd.DataFrame:
    proteins = []
    for row in primary.to_dict("records"):
        protein_id = str(row["protein_id"])
        positions = masks.loc[
            masks["protein_id"].eq(protein_id), "canonical_position"
        ].astype(int).tolist()
        proteins.append(
            {
                "protein_id": protein_id,
                "uniprot_accession": str(row["canonical_accession"]),
                "canonical_wt_sequence": str(row["canonical_sequence"]),
                "mask_positions": positions,
            }
        )
    generated = build_fixed_probe_candidates({"proteins": proteins}).rename(
        columns={"position": "canonical_position", "mut_aa": "candidate_aa"}
    )
    metadata = primary.set_index("protein_id")
    generated.insert(0, "v2_probe_index", range(1, len(generated) + 1))
    generated["cluster_id_30"] = generated["protein_id"].map(
        metadata["cluster_id_30"]
    )
    generated["source_stratum"] = generated["protein_id"].map(
        metadata["source_stratum"]
    )
    generated["mutation"] = (
        generated["wt_aa"]
        + generated["canonical_position"].astype(str)
        + generated["candidate_aa"]
    )
    generated["probe_origin"] = "FROZEN_19AA_SINGLE_MUTANT_ENUMERATION"
    if (
        len(generated) != 19 * len(masks)
        or generated.duplicated(
            ["protein_id", "canonical_position", "candidate_aa"]
        ).any()
        or not generated.groupby(["protein_id", "canonical_position"])
        .size()
        .eq(19)
        .all()
        or generated["candidate_aa"].eq(generated["wt_aa"]).any()
    ):
        raise FinalConfirmatoryProtocolError(
            BLOCKED_PROBE_INTEGRITY,
            "fixed_probe_cardinality_mismatch",
            "Fixed-probe space is incomplete or duplicated",
        )
    return generated.reset_index(drop=True)


def _probe_reuse_audit(
    inputs: FinalConfirmatoryInputs,
    primary: pd.DataFrame,
    probes: pd.DataFrame,
) -> pd.DataFrame:
    v1_ids = set(inputs.v1_probes["protein_id"].astype(str))
    rows = []
    scientific_columns_new = [
        "canonical_position",
        "wt_aa",
        "candidate_aa",
        "sequence_hash",
    ]
    scientific_columns_old = [
        "mutation_position",
        "wt_aa",
        "mutant_aa",
        "sequence_hash",
    ]
    for protein in primary.to_dict("records"):
        protein_id = str(protein["protein_id"])
        current = probes.loc[probes["protein_id"].eq(protein_id), scientific_columns_new]
        available = protein_id in v1_ids
        compatible = False
        if available:
            historical = inputs.v1_probes.loc[
                inputs.v1_probes["protein_id"].eq(protein_id),
                scientific_columns_old,
            ].copy()
            historical.columns = scientific_columns_new
            compatible = (
                len(current) == len(historical)
                and current.reset_index(drop=True).astype(str).equals(
                    historical.reset_index(drop=True).astype(str)
                )
                and inputs.v1_probe_manifest.get("alphabet")
                == "".join(STANDARD_AMINO_ACIDS)
                and inputs.v1_probe_manifest.get("identity_invariants", {}).get(
                    "backbone_independent"
                )
                is True
                and inputs.v1_probe_manifest.get("identity_invariants", {}).get(
                    "hamming_distance"
                )
                == 1
                and inputs.v1_probe_manifest.get("identity_invariants", {}).get(
                    "mutation_position_domain"
                )
                == "frozen_common_mask"
            )
        count = len(current)
        rows.append(
            {
                "protein_id": protein_id,
                "cluster_id_30": str(protein["cluster_id_30"]),
                "historical_probe_available": available,
                "probe_reuse_status": (
                    "PROBE_REUSE_COMPATIBLE"
                    if compatible
                    else "PROBE_REUSE_INCOMPATIBLE"
                ),
                "sequence_compatible": compatible,
                "mask_compatible": compatible,
                "position_semantics_compatible": compatible,
                "candidate_enumeration_compatible": compatible,
                "probe_contract_compatible": compatible,
                "n_reused_probe_definitions": count if compatible else 0,
                "n_generated_probe_definitions": 0 if compatible else count,
            }
        )
    audit = pd.DataFrame(rows)
    if len(audit) != 127:
        raise FinalConfirmatoryProtocolError(
            BLOCKED_PROBE_INTEGRITY,
            "probe_reuse_audit_cardinality_mismatch",
            "Probe reuse audit must contain 127 rows",
        )
    return audit


def _scoring_plan(
    inputs: FinalConfirmatoryInputs,
    primary: pd.DataFrame,
    masks: pd.DataFrame,
    probes: pd.DataFrame,
) -> pd.DataFrame:
    probe_groups = {
        key: value for key, value in probes.groupby("protein_id", sort=False)
    }
    rows = []
    for protein in primary.to_dict("records"):
        protein_id = str(protein["protein_id"])
        protein_probes = probe_groups[protein_id]
        probe_binding = sha256_canonical(
            {
                "sequence_hashes": protein_probes["sequence_hash"].astype(str).tolist()
            }
        )
        mask_positions = masks.loc[
            masks["protein_id"].eq(protein_id), "canonical_position"
        ].astype(int).tolist()
        mask_binding = sha256_canonical(
            {
                "protein_id": protein_id,
                "canonical_positions": mask_positions,
                "canonical_sequence_sha256": str(
                    protein["canonical_sequence_sha256"]
                ),
            }
        )
        realizations = {
            repeat: make_decoding_realization(
                protein_id=protein_id,
                mask_length=int(protein["common_mask_count"]),
                repeat_index=repeat,
                seed=repeat,
                protocol_version=SCORING_PROTOCOL_VERSION,
            )
            for repeat in _REPEATS
        }
        for condition, ref, digest in (
            ("PDB", protein["pdb_structure_ref"], protein["pdb_structure_sha256"]),
            ("AFDB", protein["afdb_structure_ref"], protein["afdb_structure_sha256"]),
        ):
            for repeat in _REPEATS:
                realization = realizations[repeat]
                rows.append(
                    {
                        "logical_shard_id": f"{protein_id}::{condition}::r{repeat:02d}",
                        "protein_id": protein_id,
                        "cluster_id_30": str(protein["cluster_id_30"]),
                        "source_stratum": str(protein["source_stratum"]),
                        "structure_condition": condition,
                        "structure_artifact_reference": str(ref),
                        "structure_sha256": str(digest),
                        "common_mask_binding": mask_binding,
                        "probe_manifest_binding": (
                            f"{inputs.output_root_ref}/"
                            "scale1b_v2_probe_manifest.json"
                        ),
                        "protein_probe_identity_sha256": probe_binding,
                        "repeat": repeat,
                        "seed": repeat,
                        "explicit_realization_id": realization.fingerprint,
                        "realization_algorithm": realization.algorithm,
                        "proteinmpnn_source_commit": inputs.model_identity.implementation_commit,
                        "checkpoint_sha256": inputs.model_identity.checkpoint_sha256,
                        "score_contract_id": SCORING_PROTOCOL_VERSION,
                    }
                )
    plan = pd.DataFrame(rows)
    if (
        len(plan) != 7_620
        or plan["logical_shard_id"].duplicated().any()
        or not plan.groupby(["protein_id", "repeat"])[
            "explicit_realization_id"
        ].nunique().eq(1).all()
    ):
        raise FinalConfirmatoryProtocolError(
            BLOCKED_FINAL_MANIFEST,
            "logical_scoring_plan_mismatch",
            "Logical scoring plan does not contain 127 x 2 x 30 paired shards",
        )
    return plan


def build_final_confirmatory_protocol(
    inputs: FinalConfirmatoryInputs,
) -> FinalConfirmatoryResult:
    """Build one canonical structured result; perform no model execution."""
    primary, secondary, clusters = _freeze_membership(inputs)
    masks = _primary_masks(inputs, primary)
    probes = _fixed_probes(primary, masks)
    reuse = _probe_reuse_audit(inputs, primary, probes)
    reuse_status = reuse.set_index("protein_id")["probe_reuse_status"]
    probes["probe_origin"] = probes["protein_id"].map(reuse_status).map(
        {
            "PROBE_REUSE_COMPATIBLE": "REUSED_EXACT_SCIENTIFIC_DEFINITION",
            "PROBE_REUSE_INCOMPATIBLE": "GENERATED_FROZEN_V2_CONTRACT",
        }
    )
    plan = _scoring_plan(inputs, primary, masks, probes)
    common_positions = len(masks)
    fixed_probe_n = len(probes)
    source_counts = primary["source_stratum"].value_counts().to_dict()
    summary = {
        "schema_version": "dual-uq.final-confirmatory-freeze-summary.v1",
        "status": FINAL_PROTOCOL_FROZEN,
        "n_admitted_original": 135,
        "n_admitted_wave1": 49,
        "n_admitted_wave2": 15,
        "n_unique_admitted_proteins": 199,
        "n_duplicate_identity_bindings": 0,
        "n_conflicting_admission_bindings": 0,
        "n_primary_clusters": 127,
        "n_primary_proteins": 127,
        "n_secondary_proteins": 72,
        "n_original_stratum_primaries": int(source_counts.get(_ORIGINAL, 0)),
        "n_wave1_primaries": int(source_counts.get(_WAVE1, 0)),
        "n_wave2_primaries": int(source_counts.get(_WAVE2, 0)),
        "n_common_mask_positions": common_positions,
        "n_fixed_probes": fixed_probe_n,
        "n_probe_reuse_compatible": int(
            reuse["probe_reuse_status"].eq("PROBE_REUSE_COMPATIBLE").sum()
        ),
        "n_probe_generated_fresh": int(
            reuse["probe_reuse_status"].eq("PROBE_REUSE_INCOMPATIBLE").sum()
        ),
        "logical_scoring_shards": len(plan),
        "candidate_score_rows": fixed_probe_n * 2 * 30,
        "WT_score_rows": 127 * 2 * 30,
        "proteinmpnn_forward_executions": 0,
    }
    h1_binding = {
        "path": "experiments/p2_design_baseline/stage0/h1_confirmatory_audit_manifest.json",
        "sha256": next(
            record["sha256"]
            for record in inputs.input_artifacts
            if record["path"].endswith("h1_confirmatory_audit_manifest.json")
        ),
        "analysis_protocol_identity": inputs.h1_manifest["analysis_protocol"]["identity"],
        "endpoint_definitions": inputs.h1_manifest["analysis_protocol"][
            "primary_variables"
        ],
        "endpoints_computed_in_this_task": False,
    }
    probe_manifest = {
        "schema_version": "dual-uq.final-confirmatory-probe-manifest.v1",
        "status": FINAL_PROTOCOL_FROZEN,
        "primary_cohort_binding": {},
        "primary_common_mask_binding": {},
        "fixed_probe_binding": {},
        "probe_contract": {
            "identity": "frozen_common_mask_19aa_single_mutant_v1",
            "alphabet_order": list(STANDARD_AMINO_ACIDS),
            "position_semantics": "canonical_uniprot_position_1based",
            "candidate_order": "primary_order_then_position_then_canonical_alphabet",
            "probe_reuse_semantics": "exact_scientific_identity_without_structure_sha",
        },
    }
    scoring_protocol = {
        "schema_version": "dual-uq.final-confirmatory-scoring-protocol.v1",
        "status": FINAL_PROTOCOL_FROZEN,
        "model": {
            "family": "ProteinMPNN",
            "role": "stage0_internal_inverse_folding_scorer",
            "implementation_path": "third_party/ProteinMPNN",
            "implementation_commit": inputs.model_identity.implementation_commit,
            "checkpoint_path": (
                "third_party/ProteinMPNN/vanilla_model_weights/v_48_020.pt"
            ),
            "checkpoint_sha256": inputs.model_identity.checkpoint_sha256,
        },
        "score_semantics": inputs.v1_scoring_protocol["score_semantics"],
        "wt_semantics": {
            "formula": "delta_score_vs_wt=candidate_score-matching_wt_score",
            "matching": "same_protein_backbone_repeat_explicit_realization",
            "cross_condition_normalization_forbidden": True,
            "computed_in_this_task": False,
        },
        "structure_conditions": ["PDB", "AFDB"],
        "stochasticity": {
            "repeat_count": 30,
            "repeat_indices": list(_REPEATS),
            "seeds": list(_REPEATS),
            "realization_algorithm": DECODING_REALIZATION_ALGORITHM,
            "pairing": (
                "one explicit realization per protein/repeat shared across "
                "PDB/AFDB/WT/all probes"
            ),
        },
        "logical_cardinality": {
            "protein_count": 127,
            "condition_count": 2,
            "repeat_count": 30,
            "logical_scoring_shards": 7_620,
            "candidate_score_rows": summary["candidate_score_rows"],
            "WT_score_rows": 7_620,
        },
        "historical_result_reuse_contract": {
            "criteria": [
                "canonical protein identity and canonical sequence",
                "PDB and AFDB structure identity/SHA",
                "common mask",
                "scientific probe identity/order",
                "ProteinMPNN implementation commit and checkpoint SHA",
                "score contract, repeats, seeds, and realization semantics",
                "WT/probe and PDB/AFDB pairing semantics",
            ],
            "exact_compatibility_required": True,
            "unexecuted_v1_plan_is_score_result": False,
            "execution_inventory_deferred": True,
        },
        "h1_endpoint_binding": h1_binding,
        "proteinmpnn_forward_executions": 0,
    }
    manifest = {
        "schema_version": "dual-uq.final-confirmatory-freeze-manifest.v1",
        "status": FINAL_PROTOCOL_FROZEN,
        "upstream_artifacts": list(inputs.input_artifacts),
        "outputs": {},
        "manifest_self_hash_policy": "reported_externally_to_avoid_recursive_self_hash",
        "FINAL_NR_CLUSTERS": 127,
        "PRIMARY_CONFIRMATORY_TARGET_REACHED": True,
        "ONE_PRIMARY_PER_CLUSTER": True,
        "SECONDARIES_NOT_COUNTED_AS_PRIMARY_N": True,
        "V2_PRIMARY_SCORING": True,
        "V2_SECONDARY_SCORING": False,
        "ADMISSION_CONTRACT_UNCHANGED": True,
        "REDUNDANCY_CONVENTION_UNCHANGED": True,
        "COMMON_MASK_CONTRACT_UNCHANGED": True,
        "FIXED_PROBE_CONTRACT_UNCHANGED": True,
        "PROTEINMPNN_IMPLEMENTATION_FROZEN": True,
        "PROTEINMPNN_CHECKPOINT_FROZEN": True,
        "PROTEINMPNN_SCORE_CONTRACT_UNCHANGED": True,
        "REPEAT_CONTRACT_UNCHANGED": True,
        "HISTORICAL_PROBE_REUSE_EXACT_ONLY": True,
        "HISTORICAL_SCORE_RESULT_REUSE_RULE_FROZEN": True,
        "UNEXECUTED_V1_PLAN_NOT_COUNTED_AS_SCORE_RESULT": True,
        "ORIGINAL_FRAME_PROVENANCE_PRESERVED": True,
        "EXPANSION_WAVE1_PROVENANCE_PRESERVED": True,
        "EXPANSION_WAVE2_PROVENANCE_PRESERVED": True,
        "COMBINED_PREVALENCE_ESTIMATION_FORBIDDEN": True,
        "NO_OUTCOME_DEPENDENT_SELECTION": True,
        "NO_H1_ANALYSIS": True,
        "PROTEINMPNN_FORWARD_EXECUTIONS": 0,
        "FINAL_COHORT_MEMBERSHIP_FROZEN": True,
        "FINAL_PROBE_SPACE_FROZEN": True,
        "FINAL_SCORING_PROTOCOL_FROZEN": True,
        "PROTOCOL_ENGINEERING_CLOSED": True,
        "ARM_A_DATASET_EXPANSION_CLOSED": True,
    }
    return FinalConfirmatoryResult(
        status=FINAL_PROTOCOL_FROZEN,
        primary_cohort=primary,
        secondary_pool=secondary,
        cluster_reconciliation=clusters,
        primary_common_masks=masks,
        fixed_probes=probes,
        probe_reuse_audit=reuse,
        scoring_plan=plan,
        probe_manifest=probe_manifest,
        scoring_protocol=scoring_protocol,
        summary=summary,
        manifest=manifest,
        input_artifacts=inputs.input_artifacts,
    )


def _json_bytes(payload: dict[str, Any]) -> bytes:
    rendered = (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if b"/home/" in rendered or b"/mnt/" in rendered:
        raise FinalConfirmatoryProtocolError(
            BLOCKED_FINAL_MANIFEST,
            "absolute_path_leakage",
            "Final protocol artifact contains a machine absolute path",
        )
    return rendered


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def _write_immutable(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_file() and path.read_bytes() == payload:
            return "reused_identical"
        raise FinalConfirmatoryProtocolError(
            BLOCKED_FINAL_MANIFEST,
            "immutable_output_conflict",
            f"Immutable output differs: {path.name}",
        )
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            temporary = Path(handle.name)
        os.link(temporary, path)
        return "created"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _rehash_inputs(paths: ProjectPaths, result: FinalConfirmatoryResult) -> None:
    for record in result.input_artifacts:
        path = paths.resolve_logical(str(record["path"]))
        if record.get("kind") == "git_repository":
            try:
                commit = subprocess.check_output(
                    ["git", "-C", str(path), "rev-parse", "HEAD"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                clean = implementation_worktree_is_clean(path)
            except (OSError, subprocess.CalledProcessError, ProteinMPNNScoringError) as exc:
                raise FinalConfirmatoryProtocolError(
                    BLOCKED_SCORER_INTEGRITY,
                    "implementation_changed_before_materialization",
                    "Cannot revalidate ProteinMPNN implementation",
                ) from exc
            if commit != record["commit"] or not clean:
                raise FinalConfirmatoryProtocolError(
                    BLOCKED_SCORER_INTEGRITY,
                    "implementation_changed_before_materialization",
                    "ProteinMPNN implementation changed before materialization",
                )
        elif not path.is_file() or sha256_file(path) != record["sha256"]:
            status = (
                BLOCKED_SCORER_INTEGRITY
                if record.get("kind") == "model_checkpoint"
                else BLOCKED_INPUT_INTEGRITY
            )
            raise FinalConfirmatoryProtocolError(
                status,
                "upstream_changed_before_materialization",
                f"Frozen input changed: {record['path']}",
            )


def materialize_final_confirmatory_protocol(
    paths: ProjectPaths,
    config: FinalConfirmatoryProtocolConfig,
    result: FinalConfirmatoryResult,
) -> dict[str, str]:
    """Write all artifacts immutably; write the final manifest last."""
    _rehash_inputs(paths, result)
    expected_probe_manifest_ref = (
        f"{config.output_root_ref}/scale1b_v2_probe_manifest.json"
    )
    if set(result.scoring_plan["probe_manifest_binding"].astype(str)) != {
        expected_probe_manifest_ref
    }:
        raise FinalConfirmatoryProtocolError(
            BLOCKED_FINAL_MANIFEST,
            "output_binding_mismatch",
            "Scoring-plan probe binding differs from the configured output root",
        )
    root = paths.resolve_logical(config.output_root_ref)
    names = {
        "primary_cohort": "scale1b_v2_primary_cohort.parquet",
        "secondary_pool": "scale1b_v2_secondary_pool.parquet",
        "cluster_reconciliation": "scale1b_v2_cluster_reconciliation.parquet",
        "primary_common_masks": "scale1b_v2_primary_common_masks.parquet",
        "fixed_probes": "scale1b_v2_fixed_probes.parquet",
        "probe_reuse_audit": "scale1b_v2_probe_reuse_audit.parquet",
        "probe_manifest": "scale1b_v2_probe_manifest.json",
        "scoring_plan": "scale1b_v2_proteinmpnn_scoring_plan.parquet",
        "scoring_protocol": "scale1b_v2_scoring_protocol.json",
        "freeze_summary": "scale1b_v2_freeze_summary.json",
        "freeze_manifest": "scale1b_v2_freeze_manifest.json",
    }
    frames = {
        "primary_cohort": result.primary_cohort,
        "secondary_pool": result.secondary_pool,
        "cluster_reconciliation": result.cluster_reconciliation,
        "primary_common_masks": result.primary_common_masks,
        "fixed_probes": result.fixed_probes,
        "probe_reuse_audit": result.probe_reuse_audit,
        "scoring_plan": result.scoring_plan,
    }
    payloads = {name: _parquet_bytes(frame) for name, frame in frames.items()}
    output_records: dict[str, dict[str, Any]] = {}
    for name, payload in payloads.items():
        output_records[name] = {
            "path": paths.logical_ref(root / names[name]),
            "rows": len(frames[name]),
            "sha256": sha256_bytes(payload),
        }
    probe_manifest = dict(result.probe_manifest)
    probe_manifest["primary_cohort_binding"] = output_records["primary_cohort"]
    probe_manifest["primary_common_mask_binding"] = output_records[
        "primary_common_masks"
    ]
    probe_manifest["fixed_probe_binding"] = output_records["fixed_probes"]
    json_payloads = {
        "probe_manifest": _json_bytes(probe_manifest),
        "scoring_protocol": _json_bytes(result.scoring_protocol),
        "freeze_summary": _json_bytes(result.summary),
    }
    for name, payload in json_payloads.items():
        output_records[name] = {
            "path": paths.logical_ref(root / names[name]),
            "sha256": sha256_bytes(payload),
        }
    manifest = dict(result.manifest)
    manifest["outputs"] = output_records
    manifest["outputs"]["freeze_manifest"] = {
        "path": paths.logical_ref(root / names["freeze_manifest"]),
        "self_hash_policy": "reported_externally_to_avoid_recursive_self_hash",
    }
    manifest_payload = _json_bytes(manifest)
    statuses: dict[str, str] = {}
    for name in (
        "primary_cohort",
        "secondary_pool",
        "cluster_reconciliation",
        "primary_common_masks",
        "fixed_probes",
        "probe_reuse_audit",
        "scoring_plan",
    ):
        statuses[name] = _write_immutable(root / names[name], payloads[name])
    for name in ("probe_manifest", "scoring_protocol", "freeze_summary"):
        statuses[name] = _write_immutable(root / names[name], json_payloads[name])
    statuses["freeze_manifest"] = _write_immutable(
        root / names["freeze_manifest"], manifest_payload
    )
    return statuses


def run_final_confirmatory_protocol(
    paths: ProjectPaths,
    config: FinalConfirmatoryProtocolConfig | None = None,
) -> dict[str, Any]:
    """Validate, build, and immutably materialize the final protocol."""
    effective = config or FinalConfirmatoryProtocolConfig()
    inputs = validate_final_confirmatory_inputs(paths, effective)
    result = build_final_confirmatory_protocol(inputs)
    statuses = materialize_final_confirmatory_protocol(paths, effective, result)
    manifest_path = paths.resolve_logical(effective.output_root_ref) / (
        "scale1b_v2_freeze_manifest.json"
    )
    return {
        "status": result.status,
        "summary": result.summary,
        "write_status": statuses,
        "manifest_path": paths.logical_ref(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "proteinmpnn_forward_executions": 0,
    }
