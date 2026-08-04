"""Lightweight immutable models for generic dataset derivation."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Literal

StageStatus = Literal["complete", "failed", "unobservable", "skipped_dependency"]
_STAGE_STATUSES = {"complete", "failed", "unobservable", "skipped_dependency"}
_DERIVATION_STAGE_ORDER = (
    "raw",
    "identity",
    "pair_qc",
    "mapping",
    "fragment",
    "pae",
    "confidence",
    "observability",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


class DerivationError(RuntimeError):
    """A structured derivation-contract failure."""

    def __init__(self, code: str, message: str) -> None:
        self.code = _required_text(code, "failure code")
        super().__init__(message)


def _required_text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required")
    return value.strip()


def _immutable_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({key: _immutable_value(item) for key, item in value.items()})


def _immutable_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _immutable_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_immutable_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_immutable_value(item) for item in value)
    return value


@dataclass(frozen=True)
class BiologicalIdentity:
    pair_id: str
    pdb_id: str
    chain_id: str
    uniprot_accession: str
    polymer_entity_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "pair_id", _required_text(self.pair_id, "pair_id"))
        object.__setattr__(self, "pdb_id", _required_text(self.pdb_id, "pdb_id").lower())
        object.__setattr__(self, "chain_id", _required_text(self.chain_id, "chain_id"))
        object.__setattr__(
            self,
            "uniprot_accession",
            _required_text(self.uniprot_accession, "uniprot_accession").upper(),
        )
        object.__setattr__(
            self,
            "polymer_entity_id",
            _required_text(self.polymer_entity_id, "polymer_entity_id"),
        )


@dataclass(frozen=True)
class LogicalAssetRef:
    asset_type: str
    logical_path: str
    sha256: str | None
    provenance: str
    expected_model_identity: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "asset_type", _required_text(self.asset_type, "asset_type"))
        path = _required_text(self.logical_path, "logical_path")
        logical = PurePosixPath(path)
        if (
            "\\" in path
            or logical.is_absolute()
            or any(part in {"", ".", ".."} for part in logical.parts)
        ):
            raise ValueError("asset requires a portable logical path")
        object.__setattr__(self, "logical_path", logical.as_posix())
        if self.sha256 is not None and _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("asset sha256 must be 64 lowercase hexadecimal characters")
        object.__setattr__(self, "provenance", _required_text(self.provenance, "provenance"))
        if self.expected_model_identity is not None:
            object.__setattr__(
                self,
                "expected_model_identity",
                _required_text(self.expected_model_identity, "expected_model_identity"),
            )


@dataclass(frozen=True)
class CandidateContext:
    candidate_index: int
    identity: BiologicalIdentity
    exact_afdb_accession: str
    expected_afdb_model_identity: str
    assets: tuple[LogicalAssetRef, ...]
    source_bindings: tuple[tuple[str, str], ...]
    protocol_binding: str
    selection_roles: tuple[str, ...] = ()
    selection_reason: str | None = None
    prederivation_evidence: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.candidate_index, bool) or self.candidate_index < 1:
            raise ValueError("candidate_index must be a positive integer")
        accession = _required_text(
            self.exact_afdb_accession, "exact AFDB accession"
        ).upper()
        if accession != self.identity.uniprot_accession:
            raise ValueError("exact AFDB accession must equal biological UniProt identity")
        object.__setattr__(self, "exact_afdb_accession", accession)
        object.__setattr__(
            self,
            "expected_afdb_model_identity",
            _required_text(
                self.expected_afdb_model_identity, "expected_afdb_model_identity"
            ),
        )
        asset_types = [asset.asset_type for asset in self.assets]
        if len(asset_types) != len(set(asset_types)):
            raise ValueError("duplicate asset type in CandidateContext")
        object.__setattr__(self, "protocol_binding", _required_text(self.protocol_binding, "protocol_binding"))
        if self.selection_reason is not None:
            object.__setattr__(
                self, "selection_reason", _required_text(self.selection_reason, "selection_reason")
            )

    def asset(self, asset_type: str) -> LogicalAssetRef:
        matches = [asset for asset in self.assets if asset.asset_type == asset_type]
        if len(matches) != 1:
            raise KeyError(f"Expected one logical asset {asset_type!r}; found {len(matches)}")
        return matches[0]

    def evidence(self) -> Mapping[str, object]:
        return MappingProxyType(dict(self.prederivation_evidence))


@dataclass(frozen=True)
class DerivationConfig:
    protocol_version: str
    protocol_binding: str
    preflight_thresholds: Mapping[str, float]
    observability_thresholds: Mapping[str, object]
    schema_version: str = "dataset-a.derivation.v1"
    preflight: Mapping[str, object] = field(default_factory=dict)
    scope: Mapping[str, object] = field(default_factory=dict)
    selection_policy: Mapping[str, object] = field(default_factory=dict)
    report_metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "protocol_version", _required_text(self.protocol_version, "protocol_version")
        )
        object.__setattr__(
            self, "protocol_binding", _required_text(self.protocol_binding, "protocol_binding")
        )
        object.__setattr__(
            self, "preflight_thresholds", _immutable_mapping(self.preflight_thresholds)
        )
        object.__setattr__(
            self, "observability_thresholds", _immutable_mapping(self.observability_thresholds)
        )
        object.__setattr__(
            self, "schema_version", _required_text(self.schema_version, "schema_version")
        )
        object.__setattr__(self, "preflight", _immutable_mapping(self.preflight))
        object.__setattr__(self, "scope", _immutable_mapping(self.scope))
        object.__setattr__(
            self, "selection_policy", _immutable_mapping(self.selection_policy)
        )
        object.__setattr__(
            self, "report_metadata", _immutable_mapping(self.report_metadata)
        )


@dataclass(frozen=True)
class StageResult:
    stage: str
    status: StageStatus
    primary_failure_code: str | None = None
    warnings: tuple[str, ...] = ()
    metrics: Mapping[str, object] = field(default_factory=dict)
    artifacts: tuple[LogicalAssetRef, ...] = ()
    dependent_unavailable: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", _required_text(self.stage, "stage"))
        if self.status not in _STAGE_STATUSES:
            raise ValueError(f"invalid stage status: {self.status}")
        if self.status in {"failed", "unobservable"} and not self.primary_failure_code:
            raise ValueError("failed and unobservable stages require a failure code")
        if self.status in {"complete", "skipped_dependency"} and self.primary_failure_code:
            raise ValueError(f"{self.status} stage must not have a primary failure code")
        if self.status == "skipped_dependency" and not self.dependent_unavailable:
            raise ValueError("skipped_dependency requires an upstream dependency")
        object.__setattr__(self, "metrics", _immutable_mapping(self.metrics))


def skipped_dependency_results(
    *, upstream_stage: str, stages: Sequence[str]
) -> tuple[StageResult, ...]:
    upstream = _required_text(upstream_stage, "upstream_stage")
    return tuple(
        StageResult(
            stage=stage,
            status="skipped_dependency",
            dependent_unavailable=(upstream,),
        )
        for stage in stages
    )


@dataclass(frozen=True)
class CandidateDerivationResult:
    context: CandidateContext
    stages: tuple[StageResult, ...]
    evidence: Mapping[str, object]
    report_record: Mapping[str, object]

    def __post_init__(self) -> None:
        names = [stage.stage for stage in self.stages]
        if len(names) != len(set(names)):
            raise ValueError("duplicate stage in candidate derivation result")
        try:
            positions = [_DERIVATION_STAGE_ORDER.index(name) for name in names]
        except ValueError as exc:
            raise ValueError("unknown candidate derivation stage") from exc
        if positions != sorted(positions):
            raise ValueError("candidate derivation stage order is invalid")
        object.__setattr__(self, "evidence", _immutable_mapping(self.evidence))
        object.__setattr__(self, "report_record", _immutable_mapping(self.report_record))


@dataclass(frozen=True)
class DerivationRunResult:
    schema_version: str
    preflight: Mapping[str, object]
    candidates: tuple[CandidateDerivationResult, ...]
    summary: Mapping[str, object]
    attrition_bias_probe: Mapping[str, object]
    scope: Mapping[str, object]
    selection_policy: Mapping[str, object] = field(default_factory=dict)
    report_metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _required_text(self.schema_version, "schema_version"))
        object.__setattr__(self, "preflight", _immutable_mapping(self.preflight))
        object.__setattr__(self, "summary", _immutable_mapping(self.summary))
        object.__setattr__(
            self, "attrition_bias_probe", _immutable_mapping(self.attrition_bias_probe)
        )
        object.__setattr__(self, "scope", _immutable_mapping(self.scope))
        object.__setattr__(
            self, "selection_policy", _immutable_mapping(self.selection_policy)
        )
        object.__setattr__(
            self, "report_metadata", _immutable_mapping(self.report_metadata)
        )
