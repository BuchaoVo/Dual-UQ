"""Reusable dataset construction and derivation APIs.

Heavy execution dependencies are loaded only when their public symbols are
requested. This keeps lightweight dataset services importable in dedicated
model environments that do not need Parquet planning/execution support.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from dual_uq.core.paths import ProjectPaths

__all__ = [
    "BiologicalIdentity",
    "CandidateContext",
    "CandidateDerivationResult",
    "DatasetPaths",
    "DatasetExperimentPaths",
    "DatasetRelease",
    "DatasetTask",
    "DerivationConfig",
    "DerivationError",
    "DerivationRunResult",
    "FailureRecord",
    "LogicalAssetRef",
    "MappingRecord",
    "PlanningOptions",
    "ProjectPaths",
    "ProteinRecord",
    "RetryPolicy",
    "StageResult",
    "StructureRecord",
    "TaskExecutionOutput",
    "TaskExecutionResult",
    "execute_tasks",
    "plan_tasks",
    "run_derivation",
]

_MODEL_EXPORTS = {
    "BiologicalIdentity",
    "CandidateContext",
    "CandidateDerivationResult",
    "DatasetRelease",
    "DerivationConfig",
    "DerivationError",
    "DerivationRunResult",
    "FailureRecord",
    "LogicalAssetRef",
    "MappingRecord",
    "ProteinRecord",
    "StageResult",
    "StructureRecord",
}
_PIPELINE_EXPORTS = {
    "DatasetTask",
    "PlanningOptions",
    "RetryPolicy",
    "TaskExecutionOutput",
    "TaskExecutionResult",
    "execute_tasks",
    "plan_tasks",
    "run_derivation",
}


def __getattr__(name: str) -> Any:
    if name in _MODEL_EXPORTS:
        value = getattr(import_module("dual_uq.dataset.models"), name)
    elif name == "DatasetPaths":
        value = getattr(import_module("dual_uq.dataset.paths"), name)
    elif name == "DatasetExperimentPaths":
        value = getattr(import_module("dual_uq.dataset.paths"), name)
    elif name in _PIPELINE_EXPORTS:
        value = getattr(import_module("dual_uq.dataset.pipeline"), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
