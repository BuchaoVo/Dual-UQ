"""Portable Dataset manifest, sequence and tool-output storage contracts."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ManifestValidationIssue",
    "ManifestValidationResult",
    "load_protein_manifest",
    "validate_protein_manifest",
    "write_protein_manifest",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("dual_uq.dataset.storage.manifests"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
