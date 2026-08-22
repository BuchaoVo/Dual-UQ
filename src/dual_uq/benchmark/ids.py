"""Deterministic IDs for immutable scientific identities."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def canonical_id(kind: str, fields: Mapping[str, Any]) -> str:
    if not kind or not fields:
        raise ValueError("kind and scientific identity fields are required")
    payload = {"kind": kind, "fields": dict(sorted(fields.items()))}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return f"{kind}:{hashlib.sha256(encoded).hexdigest()[:24]}"


def protein_id(uniprot_id: str) -> str:
    value = str(uniprot_id).strip().upper()
    if not value:
        raise ValueError("uniprot_id must be non-empty")
    return canonical_id("protein", {"uniprot_id": value})


def structure_id(source_structure_id: str, semantics: str, condition_label: str, protein: str) -> str:
    return canonical_id("structure", {"source_structure_id": source_structure_id, "semantics": semantics, "condition_label": condition_label, "protein_id": protein})


def pair_id(protein: str, semantics: str, condition_1_structure_id: str, condition_2_structure_id: str) -> str:
    return canonical_id("pair", {"protein_id": protein, "semantics": semantics, "condition_1_structure_id": condition_1_structure_id, "condition_2_structure_id": condition_2_structure_id})


def instance_id(pair: str) -> str:
    return canonical_id("instance", {"pair_id": pair})
