from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _normalize_canonical(value: Any) -> Any:
    value_type = type(value)
    if value is None or value_type in {bool, int, str}:
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError("Canonical floats must be finite.")
        return value
    if value_type in {list, tuple}:
        return [_normalize_canonical(item) for item in value]
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("Canonical mapping keys must be strings.")
            normalized[key] = _normalize_canonical(item)
        return normalized
    raise TypeError(f"Unsupported canonical value type: {value_type.__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize an explicitly JSON-compatible scientific object canonically."""
    normalized = _normalize_canonical(value)
    payload = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return payload.encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    """Return a lowercase SHA-256 digest for exact bytes."""
    if type(data) is not bytes:
        raise TypeError("sha256_bytes requires bytes.")
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file's raw bytes without newline or encoding normalization."""
    if type(chunk_size) is not int:
        raise TypeError("chunk_size must be an integer, not bool or another type.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_canonical(value: Any) -> str:
    """Hash the canonical UTF-8 JSON representation of a scientific object."""
    return sha256_bytes(canonical_json_bytes(value))


# ProteinMPNN passes its seed to NumPy's legacy RandomState, whose strict
# upper bound is 2**32 - 1. Zero means "choose a random seed", so canonical
# scientific tool seeds use the shared nonzero range below.
MIN_TOOL_SEED = 1
MAX_TOOL_SEED = 2**32 - 1
_SHA256_HEX_PATTERN = re.compile(r"[0-9a-f]{64}")


def _validate_nonnegative_int(value: object, field: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be an integer, not bool or another type.")
    if value < 0:
        raise ValueError(f"{field} must be non-negative.")
    return value


def _validate_identity_string(value: object, field: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field} must be a string.")
    if not value or value != value.strip():
        raise ValueError(f"{field} must be non-empty with no surrounding whitespace.")
    if "\0" in value:
        raise ValueError(f"{field} must not contain NUL.")
    return value


def derive_seed_digest(
    *,
    base_seed: int,
    pipeline_version: str,
    protein_id: str,
    stage: str,
    candidate_id: str | None,
    replicate: int,
) -> str:
    """Return the full SHA-256 identity for one scientific random stream."""
    checked_base_seed = _validate_nonnegative_int(base_seed, "base_seed")
    checked_replicate = _validate_nonnegative_int(replicate, "replicate")
    checked_pipeline_version = _validate_identity_string(
        pipeline_version, "pipeline_version"
    )
    checked_protein_id = _validate_identity_string(protein_id, "protein_id")
    checked_stage = _validate_identity_string(stage, "stage")

    if candidate_id is None:
        candidate = {"present": False}
    else:
        candidate = {
            "present": True,
            "id": _validate_identity_string(candidate_id, "candidate_id"),
        }

    return sha256_canonical(
        {
            "base_seed": checked_base_seed,
            "pipeline_version": checked_pipeline_version,
            "protein_id": checked_protein_id,
            "stage": checked_stage,
            "candidate": candidate,
            "replicate": checked_replicate,
        }
    )


def seed_int_from_digest(seed_digest: str) -> int:
    """Map the first eight digest bytes into the shared nonzero tool range."""
    if type(seed_digest) is not str:
        raise TypeError("seed_digest must be a lowercase hexadecimal string.")
    if _SHA256_HEX_PATTERN.fullmatch(seed_digest) is None:
        raise ValueError("seed_digest must be 64 lowercase hexadecimal characters.")
    unsigned_prefix = int.from_bytes(bytes.fromhex(seed_digest[:16]), "big")
    return unsigned_prefix % MAX_TOOL_SEED + MIN_TOOL_SEED


def derive_seed(
    *,
    base_seed: int,
    pipeline_version: str,
    protein_id: str,
    stage: str,
    candidate_id: str | None,
    replicate: int,
) -> int:
    """Derive a deterministic integer seed for a scientific random stream."""
    digest = derive_seed_digest(
        base_seed=base_seed,
        pipeline_version=pipeline_version,
        protein_id=protein_id,
        stage=stage,
        candidate_id=candidate_id,
        replicate=replicate,
    )
    return seed_int_from_digest(digest)
