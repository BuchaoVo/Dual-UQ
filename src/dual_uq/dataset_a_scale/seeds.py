from __future__ import annotations

import re

from dual_uq.dataset_a_scale.hashing import sha256_canonical

# ProteinMPNN passes its seed to NumPy's legacy RandomState, whose strict
# upper bound is 2**32 - 1. ProteinMPNN treats zero as "choose a random seed",
# so formal Dataset A seeds use the shared nonzero range below.
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
