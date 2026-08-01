from __future__ import annotations

import hashlib
import json
import math
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
