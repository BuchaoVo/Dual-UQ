from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _as_float_array(value: Any) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.size == 0:
        return None
    return array


def load_plddt(path: str | Path, expected_length: int | None = None) -> np.ndarray:
    """Load AFDB pLDDT JSON across common schema variants."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    candidates: list[Any] = []
    if isinstance(data, list):
        candidates.append(data)
        if data and isinstance(data[0], dict):
            candidates.extend(
                data[0].get(key)
                for key in ("confidenceScore", "plddt", "confidence", "scores")
                if key in data[0]
            )
    elif isinstance(data, dict):
        candidates.extend(
            data.get(key)
            for key in ("confidenceScore", "plddt", "confidence", "scores")
            if key in data
        )

    for candidate in candidates:
        array = _as_float_array(candidate)
        if array is None:
            continue
        array = array.reshape(-1)
        if expected_length is None or len(array) == expected_length:
            return array

    raise ValueError(
        f"Unable to parse pLDDT from {path}. "
        f"Expected length: {expected_length!r}; JSON type: {type(data).__name__}"
    )


def load_pae(path: str | Path, expected_length: int | None = None) -> np.ndarray:
    """Load AFDB PAE JSON across matrix and flattened schema variants."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    record: Any = data[0] if isinstance(data, list) and data else data

    if isinstance(record, dict):
        for key in ("predicted_aligned_error", "pae"):
            if key in record:
                array = _as_float_array(record[key])
                if array is not None and array.ndim == 2:
                    if expected_length is None or array.shape == (
                        expected_length,
                        expected_length,
                    ):
                        return array

        residue1 = record.get("residue1")
        residue2 = record.get("residue2")
        distance = record.get("distance")
        if residue1 is not None and residue2 is not None and distance is not None:
            i = np.asarray(residue1, dtype=int)
            j = np.asarray(residue2, dtype=int)
            d = np.asarray(distance, dtype=float)
            if not (len(i) == len(j) == len(d)):
                raise ValueError("Flattened PAE arrays have inconsistent lengths.")
            n = expected_length or int(max(i.max(), j.max()))
            matrix = np.full((n, n), np.nan, dtype=float)
            offset = 1 if min(i.min(), j.min()) >= 1 else 0
            matrix[i - offset, j - offset] = d
            return matrix

    array = _as_float_array(record)
    if array is not None and array.ndim == 2:
        if expected_length is None or array.shape == (expected_length, expected_length):
            return array

    raise ValueError(
        f"Unable to parse PAE from {path}. "
        f"Expected shape: {(expected_length, expected_length) if expected_length else None}"
    )
