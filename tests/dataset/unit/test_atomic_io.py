from __future__ import annotations

import json
from pathlib import Path

import pytest

from dual_uq.core.atomic_io import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_new_bytes,
    atomic_write_text,
)


def test_atomic_writers_preserve_exact_requested_bytes(tmp_path: Path) -> None:
    bytes_path = tmp_path / "nested/payload.bin"
    text_path = tmp_path / "nested/payload.txt"
    json_path = tmp_path / "nested/payload.json"

    atomic_write_bytes(bytes_path, b"\x00payload\n")
    atomic_write_text(text_path, "portable\n", encoding="utf-8")
    atomic_write_json(json_path, {"z": 1, "a": [2, 3]})

    assert bytes_path.read_bytes() == b"\x00payload\n"
    assert text_path.read_bytes() == b"portable\n"
    assert json_path.read_text(encoding="utf-8") == json.dumps(
        {"a": [2, 3], "z": 1},
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def test_atomic_write_new_bytes_refuses_to_replace_existing_output(
    tmp_path: Path,
) -> None:
    path = tmp_path / "immutable/result.bin"

    atomic_write_new_bytes(path, b"first\n")

    assert path.read_bytes() == b"first\n"
    with pytest.raises(FileExistsError):
        atomic_write_new_bytes(path, b"second\n")
    assert path.read_bytes() == b"first\n"
