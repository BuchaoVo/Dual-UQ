from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dual_uq.core.artifacts import (
    json_safe,
    write_immutable_bytes,
    write_immutable_json,
    write_immutable_parquet,
)


def test_json_safe_preserves_report_serialization_semantics() -> None:
    value = {
        "values": (np.float64(1.5), np.float64("nan")),
        "count": np.int64(2),
    }

    assert json_safe(value) == {"values": [1.5, None], "count": 2}


def test_immutable_bytes_are_created_reused_and_protected(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "artifact.bin"

    assert write_immutable_bytes(path, b"first\n") == "created"
    assert write_immutable_bytes(path, b"first\n") == "reused_identical"
    with pytest.raises(RuntimeError, match="non-identical"):
        write_immutable_bytes(path, b"second\n")


def test_immutable_parquet_compares_table_content(tmp_path: Path) -> None:
    path = tmp_path / "table.parquet"
    frame = pd.DataFrame({"protein_id": ["P1", "P2"], "value": [1.0, 2.0]})

    assert write_immutable_parquet(path, frame) == "created"
    assert write_immutable_parquet(path, frame.copy()) == "reused_identical"
    with pytest.raises(RuntimeError, match="non-identical"):
        write_immutable_parquet(path, frame.assign(value=[1.0, 3.0]))


def test_immutable_json_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"

    assert write_immutable_json(path, {"z": 1, "name": "蛋白质"}) == "created"
    assert path.read_text(encoding="utf-8").endswith("\n")
    assert write_immutable_json(path, {"name": "蛋白质", "z": 1}) == "reused_identical"
