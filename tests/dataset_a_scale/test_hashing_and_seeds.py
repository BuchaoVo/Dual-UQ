from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dual_uq.dataset_a_scale.hashing import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_canonical,
    sha256_file,
)
from dual_uq.dataset_a_scale.seeds import (
    MAX_TOOL_SEED,
    MIN_TOOL_SEED,
    derive_seed,
    derive_seed_digest,
    seed_int_from_digest,
)


def _seed_identity(**overrides: object) -> dict[str, object]:
    identity: dict[str, object] = {
        "base_seed": 20260730,
        "pipeline_version": "protocol_v1",
        "protein_id": "index103",
        "stage": "P3",
        "candidate_id": "C7",
        "replicate": 4,
    }
    identity.update(overrides)
    return identity


def _derive_digest(identity: dict[str, object]) -> str:
    return derive_seed_digest(**identity)  # type: ignore[arg-type]


def _derive_seed(identity: dict[str, object]) -> int:
    return derive_seed(**identity)  # type: ignore[arg-type]


def test_canonical_mapping_key_order_is_irrelevant() -> None:
    assert canonical_json_bytes({"a": 1, "b": 2}) == canonical_json_bytes(
        {"b": 2, "a": 1}
    )


def test_nested_mapping_key_order_is_irrelevant() -> None:
    left = {"protein": {"id": "index103", "tier": 2}, "stage": "P3"}
    right = {"stage": "P3", "protein": {"tier": 2, "id": "index103"}}
    assert sha256_canonical(left) == sha256_canonical(right)


def test_list_order_remains_semantically_significant() -> None:
    assert sha256_canonical({"sites": [1, 2]}) != sha256_canonical(
        {"sites": [2, 1]}
    )


def test_tuple_has_json_array_semantics() -> None:
    assert canonical_json_bytes((1, "x")) == canonical_json_bytes([1, "x"])


def test_canonical_unicode_is_stable_utf8_without_bom() -> None:
    expected = '{"name":"蛋白质"}'.encode()
    assert canonical_json_bytes({"name": "蛋白质"}) == expected
    assert not expected.startswith(b"\xef\xbb\xbf")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_float_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        canonical_json_bytes(value)


@pytest.mark.parametrize(
    "value",
    [
        object(),
        {1, 2},
        b"abc",
        Path("input.cif"),
        np.int64(1),
        pd.Series([1]),
    ],
)
def test_unsupported_object_is_rejected(value: object) -> None:
    with pytest.raises(TypeError, match="Unsupported canonical value"):
        canonical_json_bytes(value)


def test_non_string_mapping_key_is_rejected() -> None:
    with pytest.raises(TypeError, match="string"):
        canonical_json_bytes({1: "x"})


def test_int_and_float_keep_distinct_typed_identity() -> None:
    assert canonical_json_bytes(1) == b"1"
    assert canonical_json_bytes(1.0) == b"1.0"
    assert sha256_canonical(1) != sha256_canonical(1.0)


def test_sha256_bytes_matches_known_digest() -> None:
    assert sha256_bytes(b"abc") == (
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )


def test_sha256_file_hashes_exact_raw_bytes(tmp_path: Path) -> None:
    path = tmp_path / "input.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == hashlib.sha256(b"abc").hexdigest()


def test_file_hash_does_not_normalize_newlines(tmp_path: Path) -> None:
    lf = tmp_path / "lf.tsv"
    crlf = tmp_path / "crlf.tsv"
    lf.write_bytes(b"a\tb\n1\t2\n")
    crlf.write_bytes(b"a\tb\r\n1\t2\r\n")
    assert sha256_file(lf) != sha256_file(crlf)


@pytest.mark.parametrize("chunk_size", [0, -1, False, 1.5])
def test_sha256_file_rejects_invalid_chunk_size(
    tmp_path: Path, chunk_size: object
) -> None:
    path = tmp_path / "nonempty.bin"
    path.write_bytes(b"scientific input")
    with pytest.raises((TypeError, ValueError), match="chunk_size"):
        sha256_file(path, chunk_size=chunk_size)  # type: ignore[arg-type]


def test_seed_is_deterministic_for_retry() -> None:
    identity = _seed_identity()
    assert _derive_digest(identity) == _derive_digest(identity)
    assert _derive_seed(identity) == _derive_seed(identity)


def test_seed_digest_is_lowercase_sha256_hex() -> None:
    digest = _derive_digest(_seed_identity())
    assert re.fullmatch(r"[0-9a-f]{64}", digest)


def test_seed_identity_has_frozen_digest_and_integer_fixture() -> None:
    identity = _seed_identity()
    assert _derive_digest(identity) == (
        "89991149c88ebfa8251dacf3c4b0d649"
        "0f5b4e87b1a00b5780ee4fcd54fd255f"
    )
    assert _derive_seed(identity) == 1378341107


def test_candidate_absence_has_distinct_namespace() -> None:
    absent = _derive_digest(_seed_identity(candidate_id=None))
    literal_none = _derive_digest(_seed_identity(candidate_id="None"))
    assert absent != literal_none


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("base_seed", 20260731),
        ("pipeline_version", "protocol_v2"),
        ("protein_id", "index36"),
        ("stage", "P4"),
        ("candidate_id", "C8"),
        ("replicate", 5),
    ],
)
def test_seed_changes_across_identity_namespaces(field: str, value: object) -> None:
    baseline = _derive_digest(_seed_identity())
    changed = _derive_digest(_seed_identity(**{field: value}))
    assert changed != baseline


def test_seed_does_not_depend_on_candidate_order() -> None:
    target = _seed_identity(candidate_id="C7")
    first = {candidate: _derive_seed(_seed_identity(candidate_id=candidate)) for candidate in ["C1", "C7"]}
    second = {
        candidate: _derive_seed(_seed_identity(candidate_id=candidate))
        for candidate in ["C20", "C10", "C7", "C1"]
    }
    assert first["C7"] == second["C7"] == _derive_seed(target)


def test_seed_does_not_depend_on_protein_order() -> None:
    first = {
        protein: _derive_seed(_seed_identity(protein_id=protein))
        for protein in ["P103", "P36"]
    }
    second = {
        protein: _derive_seed(_seed_identity(protein_id=protein))
        for protein in ["P1", "P2", "P36", "P103"]
    }
    assert first["P103"] == second["P103"]


def test_replicate_can_be_derived_independently() -> None:
    direct = _derive_seed(_seed_identity(replicate=12))
    loop = {
        replicate: _derive_seed(_seed_identity(replicate=replicate))
        for replicate in range(13)
    }
    assert direct == loop[12]


@pytest.mark.parametrize("value", [True, False, -1, 1.5, "1"])
def test_invalid_base_seed_is_rejected(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="base_seed"):
        _derive_digest(_seed_identity(base_seed=value))


@pytest.mark.parametrize("value", [True, -1, 1.0, "1"])
def test_invalid_replicate_is_rejected(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="replicate"):
        _derive_digest(_seed_identity(replicate=value))


@pytest.mark.parametrize("field", ["pipeline_version", "protein_id", "stage"])
@pytest.mark.parametrize("value", ["", "   ", " leading", "trailing "])
def test_noncanonical_identity_string_is_rejected(field: str, value: str) -> None:
    with pytest.raises(ValueError, match=field):
        _derive_digest(_seed_identity(**{field: value}))


@pytest.mark.parametrize("value", [1, True, Path("P1")])
def test_non_string_identity_is_rejected(value: object) -> None:
    with pytest.raises(TypeError, match="protein_id"):
        _derive_digest(_seed_identity(protein_id=value))


@pytest.mark.parametrize("value", ["", " C7", "C7 ", 7])
def test_invalid_present_candidate_identity_is_rejected(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="candidate_id"):
        _derive_digest(_seed_identity(candidate_id=value))


@pytest.mark.parametrize("field", ["pipeline_version", "protein_id", "stage", "candidate_id"])
def test_nul_in_identity_is_rejected(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _derive_digest(_seed_identity(**{field: "left\0right"}))


def test_seed_integer_is_within_supported_range() -> None:
    assert MIN_TOOL_SEED == 1
    assert MAX_TOOL_SEED == 2**32 - 1
    for replicate in range(100):
        seed = _derive_seed(_seed_identity(replicate=replicate))
        assert MIN_TOOL_SEED <= seed <= MAX_TOOL_SEED


def test_seed_integer_conversion_uses_first_eight_digest_bytes() -> None:
    digest = "ff" * 8 + "00" * 24
    expected = int.from_bytes(bytes.fromhex(digest[:16]), "big") % MAX_TOOL_SEED + 1
    assert seed_int_from_digest(digest) == expected


@pytest.mark.parametrize(
    "digest",
    ["", "0" * 63, "0" * 65, "g" * 64, "A" * 64],
)
def test_seed_integer_rejects_noncanonical_digest(digest: str) -> None:
    with pytest.raises(ValueError, match="digest"):
        seed_int_from_digest(digest)
