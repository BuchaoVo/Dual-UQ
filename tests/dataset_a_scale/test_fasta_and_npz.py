from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dual_uq.dataset_a_scale.proteinmpnn import (
    PROTEINMPNN_ALPHABET,
    FastaRecord,
    find_unique_score_npz,
    load_probability_npz,
    load_score_only_npz,
    read_single_candidate_fasta,
    sequence_sha256,
    validate_protein_sequence,
    write_single_candidate_fasta,
)

SEQUENCE = "ACDEFGHIKLMNPQRSTVWY"


def _indices(sequence: str = SEQUENCE) -> np.ndarray:
    return np.asarray(
        [PROTEINMPNN_ALPHABET.index(residue) for residue in sequence],
        dtype=np.int64,
    )


def _write_score_npz(
    path: Path,
    *,
    sequence: str = SEQUENCE,
    repeats: int = 3,
    **overrides: np.ndarray,
) -> None:
    payload: dict[str, np.ndarray] = {
        "score": np.linspace(1.0, 2.0, repeats, dtype=np.float32),
        "global_score": np.linspace(2.0, 3.0, repeats, dtype=np.float32),
        "S": _indices(sequence),
        "seq_str": np.asarray(sequence),
    }
    payload.update(overrides)
    np.savez(path, **payload)


def _write_probability_npz(
    path: Path,
    *,
    sequence: str = SEQUENCE,
    repeats: int = 3,
    alphabet_size: int = len(PROTEINMPNN_ALPHABET),
    **overrides: np.ndarray,
) -> None:
    length = len(sequence)
    payload: dict[str, np.ndarray] = {
        "log_p": np.zeros((repeats, length, alphabet_size), dtype=np.float32),
        "S": _indices(sequence),
        "mask": np.ones(length, dtype=np.float32),
        "design_mask": np.ones(length, dtype=np.float32),
    }
    payload.update(overrides)
    np.savez(path, **payload)


def test_single_candidate_fasta_round_trip_and_lf_output(tmp_path: Path) -> None:
    path = tmp_path / "candidate.fa"
    expected = FastaRecord("candidate-01", SEQUENCE)

    write_single_candidate_fasta(path, expected)

    assert path.read_bytes() == f">candidate-01\n{SEQUENCE}\n".encode("ascii")
    assert read_single_candidate_fasta(path) == expected


def test_reader_accepts_crlf_single_candidate_fasta(tmp_path: Path) -> None:
    path = tmp_path / "candidate.fa"
    path.write_bytes(f">candidate-01\r\n{SEQUENCE}\r\n".encode("ascii"))
    assert read_single_candidate_fasta(path) == FastaRecord("candidate-01", SEQUENCE)


@pytest.mark.parametrize(
    "payload",
    [
        f">candidate-01\r{SEQUENCE}\r".encode("ascii"),
        f">candidate-01\n{SEQUENCE[:10]}\r\n{SEQUENCE[10:]}\n".encode("ascii"),
        f">candidate-01\x0b{SEQUENCE}\x0b".encode("ascii"),
    ],
)
def test_reader_rejects_non_lf_crlf_or_mixed_line_endings(
    tmp_path: Path, payload: bytes
) -> None:
    path = tmp_path / "candidate.fa"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="line endings"):
        read_single_candidate_fasta(path)


@pytest.mark.parametrize(
    "sequence",
    ["", "ACDX", "ACDB", "ACDZ", "ACDJ", "ACDU", "ACDO", "acde", "AC DE", "AC\nDE"],
)
def test_invalid_protein_sequence_is_rejected(sequence: str) -> None:
    with pytest.raises(ValueError, match="sequence"):
        validate_protein_sequence(sequence)


@pytest.mark.parametrize(
    "candidate_id",
    [
        "",
        " leading",
        "trailing ",
        "bad\nline",
        "bad\rline",
        "bad\tid",
        "bad\x1fid",
        "bad\0id",
        ">nested",
    ],
)
def test_invalid_candidate_id_is_rejected(candidate_id: str) -> None:
    with pytest.raises(ValueError, match="candidate_id"):
        FastaRecord(candidate_id, SEQUENCE)


def test_reader_rejects_multiple_fasta_records(tmp_path: Path) -> None:
    path = tmp_path / "multiple.fa"
    path.write_text(f">one\n{SEQUENCE}\n>two\n{SEQUENCE}\n", encoding="ascii")
    with pytest.raises(ValueError, match="exactly one"):
        read_single_candidate_fasta(path)


def test_writer_does_not_overwrite_conflicting_existing_fasta(tmp_path: Path) -> None:
    path = tmp_path / "candidate.fa"
    path.write_text(">old\nACDE\n", encoding="ascii")
    historical = path.read_bytes()

    with pytest.raises(FileExistsError, match="different content"):
        write_single_candidate_fasta(path, FastaRecord("new", SEQUENCE))

    assert path.read_bytes() == historical


def test_writer_allows_idempotent_identical_existing_fasta(tmp_path: Path) -> None:
    path = tmp_path / "candidate.fa"
    record = FastaRecord("candidate-01", SEQUENCE)
    write_single_candidate_fasta(path, record)
    historical = path.read_bytes()
    write_single_candidate_fasta(path, record)
    assert path.read_bytes() == historical


def test_sequence_sha_depends_only_on_sequence(tmp_path: Path) -> None:
    first = FastaRecord("first", SEQUENCE)
    second = FastaRecord("second", SEQUENCE)
    write_single_candidate_fasta(tmp_path / "one.fa", first)
    write_single_candidate_fasta(tmp_path / "nested/two.fa", second)
    assert sequence_sha256(first.sequence) == sequence_sha256(second.sequence)


def test_find_unique_score_npz_selects_only_fasta_1(tmp_path: Path) -> None:
    expected = tmp_path / "score_only/target_fasta_1.npz"
    expected.parent.mkdir()
    _write_score_npz(expected)
    _write_score_npz(expected.parent / "target_fasta_2.npz")
    np.savez(expected.parent / "target_pdb.npz", unrelated=np.asarray([1]))
    assert find_unique_score_npz(tmp_path) == expected


def test_find_unique_score_npz_rejects_zero_matches(tmp_path: Path) -> None:
    _write_score_npz(tmp_path / "target_fasta_2.npz")
    with pytest.raises(FileNotFoundError, match="fasta_1"):
        find_unique_score_npz(tmp_path)


def test_find_unique_score_npz_rejects_multiple_matches(tmp_path: Path) -> None:
    for directory in (tmp_path / "one", tmp_path / "two"):
        directory.mkdir()
        _write_score_npz(directory / "target_fasta_1.npz")
    with pytest.raises(ValueError, match="multiple"):
        find_unique_score_npz(tmp_path)


def test_valid_score_only_npz(tmp_path: Path) -> None:
    path = tmp_path / "target_fasta_1.npz"
    _write_score_npz(path, repeats=3)
    result = load_score_only_npz(path, expected_sequence=SEQUENCE, expected_repeats=3)
    assert result.repeat_count == 3
    assert result.sequence == SEQUENCE
    assert result.scores.shape == (3,)
    assert result.sequence_indices.shape == (len(SEQUENCE),)


def test_score_loader_rejects_missing_key(tmp_path: Path) -> None:
    path = tmp_path / "target_fasta_1.npz"
    np.savez(path, score=np.ones(3), global_score=np.ones(3), S=_indices())
    with pytest.raises(ValueError, match="missing.*seq_str"):
        load_score_only_npz(path, expected_sequence=SEQUENCE, expected_repeats=3)


def test_score_loader_rejects_wrong_sequence(tmp_path: Path) -> None:
    path = tmp_path / "target_fasta_1.npz"
    wrong = _indices().copy()
    wrong[-1] = 0
    _write_score_npz(path, S=wrong)
    with pytest.raises(ValueError, match="sequence"):
        load_score_only_npz(path, expected_sequence=SEQUENCE, expected_repeats=3)


def test_score_loader_rejects_wrong_later_repeat_sequence(tmp_path: Path) -> None:
    path = tmp_path / "target_fasta_1.npz"
    repeated = np.repeat(_indices()[None, :], 3, axis=0)
    repeated[2, -1] = 0
    _write_score_npz(path, S=repeated)
    with pytest.raises(ValueError, match="S shape"):
        load_score_only_npz(path, expected_sequence=SEQUENCE, expected_repeats=3)


def test_score_loader_rejects_wrong_repeat_count(tmp_path: Path) -> None:
    path = tmp_path / "target_fasta_1.npz"
    _write_score_npz(path, repeats=2)
    with pytest.raises(ValueError, match="repeat"):
        load_score_only_npz(path, expected_sequence=SEQUENCE, expected_repeats=3)


@pytest.mark.parametrize(
    "field,value",
    [
        ("score", np.ones((3, 1), dtype=np.float32)),
        ("global_score", np.ones((3, 1), dtype=np.float32)),
        ("S", np.ones((1, len(SEQUENCE)), dtype=np.int64)),
        ("seq_str", np.asarray([SEQUENCE])),
    ],
)
def test_score_loader_rejects_wrong_shape(
    tmp_path: Path, field: str, value: np.ndarray
) -> None:
    path = tmp_path / "target_fasta_1.npz"
    _write_score_npz(path, **{field: value})
    with pytest.raises(ValueError, match="shape"):
        load_score_only_npz(path, expected_sequence=SEQUENCE, expected_repeats=3)


@pytest.mark.parametrize("field,bad", [("score", np.nan), ("global_score", np.inf)])
def test_score_loader_rejects_nonfinite_values(
    tmp_path: Path, field: str, bad: float
) -> None:
    path = tmp_path / "target_fasta_1.npz"
    values = np.ones(3, dtype=np.float32)
    values[1] = bad
    _write_score_npz(path, **{field: values})
    with pytest.raises(ValueError, match="finite"):
        load_score_only_npz(path, expected_sequence=SEQUENCE, expected_repeats=3)


def test_score_loader_rejects_invalid_s_index(tmp_path: Path) -> None:
    path = tmp_path / "target_fasta_1.npz"
    invalid = _indices().copy()
    invalid[0] = len(PROTEINMPNN_ALPHABET)
    _write_score_npz(path, S=invalid)
    with pytest.raises(ValueError, match="S index"):
        load_score_only_npz(path, expected_sequence=SEQUENCE, expected_repeats=3)


def test_score_loader_rejects_object_dtype(tmp_path: Path) -> None:
    path = tmp_path / "target_fasta_1.npz"
    _write_score_npz(path, score=np.asarray([1, 2, 3], dtype=object))
    with pytest.raises(ValueError, match="Object arrays|dtype"):
        load_score_only_npz(path, expected_sequence=SEQUENCE, expected_repeats=3)


def test_valid_probability_npz(tmp_path: Path) -> None:
    path = tmp_path / "probabilities.npz"
    _write_probability_npz(path, repeats=3)
    result = load_probability_npz(path, expected_sequence=SEQUENCE, expected_repeats=3)
    assert result.repeat_count == 3
    assert result.log_probabilities.shape == (
        3,
        len(SEQUENCE),
        len(PROTEINMPNN_ALPHABET),
    )
    assert result.sequence == SEQUENCE


def test_probability_loader_rejects_missing_key(tmp_path: Path) -> None:
    path = tmp_path / "probabilities.npz"
    np.savez(path, log_p=np.zeros((3, len(SEQUENCE), 21)), S=_indices())
    with pytest.raises(ValueError, match="missing"):
        load_probability_npz(path, expected_sequence=SEQUENCE, expected_repeats=3)


def test_probability_loader_rejects_wrong_sequence(tmp_path: Path) -> None:
    path = tmp_path / "probabilities.npz"
    wrong = _indices().copy()
    wrong[1] = 0
    _write_probability_npz(path, S=wrong)
    with pytest.raises(ValueError, match="sequence"):
        load_probability_npz(path, expected_sequence=SEQUENCE, expected_repeats=3)


def test_probability_loader_rejects_wrong_repeat_count(tmp_path: Path) -> None:
    path = tmp_path / "probabilities.npz"
    _write_probability_npz(path, repeats=2)
    with pytest.raises(ValueError, match="repeat"):
        load_probability_npz(path, expected_sequence=SEQUENCE, expected_repeats=3)


def test_probability_loader_rejects_wrong_length(tmp_path: Path) -> None:
    path = tmp_path / "probabilities.npz"
    _write_probability_npz(
        path,
        log_p=np.zeros((3, len(SEQUENCE) - 1, 21), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="log_p shape"):
        load_probability_npz(path, expected_sequence=SEQUENCE, expected_repeats=3)


def test_probability_loader_rejects_wrong_alphabet_dimension(tmp_path: Path) -> None:
    path = tmp_path / "probabilities.npz"
    _write_probability_npz(path, alphabet_size=20)
    with pytest.raises(ValueError, match="log_p shape"):
        load_probability_npz(path, expected_sequence=SEQUENCE, expected_repeats=3)


def test_probability_loader_rejects_wrong_rank(tmp_path: Path) -> None:
    path = tmp_path / "probabilities.npz"
    _write_probability_npz(path, log_p=np.zeros((len(SEQUENCE), 21), dtype=np.float32))
    with pytest.raises(ValueError, match="log_p shape"):
        load_probability_npz(path, expected_sequence=SEQUENCE, expected_repeats=3)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_probability_loader_rejects_nonfinite_values(tmp_path: Path, bad: float) -> None:
    path = tmp_path / "probabilities.npz"
    log_p = np.zeros((3, len(SEQUENCE), 21), dtype=np.float32)
    log_p[1, 2, 3] = bad
    _write_probability_npz(path, log_p=log_p)
    with pytest.raises(ValueError, match="finite"):
        load_probability_npz(path, expected_sequence=SEQUENCE, expected_repeats=3)


def test_probability_loader_rejects_invalid_s_index(tmp_path: Path) -> None:
    path = tmp_path / "probabilities.npz"
    invalid = _indices().copy()
    invalid[0] = -1
    _write_probability_npz(path, S=invalid)
    with pytest.raises(ValueError, match="S index"):
        load_probability_npz(path, expected_sequence=SEQUENCE, expected_repeats=3)


def test_probability_loader_rejects_repeat_expanded_s_even_if_first_is_correct(
    tmp_path: Path,
) -> None:
    path = tmp_path / "probabilities.npz"
    repeated = np.repeat(_indices()[None, :], 3, axis=0)
    repeated[2, -1] = 0
    _write_probability_npz(path, S=repeated)
    with pytest.raises(ValueError, match="S shape"):
        load_probability_npz(path, expected_sequence=SEQUENCE, expected_repeats=3)
