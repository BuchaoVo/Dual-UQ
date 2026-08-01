from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dual_uq.dataset_a_scale.hashing import sha256_bytes

STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
PROTEINMPNN_ALPHABET = f"{STANDARD_AMINO_ACIDS}X"

_SCORE_KEYS = frozenset({"score", "global_score", "S", "seq_str"})
_PROBABILITY_KEYS = frozenset({"log_p", "S", "mask", "design_mask"})


def validate_protein_sequence(sequence: object) -> str:
    """Validate a non-empty uppercase sequence from the standard 20 amino acids."""
    if type(sequence) is not str:
        raise TypeError("sequence must be a string.")
    if not sequence:
        raise ValueError("sequence must not be empty.")
    invalid = sorted(set(sequence).difference(STANDARD_AMINO_ACIDS))
    if invalid:
        raise ValueError(
            "sequence contains characters outside the standard uppercase 20-AA "
            f"alphabet: {invalid}"
        )
    return sequence


def _validate_candidate_id(candidate_id: object) -> str:
    if type(candidate_id) is not str:
        raise TypeError("candidate_id must be a string.")
    if (
        not candidate_id
        or candidate_id != candidate_id.strip()
        or ">" in candidate_id
        or any(ord(character) < 32 or ord(character) == 127 for character in candidate_id)
    ):
        raise ValueError(
            "candidate_id must be non-empty, canonical, and contain no FASTA "
            "control characters."
        )
    return candidate_id


@dataclass(frozen=True)
class FastaRecord:
    candidate_id: str
    sequence: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "candidate_id", _validate_candidate_id(self.candidate_id)
        )
        object.__setattr__(
            self, "sequence", validate_protein_sequence(self.sequence)
        )


@dataclass(frozen=True)
class ScoreOnlyResult:
    path: Path
    sequence: str
    sequence_indices: np.ndarray
    scores: np.ndarray
    global_scores: np.ndarray

    @property
    def repeat_count(self) -> int:
        return int(self.scores.shape[0])


@dataclass(frozen=True)
class ProbabilityResult:
    path: Path
    sequence: str
    sequence_indices: np.ndarray
    log_probabilities: np.ndarray
    mask: np.ndarray
    design_mask: np.ndarray

    @property
    def repeat_count(self) -> int:
        return int(self.log_probabilities.shape[0])


def sequence_sha256(sequence: str) -> str:
    """Hash only the validated ASCII sequence, independent of FASTA metadata."""
    checked = validate_protein_sequence(sequence)
    return sha256_bytes(checked.encode("ascii"))


def _fasta_bytes(record: FastaRecord) -> bytes:
    return f">{record.candidate_id}\n{record.sequence}\n".encode("ascii")


def write_single_candidate_fasta(path: str | Path, record: FastaRecord) -> None:
    """Write one LF FASTA record without overwriting different history."""
    if not isinstance(record, FastaRecord):
        raise TypeError("record must be a FastaRecord.")
    destination = Path(path)
    payload = _fasta_bytes(record)
    if destination.exists():
        if not destination.is_file() or destination.read_bytes() != payload:
            raise FileExistsError(
                f"Refusing to overwrite existing FASTA with different content: "
                f"{destination}"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(payload)


def read_single_candidate_fasta(path: str | Path) -> FastaRecord:
    """Read exactly one FASTA record, accepting LF or CRLF line endings."""
    source = Path(path)
    payload = source.read_bytes()
    has_crlf = b"\r\n" in payload
    without_crlf = payload.replace(b"\r\n", b"")
    has_bare_cr = b"\r" in without_crlf
    has_mixed_lf = has_crlf and b"\n" in without_crlf
    has_other_separator = any(
        separator in payload
        for separator in (b"\x0b", b"\x0c", b"\x1c", b"\x1d", b"\x1e", b"\x85")
    )
    if has_bare_cr or has_mixed_lf or has_other_separator:
        raise ValueError(f"FASTA line endings must be consistently LF or CRLF: {source}")
    text = payload.decode("ascii")
    lines = text.splitlines()
    header_indices = [index for index, line in enumerate(lines) if line.startswith(">")]
    if header_indices != [0]:
        raise ValueError(
            f"FASTA must contain exactly one record with its header first: {source}"
        )
    sequence_lines = lines[1:]
    if not sequence_lines or any(not line for line in sequence_lines):
        raise ValueError(f"FASTA sequence must contain no empty lines: {source}")
    return FastaRecord(lines[0][1:], "".join(sequence_lines))


def find_unique_score_npz(output_dir: str | Path) -> Path:
    """Find the sole FASTA-derived score output without fallback or guessing."""
    root = Path(output_dir)
    matches = sorted(path for path in root.rglob("*_fasta_1.npz") if path.is_file())
    if not matches:
        raise FileNotFoundError(f"No *_fasta_1.npz score output found under {root}.")
    if len(matches) > 1:
        raise ValueError(f"Found multiple *_fasta_1.npz score outputs: {matches}")
    return matches[0]


def _expected_repeats(value: object) -> int:
    if type(value) is not int:
        raise TypeError("expected_repeats must be an integer.")
    if value <= 0:
        raise ValueError("expected_repeats must be positive.")
    return value


def _require_keys(path: Path, actual: set[str], required: frozenset[str]) -> None:
    missing = sorted(required.difference(actual))
    if missing:
        raise ValueError(f"{path}: missing required NPZ keys {missing}.")


def _numeric_array(array: np.ndarray, field_name: str) -> np.ndarray:
    if array.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(f"{field_name} must have a real numeric dtype, got {array.dtype}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{field_name} must contain only finite values.")
    return np.array(array, copy=True)


def _sequence_indices(array: np.ndarray, expected_sequence: str) -> np.ndarray:
    expected = np.asarray(
        [PROTEINMPNN_ALPHABET.index(residue) for residue in expected_sequence],
        dtype=np.int64,
    )
    if array.shape != expected.shape:
        raise ValueError(f"S shape {array.shape}; expected {expected.shape}.")
    if array.dtype.kind not in {"i", "u"}:
        raise ValueError(f"S must have an integer dtype, got {array.dtype}.")
    observed = np.array(array, copy=True)
    if ((observed < 0) | (observed >= len(PROTEINMPNN_ALPHABET))).any():
        raise ValueError("S index is outside the ProteinMPNN alphabet.")
    if not np.array_equal(observed, expected):
        raise ValueError("NPZ sequence indices do not match the expected sequence.")
    return observed


def _scalar_sequence(array: np.ndarray, expected_sequence: str) -> None:
    if array.shape != ():
        raise ValueError(f"seq_str shape {array.shape}; expected scalar shape ().")
    if array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"seq_str must have a string dtype, got {array.dtype}.")
    value = array.item()
    if isinstance(value, bytes):
        value = value.decode("ascii")
    if value != expected_sequence:
        raise ValueError("NPZ seq_str does not match the expected sequence.")


def load_score_only_npz(
    path: str | Path,
    *,
    expected_sequence: str,
    expected_repeats: int,
) -> ScoreOnlyResult:
    """Load the exact official score-only schema with no reshape or coercion."""
    source = Path(path)
    sequence = validate_protein_sequence(expected_sequence)
    repeats = _expected_repeats(expected_repeats)
    with np.load(source, allow_pickle=False) as data:
        _require_keys(source, set(data.files), _SCORE_KEYS)
        score = _numeric_array(data["score"], "score")
        global_score = _numeric_array(data["global_score"], "global_score")
        sequence_indices = _sequence_indices(data["S"], sequence)
        _scalar_sequence(data["seq_str"], sequence)

    expected_score_shape = (repeats,)
    if score.shape != expected_score_shape:
        raise ValueError(
            f"score shape {score.shape}; expected {expected_score_shape} for the "
            "requested repeat count."
        )
    if global_score.shape != expected_score_shape:
        raise ValueError(
            f"global_score shape {global_score.shape}; expected "
            f"{expected_score_shape} for the requested repeat count."
        )
    return ScoreOnlyResult(
        source,
        sequence,
        sequence_indices,
        score,
        global_score,
    )


def load_probability_npz(
    path: str | Path,
    *,
    expected_sequence: str,
    expected_repeats: int,
) -> ProbabilityResult:
    """Load conditional or unconditional probability-only output strictly."""
    source = Path(path)
    sequence = validate_protein_sequence(expected_sequence)
    repeats = _expected_repeats(expected_repeats)
    length = len(sequence)
    with np.load(source, allow_pickle=False) as data:
        _require_keys(source, set(data.files), _PROBABILITY_KEYS)
        log_probabilities = _numeric_array(data["log_p"], "log_p")
        sequence_indices = _sequence_indices(data["S"], sequence)
        mask = _numeric_array(data["mask"], "mask")
        design_mask = _numeric_array(data["design_mask"], "design_mask")

    expected_probability_shape = (
        repeats,
        length,
        len(PROTEINMPNN_ALPHABET),
    )
    if log_probabilities.shape != expected_probability_shape:
        raise ValueError(
            f"log_p shape {log_probabilities.shape}; expected "
            f"{expected_probability_shape} for the requested repeat count."
        )
    expected_position_shape = (length,)
    if mask.shape != expected_position_shape:
        raise ValueError(f"mask shape {mask.shape}; expected {expected_position_shape}.")
    if design_mask.shape != expected_position_shape:
        raise ValueError(
            f"design_mask shape {design_mask.shape}; expected "
            f"{expected_position_shape}."
        )
    return ProbabilityResult(
        source,
        sequence,
        sequence_indices,
        log_probabilities,
        mask,
        design_mask,
    )
