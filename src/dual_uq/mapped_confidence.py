from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .structure_io import join_residue_mapping_to_ca

REQUIRED_COLUMNS = {
    "uniprot_residue_number",
    "mapped",
    "observed_ca",
    "plddt",
    "auth_asym_id",
    "auth_seq_id",
    "insertion_code",
}


@dataclass(frozen=True)
class MappedConfidenceSegment:
    threshold: float
    start_uniprot: int
    end_uniprot: int
    residue_count: int
    positions: tuple[int, ...]
    min_plddt: float
    median_plddt: float
    observed_ca_fraction: float
    distance_to_mapped_n_terminus: int
    distance_to_mapped_c_terminus: int
    is_n_terminal: bool
    is_c_terminal: bool
    is_internal: bool

    @property
    def is_terminal(self) -> bool:
        return self.is_n_terminal or self.is_c_terminal


@dataclass(frozen=True)
class InputValidation:
    valid: bool
    reason: str
    residue_table: pd.DataFrame | None = None


@dataclass(frozen=True)
class MappedConfidenceSummary:
    mapped_start: int
    mapped_end: int
    mapped_position_count: int
    scorable_position_count: int
    scorable_fraction: float
    mapped_plddt_min: float
    mapped_plddt_q10: float
    mapped_plddt_median: float
    below_70_position_count: int
    below_80_position_count: int
    below_70_segment_count: int
    below_80_segment_count: int
    longest_below_70_segment: MappedConfidenceSegment | None
    longest_below_80_segment: MappedConfidenceSegment | None
    longest_internal_below_70_segment: MappedConfidenceSegment | None
    longest_internal_below_80_segment: MappedConfidenceSegment | None
    missing_ca_interruption_count: int
    unmapped_position_interruption_count: int
    is_low_conf_local: bool
    scoring_status: str
    scoring_reason: str

    @staticmethod
    def _length(segment: MappedConfidenceSegment | None) -> int:
        return 0 if segment is None else segment.residue_count

    @property
    def longest_below_70_length(self) -> int:
        return self._length(self.longest_below_70_segment)

    @property
    def longest_below_80_length(self) -> int:
        return self._length(self.longest_below_80_segment)

    @property
    def longest_internal_below_70_length(self) -> int:
        return self._length(self.longest_internal_below_70_segment)

    @property
    def longest_internal_below_80_length(self) -> int:
        return self._length(self.longest_internal_below_80_segment)


def _invalid(reason: str) -> InputValidation:
    return InputValidation(valid=False, reason=reason)


def _normalise_table(residue_table: pd.DataFrame) -> InputValidation:
    missing = sorted(REQUIRED_COLUMNS - set(residue_table.columns))
    if missing:
        return _invalid("missing_required_columns:" + ",".join(missing))

    table = residue_table.copy()
    if "fragment_covered" not in table:
        table["fragment_covered"] = True
    if table.empty:
        return _invalid("no_mapped_residues")

    positions = pd.to_numeric(
        table["uniprot_residue_number"],
        errors="coerce",
    )
    invalid_positions = (
        positions.isna()
        | positions.mod(1).ne(0)
        | positions.le(0)
    )
    if invalid_positions.any():
        return _invalid("invalid_uniprot_residue_number")
    table["uniprot_residue_number"] = positions.astype(int)

    for column in ("mapped", "observed_ca", "fragment_covered"):
        valid_boolean = table[column].map(
            lambda value: isinstance(value, (bool, np.bool_))
        )
        if not valid_boolean.all():
            return _invalid(f"invalid_boolean_field:{column}")
        table[column] = table[column].astype(bool)

    relevant_columns = list(table.columns)
    for _, duplicates in table.groupby(
        "uniprot_residue_number",
        sort=False,
        dropna=False,
    ):
        if len(duplicates.drop_duplicates(relevant_columns)) > 1:
            return _invalid("conflicting_duplicate_uniprot_position")
    table = table.drop_duplicates(
        "uniprot_residue_number",
        keep="first",
    ).copy()

    mapped = table["mapped"]
    if not mapped.any():
        return _invalid("no_mapped_residues")

    plddt = pd.to_numeric(table["plddt"], errors="coerce")
    invalid_plddt = (
        mapped
        & (
            plddt.isna()
            | ~np.isfinite(plddt)
            | plddt.lt(0.0)
            | plddt.gt(100.0)
        )
    )
    if invalid_plddt.any():
        return _invalid("invalid_mapped_plddt")
    table["plddt"] = plddt.astype(float)

    auth_chain = table["auth_asym_id"].map(
        lambda value: "" if pd.isna(value) else str(value).strip()
    )
    auth_sequence = pd.to_numeric(table["auth_seq_id"], errors="coerce")
    explicit_keys = (
        auth_chain.ne("")
        & ~auth_chain.isin({".", "?"})
        & auth_sequence.notna()
        & np.isfinite(auth_sequence)
        & auth_sequence.mod(1).eq(0)
        & table["insertion_code"].notna()
    )
    if not explicit_keys.loc[mapped].all():
        return _invalid("missing_explicit_auth_residue_key")

    table = table.sort_values(
        "uniprot_residue_number",
        kind="mergesort",
    ).reset_index(drop=True)
    return InputValidation(valid=True, reason="valid", residue_table=table)


def validate_mapped_confidence_input(
    residue_table: pd.DataFrame,
    *,
    selected_model_entity_id: Any,
    plddt_model_entity_id: Any,
    fragment_start: int,
    fragment_end: int,
) -> InputValidation:
    def clean_model_id(value: Any) -> str:
        if value is None or pd.isna(value):
            return ""
        cleaned = str(value).strip()
        return "" if cleaned.lower() in {"none", "nan", ".", "?"} else cleaned

    selected_model = clean_model_id(selected_model_entity_id)
    plddt_model = clean_model_id(plddt_model_entity_id)
    if not selected_model or not plddt_model or selected_model != plddt_model:
        return _invalid("afdb_model_entity_id_mismatch")

    try:
        start = int(fragment_start)
        end = int(fragment_end)
    except (TypeError, ValueError):
        return _invalid("invalid_afdb_fragment_interval")
    if start < 1 or end < start:
        return _invalid("invalid_afdb_fragment_interval")

    validation = _normalise_table(residue_table)
    if not validation.valid or validation.residue_table is None:
        return validation
    table = validation.residue_table
    mapped = table["mapped"]
    mapped_positions = table.loc[mapped, "uniprot_residue_number"]
    coverage_mismatch = (
        ~table.loc[mapped, "fragment_covered"]
        | ~mapped_positions.between(start, end, inclusive="both")
    )
    if coverage_mismatch.any():
        return _invalid("afdb_fragment_coverage_mismatch")

    scorable = mapped & table["observed_ca"] & table["fragment_covered"]
    if not scorable.any():
        return _invalid("no_scorable_residues")
    return InputValidation(valid=True, reason="valid", residue_table=table)


def build_mapped_confidence_residue_table(
    mapping: pd.DataFrame,
    pdb_ca_table: pd.DataFrame,
    plddt: np.ndarray,
    *,
    fragment_start: int,
    fragment_end: int,
) -> pd.DataFrame:
    observed, _ = join_residue_mapping_to_ca(mapping, pdb_ca_table)
    observed_positions = set(
        pd.to_numeric(
            observed["uniprot_residue_number"], errors="coerce"
        ).dropna().astype(int)
    )
    table = mapping.copy()
    table["observed_ca"] = (
        pd.to_numeric(table["uniprot_residue_number"], errors="coerce")
        .isin(observed_positions)
        .astype(bool)
    )
    table["mapped"] = True
    positions = pd.to_numeric(
        table["uniprot_residue_number"],
        errors="coerce",
    )
    table["fragment_covered"] = positions.between(
        fragment_start,
        fragment_end,
        inclusive="both",
    )
    table["plddt"] = np.nan
    covered = table["fragment_covered"] & positions.notna()
    indices = positions.loc[covered].astype(int).to_numpy() - fragment_start
    if len(indices) and (indices.min() < 0 or indices.max() >= len(plddt)):
        raise IndexError("mapped_position_outside_plddt_array")
    table.loc[covered, "plddt"] = plddt[indices]
    return table


def _best_segment(
    segments: list[MappedConfidenceSegment],
) -> MappedConfidenceSegment | None:
    if not segments:
        return None
    return min(
        segments,
        key=lambda segment: (-segment.residue_count, segment.start_uniprot),
    )


def find_low_confidence_segments(
    residue_table: pd.DataFrame,
    *,
    threshold: float,
    terminal_buffer: int = 5,
) -> list[MappedConfidenceSegment]:
    if terminal_buffer < 0:
        raise ValueError("terminal_buffer must be non-negative")
    validation = _normalise_table(residue_table)
    if not validation.valid or validation.residue_table is None:
        raise ValueError(validation.reason)
    table = validation.residue_table
    mapped = table.loc[table["mapped"]]
    mapped_start = int(mapped["uniprot_residue_number"].min())
    mapped_end = int(mapped["uniprot_residue_number"].max())
    eligible = table.loc[
        table["mapped"]
        & table["observed_ca"]
        & table["fragment_covered"]
        & table["plddt"].lt(float(threshold))
    ].copy()
    if eligible.empty:
        return []

    positions = eligible["uniprot_residue_number"].astype(int).tolist()
    runs: list[list[int]] = [[positions[0]]]
    for position in positions[1:]:
        if position == runs[-1][-1] + 1:
            runs[-1].append(position)
        else:
            runs.append([position])

    segments: list[MappedConfidenceSegment] = []
    indexed = eligible.set_index("uniprot_residue_number")
    for run in runs:
        scores = indexed.loc[run, "plddt"].to_numpy(dtype=float)
        start = run[0]
        end = run[-1]
        distance_n = start - mapped_start
        distance_c = mapped_end - end
        is_n_terminal = distance_n < terminal_buffer
        is_c_terminal = distance_c < terminal_buffer
        segments.append(
            MappedConfidenceSegment(
                threshold=float(threshold),
                start_uniprot=start,
                end_uniprot=end,
                residue_count=len(run),
                positions=tuple(run),
                min_plddt=float(np.min(scores)),
                median_plddt=float(np.median(scores)),
                observed_ca_fraction=1.0,
                distance_to_mapped_n_terminus=distance_n,
                distance_to_mapped_c_terminus=distance_c,
                is_n_terminal=is_n_terminal,
                is_c_terminal=is_c_terminal,
                is_internal=not is_n_terminal and not is_c_terminal,
            )
        )
    return segments


def summarize_mapped_confidence(
    residue_table: pd.DataFrame,
    *,
    terminal_buffer: int = 5,
    minimum_internal_run: int = 5,
) -> MappedConfidenceSummary:
    if minimum_internal_run < 1:
        raise ValueError("minimum_internal_run must be positive")
    validation = _normalise_table(residue_table)
    if not validation.valid or validation.residue_table is None:
        raise ValueError(validation.reason)
    table = validation.residue_table
    mapped = table["mapped"]
    scorable = mapped & table["observed_ca"] & table["fragment_covered"]
    if not scorable.any():
        raise ValueError("no_scorable_residues")

    mapped_table = table.loc[mapped]
    scorable_table = table.loc[scorable]
    mapped_positions = mapped_table["uniprot_residue_number"].astype(int)
    scores = scorable_table["plddt"].to_numpy(dtype=float)
    below_70 = find_low_confidence_segments(
        table,
        threshold=70.0,
        terminal_buffer=terminal_buffer,
    )
    below_80 = find_low_confidence_segments(
        table,
        threshold=80.0,
        terminal_buffer=terminal_buffer,
    )
    internal_70 = [segment for segment in below_70 if segment.is_internal]
    internal_80 = [segment for segment in below_80 if segment.is_internal]
    longest_internal_80 = _best_segment(internal_80)
    sorted_positions = np.sort(mapped_positions.unique())
    gap_count = int(np.maximum(np.diff(sorted_positions) - 1, 0).sum())

    return MappedConfidenceSummary(
        mapped_start=int(mapped_positions.min()),
        mapped_end=int(mapped_positions.max()),
        mapped_position_count=int(mapped_positions.nunique()),
        scorable_position_count=len(scorable_table),
        scorable_fraction=float(len(scorable_table) / mapped_positions.nunique()),
        mapped_plddt_min=float(np.min(scores)),
        mapped_plddt_q10=float(np.quantile(scores, 0.10)),
        mapped_plddt_median=float(np.median(scores)),
        below_70_position_count=int((scores < 70.0).sum()),
        below_80_position_count=int((scores < 80.0).sum()),
        below_70_segment_count=len(below_70),
        below_80_segment_count=len(below_80),
        longest_below_70_segment=_best_segment(below_70),
        longest_below_80_segment=_best_segment(below_80),
        longest_internal_below_70_segment=_best_segment(internal_70),
        longest_internal_below_80_segment=longest_internal_80,
        missing_ca_interruption_count=int(
            (mapped & ~table["observed_ca"]).sum()
        ),
        unmapped_position_interruption_count=gap_count,
        is_low_conf_local=(
            longest_internal_80 is not None
            and longest_internal_80.residue_count >= minimum_internal_run
        ),
        scoring_status="success",
        scoring_reason="scored",
    )
