"""Independent pair comparability facts."""

from dataclasses import dataclass

import pandas as pd

from dual_uq.construction.mapping import COMPARABILITY_MAPPING_COLUMNS


@dataclass(frozen=True, slots=True)
class ComparabilityFacts:
    sequence_comparable: bool
    mapping_comparable: bool
    coordinate_comparable: bool
    construct_comparable: bool
    assembly_comparable: bool
    sequence_identity: float | None = None
    mapping_fraction: float | None = None
    coordinate_fraction: float | None = None
    construct_overlap_fraction: float | None = None
    common_mapped_count: int = 0
    common_coordinate_visible_count: int = 0
    coordinate_bearing_mismatch_count: int = 0


def compare_pair(
    *,
    sequence_identity: float | None,
    mapping_fraction: float | None,
    coordinate_fraction: float | None,
    construct_overlap: bool,
    assembly_comparable: bool,
    min_sequence_identity: float = 0.90,
    min_mapping_fraction: float = 0.90,
    min_coordinate_fraction: float = 0.80,
    construct_overlap_fraction: float | None = None,
    common_mapped_count: int = 0,
    common_coordinate_visible_count: int = 0,
    coordinate_bearing_mismatch_count: int = 0,
) -> ComparabilityFacts:
    return ComparabilityFacts(
        sequence_comparable=sequence_identity is not None and sequence_identity >= min_sequence_identity,
        mapping_comparable=mapping_fraction is not None and mapping_fraction >= min_mapping_fraction,
        coordinate_comparable=coordinate_fraction is not None and coordinate_fraction >= min_coordinate_fraction,
        construct_comparable=bool(construct_overlap),
        assembly_comparable=bool(assembly_comparable),
        sequence_identity=sequence_identity,
        mapping_fraction=mapping_fraction,
        coordinate_fraction=coordinate_fraction,
        construct_overlap_fraction=construct_overlap_fraction,
        common_mapped_count=int(common_mapped_count),
        common_coordinate_visible_count=int(common_coordinate_visible_count),
        coordinate_bearing_mismatch_count=int(coordinate_bearing_mismatch_count),
    )


def compare_mapped_pair(
    *,
    mapping: pd.DataFrame,
    canonical_length: int,
    sequence_identity: float | None,
    construct_overlap_fraction: float | None,
    assembly_comparable: bool,
    min_sequence_identity: float = 0.90,
    min_mapping_fraction: float = 0.90,
    min_coordinate_fraction: float = 0.80,
    min_construct_overlap_fraction: float = 0.90,
) -> ComparabilityFacts:
    """Derive generic pair facts from the canonical condition-pair mapping."""

    missing = sorted(COMPARABILITY_MAPPING_COLUMNS - set(mapping.columns))
    if missing:
        raise ValueError(f"canonical pair mapping lacks comparability fields: {missing}")
    denominator = int(canonical_length)
    if denominator < 0:
        raise ValueError("canonical length must be non-negative")
    common_mapped_count = int(mapping["common_mapped"].fillna(False).astype(bool).sum())
    common_visible_count = int(
        mapping["common_coordinate_visible"].fillna(False).astype(bool).sum()
    )
    mapping_fraction = common_mapped_count / denominator if denominator else None
    coordinate_fraction = common_visible_count / denominator if denominator else None
    mismatch_count = 0
    canonical_aa = mapping["canonical_aa"]
    for prefix in ("condition_1", "condition_2"):
        visible = mapping[f"{prefix}_coordinate_visible"].fillna(False).astype(bool)
        aa = mapping[f"{prefix}_aa"]
        mismatch_count += int((visible & aa.notna() & canonical_aa.notna() & aa.ne(canonical_aa)).sum())
    construct_comparable = (
        construct_overlap_fraction is not None
        and construct_overlap_fraction >= min_construct_overlap_fraction
    )
    return compare_pair(
        sequence_identity=sequence_identity,
        mapping_fraction=mapping_fraction,
        coordinate_fraction=coordinate_fraction,
        construct_overlap=construct_comparable,
        assembly_comparable=assembly_comparable,
        min_sequence_identity=min_sequence_identity,
        min_mapping_fraction=min_mapping_fraction,
        min_coordinate_fraction=min_coordinate_fraction,
        construct_overlap_fraction=construct_overlap_fraction,
        common_mapped_count=common_mapped_count,
        common_coordinate_visible_count=common_visible_count,
        coordinate_bearing_mismatch_count=mismatch_count,
    )
