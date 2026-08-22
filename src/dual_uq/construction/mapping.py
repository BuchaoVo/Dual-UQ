"""Generic condition-aware residue mapping boundary.

Source adapters may use different residue-table column names, but once a side
has been normalized to the small representation accepted here, this module is
the single mapping operation used by new construction code.  In particular,
absence of a row, an explicitly unmapped row, and an unobserved coordinate are
kept as different states.
"""

import pandas as pd

CANONICAL_PAIR_MAPPING_COLUMNS: tuple[str, ...] = (
    "pair_id",
    "protein_id",
    "canonical_position",
    "canonical_aa",
    "condition_1_residue_id",
    "condition_2_residue_id",
    "condition_1_aa",
    "condition_2_aa",
    "condition_1_mapped",
    "condition_2_mapped",
    "condition_1_coordinate_visible",
    "condition_2_coordinate_visible",
    "common_mapped",
    "common_coordinate_visible",
    "condition_1_missing_reason",
    "condition_2_missing_reason",
    "mapping_status",
)

COMPARABILITY_MAPPING_COLUMNS = frozenset(
    {
        "common_mapped",
        "common_coordinate_visible",
        "canonical_aa",
        "condition_1_aa",
        "condition_2_aa",
        "condition_1_coordinate_visible",
        "condition_2_coordinate_visible",
    }
)


def _validate_canonical(canonical: pd.DataFrame) -> None:
    required = {"canonical_position", "canonical_aa"}
    if not required <= set(canonical.columns):
        raise ValueError("canonical mapping requires canonical_position and canonical_aa")
    positions = pd.to_numeric(canonical["canonical_position"], errors="coerce")
    if positions.isna().any() or positions.mod(1).ne(0).any() or positions.le(0).any():
        raise ValueError("canonical positions must be positive integers")
    if positions.duplicated().any():
        raise ValueError("canonical mapping contains duplicate positions")


def _keyed(frame: pd.DataFrame) -> dict[int, dict[str, object]]:
    required = {"canonical_position", "residue_id", "aa", "coordinate_visible"}
    if not required <= set(frame.columns):
        raise ValueError("condition mapping lacks required fields")
    positions = pd.to_numeric(frame["canonical_position"], errors="coerce")
    if positions.isna().any() or positions.mod(1).ne(0).any() or positions.le(0).any():
        raise ValueError("condition positions must be positive integers")
    if positions.duplicated().any():
        raise ValueError("condition mapping contains duplicate positions")
    normalized = frame.copy()
    normalized["canonical_position"] = positions.astype(int)
    return normalized.set_index("canonical_position").to_dict(orient="index")


def _mapped(row: dict[str, object] | None) -> bool:
    return row is not None and bool(row.get("mapped", True))


def _missing_reason(row: dict[str, object] | None, mapped: bool, visible: bool) -> str | None:
    if visible:
        return None
    explicit = None if row is None else row.get("missing_reason")
    if explicit is not None and not pd.isna(explicit) and str(explicit).strip():
        return str(explicit)
    return "COORDINATES_UNAVAILABLE" if mapped else "UNMAPPED"


def map_condition_pair(
    *,
    protein_id: str,
    pair_id: str,
    condition_1_label: str,
    condition_2_label: str,
    canonical: pd.DataFrame,
    condition_1: pd.DataFrame,
    condition_2: pd.DataFrame,
) -> pd.DataFrame:
    _validate_canonical(canonical)
    left, right = _keyed(condition_1), _keyed(condition_2)
    rows = []
    for item in canonical.to_dict(orient="records"):
        position = int(item["canonical_position"])
        left_row, right_row = left.get(position), right.get(position)
        l_mapped, r_mapped = _mapped(left_row), _mapped(right_row)
        l_visible = bool(l_mapped and left_row and left_row["coordinate_visible"])
        r_visible = bool(r_mapped and right_row and right_row["coordinate_visible"])
        rows.append(
            {
                "pair_id": pair_id,
                "protein_id": protein_id,
                "canonical_position": position,
                "canonical_aa": item["canonical_aa"],
                "condition_1_residue_id": None if not l_mapped else left_row["residue_id"],
                "condition_2_residue_id": None if not r_mapped else right_row["residue_id"],
                "condition_1_aa": None if not l_mapped else left_row["aa"],
                "condition_2_aa": None if not r_mapped else right_row["aa"],
                "condition_1_mapped": l_mapped,
                "condition_2_mapped": r_mapped,
                "condition_1_coordinate_visible": l_visible,
                "condition_2_coordinate_visible": r_visible,
                "common_mapped": l_mapped and r_mapped,
                "common_coordinate_visible": l_visible and r_visible,
                "condition_1_missing_reason": _missing_reason(left_row, l_mapped, l_visible),
                "condition_2_missing_reason": _missing_reason(right_row, r_mapped, r_visible),
                "mapping_status": "COMMON_VISIBLE"
                if l_visible and r_visible
                else "MAPPED_NOT_VISIBLE"
                if l_mapped and r_mapped
                else "PARTIAL_OR_UNMAPPED",
            }
        )
    return pd.DataFrame(rows, columns=CANONICAL_PAIR_MAPPING_COLUMNS)
