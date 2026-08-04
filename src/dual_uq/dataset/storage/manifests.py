from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import pandas as pd

from dual_uq.core.atomic_io import atomic_write_text

PROTEIN_MANIFEST_REQUIRED_COLUMNS = (
    "protein_id",
    "screening_index",
    "pair_id",
    "mechanism_label",
    "tier",
)


@dataclass(frozen=True)
class ManifestValidationIssue:
    code: str
    message: str
    column: str | None = None
    row: int | None = None


@dataclass(frozen=True)
class ManifestValidationResult:
    issues: list[ManifestValidationIssue]

    @property
    def validation_pass(self) -> bool:
        return not self.issues

    @property
    def error_codes(self) -> list[str]:
        return list(dict.fromkeys(issue.code for issue in self.issues))


def load_protein_manifest(path: Path) -> pd.DataFrame:
    """Load a protein manifest without collapsing empty TSV fields."""
    return pd.read_csv(path, sep="\t", keep_default_na=False)


def write_protein_manifest(frame: pd.DataFrame, path: Path) -> None:
    """Write a protein manifest with stable LF line endings and empty fields."""
    stream = StringIO(newline="")
    frame.to_csv(stream, sep="\t", index=False, na_rep="", lineterminator="\n")
    atomic_write_text(path, stream.getvalue())


def _blank_mask(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype(str).str.strip().eq("")


def _append_row_issues(
    issues: list[ManifestValidationIssue],
    mask: pd.Series,
    *,
    code: str,
    column: str,
    message: str,
) -> None:
    for index in mask[mask].index:
        issues.append(
            ManifestValidationIssue(
                code=code,
                message=message,
                column=column,
                row=int(index) + 2,
            )
        )


def validate_protein_manifest(frame: pd.DataFrame) -> ManifestValidationResult:
    """Return explicit manifest validation errors without applying fallbacks."""
    issues: list[ManifestValidationIssue] = []
    for column in PROTEIN_MANIFEST_REQUIRED_COLUMNS:
        if column not in frame:
            issues.append(
                ManifestValidationIssue(
                    code="missing_required_column",
                    message=f"Required manifest column {column!r} is missing.",
                    column=column,
                )
            )

    if any(column not in frame for column in PROTEIN_MANIFEST_REQUIRED_COLUMNS):
        return ManifestValidationResult(issues=issues)

    for column in ("protein_id", "pair_id", "mechanism_label"):
        _append_row_issues(
            issues,
            _blank_mask(frame[column]),
            code=f"missing_{column}",
            column=column,
            message=f"{column} must be explicit; no fallback is applied.",
        )

    screening_index = pd.to_numeric(frame["screening_index"], errors="coerce")
    invalid_screening_index = (
        screening_index.isna()
        | screening_index.mod(1).ne(0)
        | screening_index.le(0)
    )
    _append_row_issues(
        issues,
        invalid_screening_index,
        code="invalid_screening_index",
        column="screening_index",
        message="screening_index must be a positive integer.",
    )

    tier = pd.to_numeric(frame["tier"], errors="coerce")
    invalid_tier = tier.isna() | tier.mod(1).ne(0) | ~tier.isin((1, 2))
    _append_row_issues(
        issues,
        invalid_tier,
        code="invalid_tier",
        column="tier",
        message="tier must be integer 1 or 2.",
    )

    for column in ("protein_id", "screening_index", "pair_id"):
        values = frame[column].astype(str).str.strip()
        duplicated = values.ne("") & values.duplicated(keep=False)
        _append_row_issues(
            issues,
            duplicated,
            code=f"duplicate_{column}",
            column=column,
            message=f"{column} must be unique within a manifest.",
        )

    return ManifestValidationResult(issues=issues)
