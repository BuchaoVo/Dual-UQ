from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import yaml

from dual_uq.a0_classification import (
    LONG_RANGE_PAE_BINS,
    PAE_BINS,
    A0Classification,
    A0Evidence,
    DisagreementSegment,
    PaeStratumStats,
    classify_candidate,
    compute_pae_strata,
)
from dual_uq.confidence import load_plddt
from dual_uq.mapped_confidence import summarize_mapped_confidence

CANDIDATE_KEY = (
    "screening_index",
    "pdb_id",
    "chain_id",
    "uniprot_id",
)
LABEL_COLUMNS = (
    "is_easy_control",
    "is_low_conf_local",
    "is_high_pae_long_range",
    "is_high_conf_state_disagreement",
    "is_construct_difference",
    "is_missing_coordinate_stress",
)
LIFECYCLE_REQUIRED_COLUMNS = {
    *CANDIDATE_KEY,
    "pair_name",
    "pair_status",
    "geometry_status",
    "robust_status",
    "segment_context_status",
}
LIFECYCLE_STATUS_VALUES = {
    "pair_status": {
        "complete",
        "skipped_complete",
        "not_started",
        "in_progress",
        "running",
        "running_pair",
        "blocked_upstream",
        "skipped_preflight",
        "unsupported_afdb_fragment",
        "failed_pair",
    },
    "geometry_status": {
        "complete",
        "skipped_complete",
        "not_started",
        "in_progress",
        "running",
        "running_geometry",
        "blocked_upstream",
        "skipped_preflight",
        "unsupported_afdb_fragment",
        "failed_geometry",
    },
    "robust_status": {
        "complete",
        "skipped_complete",
        "not_started",
        "in_progress",
        "running",
        "running_robust",
        "blocked_upstream",
        "skipped_preflight",
        "unsupported_afdb_fragment",
        "failed_robust",
    },
    "segment_context_status": {
        "complete",
        "skipped_complete",
        "successful_no_segments",
        "not_started",
        "in_progress",
        "running",
        "running_segment_context",
        "blocked_upstream",
        "skipped_preflight",
        "unsupported_afdb_fragment",
        "failed_segment_context",
    },
}
GEOMETRY_STATISTIC_FIELDS = {
    "ca_disagreement_median": "median_aligned_ca_distance",
    "ca_disagreement_p90": "p90_aligned_ca_distance",
    "ca_disagreement_max": "max_aligned_ca_distance",
}
GEOMETRY_COMPARISON_ATOL = 1e-9
PILOT_GEOMETRY_COMPARISON_ATOL = 1e-6
GEOMETRY_COMPLETE_STATUSES = {"complete", "skipped_complete"}


class GeometryEvidence(NamedTuple):
    available: bool
    source: str | None
    ca_disagreement_median: float | None
    ca_disagreement_p90: float | None
    ca_disagreement_max: float | None
    mapped_ca_count: int | None
    validated_paths: tuple[str, ...]
    identity_match: bool
    error_reason: str | None
    evidence_mismatch: bool = False


class ConfidenceEvidence(NamedTuple):
    available: bool
    confidence_model_match: bool | None
    selected_model_entity_id: str | None
    selected_version: int | None
    fragment_start: int | None
    fragment_end: int | None
    plddt_model_entity_id: str | None
    plddt_model_version: int | None
    pae_model_entity_id: str | None
    pae_model_version: int | None
    source: str | None
    error_reason: str | None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build evidence-driven, multi-label A0 candidate classifications "
            "from existing local diagnostic artifacts."
        )
    )
    parser.add_argument("--project-root", required=True)
    parser.add_argument(
        "--config",
        default="configs/a0_selection.yaml",
    )
    parser.add_argument(
        "--output",
        default="reports/a0_candidate_summary.csv",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional screening indices or pair names to rebuild.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any candidate has a model/evidence mismatch.",
    )
    parser.add_argument(
        "--lifecycle",
        action="append",
        default=None,
        help=(
            "Normalized lifecycle CSV. Repeat for multiple cohorts. "
            "Defaults to reports/candidate_lifecycle.csv."
        ),
    )
    return parser.parse_args(argv)


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_config(root: Path, config_path: str) -> dict[str, Any]:
    path = resolve_path(root, config_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"classification config must be a mapping: {path}")
    return config


def _clean_text(value: Any, *, lower: bool = False) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", ".", "?"}:
        return None
    return text.lower() if lower else text


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _optional_int(value: Any) -> int | None:
    number = _optional_float(value)
    if number is None or not float(number).is_integer():
        return None
    return int(number)


def _optional_bool(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _candidate_key(row: Mapping[str, Any]) -> tuple[Any, str, str, str]:
    index = _optional_int(row.get("screening_index"))
    pdb_id = _clean_text(row.get("pdb_id"), lower=True)
    chain_id = _clean_text(row.get("chain_id"))
    uniprot_id = _clean_text(row.get("uniprot_id"))
    if not pdb_id or not chain_id or not uniprot_id:
        raise ValueError(
            "candidate key requires pdb_id, chain_id, and uniprot_id"
        )
    return index, pdb_id, chain_id, uniprot_id


def _pair_name(row: Mapping[str, Any]) -> str:
    _, pdb_id, chain_id, uniprot_id = _candidate_key(row)
    return f"{pdb_id}_{chain_id}__{uniprot_id}"


def _records_by_key(
    table: pd.DataFrame,
    source_name: str,
) -> dict[tuple[Any, str, str, str], dict[str, Any]]:
    records: dict[tuple[Any, str, str, str], dict[str, Any]] = {}
    for record in table.to_dict(orient="records"):
        key = _candidate_key(record)
        if key in records:
            raise ValueError(
                f"duplicate candidate key in {source_name}: {key}"
            )
        records[key] = record
    return records


def _empty_geometry_evidence(reason: str) -> GeometryEvidence:
    return GeometryEvidence(
        available=False,
        source=None,
        ca_disagreement_median=None,
        ca_disagreement_p90=None,
        ca_disagreement_max=None,
        mapped_ca_count=None,
        validated_paths=(),
        identity_match=False,
        error_reason=reason,
    )


def _geometry_failure(
    *,
    strict: bool,
    error: OSError | TypeError | ValueError,
) -> GeometryEvidence:
    if strict:
        raise error
    return _empty_geometry_evidence(f"{type(error).__name__}:{error}")


def _geometry_statistics(values: np.ndarray) -> dict[str, float]:
    return {
        "ca_disagreement_median": float(np.median(values)),
        "ca_disagreement_p90": float(
            np.quantile(values, 0.90, method="linear")
        ),
        "ca_disagreement_max": float(np.max(values)),
    }


def load_pair_geometry_evidence(
    *,
    project_root: Path,
    candidate_identity: Mapping[str, Any],
    geometry_status: str | None,
    pilot_mechanism: Mapping[str, Any],
    strict: bool,
) -> GeometryEvidence:
    if geometry_status not in GEOMETRY_COMPLETE_STATUSES:
        return _empty_geometry_evidence("geometry_not_complete")

    try:
        canonical_pair_name = _pair_name(candidate_identity)
        supplied_pair_name = _clean_text(candidate_identity.get("pair_name"))
        if supplied_pair_name != canonical_pair_name:
            raise ValueError(
                "geometry identity mismatch: "
                f"pair_name={supplied_pair_name!r}, expected={canonical_pair_name!r}"
            )
        pair_dir = (
            project_root / "data/processed/pairs" / canonical_pair_name
        )
        qc_path = pair_dir / "pair_geometry_qc.json"
        residue_path = pair_dir / "residue_geometry.parquet"
        missing_paths = [
            str(path)
            for path in (qc_path, residue_path)
            if not path.exists()
        ]
        if missing_paths:
            if not strict and len(missing_paths) == 2:
                pilot_values = {
                    field: _optional_float(pilot_mechanism.get(field))
                    for field in GEOMETRY_STATISTIC_FIELDS
                }
                finite_pilot = {
                    field: value
                    for field, value in pilot_values.items()
                    if value is not None and value >= 0.0
                }
                if finite_pilot:
                    return GeometryEvidence(
                        available=True,
                        source="legacy_pilot_only",
                        ca_disagreement_median=finite_pilot.get(
                            "ca_disagreement_median"
                        ),
                        ca_disagreement_p90=finite_pilot.get(
                            "ca_disagreement_p90"
                        ),
                        ca_disagreement_max=finite_pilot.get(
                            "ca_disagreement_max"
                        ),
                        mapped_ca_count=_optional_int(
                            pilot_mechanism.get("mapped_ca_count")
                        ),
                        validated_paths=(),
                        identity_match=False,
                        error_reason=(
                            "canonical_geometry_artifacts_absent:"
                            + ",".join(missing_paths)
                        ),
                    )
            raise FileNotFoundError(
                "missing canonical geometry artifact: "
                + ",".join(missing_paths)
            )

        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        if not isinstance(qc, dict):
            raise TypeError("geometry QC must be a JSON object")
        _, pdb_id, chain_id, uniprot_id = _candidate_key(
            candidate_identity
        )
        expected_identity = {
            "pdb_id": pdb_id,
            "chain_id": chain_id,
            "uniprot_id": uniprot_id,
        }
        observed_identity = {
            "pdb_id": _clean_text(qc.get("pdb_id"), lower=True),
            "chain_id": _clean_text(qc.get("chain_id")),
            "uniprot_id": _clean_text(qc.get("uniprot_id")),
        }
        conflicts = {
            field: (observed_identity[field], expected)
            for field, expected in expected_identity.items()
            if observed_identity[field] != expected
        }
        if conflicts:
            raise ValueError(
                "geometry identity mismatch: "
                + ",".join(
                    f"{field}={observed!r},expected={expected!r}"
                    for field, (observed, expected) in conflicts.items()
                )
            )

        residues = pd.read_parquet(residue_path)
        if "uniprot_residue_number" not in residues:
            raise ValueError(
                "residue geometry missing uniprot_residue_number"
            )
        positions = pd.to_numeric(
            residues["uniprot_residue_number"],
            errors="coerce",
        )
        invalid_positions = (
            positions.isna()
            | positions.mod(1).ne(0)
            | positions.le(0)
        )
        if invalid_positions.any():
            raise ValueError(
                "residue geometry contains invalid UniProt positions"
            )
        if positions.astype(int).duplicated().any():
            raise ValueError(
                "residue geometry contains duplicate UniProt positions"
            )

        distance_columns = [
            field
            for field in ("aligned_ca_distance", "ca_disagreement")
            if field in residues
        ]
        if not distance_columns:
            raise ValueError(
                "residue geometry missing CA disagreement field"
            )
        values = pd.to_numeric(
            residues[distance_columns[0]],
            errors="coerce",
        ).to_numpy(dtype=float)
        if (
            not np.isfinite(values).all()
            or (values < 0.0).any()
            or values.size == 0
        ):
            raise ValueError(
                "CA disagreement values must be finite, non-negative, "
                "and non-empty"
            )
        if len(distance_columns) == 2:
            alias_values = pd.to_numeric(
                residues[distance_columns[1]],
                errors="coerce",
            ).to_numpy(dtype=float)
            if (
                alias_values.shape != values.shape
                or not np.isfinite(alias_values).all()
                or not np.allclose(
                    values,
                    alias_values,
                    rtol=0.0,
                    atol=GEOMETRY_COMPARISON_ATOL,
                )
            ):
                raise ValueError(
                    "CA disagreement aliases conflict"
                )

        mapped_ca_count = _optional_int(qc.get("mapped_ca_count"))
        if mapped_ca_count != len(residues):
            raise ValueError(
                "geometry mapped_ca_count mismatch: "
                f"qc={mapped_ca_count}, residues={len(residues)}"
            )
        computed = _geometry_statistics(values)
        canonical: dict[str, float] = {}
        qc_fields_present = True
        for output_field, qc_field in GEOMETRY_STATISTIC_FIELDS.items():
            if qc_field not in qc:
                qc_fields_present = False
                canonical[output_field] = computed[output_field]
                continue
            qc_value = _optional_float(qc[qc_field])
            if qc_value is None:
                raise ValueError(
                    "invalid geometry QC statistic: "
                    f"{qc_field}={qc[qc_field]!r}"
                )
            if qc_value < 0.0 or not np.isclose(
                qc_value,
                computed[output_field],
                rtol=0.0,
                atol=GEOMETRY_COMPARISON_ATOL,
            ):
                raise ValueError(
                    "geometry QC/residue statistic conflict: "
                    f"{qc_field}={qc_value}, "
                    f"computed={computed[output_field]}"
                )
            canonical[output_field] = qc_value

        pilot_conflicts: list[str] = []
        for field, canonical_value in canonical.items():
            pilot_value = _optional_float(pilot_mechanism.get(field))
            if pilot_value is None:
                continue
            if not np.isclose(
                pilot_value,
                canonical_value,
                rtol=0.0,
                atol=PILOT_GEOMETRY_COMPARISON_ATOL,
            ):
                pilot_conflicts.append(
                    f"{field}={pilot_value},canonical={canonical_value}"
                )
        if pilot_conflicts and strict:
            raise ValueError(
                "pilot geometry conflict: " + ";".join(pilot_conflicts)
            )
        return GeometryEvidence(
            available=True,
            source=(
                "pair_geometry_qc+residue_geometry"
                if qc_fields_present
                else "residue_geometry"
            ),
            ca_disagreement_median=canonical[
                "ca_disagreement_median"
            ],
            ca_disagreement_p90=canonical["ca_disagreement_p90"],
            ca_disagreement_max=canonical["ca_disagreement_max"],
            mapped_ca_count=mapped_ca_count,
            validated_paths=(str(qc_path), str(residue_path)),
            identity_match=True,
            error_reason=(
                None
                if not pilot_conflicts
                else "pilot_geometry_conflict:" + ";".join(pilot_conflicts)
            ),
            evidence_mismatch=bool(pilot_conflicts),
        )
    except (OSError, TypeError, ValueError) as exc:
        return _geometry_failure(strict=strict, error=exc)


def enrich_pilot_mechanism_identity(
    mechanisms: pd.DataFrame,
    pilot_manifest: pd.DataFrame,
) -> pd.DataFrame:
    identity_columns = [
        "screening_index",
        "source",
        "pdb_id",
        "chain_id",
        "uniprot_id",
        "pair_name",
        "pilot_role",
    ]
    required_manifest = {"pilot_id", *identity_columns}
    missing_manifest = sorted(required_manifest - set(pilot_manifest.columns))
    if missing_manifest:
        raise ValueError(
            "geometry pilot manifest missing identity columns: "
            + ",".join(missing_manifest)
        )
    if "pilot_id" not in mechanisms:
        raise ValueError("geometry pilot mechanisms missing pilot_id")
    if mechanisms["pilot_id"].duplicated().any():
        raise ValueError("duplicate pilot_id in geometry pilot mechanisms")
    if pilot_manifest["pilot_id"].duplicated().any():
        raise ValueError("duplicate pilot_id in geometry pilot manifest")

    provided_identity = [
        column for column in identity_columns if column in mechanisms
    ]
    renamed = mechanisms.rename(
        columns={
            column: f"{column}_mechanism"
            for column in provided_identity
        }
    )
    identity = pilot_manifest[["pilot_id", *identity_columns]].copy()
    merged = renamed.merge(
        identity,
        on="pilot_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if len(merged) != len(mechanisms):
        raise ValueError("pilot identity enrichment changed row count")
    unmatched = merged["_merge"] != "both"
    if unmatched.any():
        missing = ",".join(
            merged.loc[unmatched, "pilot_id"].astype(str).tolist()
        )
        raise ValueError(f"mechanism rows missing pilot identity: {missing}")

    for column in provided_identity:
        supplied = merged[f"{column}_mechanism"]
        canonical = merged[column]
        if column == "screening_index":
            normalise = _optional_int
        else:
            normalise = (
                lambda value, identity_column=column: _clean_text(
                    value,
                    lower=identity_column == "pdb_id",
                )
            )
        conflict = pd.Series(
            [
                normalise(supplied_value) != normalise(canonical_value)
                for supplied_value, canonical_value in zip(
                    supplied,
                    canonical,
                    strict=True,
                )
            ],
            index=merged.index,
            dtype=bool,
        )
        if conflict.any():
            identifiers = ",".join(
                merged.loc[conflict, "pilot_id"].astype(str).tolist()
            )
            raise ValueError(
                f"mechanism {column} conflict: {identifiers}"
            )
    return merged.drop(
        columns=[
            *(f"{column}_mechanism" for column in provided_identity),
            "_merge",
        ]
    )


def _read_table(path: Path, *, sep: str = ",") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep=sep)


def _validate_lifecycle_table(
    table: pd.DataFrame,
    *,
    source_path: Path,
) -> pd.DataFrame:
    missing = sorted(LIFECYCLE_REQUIRED_COLUMNS - set(table.columns))
    if missing:
        raise ValueError(
            "missing_required_lifecycle_columns:"
            f"{source_path}:{','.join(missing)}"
        )

    validated = table.copy()
    indices = validated["screening_index"].map(_optional_int)
    if indices.isna().any():
        raise ValueError(f"invalid_screening_index:{source_path}")
    validated["screening_index"] = indices.astype(int)

    normalized_keys: list[tuple[int, str, str, str]] = []
    canonical_pairs: list[str] = []
    for record in validated.to_dict(orient="records"):
        key = _candidate_key(record)
        if key[0] is None:
            raise ValueError(f"invalid_screening_index:{source_path}")
        normalized_key = (int(key[0]), key[1], key[2], key[3])
        normalized_keys.append(normalized_key)
        canonical_pairs.append(
            f"{key[1]}_{key[2]}__{key[3]}"
        )
    supplied_pairs = validated["pair_name"].map(_clean_text)
    if supplied_pairs.isna().any():
        raise ValueError(f"invalid_pair_name:{source_path}")
    if supplied_pairs.tolist() != canonical_pairs:
        raise ValueError(f"pair_name_identity_conflict:{source_path}")

    for column, allowed in LIFECYCLE_STATUS_VALUES.items():
        normalized = validated[column].map(
            lambda value: _clean_text(value, lower=True)
        )
        invalid = normalized.isna() | ~normalized.isin(allowed)
        if invalid.any():
            values = sorted(
                {
                    str(value)
                    for value in validated.loc[invalid, column].tolist()
                }
            )
            raise ValueError(
                "invalid_lifecycle_status:"
                f"{source_path}:{column}:{','.join(values)}"
            )
        validated[column] = normalized

    validated["_lifecycle_key"] = normalized_keys
    _validate_lifecycle_identity(validated, source_name=str(source_path))
    cohort = (
        validated["source"].map(_clean_text)
        if "source" in validated
        else pd.Series([None] * len(validated), index=validated.index)
    )
    validated["lifecycle_cohort"] = cohort
    validated["lifecycle_source_path"] = str(source_path)
    return validated


def _validate_lifecycle_identity(
    table: pd.DataFrame,
    *,
    source_name: str,
) -> None:
    if table["_lifecycle_key"].duplicated(keep=False).any():
        raise ValueError(f"duplicate_candidate_key:{source_name}")
    if table["screening_index"].duplicated(keep=False).any():
        raise ValueError(
            f"screening_index_identity_conflict:{source_name}"
        )
    if table["pair_name"].duplicated(keep=False).any():
        raise ValueError(f"pair_name_identity_conflict:{source_name}")


def load_lifecycle_tables(
    root: Path,
    lifecycle_paths: Sequence[str] | None,
) -> pd.DataFrame:
    requested = list(lifecycle_paths or ["reports/candidate_lifecycle.csv"])
    if not requested:
        requested = ["reports/candidate_lifecycle.csv"]
    tables: list[pd.DataFrame] = []
    for value in requested:
        path = resolve_path(root, value).expanduser().resolve()
        table = _read_table(path)
        tables.append(
            _validate_lifecycle_table(table, source_path=path)
        )
    combined = pd.concat(tables, ignore_index=True, sort=False)
    _validate_lifecycle_identity(
        combined,
        source_name="combined_lifecycle_inputs",
    )
    combined = combined.sort_values(
        [*CANDIDATE_KEY, "pair_name"],
        kind="mergesort",
    ).reset_index(drop=True)
    return combined.drop(columns=["_lifecycle_key"])


def _build_candidate_universe(
    screening_pool: pd.DataFrame,
    replacement_pool: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in screening_pool.to_dict(orient="records"):
        record = dict(record)
        record["source"] = "screening_pool"
        record["pair_name"] = _pair_name(record)
        rows.append(record)
    for record in replacement_pool.to_dict(orient="records"):
        record = dict(record)
        record["source"] = "replacement_pool"
        record["pair_name"] = _pair_name(record)
        rows.append(record)
    rows.append(
        {
            "screening_index": pd.NA,
            "source": "reference_pair",
            "pdb_id": "1ake",
            "chain_id": "A",
            "uniprot_id": "P69441",
            "pair_name": "1ake_A__P69441",
            "provisional_stratum": "reference_pair",
        }
    )
    universe = pd.DataFrame(rows)
    _records_by_key(universe, "candidate universe")
    return universe


def _lookup(
    records: Mapping[tuple[Any, str, str, str], dict[str, Any]],
    key: tuple[Any, str, str, str],
) -> dict[str, Any]:
    return records.get(key, {})


def _first_value(
    records: Iterable[Mapping[str, Any]],
    field: str,
) -> Any:
    for record in records:
        if field not in record:
            continue
        value = record[field]
        if _clean_text(value) is not None or _optional_float(value) is not None:
            return value
    return None


def _empty_confidence_evidence(reason: str) -> ConfidenceEvidence:
    return ConfidenceEvidence(
        available=False,
        confidence_model_match=None,
        selected_model_entity_id=None,
        selected_version=None,
        fragment_start=None,
        fragment_end=None,
        plddt_model_entity_id=None,
        plddt_model_version=None,
        pae_model_entity_id=None,
        pae_model_version=None,
        source=None,
        error_reason=reason,
    )


def _load_json_mapping(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {description}: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"invalid {description}: {path}")
    return value


def _artifact_identity(
    url: Any,
    *,
    kind: str,
) -> tuple[str, int]:
    text = _clean_text(url)
    if text is None:
        raise ValueError(f"missing {kind} metadata URL")
    name = Path(urlparse(text).path).name
    suffix = {
        "model": "model",
        "plddt": "confidence",
        "pae": "predicted_aligned_error",
    }[kind]
    match = re.fullmatch(
        rf"(.+)-{suffix}_v([1-9][0-9]*)\.(?:cif|json)",
        name,
    )
    if match is None:
        raise ValueError(f"invalid {kind} metadata URL: {name}")
    return match.group(1), int(match.group(2))


def load_pair_confidence_evidence(
    *,
    pair_dir: Path,
    candidate_identity: Mapping[str, Any],
    geometry_status: str | None,
    strict: bool,
) -> ConfidenceEvidence:
    if geometry_status not in GEOMETRY_COMPLETE_STATUSES:
        return _empty_confidence_evidence("geometry_not_complete")
    qc_path = pair_dir / "pair_qc.json"
    if not qc_path.exists():
        if strict:
            raise FileNotFoundError(f"missing canonical pair QC: {qc_path}")
        return _empty_confidence_evidence("missing_pair_qc")
    qc = _load_json_mapping(qc_path, "pair QC")
    expected_identity = {
        "pdb_id": _clean_text(candidate_identity.get("pdb_id"), lower=True),
        "chain_id": _clean_text(candidate_identity.get("chain_id")),
        "uniprot_id": _clean_text(candidate_identity.get("uniprot_id")),
    }
    actual_identity = {
        "pdb_id": _clean_text(qc.get("pdb_id"), lower=True),
        "chain_id": _clean_text(qc.get("chain_id")),
        "uniprot_id": _clean_text(qc.get("uniprot_id")),
    }
    if actual_identity != expected_identity:
        raise ValueError("canonical pair QC candidate identity mismatch")

    selected = _clean_text(qc.get("afdb_model_entity_id"))
    version = _optional_int(qc.get("afdb_version"))
    fragment_start = _optional_int(qc.get("afdb_fragment_start"))
    fragment_end = _optional_int(qc.get("afdb_fragment_end"))
    if selected is None or version is None:
        raise ValueError("invalid selected AFDB model identity or version")

    paths: dict[str, Path] = {}
    for field, kind in (
        ("afdb_model_path", "model"),
        ("plddt_path", "pLDDT"),
        ("pae_path", "PAE"),
    ):
        text = _clean_text(qc.get(field))
        if text is None:
            raise ValueError(f"missing canonical {kind} path")
        path = Path(text)
        if not path.exists():
            raise FileNotFoundError(f"missing canonical {kind} artifact: {path}")
        paths[field] = path

    artifact_parents = {path.parent for path in paths.values()}
    if len(artifact_parents) != 1:
        raise ValueError("canonical AFDB artifacts use inconsistent directories")
    artifact_parent = next(iter(artifact_parents))
    current_layout = artifact_parent.name == selected
    metadata_path = paths["afdb_model_path"].parent / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"missing canonical AFDB metadata: {metadata_path}"
        )
    metadata = _load_json_mapping(metadata_path, "AFDB metadata")
    metadata_model = _clean_text(
        metadata.get("modelEntityId") or metadata.get("entryId")
    )
    metadata_version = _optional_int(metadata.get("latestVersion"))
    metadata_start = _optional_int(
        metadata.get("uniprotStart", metadata.get("sequenceStart"))
    )
    metadata_end = _optional_int(
        metadata.get("uniprotEnd", metadata.get("sequenceEnd"))
    )
    if metadata_model != selected:
        raise ValueError("AFDB model identity conflicts with selected model")
    if metadata_version != version:
        raise ValueError("AFDB model version conflicts with selected version")
    if fragment_start is None and fragment_end is None and not current_layout:
        fragment_start, fragment_end = metadata_start, metadata_end
    if (
        fragment_start is None
        or fragment_end is None
        or fragment_start < 1
        or fragment_end < fragment_start
    ):
        raise ValueError("invalid selected AFDB fragment interval")
    if (metadata_start, metadata_end) != (fragment_start, fragment_end):
        raise ValueError("AFDB fragment interval conflicts with selected interval")

    model_id, model_version = _artifact_identity(
        metadata.get("cifUrl"),
        kind="model",
    )
    plddt_id, plddt_version = _artifact_identity(
        metadata.get("plddtDocUrl"),
        kind="plddt",
    )
    pae_id, pae_version = _artifact_identity(
        metadata.get("paeDocUrl"),
        kind="pae",
    )
    if model_id != selected or model_version != version:
        raise ValueError("model identity/version conflicts with selected model")
    if plddt_id != selected or plddt_version != version:
        raise ValueError("pLDDT model identity/version conflicts with selected model")
    if pae_id != selected or pae_version != version:
        raise ValueError("PAE model identity/version conflicts with selected model")
    return ConfidenceEvidence(
        available=True,
        confidence_model_match=True,
        selected_model_entity_id=selected,
        selected_version=version,
        fragment_start=fragment_start,
        fragment_end=fragment_end,
        plddt_model_entity_id=plddt_id,
        plddt_model_version=plddt_version,
        pae_model_entity_id=pae_id,
        pae_model_version=pae_version,
        source="pair_qc+afdb_metadata",
        error_reason=None,
    )


def _legacy_auth_fields(
    mapping: pd.DataFrame,
    *,
    allow_unmapped: bool = False,
) -> pd.DataFrame:
    number_column = (
        "pdb_residue_number_norm"
        if "pdb_residue_number_norm" in mapping
        else "pdb_residue_number"
    )
    residue_numbers = mapping[number_column].astype(str).str.strip()
    parsed = residue_numbers.str.extract(r"^(-?\d+)([A-Za-z]?)$")
    if parsed[0].isna().any() and not allow_unmapped:
        raise ValueError("unparseable legacy PDB residue number")
    auth_sequence = pd.to_numeric(parsed[0], errors="coerce").astype("Int64")
    auth_chain = mapping["pdb_chain_id"].astype(str)
    if allow_unmapped:
        auth_chain = auth_chain.where(auth_sequence.notna(), "")
    return pd.DataFrame(
        {
            "auth_asym_id": auth_chain,
            "auth_seq_id": auth_sequence,
            "insertion_code": parsed[1].fillna(""),
        },
        index=mapping.index,
    )


def _mapped_confidence_from_pair(
    pair_dir: Path,
    confidence: ConfidenceEvidence | None = None,
) -> dict[str, Any] | None:
    mapping_path = pair_dir / "residue_mapping.parquet"
    residue_path = pair_dir / "residue_geometry.parquet"
    if not mapping_path.exists() or not residue_path.exists():
        return None
    mapping = pd.read_parquet(mapping_path).copy()
    residues = pd.read_parquet(residue_path)
    if confidence is not None and confidence.available:
        if confidence.fragment_start is None or confidence.fragment_end is None:
            raise ValueError("missing canonical AFDB fragment interval")
        qc = _load_json_mapping(pair_dir / "pair_qc.json", "pair QC")
        plddt_path = Path(str(qc["plddt_path"]))
        expected_length = (
            confidence.fragment_end - confidence.fragment_start + 1
        )
        plddt = load_plddt(plddt_path, expected_length=expected_length)
        if (
            not np.isfinite(plddt).all()
            or ((plddt < 0.0) | (plddt > 100.0)).any()
        ):
            raise ValueError("invalid canonical pLDDT")
        required = {
            "uniprot_residue_number",
            "auth_asym_id",
            "auth_seq_id",
            "insertion_code",
        }
        missing_mapping = sorted(required - set(mapping.columns))
        missing_residues = sorted(required - set(residues.columns))
        if missing_mapping:
            legacy = _legacy_auth_fields(mapping, allow_unmapped=True)
            mapping = mapping.assign(**legacy.to_dict(orient="series"))
        if missing_residues:
            legacy = _legacy_auth_fields(residues)
            residues = residues.assign(**legacy.to_dict(orient="series"))
        mapping["insertion_code"] = mapping["insertion_code"].fillna("")
        residues["insertion_code"] = residues["insertion_code"].fillna("")
        auth_key = ["auth_asym_id", "auth_seq_id", "insertion_code"]
        observed_keys = residues[auth_key].drop_duplicates()
        if observed_keys.duplicated(auth_key).any():
            raise ValueError("duplicate observed auth residue key")
        table = mapping.copy()
        table["_mapped_row"] = np.arange(len(table))
        observed = observed_keys.assign(observed_ca=True)
        table = table.merge(
            observed,
            on=auth_key,
            how="left",
            validate="many_to_one",
        ).sort_values("_mapped_row", kind="mergesort")
        table["observed_ca"] = (
            table["observed_ca"].fillna(False).astype(bool)
        )
        positions = pd.to_numeric(
            table["uniprot_residue_number"],
            errors="coerce",
        )
        if (
            positions.isna().any()
            or positions.mod(1).ne(0).any()
            or positions.duplicated().any()
        ):
            raise ValueError("invalid mapped UniProt positions")
        positions = positions.astype(int)
        covered = positions.between(
            confidence.fragment_start,
            confidence.fragment_end,
            inclusive="both",
        )
        if not covered.all():
            raise ValueError("mapped position outside selected AFDB fragment")
        local_indices = positions.to_numpy() - confidence.fragment_start
        table["uniprot_residue_number"] = positions
        table["mapped"] = (
            table["auth_asym_id"].astype(str).str.strip().ne("")
            & pd.to_numeric(table["auth_seq_id"], errors="coerce").notna()
        )
        table["fragment_covered"] = covered
        table["plddt"] = plddt[local_indices]
        summary = summarize_mapped_confidence(table)
        return {
            "mapped_plddt_min": summary.mapped_plddt_min,
            "mapped_plddt_q10": summary.mapped_plddt_q10,
            "mapped_plddt_median": summary.mapped_plddt_median,
            "longest_internal_below_70_length": (
                summary.longest_internal_below_70_length
            ),
            "longest_internal_below_80_length": (
                summary.longest_internal_below_80_length
            ),
            "terminal_only_below_80": bool(
                summary.longest_below_80_length >= 5
                and summary.longest_internal_below_80_length < 5
            ),
            "is_low_conf_local": summary.is_low_conf_local,
            "low_conf_evidence_available": True,
            "mapped_confidence_source": "canonical_pair_artifacts",
        }

    mapped_positions = pd.to_numeric(
        mapping["uniprot_residue_number"],
        errors="raise",
    ).astype(int)
    positions = pd.to_numeric(
        residues["uniprot_residue_number"],
        errors="raise",
    ).astype(int)
    if positions.duplicated().any():
        raise ValueError("duplicate residue-geometry UniProt position")
    if not positions.isin(set(mapped_positions)).all():
        raise ValueError("residue geometry position absent from mapping")

    if {
        "auth_asym_id",
        "auth_seq_id",
        "insertion_code",
    }.issubset(residues.columns):
        auth = residues[
            ["auth_asym_id", "auth_seq_id", "insertion_code"]
        ].copy()
        auth["insertion_code"] = auth["insertion_code"].fillna("")
    else:
        auth = _legacy_auth_fields(residues)
    residue_table = auth.assign(
        uniprot_residue_number=positions,
        mapped=True,
        observed_ca=True,
        plddt=pd.to_numeric(residues["plddt"], errors="raise"),
        fragment_covered=True,
    )
    summary = summarize_mapped_confidence(residue_table)
    return {
        "mapped_plddt_min": summary.mapped_plddt_min,
        "mapped_plddt_q10": summary.mapped_plddt_q10,
        "mapped_plddt_median": summary.mapped_plddt_median,
        "longest_internal_below_70_length": (
            summary.longest_internal_below_70_length
        ),
        "longest_internal_below_80_length": (
            summary.longest_internal_below_80_length
        ),
        "terminal_only_below_80": bool(
            summary.longest_below_80_length > 0
            and summary.longest_internal_below_80_length == 0
        ),
        "is_low_conf_local": summary.is_low_conf_local,
        "low_conf_evidence_available": True,
        "mapped_confidence_source": "legacy_pair_geometry",
    }


def _mapped_confidence_from_report(
    record: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not record:
        return None
    success = (
        _clean_text(record.get("scoring_status")) == "success"
        and _optional_bool(record.get("confidence_model_match")) is True
        and _optional_bool(record.get("plddt_bfactor_match")) is True
    )
    return {
        "mapped_plddt_min": _optional_float(
            record.get("mapped_plddt_min")
        ),
        "mapped_plddt_q10": _optional_float(
            record.get("mapped_plddt_q10")
        ),
        "mapped_plddt_median": _optional_float(
            record.get("mapped_plddt_median")
        ),
        "longest_internal_below_70_length": _optional_int(
            record.get("longest_internal_below_70_length")
        ),
        "longest_internal_below_80_length": _optional_int(
            record.get("longest_internal_below_80_length")
        ),
        "terminal_only_below_80": _optional_bool(
            record.get("terminal_only_below_80")
        ),
        "is_low_conf_local": _optional_bool(
            record.get("is_low_conf_local")
        ),
        "low_conf_evidence_available": success,
        "mapped_confidence_source": "mapped_confidence_report",
    }


def _cross_check_mapped_confidence(
    canonical: Mapping[str, Any],
    report: Mapping[str, Any] | None,
) -> None:
    if report is None or not bool(report.get("low_conf_evidence_available")):
        return
    numeric_fields = (
        "mapped_plddt_min",
        "mapped_plddt_q10",
        "mapped_plddt_median",
        "longest_internal_below_70_length",
        "longest_internal_below_80_length",
    )
    conflicts = []
    for field in numeric_fields:
        canonical_value = _optional_float(canonical.get(field))
        report_value = _optional_float(report.get(field))
        if (
            canonical_value is None
            or report_value is None
            or not np.isclose(
                canonical_value,
                report_value,
                rtol=0.0,
                atol=1e-6,
            )
        ):
            conflicts.append(field)
    for field in ("terminal_only_below_80",):
        if _optional_bool(canonical.get(field)) != _optional_bool(
            report.get(field)
        ):
            conflicts.append(field)
    if conflicts:
        raise ValueError(
            "mapped confidence report conflicts with canonical artifacts: "
            + ",".join(conflicts)
        )


def _pae_from_pair(
    pair_dir: Path,
    *,
    model_match: bool,
    required: bool = False,
) -> tuple[dict[str, PaeStratumStats] | None, bool]:
    pairwise_path = pair_dir / "pairwise_geometry.npz"
    if not model_match:
        return None, False
    if not pairwise_path.exists():
        if required:
            raise FileNotFoundError(
                f"missing canonical pairwise artifact: {pairwise_path}"
            )
        return None, False
    with np.load(pairwise_path) as data:
        required_arrays = {"uniprot_positions", "symmetric_pae"}
        missing = sorted(required_arrays - set(data.files))
        if missing:
            raise ValueError(
                "pairwise artifact missing arrays: " + ",".join(missing)
            )
        positions = np.asarray(data["uniprot_positions"])
        pae = np.asarray(data["symmetric_pae"])
    if not np.isfinite(pae).all():
        raise ValueError("canonical symmetric PAE contains NaN or infinity")
    return compute_pae_strata(positions, pae), True


def _segments_from_pair(
    pair_dir: Path,
    *,
    segment_context_status: str | None,
    robust_status: str | None,
) -> tuple[tuple[DisagreementSegment, ...] | None, bool]:
    if (
        robust_status != "complete"
        or segment_context_status
        not in {"complete", "successful_no_segments"}
    ):
        return None, False
    segment_path = pair_dir / "disagreement_segments.csv"
    if not segment_path.exists():
        if segment_context_status == "successful_no_segments":
            return (), True
        return None, False
    try:
        table = pd.read_csv(segment_path)
    except pd.errors.EmptyDataError:
        table = pd.DataFrame()
    if table.empty:
        return (), True
    required = {
        "start_position",
        "end_position",
        "residue_count",
        "median_plddt",
        "median_disagreement",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(
            "disagreement_segments missing columns: " + ",".join(missing)
        )
    segments = tuple(
        DisagreementSegment(
            start_position=int(row.start_position),
            end_position=int(row.end_position),
            residue_count=int(row.residue_count),
            median_plddt=float(row.median_plddt),
            median_disagreement=float(row.median_disagreement),
        )
        for row in table.itertuples(index=False)
    )
    return segments, True


def _mechanism_model_match(record: Mapping[str, Any]) -> bool | None:
    if not record:
        return None
    stated = _optional_bool(record.get("confidence_model_match"))
    model_ids = {
        _clean_text(record.get(field))
        for field in (
            "selected_afdb_model_entity_id",
            "plddt_model_entity_id",
            "pae_model_entity_id",
            "geometry_model_entity_id",
        )
    }
    model_ids.discard(None)
    versions = {
        _optional_int(record.get(field))
        for field in (
            "selected_afdb_version",
            "plddt_model_version",
            "pae_model_version",
            "geometry_model_version",
        )
    }
    versions.discard(None)
    explicit_match = len(model_ids) == 1 and len(versions) == 1
    return bool(stated is True and explicit_match)


def _pae_output(
    strata: Mapping[str, PaeStratumStats] | None,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, _, _ in PAE_BINS:
        prefix = f"pae_{name}"
        stats = strata.get(name) if strata is not None else None
        output[f"{prefix}_q50"] = None if stats is None else stats.pae_q50
        output[f"{prefix}_q90"] = None if stats is None else stats.pae_q90
        output[f"{prefix}_mean"] = None if stats is None else stats.pae_mean
        output[f"{prefix}_above_10_fraction"] = (
            None if stats is None else stats.above_10_fraction
        )
        output[f"{prefix}_above_15_fraction"] = (
            None if stats is None else stats.above_15_fraction
        )
        output[f"{prefix}_pair_count"] = (
            None if stats is None else stats.pair_count
        )
    for name in LONG_RANGE_PAE_BINS:
        stats = strata.get(name) if strata is not None else None
        prefix = f"long_range_{name}"
        output[f"{prefix}_pae_q50"] = (
            None if stats is None else stats.pae_q50
        )
        output[f"{prefix}_pae_q90"] = (
            None if stats is None else stats.pae_q90
        )
        output[f"{prefix}_pae_mean"] = (
            None if stats is None else stats.pae_mean
        )
        output[f"{prefix}_above_10_fraction"] = (
            None if stats is None else stats.above_10_fraction
        )
        output[f"{prefix}_above_15_fraction"] = (
            None if stats is None else stats.above_15_fraction
        )
        output[f"{prefix}_pair_count"] = (
            None if stats is None else stats.pair_count
        )
    return output


def _best_high_conf_segment(
    segments: tuple[DisagreementSegment, ...] | None,
    config: Mapping[str, Any],
) -> DisagreementSegment | None:
    if not segments:
        return None
    thresholds = config["thresholds"][
        "high_confidence_state_disagreement"
    ]
    eligible = [
        segment
        for segment in segments
        if segment.median_plddt
        >= float(thresholds["minimum_segment_median_plddt"])
        and segment.median_disagreement
        >= float(thresholds["minimum_segment_disagreement"])
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda segment: (-segment.residue_count, segment.start_position),
    )


def _status_for_unstarted(preflight_status: str | None) -> str:
    return (
        "not_started"
        if preflight_status
        in {"pass_full_length", "warn_construct_difference"}
        else "skipped_preflight"
    )


def _base_failure_row(
    base: Mapping[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        **base,
        "full_diagnostic_complete": False,
        "main_quality_pass": False,
        "is_easy_control": False,
        "is_low_conf_local": None,
        "is_high_pae_long_range": None,
        "is_high_conf_state_disagreement": None,
        "is_construct_difference": None,
        "is_missing_coordinate_stress": None,
        "easy_control_evidence_available": False,
        "low_conf_evidence_available": False,
        "high_pae_evidence_available": False,
        "state_disagreement_evidence_available": False,
        "construct_evidence_available": False,
        "missing_coordinate_evidence_available": False,
        "primary_category": "insufficient_evidence",
        "classification_status": "insufficient_evidence",
        "classification_explanation": f"evidence_assembly_failure:{reason}",
        "selection_eligible": False,
        "evidence_assembly_status": "failure",
        "evidence_assembly_reason": reason,
        "model_evidence_mismatch": False,
    }


def _classification_fields(
    result: A0Classification,
) -> dict[str, Any]:
    return asdict(result)


def _assemble_candidate(
    candidate: Mapping[str, Any],
    *,
    root: Path,
    config: Mapping[str, Any],
    preflight: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
    replacement: Mapping[str, Any],
    pilot: Mapping[str, Any],
    mechanism: Mapping[str, Any],
    mapped_confidence: Mapping[str, Any],
    strict: bool = False,
) -> dict[str, Any]:
    key = _candidate_key(candidate)
    index, pdb_id, chain_id, uniprot_id = key
    pair_name = str(candidate["pair_name"])
    source = str(candidate["source"])
    lifecycle_record_available = bool(lifecycle)
    lifecycle_source_path = _clean_text(
        lifecycle.get("lifecycle_source_path")
    )
    lifecycle_cohort = _clean_text(lifecycle.get("lifecycle_cohort"))
    if lifecycle_record_available and lifecycle_cohort is None:
        lifecycle_cohort = _clean_text(lifecycle.get("source")) or source
    records = (mechanism, preflight, replacement, lifecycle, candidate)
    preflight_status = _clean_text(
        _first_value(records, "preflight_status")
    )
    pilot_role = _clean_text(_first_value((pilot, mechanism), "pilot_role"))

    pair_status = _clean_text(
        _first_value((mechanism, lifecycle), "pair_status")
    )
    lifecycle_geometry_status = _clean_text(lifecycle.get("geometry_status"))
    pilot_geometry_status = _clean_text(mechanism.get("geometry_status"))
    geometry_status = (
        lifecycle_geometry_status
        if lifecycle_record_available
        else pilot_geometry_status
    )
    comparable_lifecycle_geometry_status = (
        "complete"
        if lifecycle_geometry_status in GEOMETRY_COMPLETE_STATUSES
        else lifecycle_geometry_status
    )
    comparable_pilot_geometry_status = (
        "complete"
        if pilot_geometry_status in GEOMETRY_COMPLETE_STATUSES
        else pilot_geometry_status
    )
    if (
        strict
        and lifecycle_record_available
        and comparable_lifecycle_geometry_status is not None
        and comparable_pilot_geometry_status is not None
        and comparable_lifecycle_geometry_status
        != comparable_pilot_geometry_status
    ):
        raise ValueError(
            "geometry status conflict: "
            f"lifecycle={lifecycle_geometry_status},"
            f"pilot={pilot_geometry_status}"
        )
    robust_status = _clean_text(
        _first_value((mechanism, lifecycle), "robust_status")
    )
    segment_status = _clean_text(
        _first_value(
            (mechanism, lifecycle),
            "segment_context_status",
        )
    )
    if source == "replacement_pool":
        default_status = _status_for_unstarted(preflight_status)
        pair_status = pair_status or default_status
        geometry_status = geometry_status or default_status
        robust_status = robust_status or default_status
        segment_status = segment_status or default_status

    report_model_match = _optional_bool(
        mapped_confidence.get("confidence_model_match")
    )
    mechanism_model_match = _mechanism_model_match(mechanism)

    pair_dir = root / "data/processed/pairs" / pair_name
    pair_qc_exists = (pair_dir / "pair_qc.json").exists()
    canonical_confidence = (
        load_pair_confidence_evidence(
            pair_dir=pair_dir,
            candidate_identity=candidate,
            geometry_status=geometry_status,
            strict=strict,
        )
        if pair_qc_exists
        else _empty_confidence_evidence("missing_pair_qc")
    )
    model_match = canonical_confidence.confidence_model_match
    if canonical_confidence.available:
        for record_name, record, stated_match in (
            ("mechanism", mechanism, mechanism_model_match),
            ("mapped confidence", mapped_confidence, report_model_match),
        ):
            if stated_match is False:
                raise ValueError(
                    f"{record_name} model identity conflicts with canonical artifacts"
                )
            stated_model = _clean_text(
                record.get("selected_afdb_model_entity_id")
            )
            stated_version = _optional_int(
                record.get("selected_afdb_version")
            )
            if (
                stated_model is not None
                and stated_model
                != canonical_confidence.selected_model_entity_id
            ):
                raise ValueError(
                    f"{record_name} selected model conflicts with canonical artifacts"
                )
            if (
                stated_version is not None
                and stated_version != canonical_confidence.selected_version
            ):
                raise ValueError(
                    f"{record_name} selected version conflicts with canonical artifacts"
                )
    else:
        model_match = mechanism_model_match
        if model_match is None:
            model_match = report_model_match

    geometry = load_pair_geometry_evidence(
        project_root=root,
        candidate_identity=candidate,
        geometry_status=geometry_status,
        pilot_mechanism=mechanism,
        strict=strict,
    )
    report_mapped = _mapped_confidence_from_report(mapped_confidence)
    mapped = None
    if (
        geometry_status in GEOMETRY_COMPLETE_STATUSES
        and canonical_confidence.available
    ):
        mapped = _mapped_confidence_from_pair(
            pair_dir,
            canonical_confidence,
        )
        if strict and mapped is not None:
            _cross_check_mapped_confidence(mapped, report_mapped)
    else:
        mapped = report_mapped
        if (
            mapped is None
            and geometry_status in GEOMETRY_COMPLETE_STATUSES
            and pair_dir.exists()
        ):
            mapped = _mapped_confidence_from_pair(pair_dir)
    mapped = mapped or {
        "mapped_plddt_min": None,
        "mapped_plddt_q10": None,
        "mapped_plddt_median": None,
        "longest_internal_below_70_length": None,
        "longest_internal_below_80_length": None,
        "terminal_only_below_80": None,
        "is_low_conf_local": None,
        "low_conf_evidence_available": False,
        "mapped_confidence_source": None,
    }

    if geometry_status in GEOMETRY_COMPLETE_STATUSES:
        pae_strata, pae_available = _pae_from_pair(
            pair_dir,
            model_match=model_match is True,
            required=canonical_confidence.available,
        )
    else:
        pae_strata, pae_available = None, False
    segments, state_available = _segments_from_pair(
        pair_dir,
        segment_context_status=segment_status,
        robust_status=robust_status,
    )
    best_segment = _best_high_conf_segment(segments, config)

    full_coverage = _optional_float(
        _first_value(records, "full_length_mapping_coverage")
    )
    entity_coverage = _optional_float(
        _first_value(records, "entity_mapping_coverage")
    )
    sequence_identity = _optional_float(
        _first_value(records, "sequence_identity")
    )
    observed_ca = _optional_float(
        _first_value(
            records,
            "observed_ca_fraction",
        )
    )
    if observed_ca is None:
        observed_ca = _optional_float(
            _first_value(
                records,
                "observed_ca_fraction_of_mapped",
            )
        )
    if source == "reference_pair":
        preflight_status = preflight_status or "reference_quality_pass"
        full_coverage = full_coverage if full_coverage is not None else 1.0
        entity_coverage = (
            entity_coverage if entity_coverage is not None else 1.0
        )
        sequence_identity = (
            sequence_identity if sequence_identity is not None else 1.0
        )
        observed_ca = observed_ca if observed_ca is not None else 1.0

    unsupported = bool(
        preflight_status == "unsupported_afdb_fragment"
        or _clean_text(preflight.get("afdb_coverage_status"))
        == "unsupported_afdb_fragment"
        or _clean_text(replacement.get("afdb_selection_status"))
        == "unsupported_afdb_fragment"
    )
    evidence = A0Evidence(
        preflight_status=preflight_status,
        full_length_mapping_coverage=full_coverage,
        entity_mapping_coverage=entity_coverage,
        sequence_identity=sequence_identity,
        observed_ca_fraction=observed_ca,
        pair_status=pair_status,
        geometry_status=geometry_status,
        robust_status=robust_status,
        segment_context_status=segment_status,
        confidence_model_match=model_match,
        mapped_plddt_median=mapped["mapped_plddt_median"],
        longest_internal_below_80_length=mapped[
            "longest_internal_below_80_length"
        ],
        terminal_only_below_80=mapped["terminal_only_below_80"],
        low_conf_evidence_available=bool(
            mapped["low_conf_evidence_available"]
        ),
        pae_strata=pae_strata,
        pae_evidence_available=pae_available,
        ca_disagreement_p90=geometry.ca_disagreement_p90,
        disagreement_segments=segments,
        state_disagreement_evidence_available=state_available,
        pilot_role=pilot_role,
        secondary_missing_coordinate_stress=None,
        unsupported_afdb_fragment=unsupported,
        protein_length=_optional_int(
            _first_value(records, "length")
        ),
        provisional_stratum=_clean_text(
            _first_value(records, "provisional_stratum")
        ),
    )
    result = classify_candidate(evidence, config)
    base = {
        "screening_index": index,
        "source": source,
        "pdb_id": pdb_id,
        "chain_id": chain_id,
        "uniprot_id": uniprot_id,
        "pair_name": pair_name,
        "provisional_stratum": evidence.provisional_stratum,
        "pilot_role": pilot_role,
        "lifecycle_record_available": lifecycle_record_available,
        "lifecycle_source_path": lifecycle_source_path,
        "lifecycle_cohort": lifecycle_cohort,
        "preflight_status": preflight_status,
        "full_length_mapping_coverage": full_coverage,
        "entity_mapping_coverage": entity_coverage,
        "sequence_identity": sequence_identity,
        "observed_ca_fraction": observed_ca,
        "pair_status": pair_status,
        "geometry_status": geometry_status,
        "robust_status": robust_status,
        "segment_context_status": segment_status,
        "confidence_model_match": model_match,
        "selected_afdb_model_entity_id": (
            canonical_confidence.selected_model_entity_id
        ),
        "selected_afdb_version": canonical_confidence.selected_version,
        "afdb_fragment_start": canonical_confidence.fragment_start,
        "afdb_fragment_end": canonical_confidence.fragment_end,
        "plddt_model_entity_id": (
            canonical_confidence.plddt_model_entity_id
        ),
        "plddt_model_version": canonical_confidence.plddt_model_version,
        "pae_model_entity_id": canonical_confidence.pae_model_entity_id,
        "pae_model_version": canonical_confidence.pae_model_version,
        "confidence_evidence_source": canonical_confidence.source,
        "mapped_plddt_min": mapped["mapped_plddt_min"],
        "mapped_plddt_q10": mapped["mapped_plddt_q10"],
        "mapped_plddt_median": mapped["mapped_plddt_median"],
        "longest_internal_below_70_length": mapped[
            "longest_internal_below_70_length"
        ],
        "longest_internal_below_80_length": mapped[
            "longest_internal_below_80_length"
        ],
        "terminal_only_below_80": mapped["terminal_only_below_80"],
        "mapped_confidence_source": mapped["mapped_confidence_source"],
        "ca_disagreement_median": geometry.ca_disagreement_median,
        "ca_disagreement_p90": geometry.ca_disagreement_p90,
        "ca_disagreement_max": geometry.ca_disagreement_max,
        "largest_high_conf_disagreement_segment": (
            0 if best_segment is None else best_segment.residue_count
        ),
        "high_conf_segment_median_plddt": (
            None if best_segment is None else best_segment.median_plddt
        ),
        "high_conf_segment_disagreement": (
            None if best_segment is None else best_segment.median_disagreement
        ),
        **_pae_output(pae_strata),
        **_classification_fields(result),
        "evidence_assembly_status": "success",
        "evidence_assembly_reason": (
            "assembled"
            if geometry.error_reason is None
            else f"assembled;{geometry.error_reason}"
        ),
        "model_evidence_mismatch": (
            model_match is False or geometry.evidence_mismatch
        ),
    }
    return base


def _filter_only(
    universe: pd.DataFrame,
    only: list[str] | None,
) -> pd.DataFrame:
    if not only:
        return universe
    indices = {_optional_int(value) for value in only}
    indices.discard(None)
    pair_names = {value for value in only if _optional_int(value) is None}
    mask = universe["screening_index"].map(_optional_int).isin(indices)
    mask |= universe["pair_name"].isin(pair_names)
    selected = universe.loc[mask].copy()
    if selected.empty:
        raise ValueError("--only did not match any candidate")
    return selected


def _audit(table: pd.DataFrame) -> dict[str, Any]:
    label_counts = {
        label: int(table[label].map(lambda value: value is True).sum())
        for label in LABEL_COLUMNS
    }
    true_label_counts = table[list(LABEL_COLUMNS)].apply(
        lambda row: sum(value is True for value in row),
        axis=1,
    )
    overlap = table.loc[true_label_counts > 1, list(LABEL_COLUMNS)]
    overlap_combinations = (
        overlap.apply(
            lambda row: "+".join(
                label.removeprefix("is_")
                for label, value in row.items()
                if value is True
            ),
            axis=1,
        )
        .value_counts()
        .sort_index()
        .to_dict()
    )
    low = table["is_low_conf_local"].map(lambda value: value is True)
    complete = table["full_diagnostic_complete"].map(
        lambda value: value is True
    )
    return {
        "candidate_count": len(table),
        "complete_classification_count": int(
            (table["classification_status"] == "complete_classification").sum()
        ),
        "partial_classification_count": int(
            (table["classification_status"] == "partial_classification").sum()
        ),
        "insufficient_evidence_count": int(
            (table["classification_status"] == "insufficient_evidence").sum()
        ),
        "selection_eligible_count": int(
            table["selection_eligible"].map(lambda value: value is True).sum()
        ),
        "label_counts": label_counts,
        "primary_category_counts": (
            table["primary_category"].value_counts(dropna=False).to_dict()
        ),
        "multi_label_overlap_count": int((true_label_counts > 1).sum()),
        "multi_label_overlap_combinations": overlap_combinations,
        "control_count": int(
            table["primary_category"]
            .isin(
                {
                    "construct_difference_control",
                    "missing_coordinate_stress",
                }
            )
            .sum()
        ),
        "unsupported_count": int(
            (table["primary_category"] == "unsupported_afdb_fragment").sum()
        ),
        "model_evidence_mismatch_count": int(
            table["model_evidence_mismatch"]
            .map(lambda value: value is True)
            .sum()
        ),
        "low_conf_local_total": int(low.sum()),
        "low_conf_local_full_diagnostic": int((low & complete).sum()),
        "low_conf_local_partial": int(
            (
                low
                & (
                    table["classification_status"]
                    == "partial_classification"
                )
            ).sum()
        ),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    config = load_config(root, args.config)
    output = resolve_path(root, args.output)

    screening_pool = _read_table(
        root / "data/manifests/screening_pool.tsv",
        sep="\t",
    )
    replacement_pool = _read_table(
        root / "data/manifests/lower_conf_replacement_pool.tsv",
        sep="\t",
    )
    pilot_table = _read_table(
        root / "data/manifests/geometry_pilot.tsv",
        sep="\t",
    )
    mechanism_table = enrich_pilot_mechanism_identity(
        _read_table(root / "reports/geometry_pilot_mechanisms.csv"),
        pilot_table,
    )
    lifecycle_table = load_lifecycle_tables(root, args.lifecycle)
    universe = _filter_only(
        _build_candidate_universe(screening_pool, replacement_pool),
        args.only,
    )
    sources = {
        "preflight": _records_by_key(
            _read_table(
                root / "data/manifests/screening_pool_preflight.tsv",
                sep="\t",
            ),
            "screening preflight",
        ),
        "lifecycle": _records_by_key(
            lifecycle_table,
            "candidate lifecycle",
        ),
        "replacement": _records_by_key(
            replacement_pool,
            "replacement pool",
        ),
        "pilot": _records_by_key(
            pilot_table,
            "geometry pilot manifest",
        ),
        "mechanism": _records_by_key(
            mechanism_table,
            "geometry pilot mechanisms",
        ),
        "mapped_confidence": _records_by_key(
            _read_table(
                root / "reports/replacement_mapped_confidence.csv"
            ),
            "replacement mapped confidence",
        ),
    }

    rows: list[dict[str, Any]] = []
    for candidate in universe.to_dict(orient="records"):
        key = _candidate_key(candidate)
        lifecycle_record = _lookup(sources["lifecycle"], key)
        base = {
            "screening_index": key[0],
            "source": candidate["source"],
            "pdb_id": key[1],
            "chain_id": key[2],
            "uniprot_id": key[3],
            "pair_name": candidate["pair_name"],
            "provisional_stratum": _clean_text(
                candidate.get("provisional_stratum")
            ),
            "pilot_role": None,
            "lifecycle_record_available": bool(lifecycle_record),
            "lifecycle_source_path": _clean_text(
                lifecycle_record.get("lifecycle_source_path")
            ),
            "lifecycle_cohort": (
                _clean_text(lifecycle_record.get("lifecycle_cohort"))
                or (
                    str(candidate["source"])
                    if lifecycle_record
                    else None
                )
            ),
        }
        try:
            rows.append(
                _assemble_candidate(
                    candidate,
                    root=root,
                    config=config,
                    preflight=_lookup(sources["preflight"], key),
                    lifecycle=lifecycle_record,
                    replacement=_lookup(sources["replacement"], key),
                    pilot=_lookup(sources["pilot"], key),
                    mechanism=_lookup(sources["mechanism"], key),
                    mapped_confidence=_lookup(
                        sources["mapped_confidence"],
                        key,
                    ),
                    strict=args.strict,
                )
            )
        except Exception as exc:
            if args.strict:
                raise
            rows.append(_base_failure_row(base, f"{type(exc).__name__}:{exc}"))

    table = pd.DataFrame(rows)
    table["screening_index"] = pd.array(
        table["screening_index"],
        dtype="Int64",
    )
    table = table.sort_values(
        ["screening_index", "pair_name"],
        na_position="first",
        kind="mergesort",
    ).reset_index(drop=True)
    _records_by_key(table, "A0 candidate summary")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(output)

    audit = _audit(table)
    print(table.to_string(index=False))
    print("\nClassification audit")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(f"\nSaved: {output}")
    if args.strict and audit["model_evidence_mismatch_count"]:
        raise RuntimeError(
            "strict classification failed: model/evidence mismatch"
        )


if __name__ == "__main__":
    main()
