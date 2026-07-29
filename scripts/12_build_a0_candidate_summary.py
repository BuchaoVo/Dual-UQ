from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

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


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


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


def _legacy_auth_fields(mapping: pd.DataFrame) -> pd.DataFrame:
    number_column = (
        "pdb_residue_number_norm"
        if "pdb_residue_number_norm" in mapping
        else "pdb_residue_number"
    )
    residue_numbers = mapping[number_column].astype(str).str.strip()
    parsed = residue_numbers.str.extract(r"^(-?\d+)([A-Za-z]?)$")
    if parsed[0].isna().any():
        raise ValueError("unparseable legacy PDB residue number")
    return pd.DataFrame(
        {
            "auth_asym_id": mapping["pdb_chain_id"].astype(str),
            "auth_seq_id": parsed[0].astype(int),
            "insertion_code": parsed[1].fillna(""),
        },
        index=mapping.index,
    )


def _mapped_confidence_from_pair(pair_dir: Path) -> dict[str, Any] | None:
    mapping_path = pair_dir / "residue_mapping.parquet"
    residue_path = pair_dir / "residue_geometry.parquet"
    if not mapping_path.exists() or not residue_path.exists():
        return None
    mapping = pd.read_parquet(mapping_path).copy()
    residues = pd.read_parquet(residue_path)
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
        "low_conf_evidence_available": True,
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
        "low_conf_evidence_available": success,
    }


def _pae_from_pair(
    pair_dir: Path,
    *,
    model_match: bool,
) -> tuple[dict[str, PaeStratumStats] | None, bool]:
    pairwise_path = pair_dir / "pairwise_geometry.npz"
    if not model_match or not pairwise_path.exists():
        return None, False
    with np.load(pairwise_path) as data:
        positions = np.asarray(data["uniprot_positions"])
        pae = np.asarray(data["symmetric_pae"])
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
        output[f"{prefix}_pae_q90"] = (
            None if stats is None else stats.pae_q90
        )
        output[f"{prefix}_above_10_fraction"] = (
            None if stats is None else stats.above_10_fraction
        )
        output[f"{prefix}_above_15_fraction"] = (
            None if stats is None else stats.above_15_fraction
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
) -> dict[str, Any]:
    key = _candidate_key(candidate)
    index, pdb_id, chain_id, uniprot_id = key
    pair_name = str(candidate["pair_name"])
    source = str(candidate["source"])
    records = (mechanism, preflight, replacement, lifecycle, candidate)
    preflight_status = _clean_text(
        _first_value(records, "preflight_status")
    )
    pilot_role = _clean_text(_first_value((pilot, mechanism), "pilot_role"))

    pair_status = _clean_text(
        _first_value((mechanism, lifecycle), "pair_status")
    )
    geometry_status = _clean_text(
        _first_value((mechanism, lifecycle), "geometry_status")
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

    model_match = _mechanism_model_match(mechanism)
    report_model_match = _optional_bool(
        mapped_confidence.get("confidence_model_match")
    )
    if model_match is None:
        model_match = report_model_match
    elif report_model_match is False:
        model_match = False

    pair_dir = root / "data/processed/pairs" / pair_name
    mapped = _mapped_confidence_from_report(mapped_confidence)
    if mapped is None and pair_dir.exists():
        mapped = _mapped_confidence_from_pair(pair_dir)
    mapped = mapped or {
        "mapped_plddt_min": None,
        "mapped_plddt_q10": None,
        "mapped_plddt_median": None,
        "longest_internal_below_70_length": None,
        "longest_internal_below_80_length": None,
        "terminal_only_below_80": None,
        "low_conf_evidence_available": False,
    }

    pae_strata, pae_available = _pae_from_pair(
        pair_dir,
        model_match=model_match is True,
    )
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
        ca_disagreement_p90=_optional_float(
            mechanism.get("ca_disagreement_p90")
        ),
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
        "ca_disagreement_median": _optional_float(
            mechanism.get("ca_disagreement_median")
        ),
        "ca_disagreement_p90": _optional_float(
            mechanism.get("ca_disagreement_p90")
        ),
        "ca_disagreement_max": _optional_float(
            mechanism.get("ca_disagreement_max")
        ),
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
        "evidence_assembly_reason": "assembled",
        "model_evidence_mismatch": model_match is False,
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
            _read_table(root / "reports/candidate_lifecycle.csv"),
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
        }
        try:
            rows.append(
                _assemble_candidate(
                    candidate,
                    root=root,
                    config=config,
                    preflight=_lookup(sources["preflight"], key),
                    lifecycle=_lookup(sources["lifecycle"], key),
                    replacement=_lookup(sources["replacement"], key),
                    pilot=_lookup(sources["pilot"], key),
                    mechanism=_lookup(sources["mechanism"], key),
                    mapped_confidence=_lookup(
                        sources["mapped_confidence"],
                        key,
                    ),
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
