from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from dual_uq.confidence import load_plddt
from dual_uq.mapped_confidence import (
    MappedConfidenceSegment,
    MappedConfidenceSummary,
    build_mapped_confidence_residue_table,
    summarize_mapped_confidence,
    validate_mapped_confidence_input,
)
from dual_uq.sifts import parse_sifts_residue_mapping
from dual_uq.structure_io import load_chain_ca_table

OUTPUT_COLUMNS = [
    "screening_index",
    "pdb_id",
    "chain_id",
    "uniprot_id",
    "pair_name",
    "preflight_status",
    "selected_afdb_model_entity_id",
    "selected_afdb_version",
    "afdb_fragment_start",
    "afdb_fragment_end",
    "plddt_model_entity_id",
    "confidence_model_match",
    "plddt_bfactor_match",
    "plddt_bfactor_max_abs_delta",
    "mapped_start",
    "mapped_end",
    "mapped_position_count",
    "scorable_position_count",
    "scorable_fraction",
    "mapped_plddt_min",
    "mapped_plddt_q10",
    "mapped_plddt_median",
    "below_70_position_count",
    "below_80_position_count",
    "below_70_segment_count",
    "below_80_segment_count",
    "longest_below_70_start",
    "longest_below_70_end",
    "longest_below_70_length",
    "longest_below_70_is_internal",
    "longest_below_80_start",
    "longest_below_80_end",
    "longest_below_80_length",
    "longest_below_80_is_internal",
    "longest_internal_below_70_start",
    "longest_internal_below_70_end",
    "longest_internal_below_70_length",
    "longest_internal_below_80_start",
    "longest_internal_below_80_end",
    "longest_internal_below_80_length",
    "distance_to_n_terminus",
    "distance_to_c_terminus",
    "terminal_only_below_80",
    "missing_ca_interruption_count",
    "unmapped_position_interruption_count",
    "is_low_conf_local",
    "scoring_status",
    "scoring_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score mapped-region pLDDT segments from local caches only."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--replacement-pool",
        default="data/manifests/lower_conf_replacement_pool.tsv",
    )
    parser.add_argument(
        "--output",
        default="reports/replacement_mapped_confidence.csv",
    )
    parser.add_argument("--only", type=int, action="append")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--terminal-buffer", type=int, default=5)
    parser.add_argument("--minimum-run-length", type=int, default=5)
    return parser.parse_args()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _validate_candidate_keys(candidates: pd.DataFrame) -> None:
    composite_key = [
        "screening_index",
        "pdb_id",
        "chain_id",
        "uniprot_id",
    ]
    missing = sorted(set(composite_key) - set(candidates.columns))
    if missing:
        raise ValueError(f"candidate key columns missing: {missing}")
    if candidates[composite_key].isna().any(axis=None):
        raise ValueError("candidate keys must be complete and unique")
    screening_index = pd.to_numeric(
        candidates["screening_index"],
        errors="coerce",
    )
    if (
        screening_index.isna().any()
        or not np.isfinite(screening_index).all()
        or screening_index.mod(1).ne(0).any()
        or candidates["screening_index"].duplicated().any()
        or candidates.duplicated(composite_key).any()
    ):
        raise ValueError("candidate keys must be complete and unique")


def _optional_int(value: Any) -> int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or not number.is_integer():
        return None
    return int(number)


def _required_int(value: Any, reason: str) -> int:
    converted = _optional_int(value)
    if converted is None:
        raise ValueError(reason)
    return converted


def _clean_optional_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    cleaned = str(value).strip()
    return None if cleaned in {"", ".", "?"} else cleaned


def _mapping_path(root: Path, pdb_id: str) -> Path:
    base = root / "data/raw/mappings"
    candidates = [
        base / f"{pdb_id.lower()}.xml.gz",
        base / f"{pdb_id.lower()}.xml",
    ]
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError(
        f"missing_local_sifts_mapping:{pdb_id.lower()}"
    )


def _selected_prediction(
    metadata: pd.DataFrame,
    *,
    uniprot_id: str,
    model_entity_id: str,
    version: int,
    fragment_start: int,
    fragment_end: int,
) -> dict[str, Any]:
    rows = metadata.loc[
        metadata["uniprot_id"].astype(str).str.upper().eq(uniprot_id.upper())
    ]
    if len(rows) != 1:
        raise LookupError("missing_or_duplicate_local_afdb_metadata")
    records = json.loads(str(rows.iloc[0]["prediction_records_json"]))
    if not isinstance(records, list) or not all(
        isinstance(record, dict) for record in records
    ):
        raise TypeError("malformed_local_afdb_metadata")
    matches = [
        record
        for record in records
        if str(record.get("modelEntityId") or record.get("entryId"))
        == model_entity_id
        and int(record.get("latestVersion") or 0) == version
    ]
    if len(matches) != 1:
        raise LookupError("selected_afdb_model_not_in_local_metadata")
    prediction = matches[0]
    start = int(
        prediction.get("uniprotStart", prediction.get("sequenceStart"))
    )
    end = int(prediction.get("uniprotEnd", prediction.get("sequenceEnd")))
    if (start, end) != (fragment_start, fragment_end):
        raise ValueError("selected_afdb_fragment_metadata_mismatch")
    return prediction


def _selected_artifact_paths(
    root: Path,
    *,
    uniprot_id: str,
    model_entity_id: str,
    version: int,
    prediction: dict[str, Any],
) -> tuple[Path, Path]:
    expected = {
        "cifUrl": f"{model_entity_id}-model_v{version}.cif",
        "plddtDocUrl": (
            f"{model_entity_id}-confidence_v{version}.json"
        ),
    }
    names: dict[str, str] = {}
    for field, expected_name in expected.items():
        value = prediction.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"selected_afdb_metadata_missing_{field}")
        name = Path(urlparse(value).path).name
        if name != expected_name:
            raise ValueError(
                f"selected_afdb_artifact_version_mismatch:{field}"
            )
        names[field] = name
    model_dir = root / "data/raw/afdb" / uniprot_id / model_entity_id
    return model_dir / names["cifUrl"], model_dir / names["plddtDocUrl"]


def _validate_model_bfactor(
    model_path: Path,
    plddt: np.ndarray,
    *,
    model_entity_id: str,
    fragment_start: int,
    fragment_end: int,
    tolerance: float = 0.01,
) -> float:
    cif_data = MMCIF2Dict(str(model_path))
    entry_ids = cif_data.get("_entry.id", [])
    entry_ids = entry_ids if isinstance(entry_ids, list) else [entry_ids]
    if model_entity_id not in {str(value).strip() for value in entry_ids}:
        raise ValueError("afdb_model_cif_identity_mismatch")

    model = load_chain_ca_table(model_path, chain_id="A")
    local_positions = pd.to_numeric(model["label_seq_id"], errors="coerce")
    if local_positions.isna().any():
        raise ValueError("afdb_model_missing_label_seq_id")
    local_positions = local_positions.astype(int)
    expected_length = fragment_end - fragment_start + 1
    expected = np.arange(1, expected_length + 1)
    actual = np.sort(local_positions.unique())
    if len(model) != expected_length or not np.array_equal(actual, expected):
        raise ValueError("afdb_model_fragment_numbering_mismatch")

    ordered = model.assign(_local_position=local_positions).sort_values(
        "_local_position",
        kind="mergesort",
    )
    bfactor = ordered["bfactor"].to_numpy(dtype=float)
    if (
        len(bfactor) != len(plddt)
        or not np.isfinite(bfactor).all()
        or ((bfactor < 0.0) | (bfactor > 100.0)).any()
    ):
        raise ValueError("invalid_afdb_model_bfactor")
    max_delta = float(np.max(np.abs(bfactor - plddt)))
    if max_delta > tolerance:
        raise ValueError("plddt_json_bfactor_mismatch")
    return max_delta


def _segment_fields(
    prefix: str,
    segment: MappedConfidenceSegment | None,
    *,
    include_internal: bool = True,
) -> dict[str, Any]:
    values: dict[str, Any] = {
        f"{prefix}_start": None if segment is None else segment.start_uniprot,
        f"{prefix}_end": None if segment is None else segment.end_uniprot,
        f"{prefix}_length": 0 if segment is None else segment.residue_count,
    }
    if include_internal:
        values[f"{prefix}_is_internal"] = (
            False if segment is None else segment.is_internal
        )
    return values


def _summary_fields(
    summary: MappedConfidenceSummary,
    *,
    minimum_run_length: int,
) -> dict[str, Any]:
    longest_80 = summary.longest_below_80_segment
    terminal_only = (
        longest_80 is not None
        and longest_80.residue_count >= minimum_run_length
        and longest_80.is_terminal
        and (
            summary.longest_internal_below_80_length
            < minimum_run_length
        )
    )
    return {
        "mapped_start": summary.mapped_start,
        "mapped_end": summary.mapped_end,
        "mapped_position_count": summary.mapped_position_count,
        "scorable_position_count": summary.scorable_position_count,
        "scorable_fraction": summary.scorable_fraction,
        "mapped_plddt_min": summary.mapped_plddt_min,
        "mapped_plddt_q10": summary.mapped_plddt_q10,
        "mapped_plddt_median": summary.mapped_plddt_median,
        "below_70_position_count": summary.below_70_position_count,
        "below_80_position_count": summary.below_80_position_count,
        "below_70_segment_count": summary.below_70_segment_count,
        "below_80_segment_count": summary.below_80_segment_count,
        **_segment_fields(
            "longest_below_70",
            summary.longest_below_70_segment,
        ),
        **_segment_fields(
            "longest_below_80",
            summary.longest_below_80_segment,
        ),
        **_segment_fields(
            "longest_internal_below_70",
            summary.longest_internal_below_70_segment,
            include_internal=False,
        ),
        **_segment_fields(
            "longest_internal_below_80",
            summary.longest_internal_below_80_segment,
            include_internal=False,
        ),
        "distance_to_n_terminus": (
            None
            if longest_80 is None
            else longest_80.distance_to_mapped_n_terminus
        ),
        "distance_to_c_terminus": (
            None
            if longest_80 is None
            else longest_80.distance_to_mapped_c_terminus
        ),
        "terminal_only_below_80": terminal_only,
        "missing_ca_interruption_count": (
            summary.missing_ca_interruption_count
        ),
        "unmapped_position_interruption_count": (
            summary.unmapped_position_interruption_count
        ),
        "is_low_conf_local": summary.is_low_conf_local,
        "scoring_status": summary.scoring_status,
        "scoring_reason": summary.scoring_reason,
    }


def _base_output(row: Any) -> dict[str, Any]:
    model_id = _clean_optional_text(row.selected_afdb_model_entity_id)
    return {
        "screening_index": int(row.screening_index),
        "pdb_id": str(row.pdb_id).lower(),
        "chain_id": str(row.chain_id),
        "uniprot_id": str(row.uniprot_id).upper(),
        "pair_name": (
            f"{str(row.pdb_id).lower()}_{row.chain_id}__"
            f"{str(row.uniprot_id).upper()}"
        ),
        "preflight_status": str(row.preflight_status),
        "selected_afdb_model_entity_id": model_id,
        "selected_afdb_version": _optional_int(row.afdb_version),
        "afdb_fragment_start": _optional_int(row.afdb_fragment_start),
        "afdb_fragment_end": _optional_int(row.afdb_fragment_end),
        "plddt_model_entity_id": None,
        "confidence_model_match": False,
        "plddt_bfactor_match": False,
        "plddt_bfactor_max_abs_delta": None,
    }


def _score_candidate(
    row: Any,
    *,
    root: Path,
    metadata: pd.DataFrame,
    terminal_buffer: int,
    minimum_run_length: int,
) -> dict[str, Any]:
    result = _base_output(row)
    try:
        model_id = result["selected_afdb_model_entity_id"]
        if model_id is None:
            raise ValueError("missing_selected_afdb_model_entity_id")
        version = _required_int(
            result["selected_afdb_version"],
            "invalid_selected_afdb_version",
        )
        fragment_start = _required_int(
            result["afdb_fragment_start"],
            "invalid_selected_afdb_fragment_start",
        )
        fragment_end = _required_int(
            result["afdb_fragment_end"],
            "invalid_selected_afdb_fragment_end",
        )
        prediction = _selected_prediction(
            metadata,
            uniprot_id=result["uniprot_id"],
            model_entity_id=model_id,
            version=version,
            fragment_start=fragment_start,
            fragment_end=fragment_end,
        )
        plddt_model_id = str(
            prediction.get("modelEntityId") or prediction.get("entryId")
        )
        result["plddt_model_entity_id"] = plddt_model_id
        result["confidence_model_match"] = plddt_model_id == model_id
        if not result["confidence_model_match"]:
            raise ValueError("afdb_model_entity_id_mismatch")

        model_path, plddt_path = _selected_artifact_paths(
            root,
            uniprot_id=result["uniprot_id"],
            model_entity_id=model_id,
            version=version,
            prediction=prediction,
        )
        if not model_path.exists() or not plddt_path.exists():
            raise FileNotFoundError("missing_local_selected_afdb_artifact")

        expected_length = fragment_end - fragment_start + 1
        plddt = load_plddt(
            plddt_path,
            expected_length=expected_length,
        )
        if (
            not np.isfinite(plddt).all()
            or ((plddt < 0.0) | (plddt > 100.0)).any()
        ):
            raise ValueError("invalid_selected_afdb_plddt")
        max_delta = _validate_model_bfactor(
            model_path,
            plddt,
            model_entity_id=model_id,
            fragment_start=fragment_start,
            fragment_end=fragment_end,
        )
        result["plddt_bfactor_match"] = True
        result["plddt_bfactor_max_abs_delta"] = max_delta

        pdb_path = root / "data/raw/pdb" / f"{result['pdb_id']}.cif"
        if not pdb_path.exists():
            raise FileNotFoundError("missing_local_pdb_mmcif")
        mapping = parse_sifts_residue_mapping(
            _mapping_path(root, result["pdb_id"]),
            chain_id=result["chain_id"],
            uniprot_id=result["uniprot_id"],
        )
        pdb_ca = load_chain_ca_table(pdb_path, result["chain_id"])
        residue_table = build_mapped_confidence_residue_table(
            mapping,
            pdb_ca,
            plddt,
            fragment_start=fragment_start,
            fragment_end=fragment_end,
        )
        validation = validate_mapped_confidence_input(
            residue_table,
            selected_model_entity_id=model_id,
            plddt_model_entity_id=plddt_model_id,
            fragment_start=fragment_start,
            fragment_end=fragment_end,
        )
        if not validation.valid or validation.residue_table is None:
            raise ValueError(validation.reason)
        summary = summarize_mapped_confidence(
            validation.residue_table,
            terminal_buffer=terminal_buffer,
            minimum_internal_run=minimum_run_length,
        )
        result.update(
            _summary_fields(
                summary,
                minimum_run_length=minimum_run_length,
            )
        )
    except (
        FileNotFoundError,
        AttributeError,
        IndexError,
        KeyError,
        LookupError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        result.update(
            {
                "is_low_conf_local": None,
                "scoring_status": "failed",
                "scoring_reason": f"{type(exc).__name__}:{exc}",
            }
        )
    return result


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    pool_path = _resolve(root, args.replacement_pool)
    output_path = _resolve(root, args.output)
    pool = pd.read_csv(pool_path, sep="\t")
    passed = pool.loc[pool["preflight_status"].eq("pass_full_length")].copy()
    _validate_candidate_keys(passed)
    if args.only:
        requested = set(args.only)
        unknown = requested - set(passed["screening_index"].astype(int))
        if unknown:
            raise ValueError(
                f"--only indices are not pass_full_length replacements: "
                f"{sorted(unknown)}"
            )
        passed = passed.loc[
            passed["screening_index"].astype(int).isin(requested)
        ].copy()
    passed = passed.sort_values("screening_index", kind="mergesort")
    metadata = pd.read_parquet(
        root
        / "data/processed/replacements/afdb_canonical_length_cache.parquet"
    )

    rows: list[dict[str, Any]] = []
    if args.force:
        print("Force mode: recomputing selected rows from local caches.", flush=True)
    for row in passed.itertuples(index=False):
        scored = _score_candidate(
            row,
            root=root,
            metadata=metadata,
            terminal_buffer=args.terminal_buffer,
            minimum_run_length=args.minimum_run_length,
        )
        rows.append(scored)
        print(
            f"[{scored['screening_index']}] {scored['pair_name']} "
            f"{scored['scoring_status']}: {scored['scoring_reason']}",
            flush=True,
        )

    output = pd.DataFrame(rows).reindex(columns=OUTPUT_COLUMNS)
    if output["screening_index"].duplicated().any():
        raise ValueError("output screening_index values must be unique")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    output.to_csv(temporary, index=False)
    temporary.replace(output_path)
    print(f"Saved {len(output)} rows: {output_path}", flush=True)


if __name__ == "__main__":
    main()
