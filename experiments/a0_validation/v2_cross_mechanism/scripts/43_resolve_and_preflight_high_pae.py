#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


INDEX_COLUMN_NAMES = {
    "screening_index",
    "candidate_index",
    "pilot_index",
    "sample_index",
    "index",
}
PAIR_COLUMN_HINTS = {
    "pair_id",
    "pair_key",
    "pair_name",
    "pair_dir",
    "pair_path",
    "protein_id",
    "sample_id",
}
REQUIRED_PAIR_ARTIFACTS = (
    "residue_mapping.parquet",
    "residue_geometry.parquet",
    "pair_qc.json",
    "pair_geometry_qc.json",
    "robust_pair_diagnostics.json",
)
OPTIONAL_PAIR_ARTIFACTS = (
    "pairwise_geometry.npz",
    "pairwise_strata.csv",
    "disagreement_segments.csv",
    "high_confidence_disagreement.csv",
)
MECHANISM_TERMS = (
    "high_pae",
    "high-pae",
    "pae_long_range",
    "long_range",
    "long-range",
    "mechanism_label",
    "primary_category",
)
SOURCE_PATH_KEY_TERMS = (
    "pdb_path",
    "pdb_file",
    "structure_path",
    "afdb_path",
    "afdb_model",
    "model_cif",
    "cif_path",
    "selected_model",
    "pae_path",
    "plddt_path",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def json_walk(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from json_walk(child, child_prefix)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            yield from json_walk(child, child_prefix)
    else:
        yield prefix, value


def normalize_index(value: Any) -> int | None:
    try:
        if isinstance(value, bool):
            return None
        number = float(value)
        if not math.isfinite(number) or int(number) != number:
            return None
        return int(number)
    except (TypeError, ValueError):
        return None


def safe_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table suffix: {path}")


def table_index_matches(
    path: Path,
    target_index: int,
) -> list[dict[str, Any]]:
    frame = read_table(path)
    results: list[dict[str, Any]] = []
    lower_to_original = {str(column).lower(): column for column in frame.columns}
    matching_index_columns = [
        lower_to_original[name]
        for name in INDEX_COLUMN_NAMES
        if name in lower_to_original
    ]
    if not matching_index_columns:
        return results

    for index_column in matching_index_columns:
        normalized = frame[index_column].map(normalize_index)
        matches = frame.loc[normalized == target_index]
        for row_number, row in matches.iterrows():
            pair_fields = {}
            for column in frame.columns:
                lower = str(column).lower()
                if (
                    lower in PAIR_COLUMN_HINTS
                    or "pair" in lower
                    or lower.endswith("_path")
                    or lower.endswith("_dir")
                ):
                    value = row[column]
                    if pd.notna(value):
                        pair_fields[str(column)] = str(value)
            results.append({
                "source_file": str(path),
                "source_type": "table",
                "index_column": str(index_column),
                "row_number": int(row_number) if isinstance(row_number, (int, np.integer)) else str(row_number),
                "pair_fields": pair_fields,
                "row_preview": {
                    str(column): (
                        None if pd.isna(row[column]) else str(row[column])
                    )
                    for column in list(frame.columns)[:30]
                },
            })
    return results


def json_index_matches(
    path: Path,
    target_index: int,
) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    flat = list(json_walk(data))
    matching_paths = []
    for key_path, value in flat:
        final_key = re.split(r"[.\[]", key_path)[-1].rstrip("]").lower()
        if final_key in INDEX_COLUMN_NAMES and normalize_index(value) == target_index:
            matching_paths.append(key_path)
    if not matching_paths:
        return []

    pair_fields = {}
    for key_path, value in flat:
        lower = key_path.lower()
        if (
            any(term in lower for term in ("pair_id", "pair_key", "pair_dir", "pair_path"))
            and isinstance(value, (str, int, float))
        ):
            pair_fields[key_path] = str(value)

    return [{
        "source_file": str(path),
        "source_type": "json",
        "matching_index_paths": matching_paths,
        "pair_fields": pair_fields,
    }]


def candidate_pair_dirs_from_match(
    match: dict[str, Any],
    project_root: Path,
) -> set[Path]:
    candidates: set[Path] = set()
    pair_root = project_root / "data/processed/pairs"
    values = list(match.get("pair_fields", {}).values())

    for value in values:
        raw = Path(str(value))
        possible = []
        if raw.is_absolute():
            possible.append(raw)
        else:
            possible.extend([
                project_root / raw,
                pair_root / raw,
            ])
        for path in possible:
            if path.is_dir():
                candidates.add(path.resolve())

        name = raw.name
        if name:
            direct = pair_root / name
            if direct.is_dir():
                candidates.add(direct.resolve())

    return candidates


def find_index_matches(
    project_root: Path,
    target_index: int,
    max_file_mb: float,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    roots = [
        project_root / "experiments/a0_validation",
        project_root / "data/processed",
    ]
    matches: list[dict[str, Any]] = []
    scan_errors: list[dict[str, str]] = []

    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if safe_size_mb(path) > max_file_mb:
                continue
            suffix = path.suffix.lower()
            if suffix not in {".json", ".tsv", ".csv", ".parquet"}:
                continue
            try:
                if suffix == ".json":
                    matches.extend(json_index_matches(path, target_index))
                else:
                    matches.extend(table_index_matches(path, target_index))
            except Exception as exc:
                scan_errors.append({
                    "path": str(path),
                    "error": f"{type(exc).__name__}: {exc}",
                })
    return matches, scan_errors


def find_pair_dirs_by_literal_name(
    project_root: Path,
    target_index: int,
) -> set[Path]:
    pair_root = project_root / "data/processed/pairs"
    candidates: set[Path] = set()
    if not pair_root.is_dir():
        return candidates

    pattern = re.compile(rf"(^|[^0-9]){target_index}([^0-9]|$)")
    for directory in pair_root.iterdir():
        if directory.is_dir() and pattern.search(directory.name):
            candidates.add(directory.resolve())
    return candidates


def inspect_json_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    flat = list(json_walk(data))
    mechanism_hits = []
    source_path_hits = []

    for key_path, value in flat:
        rendered = str(value)
        combined = f"{key_path} {rendered}".lower()
        if any(term in combined for term in MECHANISM_TERMS):
            mechanism_hits.append({
                "key_path": key_path,
                "value": value,
            })
        if any(term in key_path.lower() for term in SOURCE_PATH_KEY_TERMS):
            source_path_hits.append({
                "key_path": key_path,
                "value": value,
                "path_exists": (
                    Path(str(value)).exists()
                    if isinstance(value, str) and value
                    else None
                ),
            })

    return {
        "path": str(path),
        "exists": True,
        "sha256": sha256_file(path),
        "mechanism_hits": mechanism_hits,
        "source_path_hits": source_path_hits,
    }


def inspect_table(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}

    frame = read_table(path)
    columns = [str(column) for column in frame.columns]
    lower_columns = {column.lower(): column for column in columns}

    position_columns = [
        column for column in columns
        if any(term in column.lower() for term in (
            "uniprot",
            "mapped_order",
            "auth_seq",
            "label_seq",
            "position",
            "residue_number",
        ))
    ]
    residue_columns = [
        column for column in columns
        if any(term in column.lower() for term in (
            "residue_one_letter",
            "amino",
            "aa_",
            "sequence",
        ))
    ]
    geometry_columns = [
        column for column in columns
        if any(term in column.lower() for term in (
            "aligned_ca_distance",
            "distance",
            "rmsd",
            "disagreement",
        ))
    ]
    confidence_columns = [
        column for column in columns
        if any(term in column.lower() for term in (
            "plddt",
            "pae",
            "confidence",
        ))
    ]

    numeric_summary = {}
    # Boolean indicator columns such as `is_high_confidence` are valid
    # confidence fields, but pandas/numpy cannot compute quantiles directly
    # on boolean arrays because that requires subtraction/interpolation.
    # Convert every numeric-like field to float before summary statistics.
    for column in dict.fromkeys(geometry_columns + confidence_columns):
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if len(values):
            values = values.astype(float)
            numeric_summary[column] = {
                "count": int(len(values)),
                "min": float(values.min()),
                "median": float(values.median()),
                "mean": float(values.mean()),
                "p90": float(values.quantile(0.9)),
                "max": float(values.max()),
            }

    return {
        "path": str(path),
        "exists": True,
        "sha256": sha256_file(path),
        "rows": int(len(frame)),
        "columns": columns,
        "position_columns": position_columns,
        "residue_columns": residue_columns,
        "geometry_columns": geometry_columns,
        "confidence_columns": confidence_columns,
        "numeric_summary": numeric_summary,
    }


def inspect_npz(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}

    arrays = {}
    pae_arrays = {}
    with np.load(path, allow_pickle=True) as data:
        for key in data.files:
            array = np.asarray(data[key])
            arrays[key] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
            if "pae" in key.lower() and np.issubdtype(array.dtype, np.number):
                values = np.asarray(array, dtype=float)
                finite = values[np.isfinite(values)]
                if finite.size:
                    pae_arrays[key] = {
                        "shape": list(values.shape),
                        "min": float(finite.min()),
                        "median": float(np.median(finite)),
                        "mean": float(finite.mean()),
                        "p90": float(np.quantile(finite, 0.9)),
                        "max": float(finite.max()),
                    }

    return {
        "path": str(path),
        "exists": True,
        "sha256": sha256_file(path),
        "arrays": arrays,
        "pae_arrays": pae_arrays,
    }


def search_mechanism_in_text_files(pair_dir: Path) -> list[dict[str, Any]]:
    hits = []
    for path in pair_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in {".json", ".csv", ".tsv"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        lower = text.lower()
        matched_terms = sorted({term for term in MECHANISM_TERMS if term in lower})
        if matched_terms:
            hits.append({
                "path": str(path),
                "matched_terms": matched_terms,
            })
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve and preflight a high-PAE A0-dev sample.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--screening-index", type=int, default=36)
    parser.add_argument("--backup-index", type=int, default=115)
    parser.add_argument(
        "--expected-mechanism",
        default="high_pae_long_range_candidate",
    )
    parser.add_argument("--max-scan-file-mb", type=float, default=64.0)
    parser.add_argument(
        "--pair-dir",
        type=Path,
        default=None,
        help="Optional explicit pair directory. Auto-resolution is used when omitted.",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_root = (
        project_root
        / "experiments/a0_validation/v2_cross_mechanism"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    matches, scan_errors = find_index_matches(
        project_root,
        args.screening_index,
        args.max_scan_file_mb,
    )

    candidate_pair_dirs: set[Path] = set()
    for match in matches:
        candidate_pair_dirs |= candidate_pair_dirs_from_match(match, project_root)
    candidate_pair_dirs |= find_pair_dirs_by_literal_name(
        project_root,
        args.screening_index,
    )

    if args.pair_dir is not None:
        explicit = args.pair_dir
        if not explicit.is_absolute():
            explicit = project_root / explicit
        candidate_pair_dirs.add(explicit.resolve())

    # Retain only directories under data/processed/pairs that actually exist.
    candidate_pair_dirs = {
        path for path in candidate_pair_dirs
        if path.is_dir()
    }

    resolution_errors = []
    if len(candidate_pair_dirs) == 0:
        resolution_errors.append(
            f"No pair directory could be resolved for screening index "
            f"{args.screening_index}."
        )
    elif len(candidate_pair_dirs) > 1:
        resolution_errors.append(
            f"Multiple pair directories resolved for screening index "
            f"{args.screening_index}: "
            f"{[str(path) for path in sorted(candidate_pair_dirs)]}"
        )

    pair_dir = (
        next(iter(candidate_pair_dirs))
        if len(candidate_pair_dirs) == 1
        else None
    )

    resolution = {
        "screening_index": args.screening_index,
        "backup_index": args.backup_index,
        "expected_mechanism": args.expected_mechanism,
        "index_matches": matches,
        "scan_errors": scan_errors,
        "candidate_pair_directories": [
            str(path) for path in sorted(candidate_pair_dirs)
        ],
        "resolved_pair_directory": str(pair_dir) if pair_dir else None,
        "resolution_errors": resolution_errors,
        "resolution_pass": pair_dir is not None and not resolution_errors,
    }

    manifest_path = (
        output_root
        / f"manifests/index{args.screening_index}_target_resolution.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(resolution, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    preflight_errors = list(resolution_errors)
    pair_artifacts = {}
    json_inspections = {}
    table_inspections = {}
    npz_inspections = {}
    mechanism_text_hits = []
    source_path_hits = []

    if pair_dir is not None:
        for filename in REQUIRED_PAIR_ARTIFACTS + OPTIONAL_PAIR_ARTIFACTS:
            path = pair_dir / filename
            pair_artifacts[filename] = {
                "path": str(path),
                "exists": path.is_file(),
            }

        missing_required = [
            filename
            for filename in REQUIRED_PAIR_ARTIFACTS
            if not pair_artifacts[filename]["exists"]
        ]
        if missing_required:
            preflight_errors.append(
                f"Missing required pair artifacts: {missing_required}"
            )

        for filename in (
            "pair_qc.json",
            "pair_geometry_qc.json",
            "robust_pair_diagnostics.json",
        ):
            path = pair_dir / filename
            try:
                record = inspect_json_file(path)
            except Exception as exc:
                record = {
                    "path": str(path),
                    "exists": path.is_file(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                preflight_errors.append(
                    f"Could not inspect {filename}: {type(exc).__name__}: {exc}"
                )
            json_inspections[filename] = record
            source_path_hits.extend(record.get("source_path_hits", []))

        for filename in (
            "residue_mapping.parquet",
            "residue_geometry.parquet",
            "pairwise_strata.csv",
            "disagreement_segments.csv",
            "high_confidence_disagreement.csv",
        ):
            path = pair_dir / filename
            if not path.is_file():
                continue
            try:
                table_inspections[filename] = inspect_table(path)
            except Exception as exc:
                table_inspections[filename] = {
                    "path": str(path),
                    "exists": True,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                preflight_errors.append(
                    f"Could not inspect {filename}: {type(exc).__name__}: {exc}"
                )

        npz_path = pair_dir / "pairwise_geometry.npz"
        if npz_path.is_file():
            try:
                npz_inspections["pairwise_geometry.npz"] = inspect_npz(npz_path)
            except Exception as exc:
                npz_inspections["pairwise_geometry.npz"] = {
                    "path": str(npz_path),
                    "exists": True,
                    "error": f"{type(exc).__name__}: {exc}",
                }

        mechanism_text_hits = search_mechanism_in_text_files(pair_dir)

    mapping_record = table_inspections.get("residue_mapping.parquet", {})
    geometry_record = table_inspections.get("residue_geometry.parquet", {})

    mapping_ready = bool(
        mapping_record.get("exists")
        and mapping_record.get("rows", 0) > 0
        and mapping_record.get("position_columns")
    )
    geometry_ready = bool(
        geometry_record.get("exists")
        and geometry_record.get("rows", 0) > 0
        and (
            geometry_record.get("geometry_columns")
            or geometry_record.get("confidence_columns")
        )
    )

    mechanism_json_hits = []
    for filename, record in json_inspections.items():
        for hit in record.get("mechanism_hits", []):
            mechanism_json_hits.append({
                "source": filename,
                **hit,
            })

    pae_numeric_evidence = []
    for filename, record in table_inspections.items():
        for column, summary in record.get("numeric_summary", {}).items():
            if "pae" in column.lower():
                pae_numeric_evidence.append({
                    "source": filename,
                    "column": column,
                    "summary": summary,
                })
    for filename, record in npz_inspections.items():
        for key, summary in record.get("pae_arrays", {}).items():
            pae_numeric_evidence.append({
                "source": filename,
                "array": key,
                "summary": summary,
            })

    mechanism_evidence_found = bool(
        mechanism_json_hits
        or mechanism_text_hits
        or pae_numeric_evidence
    )
    source_paths_resolved = any(
        hit.get("path_exists") is True
        for hit in source_path_hits
    )

    if pair_dir is not None and not mapping_ready:
        preflight_errors.append(
            "Residue mapping is not ready: no readable mapped rows or position columns."
        )
    if pair_dir is not None and not geometry_ready:
        preflight_errors.append(
            "Residue geometry/confidence data is not ready."
        )
    if pair_dir is not None and not mechanism_evidence_found:
        preflight_errors.append(
            "No high-PAE mechanism label or numeric PAE evidence was found."
        )
    if pair_dir is not None and not source_paths_resolved:
        preflight_errors.append(
            "No existing PDB/AFDB source path could be resolved from pair diagnostics."
        )

    preflight_pass = not preflight_errors

    audit = {
        "screening_index": args.screening_index,
        "backup_index": args.backup_index,
        "expected_mechanism": args.expected_mechanism,
        "resolved_pair_directory": str(pair_dir) if pair_dir else None,
        "pair_artifacts": pair_artifacts,
        "json_inspections": json_inspections,
        "table_inspections": table_inspections,
        "npz_inspections": npz_inspections,
        "mechanism_evidence": {
            "json_hits": mechanism_json_hits,
            "text_file_hits": mechanism_text_hits,
            "pae_numeric_evidence": pae_numeric_evidence,
            "evidence_found": mechanism_evidence_found,
        },
        "source_path_hits": source_path_hits,
        "readiness": {
            "mapping_ready": mapping_ready,
            "geometry_ready": geometry_ready,
            "mechanism_evidence_found": mechanism_evidence_found,
            "source_paths_resolved": source_paths_resolved,
        },
        "errors": preflight_errors,
        "preflight_pass": preflight_pass,
        "next_stage": (
            "V2B_build_generalized_paired_backbone_inputs"
            if preflight_pass
            else "resolve_primary_sample_or_review_backup_index_115"
        ),
    }

    metrics_path = (
        output_root
        / f"metrics/index{args.screening_index}_high_pae_preflight.json"
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    markdown_path = (
        output_root
        / f"V2A_INDEX{args.screening_index}_HIGH_PAE_PREFLIGHT.md"
    )
    lines = [
        f"# A0-V2A Index {args.screening_index} High-PAE Preflight",
        "",
        f"- Expected mechanism: `{args.expected_mechanism}`",
        f"- Resolved pair directory: `{pair_dir}`",
        f"- Mapping ready: `{mapping_ready}`",
        f"- Geometry/confidence ready: `{geometry_ready}`",
        f"- Mechanism evidence found: `{mechanism_evidence_found}`",
        f"- Source paths resolved: `{source_paths_resolved}`",
        f"- Preflight pass: `{preflight_pass}`",
        f"- Next stage: `{audit['next_stage']}`",
        "",
        "## Required artifacts",
        "",
    ]
    for filename in REQUIRED_PAIR_ARTIFACTS:
        record = pair_artifacts.get(filename, {})
        lines.append(f"- `{filename}`: `{record.get('exists', False)}`")

    lines += [
        "",
        "## Mapping summary",
        "",
        f"- `{mapping_record}`",
        "",
        "## Geometry/confidence summary",
        "",
        f"- `{geometry_record}`",
        "",
        "## High-PAE evidence",
        "",
        f"- JSON hits: `{mechanism_json_hits}`",
        f"- Text-file hits: `{mechanism_text_hits}`",
        f"- Numeric PAE evidence: `{pae_numeric_evidence}`",
        "",
        "## Source path evidence",
        "",
    ]
    for hit in source_path_hits:
        lines.append(f"- `{hit}`")

    if preflight_errors:
        lines += ["", "## Errors", ""]
        lines.extend(f"- {error}" for error in preflight_errors)

    lines += [
        "",
        "## Interpretation boundary",
        "",
        "本阶段只验证Index 36是否具备跨机制复制的输入条件。"
        "它不重新分类机制，也不在主样本失败时自动切换到Index 115。",
        "",
    ]
    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    print("Screening index:", args.screening_index)
    print("Resolved pair directory:", pair_dir)
    print("Candidate pair directories:", resolution["candidate_pair_directories"])
    print("Mapping ready:", mapping_ready)
    print("Geometry/confidence ready:", geometry_ready)
    print("Mechanism evidence found:", mechanism_evidence_found)
    print("Source paths resolved:", source_paths_resolved)
    print("Errors:", preflight_errors)
    print("Preflight pass:", preflight_pass)
    print("Next stage:", audit["next_stage"])
    print("Wrote:", manifest_path)
    print("Wrote:", metrics_path)
    print("Wrote:", markdown_path)
    return 0 if preflight_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
