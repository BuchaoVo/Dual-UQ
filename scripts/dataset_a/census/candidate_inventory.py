"""Build the Dataset-A metadata-only candidate inventory.

This script is deliberately offline and read-only with respect to scientific
inputs.  It inventories existing local artifacts, writes planning reports,
and never runs P0/P1/P2 or changes candidate admission.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_SOURCE = Path("data/processed/discovery/discovered_candidates.parquet")
OUTPUT_DIR = Path("reports/dataset_a_scale")
EXPECTED_TOTAL = 220
EXPECTED_ELIGIBLE = 213
INDEXING_POLICY_VERSION = "inventory-source-order-v1"
BATCH1_TARGET = 48

SAMPLING_PRIORS = (
    "easy_control_prior",
    "low_confidence_local_prior",
    "high_pae_long_range_prior",
    "state_disagreement_prior",
    "uncertain_or_unclassified",
)
POSITIVE_PRIOR_PRECEDENCE = (
    "state_disagreement_prior",
    "high_pae_long_range_prior",
    "low_confidence_local_prior",
    "easy_control_prior",
)
PRIOR_RULES = {
    "easy_control_prior": {
        "primary_category": "easy_control",
        "label_field": "is_easy_control",
        "evidence_field": "easy_control_evidence_available",
        "required_local": ("pdb_structure", "afdb_model", "afdb_pae", "residue_mapping"),
    },
    "low_confidence_local_prior": {
        "primary_category": "low_confidence_local",
        "label_field": "is_low_conf_local",
        "evidence_field": "low_conf_evidence_available",
        "required_local": ("pdb_structure", "afdb_model", "afdb_plddt", "residue_mapping"),
    },
    "high_pae_long_range_prior": {
        "primary_category": "high_pae_long_range",
        "label_field": "is_high_pae_long_range",
        "evidence_field": "high_pae_evidence_available",
        "required_local": ("pdb_structure", "afdb_model", "afdb_pae", "residue_mapping"),
    },
    "state_disagreement_prior": {
        "primary_category": "high_confidence_state_disagreement",
        "label_field": "is_high_conf_state_disagreement",
        "evidence_field": "state_disagreement_evidence_available",
        "required_local": ("pdb_structure", "afdb_model", "residue_mapping"),
    },
}
SAMPLING_PRIOR_DISCLAIMER = (
    "sampling_stratum_prior is a non-inferential sampling aid; it is not a "
    "formal P6 mechanism label and does not change candidate admission."
)
EVIDENCE_POLICY = {
    "null_policy": "Unavailable evidence remains null; availability checks remain false.",
    "identity_source_available": (
        "True only when local pair-QC or Round-1 provides a numeric sequence-identity value; "
        "canonical discovery identity is tracked separately."
    ),
    "canonical_uniprot_length": (
        "Historical preflight/lifecycle canonical length, then pair-QC length, then an explicit "
        "AFDB interval starting at 1 whose sequence length equals its end."
    ),
    "mapping_fields": "Existing pair-QC and residue_mapping.parquet only.",
    "fragment_fields": "Existing local AFDB metadata and cached preflight intervals only.",
    "PDB_CA_coverage_proxy": "Existing preflight/lifecycle value, then pair_geometry_qc.",
    "global_pLDDT_proxy": "Canonical discovery-table afdb_global_plddt.",
    "sampling_prior_positive_evidence": (
        "A non-unknown sampling prior requires a complete, quality-passing historical A0 "
        "classification, its explicit positive label/evidence flag, and the required local "
        "PDB/AFDB/PAE-or-pLDDT/mapping provenance. Global pLDDT alone never establishes a prior."
    ),
    "low_confidence_fraction_proxy": (
        "Local AFDB metadata fractionPlddtVeryLow + fractionPlddtLow when unambiguous."
    ),
    "long_range_PAE_proxy": "Maximum existing A0-summary long-range q90; never recomputed here.",
    "existing_PDB_AFDB_disagreement_proxy": (
        "Existing pair_geometry_qc median_aligned_ca_distance; never recomputed here."
    ),
    "ligand_annotation_available": "Presence of explicit local mmCIF _pdbx_nonpoly_scheme.",
    "interface_annotation_available": (
        "Null because no dedicated local interface-annotation artifact was identified."
    ),
}

INVENTORY_COLUMNS = (
    "candidate_index",
    "canonical_source_row",
    "polymer_entity_id",
    "pair_id",
    "historical_screening_index",
    "PDB",
    "chain",
    "UniProt",
    "sequence_cluster",
    "protein_family_if_available",
    "canonical_uniprot_length",
    "mapped_length",
    "mapping_coverage",
    "identity_source_available",
    "mapping_available",
    "pair_qc_status",
    "local_PDB_available",
    "local_AFDB_available",
    "local_PAE_available",
    "local_pLDDT_available",
    "fragment_metadata_available",
    "fragment_count",
    "single_fragment_full_coverage_possible",
    "PDB_CA_coverage_proxy",
    "missing_residue_proxy",
    "global_pLDDT_proxy",
    "low_confidence_fraction_proxy",
    "long_range_PAE_proxy",
    "existing_PDB_AFDB_disagreement_proxy",
    "ligand_annotation_available",
    "interface_annotation_available",
    "known_sequence_mismatch_flag",
    "known_provenance_issue",
    "estimated_P0_readiness",
    "estimated_P1_risk",
    "local_data_complete",
    "missing_local_inputs",
    "sampling_stratum_prior",
    "sampling_stratum_prior_source",
    "prior_evidence_status",
    "prior_evidence_sources",
    "prior_observability_complete",
    "supported_sampling_priors",
    "round1_member",
)


class InventoryInvariantError(RuntimeError):
    """Raised when the canonical universe cannot be identified unambiguously."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _optional_float(value: Any) -> float | None:
    if _is_missing(value) or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _optional_int(value: Any) -> int | None:
    number = _optional_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _optional_str(value: Any) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    return text or None


def _optional_bool(value: Any) -> bool | None:
    if _is_missing(value) or value == "":
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def _pair_id(pdb_id: Any, chain_id: Any, uniprot_id: Any) -> str:
    return (
        f"{str(pdb_id).strip().lower()}_{str(chain_id).strip()}__{str(uniprot_id).strip().upper()}"
    )


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def load_canonical_universe(
    source_path: Path,
    *,
    project_root: Path,
    expected_total: int = EXPECTED_TOTAL,
    expected_eligible: int = EXPECTED_ELIGIBLE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and validate the canonical eligible universe without reordering it."""
    frame = pd.read_parquet(source_path).reset_index(drop=True)
    if len(frame) != expected_total:
        raise InventoryInvariantError(
            f"canonical source row-count drift: expected {expected_total}, observed {len(frame)}"
        )
    required = {
        "polymer_entity_id",
        "pdb_id",
        "chain_id",
        "uniprot_id",
        "discovery_status",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise InventoryInvariantError(f"canonical source lacks required columns: {missing}")

    eligible = frame.loc[frame["discovery_status"].eq("eligible")].copy()
    if len(eligible) != expected_eligible:
        raise InventoryInvariantError(
            "canonical eligible row-count drift: "
            f"expected {expected_eligible}, observed {len(eligible)}"
        )

    eligible.insert(0, "canonical_source_row", eligible.index.astype(int))
    eligible.insert(0, "candidate_index", range(1, len(eligible) + 1))
    eligible["PDB"] = eligible["pdb_id"].astype(str).str.strip().str.lower()
    eligible["chain"] = eligible["chain_id"].astype(str).str.strip()
    eligible["UniProt"] = eligible["uniprot_id"].astype(str).str.strip().str.upper()
    eligible["pair_id"] = [
        _pair_id(pdb_id, chain_id, uniprot_id)
        for pdb_id, chain_id, uniprot_id in zip(
            eligible["PDB"], eligible["chain"], eligible["UniProt"], strict=True
        )
    ]

    identity_sets = {
        "polymer_entity_id": ["polymer_entity_id"],
        "pair_id": ["pair_id"],
        "(PDB, chain, UniProt)": ["PDB", "chain", "UniProt"],
    }
    duplicate_labels = [
        label
        for label, columns in identity_sets.items()
        if eligible.duplicated(columns, keep=False).any()
    ]
    if duplicate_labels:
        raise InventoryInvariantError(
            "canonical eligible identity is ambiguous for: " + ", ".join(duplicate_labels)
        )

    metadata = {
        "canonical_source_path": _relative_path(source_path, project_root),
        "source_sha256": sha256_file(source_path),
        "total_row_count": len(frame),
        "eligible_row_count": len(eligible),
        "indexing_policy_version": INDEXING_POLICY_VERSION,
    }
    return eligible.reset_index(drop=True), metadata


def _read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    separator = "\t" if path.suffix == ".tsv" else ","
    return pd.read_csv(path, sep=separator)


def _table_by_pair(path: Path) -> dict[str, dict[str, Any]]:
    frame = _read_table(path)
    if frame.empty:
        return {}
    if "pair_id" in frame.columns:
        pair_values = frame["pair_id"]
    elif "pair_name" in frame.columns:
        pair_values = frame["pair_name"]
    else:
        required = {"pdb_id", "chain_id", "uniprot_id"}
        if not required.issubset(frame.columns):
            return {}
        pair_values = [
            _pair_id(pdb, chain, uniprot)
            for pdb, chain, uniprot in zip(
                frame["pdb_id"], frame["chain_id"], frame["uniprot_id"], strict=True
            )
        ]
    records: dict[str, dict[str, Any]] = {}
    for pair_id, (_, row) in zip(pair_values, frame.iterrows(), strict=True):
        records[str(pair_id)] = row.to_dict()
    return records


def _historical_evidence(project_root: Path) -> dict[str, dict[str, Any]]:
    sources = (
        project_root / "data/manifests/screening_pool.tsv",
        project_root / "data/manifests/lower_conf_replacement_pool.tsv",
        project_root / "data/manifests/screening_pool_preflight.tsv",
        project_root / "reports/candidate_lifecycle.csv",
        project_root / "reports/replacement_candidate_lifecycle.csv",
    )
    combined: dict[str, dict[str, Any]] = {}
    for path in sources:
        for pair_id, row in _table_by_pair(path).items():
            destination = combined.setdefault(pair_id, {})
            for key, value in row.items():
                if key not in destination or _is_missing(destination[key]):
                    destination[key] = value
    return combined


def _round1_evidence(project_root: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(project_root / "reports/dataset_a_census/round1_report.json")
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        return {}
    return {
        str(record["pair_id"]): record
        for record in payload["records"]
        if isinstance(record, dict) and record.get("pair_id")
    }


def _afdb_files(root: Path, accession: str) -> dict[str, list[Path]]:
    accession_root = root / "data/raw/afdb" / accession
    if not accession_root.is_dir():
        return {"metadata": [], "model": [], "pae": [], "plddt": []}
    files = [path for path in accession_root.rglob("*") if path.is_file()]
    return {
        "metadata": sorted(path for path in files if path.name == "metadata.json"),
        "model": sorted(
            path
            for path in files
            if path.name == "model.cif" or path.name.endswith("-model_v6.cif")
        ),
        "pae": sorted(
            path
            for path in files
            if path.name == "pae.json" or "predicted_aligned_error" in path.name
        ),
        "plddt": sorted(
            path for path in files if path.name == "plddt.json" or "confidence_v" in path.name
        ),
    }


def _prediction_records(metadata_paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for path in metadata_paths:
        payload = _read_json(path)
        values = payload if isinstance(payload, list) else [payload]
        for value in values:
            if not isinstance(value, dict):
                continue
            key = (
                value.get("modelEntityId") or value.get("entryId"),
                _optional_int(value.get("uniprotStart", value.get("sequenceStart"))),
                _optional_int(value.get("uniprotEnd", value.get("sequenceEnd"))),
            )
            records[key] = value
    return list(records.values())


def _parse_fragment_intervals(value: Any) -> list[tuple[int, int]]:
    if _is_missing(value) or value == "":
        return []
    payload = value
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(payload, list):
        return []
    intervals: set[tuple[int, int]] = set()
    for item in payload:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        start, end = _optional_int(item[0]), _optional_int(item[1])
        if start is not None and end is not None and 1 <= start <= end:
            intervals.add((start, end))
    return sorted(intervals)


def _metadata_intervals(records: Iterable[Mapping[str, Any]]) -> list[tuple[int, int]]:
    intervals: set[tuple[int, int]] = set()
    for record in records:
        start = _optional_int(record.get("uniprotStart", record.get("sequenceStart")))
        end = _optional_int(record.get("uniprotEnd", record.get("sequenceEnd")))
        if start is not None and end is not None and 1 <= start <= end:
            intervals.add((start, end))
    return sorted(intervals)


def _canonical_length(
    history: Mapping[str, Any],
    pair_qc: Mapping[str, Any],
    metadata_records: Iterable[Mapping[str, Any]],
) -> int | None:
    for key in ("canonical_uniprot_length", "uniprot_length"):
        value = _optional_int(history.get(key))
        if value is not None:
            return value
    value = _optional_int(pair_qc.get("uniprot_length"))
    if value is not None:
        return value
    full_lengths = set()
    for record in metadata_records:
        start = _optional_int(record.get("uniprotStart", record.get("sequenceStart")))
        end = _optional_int(record.get("uniprotEnd", record.get("sequenceEnd")))
        sequence = _optional_str(record.get("uniprotSequence"))
        if start == 1 and end is not None and sequence is not None and len(sequence) == end:
            full_lengths.add(end)
    return next(iter(full_lengths)) if len(full_lengths) == 1 else None


def _low_confidence_fraction(records: Iterable[Mapping[str, Any]]) -> float | None:
    values = []
    for record in records:
        very_low = _optional_float(record.get("fractionPlddtVeryLow"))
        low = _optional_float(record.get("fractionPlddtLow"))
        if very_low is not None and low is not None:
            values.append(very_low + low)
    return values[0] if len(set(values)) == 1 else None


def _first_number(source: Mapping[str, Any], keys: Iterable[str]) -> float | None:
    for key in keys:
        value = _optional_float(source.get(key))
        if value is not None:
            return value
    return None


def assess_sampling_prior(
    *,
    a0_summary: Mapping[str, Any],
    local_availability: Mapping[str, bool],
    global_plddt: float | None,
    local_evidence_sources: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return an evidence-gated, non-inferential sampling-prior assessment."""
    del global_plddt  # Retained in the interface as a ranking-only feature.
    local_evidence_sources = local_evidence_sources or {}
    classification_complete = (
        _optional_str(a0_summary.get("classification_status")) == "complete_classification"
    )
    full_diagnostic_complete = _optional_bool(
        a0_summary.get("full_diagnostic_complete")
    ) is True
    main_quality_pass = _optional_bool(a0_summary.get("main_quality_pass")) is True
    classification_observed_complete = classification_complete and full_diagnostic_complete
    common_complete = classification_observed_complete and main_quality_pass

    supported: list[str] = []
    partial_positive_evidence = False
    for prior in POSITIVE_PRIOR_PRECEDENCE:
        rule = PRIOR_RULES[prior]
        label_positive = _optional_bool(a0_summary.get(rule["label_field"])) is True
        evidence_available = (
            _optional_bool(a0_summary.get(rule["evidence_field"])) is True
        )
        local_complete = all(
            bool(local_availability.get(name)) for name in rule["required_local"]
        )
        if label_positive and evidence_available and common_complete and local_complete:
            supported.append(prior)
        elif label_positive:
            partial_positive_evidence = True

    primary_category = _optional_str(a0_summary.get("primary_category"))
    primary_prior = next(
        (
            prior
            for prior, rule in PRIOR_RULES.items()
            if rule["primary_category"] == primary_category
        ),
        None,
    )
    if primary_prior in supported:
        selected = primary_prior
    elif supported:
        selected = supported[0]
    else:
        selected = "uncertain_or_unclassified"

    all_observability = ("pdb_structure", "afdb_model", "afdb_pae", "residue_mapping")
    observability_complete = common_complete and all(
        bool(local_availability.get(name)) for name in all_observability
    )
    if supported:
        evidence_status = "positive_complete"
    elif partial_positive_evidence:
        evidence_status = "positive_partial"
    elif classification_observed_complete:
        evidence_status = "observed_no_positive_prior"
    else:
        evidence_status = "unobserved"

    evidence_sources: list[str] = []
    screening_index = _optional_str(a0_summary.get("screening_index"))
    if _optional_str(a0_summary.get("classification_status")) is not None:
        record_key = f"screening_index={screening_index}" if screening_index else "matched_pair"
        evidence_sources.append(f"reports/a0_candidate_summary.csv:{record_key}")
    required_sources = {
        name
        for prior in supported
        for name in PRIOR_RULES[prior]["required_local"]
        if local_availability.get(name)
    }
    evidence_sources.extend(
        local_evidence_sources.get(name, f"local_artifact:{name}")
        for name in sorted(required_sources)
    )

    if selected != "uncertain_or_unclassified":
        source = "documented_a0_primary_category_precedence_sampling_only"
    elif evidence_status == "unobserved":
        source = "no_positive_observable_evidence"
    else:
        source = "observed_but_no_complete_positive_prior"
    return {
        "sampling_stratum_prior": selected,
        "sampling_stratum_prior_source": source,
        "prior_evidence_status": evidence_status,
        "prior_evidence_sources": evidence_sources,
        "prior_observability_complete": observability_complete,
        "supported_sampling_priors": supported,
    }


def _pdb_ligand_annotation_available(path: Path) -> bool | None:
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return "_pdbx_nonpoly_scheme." in text


def build_candidate_record(
    row: Mapping[str, Any],
    *,
    project_root: Path,
    history: Mapping[str, Any] | None = None,
    round1: Mapping[str, Any] | None = None,
    a0_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    history = history or {}
    round1 = round1 or {}
    a0_summary = a0_summary or {}
    pair_id = str(row["pair_id"])
    pdb_id = str(row["PDB"]).lower()
    accession = str(row["UniProt"]).upper()
    pair_root = project_root / "data/processed/pairs" / pair_id
    pair_qc_path = pair_root / "pair_qc.json"
    mapping_path = pair_root / "residue_mapping.parquet"
    pair_qc_payload = _read_json(pair_qc_path)
    pair_qc = pair_qc_payload if isinstance(pair_qc_payload, dict) else {}
    geometry_payload = _read_json(pair_root / "pair_geometry_qc.json")
    geometry = geometry_payload if isinstance(geometry_payload, dict) else {}
    pdb_path = project_root / "data/raw/pdb" / f"{pdb_id}.cif"
    sifts_path = project_root / "data/raw/mappings" / f"{pdb_id}.xml.gz"
    afdb = _afdb_files(project_root, accession)
    metadata_records = _prediction_records(afdb["metadata"])

    canonical_length = _canonical_length(history, pair_qc, metadata_records)
    explicit_intervals = _parse_fragment_intervals(history.get("afdb_fragment_intervals"))
    intervals = explicit_intervals or _metadata_intervals(metadata_records)
    fragment_count = len(intervals) if intervals else (len(metadata_records) or None)
    full_coverage = None
    if canonical_length is not None and intervals:
        full_coverage = any(start <= 1 and end >= canonical_length for start, end in intervals)

    ca_coverage = _first_number(
        history,
        ("observed_ca_fraction_of_mapped", "observed_ca_fraction"),
    )
    if ca_coverage is None:
        ca_coverage = _first_number(geometry, ("mapped_ca_coverage",))
    missing_residue = None if ca_coverage is None else max(0.0, 1.0 - ca_coverage)
    global_plddt = _optional_float(row.get("afdb_global_plddt"))

    long_range_values = [
        _optional_float(a0_summary.get("long_range_48_95_pae_q90")),
        _optional_float(a0_summary.get("long_range_96_plus_pae_q90")),
    ]
    long_range_pae = max((value for value in long_range_values if value is not None), default=None)
    disagreement = _optional_float(geometry.get("median_aligned_ca_distance"))

    mismatch_count = _optional_int(round1.get("mismatch_count"))
    mismatch_flag = None if mismatch_count is None else mismatch_count > 0
    provenance_codes = {
        "no_identity_source",
        "missing_mapping_column",
        "ambiguous_auth_residue_mapping",
        "invalid_atom_residue_identity",
        "pdb_amino_acid_mismatch",
        "residue_count_mismatch",
        "unsupported_afdb_fragment",
        "pair_qc_not_acceptable",
    }
    failure_code = _optional_str(round1.get("failure_code"))
    provenance_issue = failure_code if failure_code in provenance_codes else None

    local_flags = {
        "pdb_structure": pdb_path.is_file(),
        "sifts_mapping": sifts_path.is_file(),
        "pair_qc": pair_qc_path.is_file(),
        "residue_mapping": mapping_path.is_file(),
        "afdb_metadata": bool(afdb["metadata"]),
        "afdb_model": bool(afdb["model"]),
        "afdb_pae": bool(afdb["pae"]),
        "afdb_plddt": bool(afdb["plddt"]),
    }
    missing_inputs = [name for name, available in local_flags.items() if not available]
    local_complete = not missing_inputs

    outcome = _optional_str(round1.get("outcome"))
    stage_reached = _optional_str(round1.get("stage_reached"))
    preflight_status = _optional_str(history.get("preflight_status"))
    if outcome == "p0_failure":
        p0_readiness = "known_blocked"
    elif (
        stage_reached in {"p0_frozen", "p1_resolved"}
        or outcome in {"p1_failure", "p1_pairing_ok"}
        or local_complete
        and preflight_status == "pass_full_length"
    ):
        p0_readiness = "likely_ready"
    elif any(local_flags.values()):
        p0_readiness = "partial_local_inputs"
    else:
        p0_readiness = "insufficient_local_evidence"

    if outcome == "p1_failure":
        p1_risk = "known_high_risk"
    elif outcome == "p1_pairing_ok":
        p1_risk = "lower_risk"
    else:
        p1_risk = "unknown"

    local_evidence_sources = {
        "pdb_structure": _relative_path(pdb_path, project_root),
        "residue_mapping": _relative_path(mapping_path, project_root),
    }
    for evidence_name, paths in (
        ("afdb_model", afdb["model"]),
        ("afdb_pae", afdb["pae"]),
        ("afdb_plddt", afdb["plddt"]),
    ):
        if paths:
            local_evidence_sources[evidence_name] = _relative_path(paths[0], project_root)
    prior_assessment = assess_sampling_prior(
        a0_summary=a0_summary,
        local_availability=local_flags,
        global_plddt=global_plddt,
        local_evidence_sources=local_evidence_sources,
    )
    round1_member = _optional_int(round1.get("screening_index")) is not None

    record = {
        "candidate_index": _optional_int(row.get("candidate_index")),
        "canonical_source_row": _optional_int(row.get("canonical_source_row")),
        "polymer_entity_id": _optional_str(row.get("polymer_entity_id")),
        "pair_id": pair_id,
        "historical_screening_index": _optional_int(history.get("screening_index")),
        "PDB": pdb_id,
        "chain": _optional_str(row.get("chain")),
        "UniProt": accession,
        "sequence_cluster": _optional_str(row.get("sequence_cluster")),
        "protein_family_if_available": None,
        "canonical_uniprot_length": canonical_length,
        "mapped_length": _optional_int(pair_qc.get("mapped_residue_count")),
        "mapping_coverage": _optional_float(pair_qc.get("mapping_coverage")),
        "identity_source_available": (
            _optional_float(pair_qc.get("sequence_identity")) is not None
            or _optional_float(round1.get("sequence_identity_paired")) is not None
        ),
        "mapping_available": mapping_path.is_file(),
        "pair_qc_status": _optional_str(pair_qc.get("quality_flag")),
        "local_PDB_available": pdb_path.is_file(),
        "local_AFDB_available": bool(afdb["model"]),
        "local_PAE_available": bool(afdb["pae"]),
        "local_pLDDT_available": bool(afdb["plddt"]),
        "fragment_metadata_available": bool(metadata_records),
        "fragment_count": fragment_count,
        "single_fragment_full_coverage_possible": full_coverage,
        "PDB_CA_coverage_proxy": ca_coverage,
        "missing_residue_proxy": missing_residue,
        "global_pLDDT_proxy": global_plddt,
        "low_confidence_fraction_proxy": _low_confidence_fraction(metadata_records),
        "long_range_PAE_proxy": long_range_pae,
        "existing_PDB_AFDB_disagreement_proxy": disagreement,
        "ligand_annotation_available": _pdb_ligand_annotation_available(pdb_path),
        "interface_annotation_available": None,
        "known_sequence_mismatch_flag": mismatch_flag,
        "known_provenance_issue": provenance_issue,
        "estimated_P0_readiness": p0_readiness,
        "estimated_P1_risk": p1_risk,
        "local_data_complete": local_complete,
        "missing_local_inputs": missing_inputs,
        **prior_assessment,
        "round1_member": round1_member,
    }
    return {column: record[column] for column in INVENTORY_COLUMNS}


def _distribution(values: Iterable[Any]) -> dict[str, Any]:
    numeric = pd.Series(
        [value for value in (_optional_float(item) for item in values) if value is not None],
        dtype=float,
    )
    if numeric.empty:
        return {"count": 0, "min": None, "q25": None, "median": None, "q75": None, "max": None}
    return {
        "count": int(numeric.count()),
        "min": float(numeric.min()),
        "q25": float(numeric.quantile(0.25)),
        "median": float(numeric.median()),
        "q75": float(numeric.quantile(0.75)),
        "max": float(numeric.max()),
    }


def _cohort_description(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "candidate_count": len(candidates),
        "canonical_uniprot_length": _distribution(
            row["canonical_uniprot_length"] for row in candidates
        ),
        "discovery_pdb_entity_length": _distribution(
            row.get("discovery_pdb_entity_length") for row in candidates
        ),
        "global_pLDDT_proxy": _distribution(row["global_pLDDT_proxy"] for row in candidates),
        "local_data_complete_count": sum(row["local_data_complete"] for row in candidates),
        "sampling_stratum_prior_counts": dict(
            sorted(Counter(row["sampling_stratum_prior"] for row in candidates).items())
        ),
    }


def summarize_inventory(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    missing_histogram = Counter(
        missing for row in candidates for missing in row["missing_local_inputs"]
    )
    prior_counts = Counter(row["sampling_stratum_prior"] for row in candidates)
    prior_evidence_counts = Counter(row["prior_evidence_status"] for row in candidates)
    multi_prior_overlaps = [
        {
            "candidate_index": row["candidate_index"],
            "pair_id": row["pair_id"],
            "supported_sampling_priors": row["supported_sampling_priors"],
            "selected_sampling_prior": row["sampling_stratum_prior"],
        }
        for row in candidates
        if len(row["supported_sampling_priors"]) > 1
    ]
    mismatch_counts = Counter(
        "unknown"
        if row["known_sequence_mismatch_flag"] is None
        else "known_true"
        if row["known_sequence_mismatch_flag"]
        else "known_false"
        for row in candidates
    )
    provenance_counts = Counter(
        row["known_provenance_issue"] or "none_or_not_observed" for row in candidates
    )
    round1 = [row for row in candidates if row["round1_member"]]
    remaining = [row for row in candidates if not row["round1_member"]]
    return {
        "total_candidate_count": len(candidates),
        "local_data_complete_count": sum(row["local_data_complete"] for row in candidates),
        "mapping_ready_count": sum(
            row["mapping_available"] and row["pair_qc_status"] is not None for row in candidates
        ),
        "fragment_ready_count": sum(
            row["fragment_metadata_available"]
            and row["single_fragment_full_coverage_possible"] is not None
            for row in candidates
        ),
        "p0_likely_ready_count": sum(
            row["estimated_P0_readiness"] == "likely_ready" for row in candidates
        ),
        "canonical_uniprot_length_distribution": _distribution(
            row["canonical_uniprot_length"] for row in candidates
        ),
        "mapped_length_distribution": _distribution(row["mapped_length"] for row in candidates),
        "missing_local_input_histogram": dict(sorted(missing_histogram.items())),
        "sampling_stratum_prior_counts": {
            prior: prior_counts.get(prior, 0) for prior in SAMPLING_PRIORS
        },
        "prior_evidence_status_counts": dict(sorted(prior_evidence_counts.items())),
        "prior_observability_complete_count": sum(
            row["prior_observability_complete"] for row in candidates
        ),
        "multi_prior_overlap_count": len(multi_prior_overlaps),
        "multi_prior_overlaps": multi_prior_overlaps,
        "known_sequence_mismatch_counts": dict(sorted(mismatch_counts.items())),
        "known_provenance_issue_counts": dict(sorted(provenance_counts.items())),
        "round1_vs_remaining_descriptive_comparison": {
            "scope_note": (
                "Descriptive comparison only; it is not a statistical population inference."
            ),
            "observed_selection_enrichment": (
                "Round-1 is strongly enriched for locally complete inputs; its global-pLDDT "
                "distribution is modestly higher. Mechanism-rich priors in Round-1 reflect "
                "existing completed diagnostics and must not be interpreted as remaining-pool "
                "prevalence."
            ),
            "round1_26": _cohort_description(round1),
            "remaining_pool": _cohort_description(remaining),
        },
    }


def select_batch1(
    candidates: list[dict[str, Any]], *, target: int = BATCH1_TARGET
) -> list[dict[str, Any]]:
    """Select a deterministic, diverse acquisition panel from non-round1 records."""
    if target < 1:
        return []
    readiness_rank = {
        "likely_ready": 0,
        "partial_local_inputs": 1,
        "known_blocked": 2,
        "insufficient_local_evidence": 3,
    }

    def acquisition_priority(row: Mapping[str, Any]) -> int:
        prior = str(row.get("sampling_stratum_prior"))
        positive = row.get("prior_evidence_status") == "positive_complete"
        if prior in {
            "low_confidence_local_prior",
            "high_pae_long_range_prior",
            "state_disagreement_prior",
        } and positive:
            return 0
        if prior == "uncertain_or_unclassified":
            return 1
        if prior == "easy_control_prior" and positive:
            return 2
        raise InventoryInvariantError(
            f"candidate {row.get('candidate_index')} has non-unknown prior without "
            "positive_complete evidence"
        )

    def length_bucket(row: Mapping[str, Any]) -> str:
        length = _optional_int(row.get("canonical_uniprot_length"))
        if length is None:
            length = _optional_int(row.get("discovery_pdb_entity_length"))
        if length is None:
            return "unknown"
        if length < 150:
            return "lt150"
        if length < 300:
            return "150_299"
        if length < 600:
            return "300_599"
        return "ge600"

    remaining = [dict(row) for row in candidates if row.get("round1_member") is not True]

    selected: list[dict[str, Any]] = []
    used_uniprot: set[str] = set()
    used_clusters: set[str] = set()
    selected_length_buckets: Counter[str] = Counter()

    while len(selected) < target:
        eligible = [
            row
            for row in remaining
            if str(row["UniProt"]) not in used_uniprot
            and (
                _optional_str(row.get("sequence_cluster")) is None
                or _optional_str(row.get("sequence_cluster")) not in used_clusters
            )
        ]
        if not eligible:
            break
        candidate = min(
            eligible,
            key=lambda row: (
                acquisition_priority(row),
                readiness_rank.get(str(row.get("estimated_P0_readiness")), 9),
                selected_length_buckets[length_bucket(row)],
                len(row.get("missing_local_inputs") or []),
                _optional_float(row.get("global_pLDDT_proxy"))
                if _optional_float(row.get("global_pLDDT_proxy")) is not None
                else math.inf,
                int(row["candidate_index"]),
            ),
        )
        remaining.remove(candidate)
        uniprot = str(candidate["UniProt"])
        cluster = _optional_str(candidate.get("sequence_cluster"))
        prior = str(candidate["sampling_stratum_prior"])
        bucket = length_bucket(candidate)
        reason_parts = [
            "acquisition_panel",
            "post_acquisition_restratification_required",
            "non_round1_candidate",
            f"prior={prior}",
            f"prior_evidence_status={candidate.get('prior_evidence_status')}",
            f"p0_readiness={candidate['estimated_P0_readiness']}",
            f"length_bucket={bucket}",
            "unique_uniprot",
            "unique_known_sequence_cluster"
            if cluster is not None
            else "sequence_cluster_unknown",
        ]
        planned = dict(candidate)
        planned["batch1_rank"] = len(selected) + 1
        planned["selection_reason"] = ";".join(reason_parts)
        selected.append(planned)
        used_uniprot.add(uniprot)
        if cluster is not None:
            used_clusters.add(cluster)
        selected_length_buckets[bucket] += 1
    if len(selected) != target:
        raise InventoryInvariantError(
            f"cannot construct Batch-1 target {target} under identity/cluster constraints; "
            f"selected {len(selected)}"
        )
    return selected


def _feasibility(candidate_count: int) -> dict[str, Any]:
    return {
        "scenario_type": "descriptive_not_confidence_intervals",
        "candidate_universe_size": candidate_count,
        "admissions_by_scenario": {
            "below_30_percent": "0-63",
            "30_to_40_percent": "64-85",
            "at_least_40_percent": "86-213",
        },
        "A1_approximately_64": {
            "required_rate": 64 / candidate_count,
            "plausibility": "plausible_only_at_about_30_percent_or_higher",
            "note": "A 30% yield is 63.9, so A1 has essentially no margin at that boundary.",
        },
        "A2_128_to_256": {
            "required_rate_for_128": 128 / candidate_count,
            "plausibility": "128_requires_at_least_60.1_percent;_256_is_impossible_from_213",
            "note": (
                "The >=40% scenario alone does not establish support for 128, and membership "
                "caps the current universe below 256 even at 100% admission."
            ),
        },
        "likely_limiting_strata": [
            "low_confidence_local_prior",
            "high_pae_long_range_prior",
            "state_disagreement_prior",
        ],
        "external_expansion_trigger": (
            "Start external-pool planning after Batch-1 if observed conservative yield projected "
            "over 213 is below the target, or if any required uncertainty-rich stratum remains "
            "too sparse for its planned representation; do not wait for full-pool exhaustion."
        ),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def build_inventory(project_root: Path, source_path: Path) -> dict[str, Any]:
    canonical, source_metadata = load_canonical_universe(
        source_path,
        project_root=project_root,
    )
    history = _historical_evidence(project_root)
    round1 = _round1_evidence(project_root)
    a0 = _table_by_pair(project_root / "reports/a0_candidate_summary.csv")
    candidates = []
    for row in canonical.to_dict(orient="records"):
        record = build_candidate_record(
            row,
            project_root=project_root,
            history=history.get(str(row["pair_id"])),
            round1=round1.get(str(row["pair_id"])),
            a0_summary=a0.get(str(row["pair_id"])),
        )
        record["discovery_pdb_entity_length"] = _optional_int(row.get("length"))
        candidates.append(record)
    batch1 = select_batch1(candidates)
    return _json_safe(
        {
            "schema_version": "dataset-a.metadata-inventory.v1",
            "inventory_status": "metadata_only_plan_not_admission",
            "source_metadata": source_metadata,
            "candidate_index_semantics": (
                "candidate_index is a 1-based inventory convenience ID in canonical eligible "
                "source order. It is not historical screening_index, does not represent admission, "
                "and is not a permanent protein identity."
            ),
            "sampling_prior_disclaimer": SAMPLING_PRIOR_DISCLAIMER,
            "evidence_policy": EVIDENCE_POLICY,
            "summary": summarize_inventory(candidates),
            "batch1_summary": {
                "target": BATCH1_TARGET,
                "selected": len(batch1),
                "plan_only": True,
                "panel_role": "acquisition_panel",
                "post_acquisition_restratification_required": True,
                "round1_overlap_count": sum(row["round1_member"] for row in batch1),
                "unique_uniprot_count": len({row["UniProt"] for row in batch1}),
                "unique_known_sequence_cluster_count": len(
                    {row["sequence_cluster"] for row in batch1 if row["sequence_cluster"]}
                ),
                "sampling_stratum_prior_counts": dict(
                    sorted(Counter(row["sampling_stratum_prior"] for row in batch1).items())
                ),
                "prior_evidence_status_counts": dict(
                    sorted(Counter(row["prior_evidence_status"] for row in batch1).items())
                ),
                "positive_evidence_prior_count": sum(
                    row["prior_evidence_status"] == "positive_complete" for row in batch1
                ),
                "p0_readiness_counts": dict(
                    sorted(Counter(row["estimated_P0_readiness"] for row in batch1).items())
                ),
            },
            "feasibility": _feasibility(len(candidates)),
            "candidates": candidates,
            "batch1": batch1,
        }
    )


def _tsv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    return value


def _write_tsv(path: Path, records: list[dict[str, Any]], columns: Iterable[str]) -> None:
    columns = list(columns)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for record in records:
            writer.writerow({column: _tsv_value(record.get(column)) for column in columns})


def _format_distribution(distribution: Mapping[str, Any]) -> str:
    if not distribution.get("count"):
        return "no local values"
    return (
        f"n={distribution['count']}, min={distribution['min']:.3g}, "
        f"q25={distribution['q25']:.3g}, median={distribution['median']:.3g}, "
        f"q75={distribution['q75']:.3g}, max={distribution['max']:.3g}"
    )


def _markdown(payload: Mapping[str, Any]) -> str:
    source = payload["source_metadata"]
    summary = payload["summary"]
    batch = payload["batch1_summary"]
    comparison = summary["round1_vs_remaining_descriptive_comparison"]
    feasibility = payload["feasibility"]
    lines = [
        "# Dataset-A Candidate Inventory v1",
        "",
        "**Status:** metadata-only planning inventory; not candidate admission.",
        "",
        "## Canonical universe",
        "",
        f"- Source: `{source['canonical_source_path']}`",
        f"- SHA-256: `{source['source_sha256']}`",
        f"- Total rows: {source['total_row_count']}",
        f"- Eligible rows: {source['eligible_row_count']}",
        f"- Indexing policy: `{source['indexing_policy_version']}`",
        f"- Identity note: {payload['candidate_index_semantics']}",
        "",
        "## Evidence policy",
        "",
        (
            f"{payload['sampling_prior_disclaimer']} Missing local evidence remains "
            "null/unknown. No formal P6 mechanism label was recomputed."
        ),
        "",
        "## Inventory summary",
        "",
        f"- Candidate count: {summary['total_candidate_count']}",
        f"- Local-data-complete: {summary['local_data_complete_count']}",
        f"- Mapping-ready: {summary['mapping_ready_count']}",
        f"- Fragment-ready: {summary['fragment_ready_count']}",
        f"- P0-likely-ready: {summary['p0_likely_ready_count']}",
        "- Canonical UniProt length: "
        + _format_distribution(summary["canonical_uniprot_length_distribution"]),
        "- Mapped length: " + _format_distribution(summary["mapped_length_distribution"]),
        "",
        "### Proxy provenance",
        "",
    ]
    lines.extend(f"- `{name}`: {text}" for name, text in payload["evidence_policy"].items())
    lines.extend(
        [
            "",
            "### Missing local inputs",
            "",
        ]
    )
    lines.extend(
        f"- `{name}`: {count}" for name, count in summary["missing_local_input_histogram"].items()
    )
    lines.extend(["", "### Sampling priors", ""])
    lines.extend(
        f"- `{name}`: {count}" for name, count in summary["sampling_stratum_prior_counts"].items()
    )
    lines.extend(
        [
            "",
            "### Prior observability audit",
            "",
            (
                "A non-unknown sampling prior requires explicit positive evidence and complete "
                "required local provenance. Global pLDDT is ranking metadata only."
            ),
            "",
        ]
    )
    lines.extend(
        f"- Evidence status `{name}`: {count}"
        for name, count in summary["prior_evidence_status_counts"].items()
    )
    lines.extend(
        [
            f"- Prior-observability complete: {summary['prior_observability_complete_count']}",
            f"- Multi-prior overlaps: {summary['multi_prior_overlap_count']}",
        ]
    )
    for overlap in summary["multi_prior_overlaps"]:
        lines.append(
            f"  - Index {overlap['candidate_index']} `{overlap['pair_id']}`: "
            + ", ".join(overlap["supported_sampling_priors"])
            + f"; selected `{overlap['selected_sampling_prior']}` by documented A0 precedence."
        )
    lines.extend(
        [
            "",
            "## Round-1 versus remaining pool",
            "",
            comparison["scope_note"],
            comparison["observed_selection_enrichment"],
            "",
            f"- Round-1 count: {comparison['round1_26']['candidate_count']}",
            f"- Remaining count: {comparison['remaining_pool']['candidate_count']}",
            "- Round-1 global pLDDT: "
            + _format_distribution(comparison["round1_26"]["global_pLDDT_proxy"]),
            "- Remaining global pLDDT: "
            + _format_distribution(comparison["remaining_pool"]["global_pLDDT_proxy"]),
            "- Round-1 local-data-complete: "
            + str(comparison["round1_26"]["local_data_complete_count"]),
            "- Remaining local-data-complete: "
            + str(comparison["remaining_pool"]["local_data_complete_count"]),
            "",
            "## Batch-1 acquisition panel",
            "",
            (
                f"The plan contains {batch['selected']} candidates (target {batch['target']}), "
                "with unique UniProt accessions and no repeated known sequence cluster. Unknown "
                "clusters are left unknown rather than inferred. Round-1 overlap is "
                f"{batch['round1_overlap_count']}. Every row in `batch1_plan_v1.tsv` records its "
                "selection reason. This acquisition panel is not a final inferential or formal "
                "mechanism-balanced panel. Post-acquisition re-stratification is mandatory. "
                "This is a plan only; P0/P1 were not run."
            ),
            "",
            "## Scale feasibility scenarios",
            "",
            "These are descriptive scenarios, not confidence intervals.",
            "",
            (
                f"- A1 (~64): {feasibility['A1_approximately_64']['plausibility']}. "
                f"{feasibility['A1_approximately_64']['note']}"
            ),
            (
                f"- A2 (128–256): {feasibility['A2_128_to_256']['plausibility']}. "
                f"{feasibility['A2_128_to_256']['note']}"
            ),
            "- Likely limiting sampling priors: "
            + ", ".join(feasibility["likely_limiting_strata"]),
            f"- Expansion trigger: {feasibility['external_expansion_trigger']}",
            "",
            "## Guardrails",
            "",
            (
                "No candidate was admitted or rejected. No download, P0, P1, P2, ProteinMPNN, "
                "or evaluator execution occurred."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(payload: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "candidate_inventory_v1.json"
    inventory_path = output_dir / "candidate_inventory_v1.tsv"
    batch_path = output_dir / "batch1_plan_v1.tsv"
    markdown_path = output_dir / "candidate_inventory_v1.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_tsv(inventory_path, list(payload["candidates"]), INVENTORY_COLUMNS)
    batch_columns = (
        "batch1_rank",
        "candidate_index",
        "canonical_source_row",
        "polymer_entity_id",
        "pair_id",
        "historical_screening_index",
        "PDB",
        "chain",
        "UniProt",
        "sequence_cluster",
        "sampling_stratum_prior",
        "sampling_stratum_prior_source",
        "prior_evidence_status",
        "prior_evidence_sources",
        "prior_observability_complete",
        "supported_sampling_priors",
        "estimated_P0_readiness",
        "estimated_P1_risk",
        "local_data_complete",
        "missing_local_inputs",
        "selection_reason",
    )
    _write_tsv(batch_path, list(payload["batch1"]), batch_columns)
    markdown_path.write_text(_markdown(payload), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    source = args.source or project_root / CANONICAL_SOURCE
    output_dir = args.output_dir or project_root / OUTPUT_DIR
    payload = build_inventory(project_root, source.resolve())
    write_outputs(payload, output_dir.resolve())
    print(
        json.dumps(
            {
                "candidate_count": payload["summary"]["total_candidate_count"],
                "batch1_count": payload["batch1_summary"]["selected"],
                "output_dir": str(output_dir.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
