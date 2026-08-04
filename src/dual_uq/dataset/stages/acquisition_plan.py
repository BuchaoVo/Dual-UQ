"""Build the offline Dataset-A Batch-1 acquisition authorization plan."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from dual_uq.core.atomic_io import atomic_write_text
from dual_uq.core.hashing import sha256_file
from dual_uq.core.paths import ProjectPaths

PROJECT_ROOT = ProjectPaths.discover(anchor=Path(__file__)).repository_root
DISCOVERY_SOURCE = Path("data/processed/discovery/discovered_candidates.parquet")
INVENTORY_PATH = Path("artifacts/dataset/reports/census/candidate_inventory_v1.json")
BATCH1_PATH = Path("artifacts/dataset/reports/census/batch1_plan_v1.tsv")
OUTPUT_DIR = Path("artifacts/dataset/reports/acquisition")
EXPECTED_DISCOVERY_SHA256 = (
    "dddb21ef1e41babb71827ba729eabe5c568ab6a7d6d6c5953f6da2e10bd98436"
)
EXPECTED_BATCH_SIZE = 48

ASSET_STATUSES = {
    "available_local",
    "requires_external_acquisition",
    "derivable_locally_after_acquisition",
    "not_required",
    "unresolved",
}
ASSET_FIELDS = (
    "canonical_identity_metadata",
    "pair_qc_source_inputs",
    "residue_mapping_source_inputs",
    "pdb_mmcif",
    "sifts",
    "afdb_metadata",
    "afdb_structure",
    "afdb_pae",
    "afdb_confidence",
    "canonical_sequence_metadata",
)
EXTERNAL_ASSETS = (
    "pdb_mmcif",
    "sifts",
    "afdb_metadata",
    "afdb_structure",
    "afdb_pae",
    "afdb_confidence",
)
LOCAL_DERIVATIONS = (
    "canonical_sequence_metadata_extraction",
    "pair_qc",
    "residue_mapping",
    "fragment_resolution",
    "p0_eligibility_inputs",
    "mechanism_observability_features",
)
ACQUISITION_CLASSES = (
    "ready_without_external_acquisition",
    "needs_mapping_derivation_only",
    "needs_pdb_side_acquisition",
    "needs_afdb_side_acquisition",
    "needs_both_structure_sides",
    "blocked_by_identity_metadata",
)
IDENTITY_FIELDS = (
    "candidate_index",
    "canonical_source_row",
    "polymer_entity_id",
    "pair_id",
    "PDB",
    "chain",
    "UniProt",
)


class AcquisitionInvariantError(RuntimeError):
    """Raised when the repaired inventory or acquisition panel has drifted."""


def _present(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AcquisitionInvariantError(f"expected JSON object: {path}")
    return payload


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _metadata_has_canonical_sequence(project_root: Path, accession: str) -> bool:
    root = project_root / "data/raw/afdb" / accession
    for path in sorted(root.rglob("metadata.json")) if root.is_dir() else []:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        records = payload if isinstance(payload, list) else [payload]
        for record in records:
            if not isinstance(record, dict):
                continue
            observed_accession = record.get("uniprotAccession")
            sequence = record.get("uniprotSequence") or record.get("sequence")
            if str(observed_accession).upper() == accession.upper() and _present(sequence):
                return True
    return False


def _external_asset_availability(candidate: Mapping[str, Any]) -> dict[str, bool]:
    missing = set(candidate.get("missing_local_inputs") or [])
    return {
        "pdb_mmcif": bool(candidate.get("local_PDB_available"))
        and "pdb_structure" not in missing,
        "sifts": "sifts_mapping" not in missing,
        "afdb_metadata": bool(candidate.get("fragment_metadata_available"))
        and "afdb_metadata" not in missing,
        "afdb_structure": bool(candidate.get("local_AFDB_available"))
        and "afdb_model" not in missing,
        "afdb_pae": bool(candidate.get("local_PAE_available"))
        and "afdb_pae" not in missing,
        "afdb_confidence": bool(candidate.get("local_pLDDT_available"))
        and "afdb_plddt" not in missing,
    }


def _required_local_derivations(candidate: Mapping[str, Any]) -> list[str]:
    missing = set(candidate.get("missing_local_inputs") or [])
    derivations: list[str] = []
    if "pair_qc" in missing or not _present(candidate.get("pair_qc_status")):
        derivations.append("pair_qc")
    if "residue_mapping" in missing or not bool(candidate.get("mapping_available")):
        derivations.append("residue_mapping")
    derivations.extend(
        ["fragment_resolution", "p0_eligibility_inputs", "mechanism_observability_features"]
    )
    return [name for name in LOCAL_DERIVATIONS if name in derivations]


def build_acquisition_record(
    candidate: Mapping[str, Any], *, project_root: Path
) -> dict[str, Any]:
    identity_complete = all(_present(candidate.get(field)) for field in IDENTITY_FIELDS)
    available = _external_asset_availability(candidate)
    required_external = [name for name in EXTERNAL_ASSETS if not available[name]]
    required_derivations = _required_local_derivations(candidate)

    pdb_side_missing = any(name in required_external for name in ("pdb_mmcif", "sifts"))
    afdb_side_missing = any(name.startswith("afdb_") for name in required_external)
    if not identity_complete:
        primary_class = "blocked_by_identity_metadata"
    elif pdb_side_missing and afdb_side_missing:
        primary_class = "needs_both_structure_sides"
    elif pdb_side_missing:
        primary_class = "needs_pdb_side_acquisition"
    elif afdb_side_missing:
        primary_class = "needs_afdb_side_acquisition"
    elif "pair_qc" in required_derivations or "residue_mapping" in required_derivations:
        primary_class = "needs_mapping_derivation_only"
    else:
        primary_class = "ready_without_external_acquisition"

    if primary_class not in ACQUISITION_CLASSES:
        raise AcquisitionInvariantError(f"invalid acquisition class: {primary_class}")
    canonical_sequence_available = (
        available["afdb_metadata"]
        and _present(candidate.get("UniProt"))
        and _metadata_has_canonical_sequence(project_root, str(candidate["UniProt"]))
    )
    if not canonical_sequence_available and not available["afdb_metadata"]:
        required_derivations = [
            name
            for name in LOCAL_DERIVATIONS
            if name == "canonical_sequence_metadata_extraction"
            or name in required_derivations
        ]
    source_inputs_available = not required_external
    asset_statuses = {
        "canonical_identity_metadata": "available_local" if identity_complete else "unresolved",
        "pair_qc_source_inputs": (
            "available_local" if source_inputs_available else "requires_external_acquisition"
        ),
        "residue_mapping_source_inputs": (
            "available_local" if source_inputs_available else "requires_external_acquisition"
        ),
        **{
            name: "available_local" if available[name] else "requires_external_acquisition"
            for name in EXTERNAL_ASSETS
        },
        "canonical_sequence_metadata": (
            "available_local"
            if canonical_sequence_available
            else "derivable_locally_after_acquisition"
            if not available["afdb_metadata"]
            else "unresolved"
        ),
    }
    if set(asset_statuses) != set(ASSET_FIELDS) or not set(asset_statuses.values()).issubset(
        ASSET_STATUSES
    ):
        raise AcquisitionInvariantError("invalid or incomplete asset-status audit")

    record = {field: candidate.get(field) for field in IDENTITY_FIELDS}
    record.update(
        {
            "current_readiness": candidate.get("estimated_P0_readiness"),
            "primary_acquisition_class": primary_class,
            "required_external_assets": required_external,
            "required_local_derivations": required_derivations,
            "sampling_stratum_prior": candidate.get("sampling_stratum_prior"),
            "prior_evidence_status": candidate.get("prior_evidence_status"),
            "selection_reason": candidate.get("selection_reason"),
            "target_observability": (
                "complete_pair_mapping_confidence_pae_and_geometry_evidence_for_"
                "post_acquisition_restratification"
            ),
            "asset_statuses": asset_statuses,
            "round1_member": bool(candidate.get("round1_member")),
        }
    )
    for asset in EXTERNAL_ASSETS:
        record[f"requires_{asset}"] = asset in required_external
    return record


def _verify_repaired_inventory(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates = payload.get("candidates")
    batch = payload.get("batch1")
    if not isinstance(candidates, list) or not isinstance(batch, list):
        raise AcquisitionInvariantError("inventory lacks candidates or Batch-1 records")
    if len(batch) != EXPECTED_BATCH_SIZE:
        raise AcquisitionInvariantError(f"expected 48 Batch-1 candidates, observed {len(batch)}")
    invalid_positive = [
        row.get("candidate_index")
        for row in candidates
        if row.get("sampling_stratum_prior") != "uncertain_or_unclassified"
        and not (
            row.get("prior_evidence_status") == "positive_complete"
            and row.get("prior_observability_complete") is True
            and bool(row.get("prior_evidence_sources"))
        )
    ]
    if invalid_positive:
        raise AcquisitionInvariantError(
            f"non-unknown priors lack positive observable evidence: {invalid_positive}"
        )
    if any(row.get("sampling_stratum_prior") != "uncertain_or_unclassified" for row in batch):
        raise AcquisitionInvariantError("Batch-1 prior invariant failed")
    if any(row.get("round1_member") is not False for row in batch):
        raise AcquisitionInvariantError("Batch-1 overlaps Round-1")
    if len({row.get("UniProt") for row in batch}) != EXPECTED_BATCH_SIZE:
        raise AcquisitionInvariantError("Batch-1 UniProt identity is not unique")
    return [dict(row) for row in batch]


def _h6_package(external_counts: Mapping[str, int]) -> dict[str, Any]:
    definitions = {
        "pdb_mmcif": (
            "PDB archive",
            "data/raw/pdb/{PDB}.cif",
            "PDB accession {PDB} -> deposited mmCIF raw bytes",
            "experimental coordinates and residue provenance",
        ),
        "sifts": (
            "PDBe SIFTS",
            "data/raw/mappings/{PDB}.xml.gz",
            "PDB accession {PDB} -> SIFTS XML raw bytes",
            "PDB-to-UniProt residue mapping provenance",
        ),
        "afdb_metadata": (
            "AlphaFold DB",
            "data/raw/afdb/{UniProt}/metadata.json",
            "AFDB prediction API record -> identity-validated metadata JSON",
            "fragment/model identity and canonical sequence metadata",
        ),
        "afdb_structure": (
            "AlphaFold DB",
            "data/raw/afdb/{UniProt}/{model_id}/model.cif",
            "AFDB metadata.cifUrl raw mmCIF bytes -> local model.cif",
            "predicted structural condition",
        ),
        "afdb_pae": (
            "AlphaFold DB",
            "data/raw/afdb/{UniProt}/{model_id}/pae.json",
            "AFDB metadata.paeDocUrl raw JSON bytes -> local pae.json",
            "long-range uncertainty observability",
        ),
        "afdb_confidence": (
            "AlphaFold DB",
            "data/raw/afdb/{UniProt}/{model_id}/plddt.json",
            (
                "AFDB metadata.plddtDocUrl raw JSON bytes -> local plddt.json; "
                "no numeric transform"
            ),
            "mapped local-confidence observability",
        ),
    }
    entries = [
        {
            "external_source_category": definitions[name][0],
            "asset_type": name,
            "candidate_count": int(external_counts[name]),
            "expected_local_destination": definitions[name][1],
            "source_binding": definitions[name][2],
            "purpose": definitions[name][3],
        }
        for name in EXTERNAL_ASSETS
    ]
    local_dependencies = {
        "canonical_sequence_metadata_extraction": (
            "identity-validated AFDB metadata.uniprotAccession plus "
            "uniprotSequence/sequence"
        ),
        "pair_qc": "PDB mmCIF + SIFTS + selected AFDB metadata/model",
        "residue_mapping": "PDB mmCIF + SIFTS + canonical UniProt sequence metadata",
        "fragment_resolution": "mapped UniProt interval + complete AFDB prediction metadata",
        "p0_eligibility_inputs": "pair QC + residue mapping + verified raw-input identities",
        "mechanism_observability_features": (
            "paired structures + residue mapping + AFDB PAE/confidence"
        ),
    }
    return {
        "authorization_scope": "human_authorization_request_external_assets_only",
        "entries": entries,
        "local_derivations": [
            {
                "local_derivation": name,
                "candidate_count": 0,
                "input_dependency": local_dependencies[name],
            }
            for name in LOCAL_DERIVATIONS
        ],
        "prohibited_actions": [
            "model_download",
            "training",
            "evaluator_execution",
            "proteinmpnn_generation",
        ],
    }


def build_plan(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    discovery = project_root / DISCOVERY_SOURCE
    inventory_path = project_root / INVENTORY_PATH
    batch_path = project_root / BATCH1_PATH
    discovery_sha = sha256_file(discovery)
    if discovery_sha != EXPECTED_DISCOVERY_SHA256:
        raise AcquisitionInvariantError("canonical discovery source SHA drift")
    inventory = _read_json(inventory_path)
    source_metadata = inventory.get("source_metadata", {})
    if source_metadata.get("source_sha256") != discovery_sha:
        raise AcquisitionInvariantError("inventory discovery-source binding mismatch")
    batch = _verify_repaired_inventory(inventory)
    batch_tsv = _read_tsv(batch_path)
    json_identity = [(str(row["candidate_index"]), str(row["pair_id"])) for row in batch]
    tsv_identity = [(row["candidate_index"], row["pair_id"]) for row in batch_tsv]
    if json_identity != tsv_identity:
        raise AcquisitionInvariantError("Batch-1 TSV identity/order differs from inventory JSON")

    records = [build_acquisition_record(row, project_root=project_root) for row in batch]
    external_counts = {
        asset: sum(record[f"requires_{asset}"] for record in records)
        for asset in EXTERNAL_ASSETS
    }
    local_counts = {
        name: sum(name in record["required_local_derivations"] for record in records)
        for name in LOCAL_DERIVATIONS
    }
    asset_status_counts = {
        field: dict(sorted(Counter(row["asset_statuses"][field] for row in records).items()))
        for field in ASSET_FIELDS
    }
    h6_package = _h6_package(external_counts)
    for entry in h6_package["local_derivations"]:
        entry["candidate_count"] = local_counts[entry["local_derivation"]]
    return {
        "schema_version": "dataset-a.batch1-acquisition-plan.v1",
        "plan_status": "offline_plan_requires_human_acquisition_authorization",
        "input_bindings": {
            "discovery_source_path": DISCOVERY_SOURCE.as_posix(),
            "discovery_source_sha256": discovery_sha,
            "candidate_inventory_path": INVENTORY_PATH.as_posix(),
            "candidate_inventory_sha256": sha256_file(inventory_path),
            "batch1_plan_path": BATCH1_PATH.as_posix(),
            "batch1_plan_sha256": sha256_file(batch_path),
        },
        "repaired_prior_invariant": {
            "global_plddt_alone_establishes_easy_control_prior": False,
            "all_non_unknown_priors_require_positive_observable_evidence": True,
            "batch1_uncertain_or_unclassified_count": len(records),
        },
        "panel_semantics": {
            "current_role": "acquisition_panel",
            "final_inferential_panel": False,
            "formal_mechanism_balanced_panel": False,
            "post_acquisition_restratification_required": True,
        },
        "post_acquisition_flow": [
            "acquire_assets",
            "verify_hashes_and_identities",
            "construct_pair_qc_and_residue_mapping",
            "evaluate_fragment_and_model_identity",
            "recompute_sampling_priors",
            "select_final_round2_census_panel",
            "run_final_p0_p1_protocol_only_after_selection",
        ],
        "post_acquisition_outcome_states": [
            "still_unobservable",
            "observed_no_positive_prior",
            "easy_control_prior",
            "low_confidence_local_prior",
            "high_pae_long_range_prior",
            "state_disagreement_prior",
        ],
        "pre_acquisition_mechanism_prevalence_claim_allowed": False,
        "asset_provenance": {
            "canonical_sequence_metadata": {
                "resolution": "deterministic_extraction_from_planned_afdb_metadata",
                "identity_field": "uniprotAccession",
                "sequence_fields_in_priority_order": ["uniprotSequence", "sequence"],
                "independent_external_acquisition_required": False,
                "no_sequence_identity_inference": True,
            },
            "afdb_confidence": {
                "classification": "direct_external_afdb_asset",
                "metadata_source_field": "plddtDocUrl",
                "local_normalized_filename": "plddt.json",
                "content_transform": "none_raw_downloaded_bytes",
            },
        },
        "summary": {
            "candidate_count": len(records),
            "external_acquisition_counts": external_counts,
            "local_derivation_counts": local_counts,
            "acquisition_class_counts": dict(
                sorted(Counter(row["primary_acquisition_class"] for row in records).items())
            ),
            "asset_status_counts": asset_status_counts,
            "identity_provenance_blocker_count": sum(
                row["primary_acquisition_class"] == "blocked_by_identity_metadata"
                for row in records
            ),
        },
        "h6_authorization_package": h6_package,
        "source_batch_identity": [
            {field: row.get(field) for field in IDENTITY_FIELDS} for row in batch
        ],
        "records": records,
    }


def _tsv_value(value: Any) -> str | int | bool | None:
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _write_tsv(path: Path, records: list[dict[str, Any]]) -> None:
    columns = (
        *IDENTITY_FIELDS,
        "current_readiness",
        "primary_acquisition_class",
        "required_external_assets",
        "required_local_derivations",
        "sampling_stratum_prior",
        "prior_evidence_status",
        "selection_reason",
        "target_observability",
        "asset_statuses",
        *(f"requires_{asset}" for asset in EXTERNAL_ASSETS),
    )
    stream = StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=columns, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    for record in records:
        writer.writerow({name: _tsv_value(record.get(name)) for name in columns})
    atomic_write_text(path, stream.getvalue())


def _markdown(payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    bindings = payload["input_bindings"]
    lines = [
        "# Dataset-A Batch-1 Acquisition Plan v1",
        "",
        "**Status:** offline acquisition plan; human authorization required.",
        "",
        (
            "The 48 candidates remain an acquisition panel, not a final inferential or formal "
            "mechanism-balanced panel. Acquisition success does not imply Dataset-A admission."
        ),
        "",
        "## Input bindings",
        "",
        f"- Discovery: `{bindings['discovery_source_path']}` — `{bindings['discovery_source_sha256']}`",
        f"- Inventory: `{bindings['candidate_inventory_path']}` — `{bindings['candidate_inventory_sha256']}`",
        f"- Batch-1: `{bindings['batch1_plan_path']}` — `{bindings['batch1_plan_sha256']}`",
        "",
        "## Repaired-prior invariant",
        "",
        "- global pLDDT alone cannot establish `easy_control_prior`.",
        "- Every non-unknown prior in the inventory has positive observable evidence.",
        "- All 48 acquisition candidates are currently `uncertain_or_unclassified`.",
        "",
        "## External acquisition counts",
        "",
        "| Asset | Candidates |",
        "|---|---:|",
    ]
    lines.extend(
        f"| `{name}` | {count} |"
        for name, count in summary["external_acquisition_counts"].items()
    )
    lines.extend(["", "## Local derivations after acquisition", "", "| Derivation | Candidates |", "|---|---:|"])
    lines.extend(
        f"| `{name}` | {count} |" for name, count in summary["local_derivation_counts"].items()
    )
    lines.extend(["", "## Primary acquisition classes", ""])
    lines.extend(
        f"- `{name}`: {count}" for name, count in summary["acquisition_class_counts"].items()
    )
    lines.extend(
        [
            "",
            "## Required post-acquisition flow",
            "",
            "```text",
            "acquire assets",
            "→ verify hashes / identities",
            "→ construct pair QC + mapping",
            "→ evaluate fragment/model identity",
            "→ recompute sampling priors",
            "→ select final Round-2 census panel",
            "→ only then run final P0/P1 protocol",
            "```",
            "",
            (
                "Post-acquisition outputs must distinguish `still_unobservable`, "
                "`observed_no_positive_prior`, `easy_control_prior`, "
                "`low_confidence_local_prior`, `high_pae_long_range_prior`, and "
                "`state_disagreement_prior`. No mechanism prevalence claim may use the "
                "pre-acquisition distribution."
            ),
            "",
            "## H6 human-authorization package",
            "",
            (
                "Canonical sequence metadata requires no independent UniProt acquisition. For "
                "the 38 missing records it is extracted locally, without sequence-identity "
                "inference, from the already planned identity-validated AFDB metadata fields "
                "`uniprotAccession` and `uniprotSequence`/`sequence`."
            ),
            "",
            (
                "AFDB confidence JSON is a direct external asset referenced by "
                "`metadata.plddtDocUrl`. Its raw bytes are stored under the normalized local "
                "filename `plddt.json`; no numeric transformation creates that file."
            ),
            "",
            (
                "| External source category | Asset | Candidates | Destination / source "
                "binding | Purpose |"
            ),
            "|---|---|---:|---|---|",
        ]
    )
    for entry in payload["h6_authorization_package"]["entries"]:
        lines.append(
            f"| {entry['external_source_category']} | `{entry['asset_type']}` | "
            f"{entry['candidate_count']} | `{entry['expected_local_destination']}`; "
            f"{entry['source_binding']} | "
            f"{entry['purpose']} |"
        )
    lines.extend(
        [
            "",
            "### Local derivations after authorized acquisition",
            "",
            "| Local derivation | Candidates | Input dependency |",
            "|---|---:|---|",
        ]
    )
    lines.extend(
        f"| `{entry['local_derivation']}` | {entry['candidate_count']} | "
        f"{entry['input_dependency']} |"
        for entry in payload["h6_authorization_package"]["local_derivations"]
    )
    lines.extend(
        [
            "",
            (
                "This package requests human authorization only for the listed external assets; "
                "it is not itself an authorization. It does not authorize executable model "
                "downloads, training, evaluator execution, ProteinMPNN generation, P0/P1/P2 "
                "execution, or candidate admission. AFDB structure files above are scientific "
                "data assets, not executable model weights."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(payload: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        output_dir / "batch1_acquisition_plan_v1.json",
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    _write_tsv(output_dir / "batch1_acquisition_plan_v1.tsv", list(payload["records"]))
    atomic_write_text(
        output_dir / "batch1_acquisition_plan_v1.md", _markdown(payload)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = (args.output_dir or project_root / OUTPUT_DIR).resolve()
    payload = build_plan(project_root)
    write_outputs(payload, output_dir)
    print(
        json.dumps(
            {
                "candidate_count": payload["summary"]["candidate_count"],
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
