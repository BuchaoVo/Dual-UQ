"""Offline eight-candidate entrypoint for the generic dataset derivation API."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from dual_uq.core.hashing import sha256_canonical, sha256_file
from dual_uq.core.paths import ProjectPaths
from dual_uq.dataset.models import (
    BiologicalIdentity,
    CandidateContext,
    DerivationConfig,
    DerivationError,
    LogicalAssetRef,
)
from dual_uq.dataset.paths import DatasetPaths
from dual_uq.dataset.pipeline import run_derivation
from dual_uq.dataset.policies.identity import extract_prediction_record_sequence
from dual_uq.dataset.reporting import build_report, write_report_mapping

DerivePilotError = DerivationError

EXPECTED_SOURCE_BINDINGS = {
    "discovery_source_sha256": "dddb21ef1e41babb71827ba729eabe5c568ab6a7d6d6c5953f6da2e10bd98436",
    "candidate_inventory_sha256": "05ee31ce8e13b699aa36ae0501a934f07b487ca9a5825e1c07a3c8c0e8f15fdf",
    "batch1_plan_sha256": "013ea4d4dd78a0269cf08afb7fe9294fb3ac1ba27bc131b875ff8ba43fec8f15",
}


def _protocol_binding(
    preflight_thresholds: Mapping[str, Any],
    observability_thresholds: Mapping[str, Any],
) -> str:
    return sha256_canonical(
        {
            "protocol_version": "dataset-a.derive-pilot.v1",
            "preflight_thresholds": dict(preflight_thresholds),
            "observability_thresholds": dict(observability_thresholds),
        }
    )


def _known_number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def validate_preflight_contract(
    current_state: Mapping[str, Any],
    *,
    expected_bindings: Mapping[str, str],
    acquisition_candidate_indices: Sequence[int],
) -> dict[str, Any]:
    """Require the frozen 48-candidate acquisition baseline before derivation."""
    summary = current_state.get("summary")
    bindings = current_state.get("source_bindings")
    if not isinstance(summary, Mapping) or not isinstance(bindings, Mapping):
        raise DerivationError("invalid_acquisition_state", "ACQ-5 state is incomplete")
    expected_summary = {
        "candidate_count": 48,
        "candidate_raw_complete": 48,
        "record_level_identity_resolved": 48,
        "current_candidate_identity_failures": 0,
    }
    observed = {key: summary.get(key) for key in expected_summary}
    if observed != expected_summary:
        raise DerivationError(
            "acquisition_baseline_drift",
            f"ACQ-5 completeness drift: expected {expected_summary}, found {observed}",
        )
    indices = [int(value) for value in acquisition_candidate_indices]
    if len(indices) != 48 or len(set(indices)) != 48:
        raise DerivationError(
            "acquisition_membership_drift", "Batch-1 must contain 48 unique candidates"
        )
    drift = {
        key: {"expected": expected, "observed": bindings.get(key)}
        for key, expected in expected_bindings.items()
        if bindings.get(key) != expected
    }
    if drift:
        raise DerivationError("source_binding_drift", f"binding drift: {drift}")
    return {
        "preflight_pass": True,
        "candidate_count": 48,
        "raw_complete_count": 48,
        "exact_record_resolved_count": 48,
        "current_identity_failure_count": 0,
        "source_bindings": dict(expected_bindings),
    }


def select_pilot_panel(
    inventory: Sequence[Mapping[str, Any]],
    acquisition: Sequence[Mapping[str, Any]],
    resolution: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select the deterministic pre-derivation eight-candidate stress panel."""
    inventory_by_index = {int(row["candidate_index"]): dict(row) for row in inventory}
    acquisition_by_index = {
        int(row["candidate_index"]): dict(row) for row in acquisition
    }
    resolution_by_index = {
        int(row["candidate_index"]): dict(row) for row in resolution
    }
    missing = sorted(set(acquisition_by_index) - set(inventory_by_index))
    if missing:
        raise DerivationError(
            "panel_binding_mismatch",
            f"Acquisition candidates missing from inventory: {missing}",
        )
    inventory_by_index = {
        index: inventory_by_index[index] for index in acquisition_by_index
    }
    selected: dict[int, dict[str, Any]] = {}

    def add(index: int, role: str, reason: str) -> None:
        row = selected.setdefault(
            index,
            {
                "candidate_index": index,
                "pair_id": acquisition_by_index[index].get("pair_id"),
                "prederivation_canonical_length": inventory_by_index[index].get(
                    "canonical_uniprot_length"
                ),
                "prederivation_global_plddt_proxy": inventory_by_index[index].get(
                    "global_pLDDT_proxy"
                ),
                "prederivation_metadata_record_count": resolution_by_index.get(
                    index, {}
                ).get("record_count", 1),
                "prederivation_pdb_entity_length": inventory_by_index[index].get(
                    "discovery_pdb_entity_length"
                ),
                "selection_roles": [],
                "selection_reason": [],
            },
        )
        row["selection_roles"].append(role)
        row["selection_reason"].append(reason)

    known_length = [
        (float(value), index)
        for index, row in inventory_by_index.items()
        if (value := _known_number(row.get("canonical_uniprot_length"))) is not None
    ]
    known_confidence = [
        (float(value), index)
        for index, row in inventory_by_index.items()
        if (value := _known_number(row.get("global_pLDDT_proxy"))) is not None
    ]
    add(
        min(known_length, key=lambda item: (item[0], item[1]))[1],
        "A_short_simple",
        "shortest known canonical length; candidate-index tie-break",
    )
    add(
        max(known_length, key=lambda item: (item[0], -item[1]))[1],
        "B_longest",
        "longest known canonical length; candidate-index tie-break",
    )
    add(
        min(known_confidence, key=lambda item: (item[0], item[1]))[1],
        "C_lower_global_plddt",
        "lowest known global-pLDDT proxy",
    )
    former = [
        index
        for index, row in resolution_by_index.items()
        if row.get("resolution_status") == "exact_record_resolved"
        and row.get("payload_provenance") == "acq3_immutable_rejected_evidence"
    ]
    if not former:
        raise DerivationError(
            "missing_multirecord_stress", "No former multi-record candidate"
        )
    largest_collection = min(
        former,
        key=lambda index: (
            -int(resolution_by_index[index].get("record_count", 0)),
            index,
        ),
    )
    add(
        largest_collection,
        "D_former_acq4_multirecord",
        "largest AFDB metadata collection; candidate-index tie-break",
    )
    class_roles = {
        "needs_pdb_side_acquisition": (
            "E_needs_pdb_side",
            "earliest Batch-1 needs-PDB-side candidate",
        ),
        "needs_both_structure_sides": (
            "F_needs_both_sides",
            "earliest Batch-1 needs-both-sides candidate",
        ),
        "needs_mapping_derivation_only": (
            "G_mapping_only",
            "earliest Batch-1 mapping-only candidate",
        ),
    }
    for acquisition_class, (role, reason) in class_roles.items():
        choices = [
            index
            for index, row in acquisition_by_index.items()
            if row.get("primary_acquisition_class") == acquisition_class
        ]
        if not choices:
            raise DerivationError(
                "missing_stress_role", f"No candidate for {acquisition_class}"
            )
        add(
            min(
                choices,
                key=lambda item: (
                    int(acquisition_by_index[item].get("batch1_rank", 10**9)),
                    item,
                ),
            ),
            role,
            reason,
        )
    complex_candidate = min(
        former,
        key=lambda index: (
            -int(resolution_by_index[index].get("record_count", 0)),
            -(
                _known_number(
                    inventory_by_index[index].get("canonical_uniprot_length")
                )
                or -1
            ),
            index,
        ),
    )
    add(
        complex_candidate,
        "H_prederivation_complexity",
        "largest known exact metadata collection then canonical length",
    )
    for index in sorted(
        former,
        key=lambda item: (
            _known_number(
                inventory_by_index[item].get("global_pLDDT_proxy")
            )
            is None,
            _known_number(inventory_by_index[item].get("global_pLDDT_proxy"))
            or float("inf"),
            item,
        ),
    ):
        if len(selected) >= 8:
            break
        add(
            index,
            "multirecord_stress_fill",
            "former ACQ-4 case; lower-pLDDT deterministic fill",
        )
    if len(selected) != 8:
        raise DerivationError(
            "pilot_panel_size", f"Expected 8 pilot candidates; found {len(selected)}"
        )
    for row in selected.values():
        row["selection_roles"] = sorted(row["selection_roles"])
        row["selection_reason"] = "; ".join(row["selection_reason"])
    return [selected[index] for index in sorted(selected)]


def attach_batch1_ranks(
    acquisition: Sequence[Mapping[str, Any]],
    plan: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Bind acquisition records to the frozen Batch-1 ordering."""
    ranks = {int(row["candidate_index"]): int(row["batch1_rank"]) for row in plan}
    indices = [int(row["candidate_index"]) for row in acquisition]
    if set(indices) != set(ranks) or len(ranks) != len(plan):
        raise DerivationError(
            "batch1_rank_binding_mismatch",
            "Acquisition membership does not exactly match frozen Batch-1 ranks",
        )
    return [
        {**dict(row), "batch1_rank": ranks[int(row["candidate_index"])]}
        for row in acquisition
    ]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DerivationError("invalid_input_json", f"Unable to read {path}") from exc
    if not isinstance(value, dict):
        raise DerivationError("invalid_input_json", f"Expected JSON object at {path}")
    return value


def _validated_preflight(paths: ProjectPaths) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset_paths = DatasetPaths.from_project(paths)
    report_paths = {
        "inventory": dataset_paths.reports / "census/candidate_inventory_v1.json",
        "batch1": dataset_paths.reports
        / "acquisition/batch1_acquisition_plan_v1.json",
        "state": dataset_paths.reports
        / "acquisition/batch1_acquisition_current_state_v2.json",
    }
    inputs = {name: _load_json(path) for name, path in report_paths.items()}
    records = inputs["batch1"].get("records")
    if not isinstance(records, list):
        raise DerivationError(
            "invalid_batch1_plan", "Batch-1 acquisition records missing"
        )
    preflight = validate_preflight_contract(
        inputs["state"],
        expected_bindings=EXPECTED_SOURCE_BINDINGS,
        acquisition_candidate_indices=[row["candidate_index"] for row in records],
    )
    bindings = inputs["state"]["source_bindings"]
    bound_paths = {
        "discovery_source_sha256": paths.resolve_logical(
            bindings["discovery_source_path"]
        ),
        "candidate_inventory_sha256": paths.resolve_logical(
            bindings["candidate_inventory_path"]
        ),
        "batch1_plan_sha256": paths.resolve_logical(bindings["batch1_plan_path"]),
    }
    observed = {key: sha256_file(path) for key, path in bound_paths.items()}
    if observed != EXPECTED_SOURCE_BINDINGS:
        raise DerivationError(
            "source_binding_drift", f"binding drift on disk: {observed}"
        )
    preflight["validated_paths"] = {
        key: paths.logical_ref(path) for key, path in bound_paths.items()
    }
    preflight["acq5_report"] = paths.logical_ref(report_paths["state"])
    return preflight, inputs


def _successful_assets(
    paths: ProjectPaths, current_state: Mapping[str, Any]
) -> dict[tuple[int, str], dict[str, Any]]:
    dataset_paths = DatasetPaths.from_project(paths)
    acq1 = _load_json(
        dataset_paths.reports / "acquisition/batch1_acquisition_run_v1.json"
    )
    rows = [
        dict(row)
        for row in acq1.get("records", [])
        if row.get("status") in {"downloaded_new", "reused_validated"}
    ]
    rows.extend(
        dict(row)
        for row in current_state.get("acq5_records", [])
        if row.get("status") in {"downloaded_new", "reused_validated"}
    )
    return {(int(row["candidate_index"]), str(row["asset_type"])): row for row in rows}


def _metadata_asset(
    paths: ProjectPaths,
    *,
    index: int,
    accession: str,
    resolution: Mapping[int, Mapping[str, Any]],
    assets: Mapping[tuple[int, str], Mapping[str, Any]],
) -> LogicalAssetRef:
    if index in resolution:
        record = resolution[index]
        return LogicalAssetRef(
            "afdb_metadata",
            str(record["payload_path"]),
            str(record["payload_SHA256"]),
            str(record["payload_provenance"]),
        )
    if (asset := assets.get((index, "afdb_metadata"))) and asset.get("local_path"):
        return LogicalAssetRef(
            "afdb_metadata",
            str(asset["local_path"]),
            str(asset.get("SHA256")) if asset.get("SHA256") else None,
            "acq1_canonical_success",
        )
    root = paths.raw_root / "afdb" / accession
    matches = sorted(root.glob("metadata.json")) + sorted(root.glob("*/metadata.json"))
    if len(matches) != 1:
        raise DerivationError(
            "metadata_path_ambiguity",
            f"Expected one local metadata path for {accession}",
        )
    return LogicalAssetRef(
        "afdb_metadata",
        paths.logical_ref(matches[0]),
        None,
        "preexisting_local_canonical",
    )


def _dependent_asset(
    paths: ProjectPaths,
    *,
    index: int,
    accession: str,
    model_id: str,
    asset_type: str,
    assets: Mapping[tuple[int, str], Mapping[str, Any]],
) -> LogicalAssetRef:
    if (bound := assets.get((index, asset_type))) and bound.get("local_path"):
        return LogicalAssetRef(
            asset_type,
            str(bound["local_path"]),
            str(bound.get("SHA256")) if bound.get("SHA256") else None,
            "acquisition_ledger_bound",
            str(bound.get("exact_record_model_identity"))
            if bound.get("exact_record_model_identity")
            else None,
        )
    names = {
        "afdb_structure": ("model.cif", f"{model_id}-model_v6.cif"),
        "afdb_pae": ("pae.json", f"{model_id}-predicted_aligned_error_v6.json"),
        "afdb_confidence": ("plddt.json", f"{model_id}-confidence_v6.json"),
    }[asset_type]
    root = paths.raw_root / "afdb" / accession
    candidates = [root / name for name in names]
    candidates.extend(root / model_id / name for name in names)
    matches = [path for path in candidates if path.is_file()]
    if not matches:
        raise DerivationError("missing_raw_asset", f"Missing {asset_type} for {model_id}")
    if len({sha256_file(path) for path in matches}) != 1:
        raise DerivationError(
            "ambiguous_local_asset_set",
            f"Multiple non-identical local {asset_type} files exist for {model_id}",
        )
    return LogicalAssetRef(
        asset_type,
        paths.logical_ref(matches[0]),
        None,
        "preexisting_local_canonical",
    )


def _side_asset(
    paths: ProjectPaths,
    *,
    index: int,
    pdb_id: str,
    asset_type: str,
    assets: Mapping[tuple[int, str], Mapping[str, Any]],
) -> LogicalAssetRef:
    bound = assets.get((index, asset_type))
    logical_path = (
        str(bound["local_path"])
        if bound and bound.get("local_path")
        else (
            f"data/raw/pdb/{pdb_id}.cif"
            if asset_type == "pdb_mmcif"
            else f"data/raw/mappings/{pdb_id}.xml.gz"
        )
    )
    return LogicalAssetRef(
        asset_type,
        logical_path,
        str(bound.get("SHA256")) if bound and bound.get("SHA256") else None,
        "acquisition_ledger_bound" if bound else "preexisting_local_canonical",
    )


def _candidate_context(
    paths: ProjectPaths,
    row: Mapping[str, Any],
    selection: Mapping[str, Any],
    *,
    resolution: Mapping[int, Mapping[str, Any]],
    assets: Mapping[tuple[int, str], Mapping[str, Any]],
    protocol_binding: str,
) -> CandidateContext:
    index = int(row["candidate_index"])
    accession = str(row["UniProt"])
    pdb_id = str(row["PDB"]).lower()
    metadata = _metadata_asset(
        paths,
        index=index,
        accession=accession,
        resolution=resolution,
        assets=assets,
    )
    exact = extract_prediction_record_sequence(
        paths.resolve_logical(metadata.logical_path).read_bytes(), accession
    )
    model_id = str(exact["model_entity_id"])
    logical_assets = (
        metadata,
        _side_asset(
            paths,
            index=index,
            pdb_id=pdb_id,
            asset_type="pdb_mmcif",
            assets=assets,
        ),
        _side_asset(
            paths,
            index=index,
            pdb_id=pdb_id,
            asset_type="sifts",
            assets=assets,
        ),
        *(
            _dependent_asset(
                paths,
                index=index,
                accession=accession,
                model_id=model_id,
                asset_type=asset_type,
                assets=assets,
            )
            for asset_type in (
                "afdb_structure",
                "afdb_pae",
                "afdb_confidence",
            )
        ),
    )
    return CandidateContext(
        candidate_index=index,
        identity=BiologicalIdentity(
            pair_id=str(row["pair_id"]),
            pdb_id=pdb_id,
            chain_id=str(row["chain"]),
            uniprot_accession=accession,
            polymer_entity_id=str(row["polymer_entity_id"]),
        ),
        exact_afdb_accession=accession,
        expected_afdb_model_identity=model_id,
        assets=logical_assets,
        source_bindings=tuple(sorted(EXPECTED_SOURCE_BINDINGS.items())),
        protocol_binding=protocol_binding,
        selection_roles=tuple(selection["selection_roles"]),
        selection_reason=str(selection["selection_reason"]),
        prederivation_evidence=(
            (
                "prederivation_canonical_length",
                selection["prederivation_canonical_length"],
            ),
            (
                "global_pLDDT_proxy",
                selection["prederivation_global_plddt_proxy"],
            ),
            (
                "prederivation_metadata_record_count",
                selection["prederivation_metadata_record_count"],
            ),
            ("pdb_entity_length", selection["prederivation_pdb_entity_length"]),
        ),
    )


def _run_pilot(paths: ProjectPaths) -> dict[str, Any]:
    dataset_paths = DatasetPaths.from_project(paths)
    preflight, inputs = _validated_preflight(paths)
    inventory_rows = inputs["inventory"].get("candidates", [])
    acquisition_rows = inputs["batch1"].get("records", [])
    plan = pd.read_csv(
        dataset_paths.reports / "census/batch1_plan_v1.tsv", sep="\t"
    ).to_dict("records")
    acquisition_rows = attach_batch1_ranks(acquisition_rows, plan)
    resolution_report = _load_json(
        dataset_paths.audits
        / "acquisition/afdb_metadata_record_resolution_v1.json"
    )
    resolution = {
        int(row["candidate_index"]): dict(row)
        for row in resolution_report.get("records", [])
    }
    selected = select_pilot_panel(
        inventory_rows, acquisition_rows, list(resolution.values())
    )
    selection = {int(row["candidate_index"]): row for row in selected}
    acquisition = {
        int(row["candidate_index"]): row for row in acquisition_rows
    }
    assets = _successful_assets(paths, inputs["state"])
    observability = yaml.safe_load(
        (
            paths.repository_root
            / "configs/legacy/a0_screening/a0_selection.yaml"
        ).read_text(encoding="utf-8")
    )
    preflight_config = yaml.safe_load(
        (
            paths.repository_root
            / "configs/legacy/a0_screening/screening_preflight.yaml"
        ).read_text(encoding="utf-8")
    )
    protocol_binding = _protocol_binding(
        preflight_config["thresholds"], observability["thresholds"]
    )
    contexts = [
        _candidate_context(
            paths,
            acquisition[index],
            selection[index],
            resolution=resolution,
            assets=assets,
            protocol_binding=protocol_binding,
        )
        for index in sorted(selection)
    ]
    config = DerivationConfig(
        protocol_version="dataset-a.derive-pilot.v1",
        protocol_binding=protocol_binding,
        preflight_thresholds=preflight_config["thresholds"],
        observability_thresholds=observability["thresholds"],
        schema_version="dataset-a.derive-pilot.v1",
        preflight=preflight,
        scope={
            "purpose": "derivation protocol stress test, not inference or admission",
            "candidate_admission_changed": False,
            "raw_evidence_mutated": False,
            "p2_started": False,
            "proteinmpnn_executed": False,
            "evaluator_executed": False,
            "network_used": False,
        },
        selection_policy={
            "deterministic": True,
            "uses_prederivation_metadata_only": True,
            "candidate_indices": [int(row["candidate_index"]) for row in selected],
        },
        report_metadata={
            "candidate_count_field": "pilot_candidate_count",
            "summary_fields": {"derive_48_started": False},
        },
    )
    return build_report(run_derivation(contexts, config, paths))


def run_pilot(project_root: Path) -> dict[str, Any]:
    """Compatibility wrapper for the portable generic derivation runner."""
    return _run_pilot(ProjectPaths.discover(project_root=project_root.resolve()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--reports-root", type=Path)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--artifacts-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = ProjectPaths.discover(
        project_root=args.project_root.resolve() if args.project_root else None,
        anchor=Path(__file__),
        data_root=args.data_root,
        reports_root=args.reports_root,
        runs_root=args.runs_root,
        artifacts_root=args.artifacts_root,
    )
    report = _run_pilot(paths)
    output_dir = args.output_dir or DatasetPaths.from_project(paths).reports / "derivation"
    if not output_dir.is_absolute():
        output_dir = paths.repository_root / output_dir
    write_report_mapping(report, output_dir)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
