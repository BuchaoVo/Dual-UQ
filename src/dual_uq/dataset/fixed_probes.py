"""Frozen common-mask and backbone-independent fixed-probe materialization."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from dual_uq.core.hashing import sha256_bytes, sha256_file
from dual_uq.dataset.policies.fragments import (
    resolve_exact_fragment,
    validate_frozen_model_artifacts,
)
from dual_uq.dataset.policies.identity import (
    extract_canonical_sequence,
    metadata_records,
)
from dual_uq.dataset.releases.intervention_admission import (
    validate_stage0_admission_bindings,
    write_immutable_release,
)
from dual_uq.dataset.services.backbone import paired_common_backbone_positions
from dual_uq.dataset.services.mapping import (
    mapping_with_provenance,
    parse_sifts_mapping_with_explicit_labels,
)
from dual_uq.dataset.storage.proteinmpnn import (
    STANDARD_AMINO_ACIDS,
    sequence_sha256,
    validate_protein_sequence,
)
from dual_uq.structure_io import load_chain_ca_table

ADMITTED_SUBSET_SHA256 = (
    "0f29fe5309a63ae0e24eda2ebdacfad861c418165bb2552b6f68e40855d6d3f5"
)
MASK_COORDINATE_SYSTEM = "uniprot_position_1based"
PROTEIN_MANIFEST_VERSION = "stage0_protein_manifest_v1"
FIXED_PROBE_DATASET_VERSION = "stage0_fixed_probe_candidates_v1"
FIXED_PROBE_COLUMNS = (
    "protein_id",
    "uniprot_accession",
    "position",
    "wt_aa",
    "mut_aa",
    "sequence_hash",
    "full_sequence",
)
_SUBSET_PATH = (
    "experiments/p2_design_baseline/stage0/"
    "stage0_intervention_admitted_v1.jsonl"
)
_ACQUISITION_RUN_PATH = "reports/dataset_a_scale/batch1_acquisition_run_v1.json"
_ACQUISITION_PLAN_PATH = "reports/dataset_a_scale/batch1_acquisition_plan_v1.json"


class FixedProbeError(ValueError):
    """A structured Stage-0 fixed-probe materialization blocker."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def _portable(value: str, field: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value:
        raise FixedProbeError("nonportable_path", f"{field} is not repository-relative")
    return value


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FixedProbeError("invalid_json", f"Unable to read JSON: {path.name}") from exc


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FixedProbeError("invalid_jsonl", f"Unable to read JSONL: {path.name}") from exc


def _stage0_1_config_sha256(config: dict[str, Any]) -> str:
    """Hash only the historical Stage-0-1 config view bound by its manifest.

    Later stages extend the shared YAML without retroactively changing the
    provenance identity of already frozen Stage-0-1 outputs.
    """
    input_fields = {
        "stage0_intervention_panel": ("panel_version", "path"),
        "stage0_intervention_admission": ("admission_version", "path"),
        "stage0_intervention_admitted_subset": (
            "admission_version",
            "subset_version",
            "path",
        ),
    }
    try:
        historical = {
            "experiment_id": config["experiment_id"],
            "inputs": {
                key: {field: config["inputs"][key][field] for field in fields}
                for key, fields in input_fields.items()
            },
        }
    except (KeyError, TypeError) as exc:
        raise FixedProbeError(
            "invalid_config", "Stage-0 config lacks the frozen Stage-0-1 view"
        ) from exc
    rendered = yaml.safe_dump(historical, sort_keys=False).encode("utf-8")
    return sha256_bytes(rendered)


def _verified_value(record: dict[str, Any], field: str) -> Any:
    value = record.get(field)
    if not isinstance(value, dict) or value.get("status") != "verified":
        raise FixedProbeError("unresolved_frozen_asset", f"{field} is not verified")
    return value


def _sifts_binding(
    *,
    candidate_index: int,
    protein_id: str,
    pdb_id: str,
    acquisition_run: dict[str, Any],
    acquisition_plan: dict[str, Any],
    root: Path,
) -> tuple[str, str, str]:
    records = [
        record
        for record in acquisition_run.get("records", [])
        if record.get("candidate_index") == candidate_index
        and record.get("pair_id") == protein_id
        and record.get("asset_type") == "sifts"
        and record.get("failure_code") is None
    ]
    if len(records) == 1:
        record = records[0]
        logical = _portable(str(record["local_path"]), "sifts_path")
        expected = str(record["SHA256"])
        mode = "acquisition_run_exact_asset_record"
    elif not records:
        plan_rows = [
            record
            for record in acquisition_plan.get("records", [])
            if record.get("candidate_index") == candidate_index
            and record.get("pair_id") == protein_id
        ]
        if len(plan_rows) != 1 or plan_rows[0].get("asset_statuses", {}).get(
            "sifts"
        ) != "available_local":
            raise FixedProbeError(
                "missing_sifts_binding", f"No exact SIFTS binding for {protein_id}"
            )
        logical = f"data/raw/mappings/{pdb_id}.xml.gz"
        expected = sha256_file(root / logical)
        mode = "canonical_local_sifts_path_from_verified_pdb_identity"
    else:
        raise FixedProbeError(
            "ambiguous_sifts_binding", f"Multiple SIFTS bindings for {protein_id}"
        )
    actual = sha256_file(root / logical)
    if actual != expected:
        raise FixedProbeError("sifts_hash_drift", f"SIFTS SHA differs for {protein_id}")
    return logical, actual, mode


def _validate_subset(config: dict[str, Any], root: Path) -> tuple[Path, list[dict[str, Any]]]:
    inputs = config.get("inputs") if isinstance(config, dict) else None
    binding = inputs.get("stage0_intervention_admitted_subset") if isinstance(inputs, dict) else None
    if not isinstance(binding, dict) or binding.get("path") != _SUBSET_PATH:
        raise FixedProbeError(
            "invalid_admitted_subset_binding", "admitted subset binding differs"
        )
    path = root / _SUBSET_PATH
    if sha256_file(path) != ADMITTED_SUBSET_SHA256:
        raise FixedProbeError("admitted_subset_hash_mismatch", "Admitted subset SHA differs")
    records = _jsonl(path)
    if len(records) != 8:
        raise FixedProbeError("admitted_subset_count_mismatch", "Expected eight records")
    identities = [(row.get("candidate_index"), row.get("protein_id")) for row in records]
    if len(set(identities)) != 8 or any(
        row.get("intervention_admission_status") != "ADMITTED" for row in records
    ):
        raise FixedProbeError("invalid_admitted_subset", "Subset identities/statuses differ")
    return path, records


def build_protein_manifest(config_path: Path, project_root: Path) -> dict[str, Any]:
    """Resolve frozen inputs and construct eight canonical common masks in memory."""
    root = project_root.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise FixedProbeError("invalid_config", "Stage-0 config must be a mapping")
    subset_path, subset = _validate_subset(config, root)
    try:
        bindings = validate_stage0_admission_bindings(config_path)
    except ValueError as exc:
        raise FixedProbeError("invalid_config", str(exc)) from exc
    panel_path = root / bindings["stage0_intervention_panel"]
    panel_bindings = {
        (
            row.get("audit_provenance", {}).get("original_panel_path"),
            row.get("audit_provenance", {}).get("original_panel_sha256"),
        )
        for row in subset
    }
    if len(panel_bindings) != 1:
        raise FixedProbeError(
            "ambiguous_panel_binding", "Admitted records do not bind one panel"
        )
    bound_panel_path, bound_panel_sha = next(iter(panel_bindings))
    if (
        bound_panel_path != bindings["stage0_intervention_panel"]
        or not isinstance(bound_panel_sha, str)
        or len(bound_panel_sha) != 64
    ):
        raise FixedProbeError(
            "invalid_panel_binding", "Admitted subset panel binding differs"
        )
    if sha256_file(panel_path) != bound_panel_sha:
        raise FixedProbeError("panel_hash_drift", "Panel SHA differs")
    panel_records = _jsonl(panel_path)
    panel_by_id = {row["protein_id"]: row for row in panel_records}
    if len(panel_by_id) != len(panel_records):
        raise FixedProbeError("ambiguous_panel_identity", "Panel protein identities repeat")
    acquisition_run_path = root / _ACQUISITION_RUN_PATH
    acquisition_plan_path = root / _ACQUISITION_PLAN_PATH
    acquisition_run = _json(acquisition_run_path)
    acquisition_plan = _json(acquisition_plan_path)

    proteins: list[dict[str, Any]] = []
    for admitted in subset:
        protein_id = str(admitted["protein_id"])
        panel = panel_by_id.get(protein_id)
        if not isinstance(panel, dict) or panel.get("candidate_index") != admitted.get(
            "candidate_index"
        ):
            raise FixedProbeError("panel_subset_identity_mismatch", protein_id)
        verification = panel.get("repository_verification")
        if not isinstance(verification, dict):
            raise FixedProbeError("missing_repository_verification", protein_id)
        pdb_record = _verified_value(verification, "pdb_backbone_path")
        afdb_record = _verified_value(verification, "afdb_backbone_path")
        metadata_record = _verified_value(verification, "afdb_metadata_path")
        entity_record = _verified_value(verification, "pdb_entity_id")
        model_record = _verified_value(verification, "afdb_model_id")
        sequence_record = _verified_value(verification, "canonical_sequence_provenance")
        pdb_path_ref = _portable(str(pdb_record["value"]), "pdb_backbone_path")
        afdb_path_ref = _portable(str(afdb_record["value"]), "afdb_backbone_path")
        metadata_ref = _portable(str(metadata_record["value"]), "afdb_metadata_path")
        pdb_path, afdb_path, metadata_path = (
            root / pdb_path_ref,
            root / afdb_path_ref,
            root / metadata_ref,
        )
        for path, expected, label in (
            (pdb_path, pdb_record["sha256"], "PDB"),
            (afdb_path, afdb_record["sha256"], "AFDB"),
            (metadata_path, metadata_record["sha256"], "metadata"),
        ):
            if sha256_file(path) != expected:
                raise FixedProbeError("frozen_asset_hash_drift", f"{label} SHA differs")
        accession = str(panel["uniprot_accession"])
        metadata_bytes = metadata_path.read_bytes()
        canonical = extract_canonical_sequence(metadata_bytes, accession)
        sequence = validate_protein_sequence(str(canonical["sequence"]))
        if canonical["sequence_sha256"] != sequence_record["value"]["sequence_sha256"]:
            raise FixedProbeError("canonical_sequence_hash_drift", protein_id)
        sifts_ref, sifts_sha, sifts_mode = _sifts_binding(
            candidate_index=int(admitted["candidate_index"]),
            protein_id=protein_id,
            pdb_id=str(panel["pdb_id"]),
            acquisition_run=acquisition_run,
            acquisition_plan=acquisition_plan,
            root=root,
        )
        mapping = parse_sifts_mapping_with_explicit_labels(
            root / sifts_ref,
            pdb_path,
            chain_id=str(panel["pdb_chain"]),
            uniprot_id=accession,
        )
        pdb_ca = load_chain_ca_table(pdb_path, str(panel["pdb_chain"]))
        mapping, gaps, join_diagnostics = mapping_with_provenance(mapping, pdb_ca)
        mapped_interval = (
            int(mapping["uniprot_position"].min()),
            int(mapping["uniprot_position"].max()),
        )
        fragment_result = resolve_exact_fragment(
            metadata_records(metadata_bytes), accession, mapped_interval
        )
        fragment = fragment_result["selected_fragment"]
        model_id = str(model_record["value"])
        if fragment is None or fragment.model_entity_id != model_id:
            raise FixedProbeError("unsupported_afdb_fragment", protein_id)
        exact_record = canonical["exact_record"]
        version = exact_record.get("latestVersion")
        if not isinstance(version, int):
            raise FixedProbeError("invalid_afdb_version", protein_id)
        validate_frozen_model_artifacts(
            exact_record,
            model_id=model_id,
            version=version,
            model_path=afdb_path,
            expected_length=fragment.model_residue_count,
        )
        mask_positions = list(
            paired_common_backbone_positions(
                mapping,
                canonical_sequence=sequence,
                pair_id=protein_id,
                pdb_path=pdb_path,
                afdb_path=afdb_path,
                fragment=fragment,
            )
        )
        proteins.append(
            {
                "candidate_index": int(admitted["candidate_index"]),
                "protein_id": protein_id,
                "intervention_admission_status": "ADMITTED",
                "uniprot_accession": accession,
                "pdb_id": str(panel["pdb_id"]),
                "pdb_chain": str(panel["pdb_chain"]),
                "pdb_entity_id": str(entity_record["value"]),
                "afdb_model_id": model_id,
                "canonical_wt_sequence": sequence,
                "canonical_sequence_length": len(sequence),
                "canonical_sequence_sha256": sequence_sha256(sequence),
                "canonical_sequence_provenance": {
                    **dict(sequence_record["value"]),
                    "source_path": metadata_ref,
                    "source_sha256": str(metadata_record["sha256"]),
                },
                "pdb_backbone_path": pdb_path_ref,
                "pdb_backbone_sha256": str(pdb_record["sha256"]),
                "afdb_backbone_path": afdb_path_ref,
                "afdb_backbone_sha256": str(afdb_record["sha256"]),
                "mapping_provenance": {
                    "sifts_path": sifts_ref,
                    "sifts_sha256": sifts_sha,
                    "sifts_binding_mode": sifts_mode,
                    "mapping_adapter": "parse_sifts_mapping_with_explicit_labels",
                    "backbone_adapter": "frozen_p1_n_ca_c_o_selection",
                    "mapped_interval": list(mapped_interval),
                    "segment_count": int(gaps["segment_count"]),
                    "gap_count": int(gaps["gap_count"]),
                    "largest_uniprot_gap": int(gaps["largest_uniprot_gap"]),
                    "gap_boundaries": gaps["gap_boundaries"],
                    "residue_join_diagnostics": join_diagnostics,
                },
                "mask_coordinate_system": MASK_COORDINATE_SYSTEM,
                "mask_positions": mask_positions,
                "mask_length": len(mask_positions),
            }
        )
    protocol_paths = {
        row.get("audit_provenance", {}).get("pdr01_protocol_path") for row in subset
    }
    if len(protocol_paths) != 1 or not isinstance(next(iter(protocol_paths)), str):
        raise FixedProbeError(
            "ambiguous_protocol_binding", "Admitted records do not bind one PDR-01"
        )
    pdr01_ref = _portable(next(iter(protocol_paths)), "pdr01_protocol_path")
    pdr01_path = root / pdr01_ref
    if not pdr01_path.is_file():
        raise FixedProbeError("missing_protocol_binding", "PDR-01 binding is absent")
    config_ref = config_path.resolve().relative_to(root).as_posix()
    return {
        "schema_version": PROTEIN_MANIFEST_VERSION,
        "experiment_id": str(config["experiment_id"]),
        "cohort": {
            "admitted_subset_path": subset_path.relative_to(root).as_posix(),
            "admitted_subset_sha256": ADMITTED_SUBSET_SHA256,
            "admission_version": subset[0]["admission_version"],
            "subset_version": subset[0]["subset_provenance"]["subset_version"],
            "panel_version": subset[0]["panel_version"],
            "record_count": len(subset),
        },
        "provenance": {
            "config_path": config_ref,
            "config_sha256": _stage0_1_config_sha256(config),
            "pdr01_protocol_path": pdr01_ref,
            "pdr01_protocol_sha256": sha256_file(pdr01_path),
            "panel_path": panel_path.relative_to(root).as_posix(),
            "panel_sha256": bound_panel_sha,
            "acquisition_run_path": _ACQUISITION_RUN_PATH,
            "acquisition_run_sha256": sha256_file(acquisition_run_path),
            "acquisition_plan_path": _ACQUISITION_PLAN_PATH,
            "acquisition_plan_sha256": sha256_file(acquisition_plan_path),
            "sequence_hash_convention": (
                "SHA256(uppercase validated 20-AA full sequence encoded as ASCII; "
                "byte-identical to UTF-8)"
            ),
            "scientific_implementation": "dual_uq.dataset.fixed_probes.v1",
        },
        "proteins": proteins,
        "summary": {
            "protein_count": len(proteins),
            "total_mask_positions": sum(record["mask_length"] for record in proteins),
        },
    }


def build_fixed_probe_candidates(manifest: dict[str, Any]) -> pd.DataFrame:
    """Generate 19 deterministic single substitutions per common-mask position."""
    rows: list[dict[str, Any]] = []
    for protein in manifest["proteins"]:
        sequence = validate_protein_sequence(str(protein["canonical_wt_sequence"]))
        for position in protein["mask_positions"]:
            position = int(position)
            wt_aa = sequence[position - 1]
            for mut_aa in STANDARD_AMINO_ACIDS:
                if mut_aa == wt_aa:
                    continue
                mutated = f"{sequence[: position - 1]}{mut_aa}{sequence[position:]}"
                rows.append(
                    {
                        "protein_id": protein["protein_id"],
                        "uniprot_accession": protein["uniprot_accession"],
                        "position": position,
                        "wt_aa": wt_aa,
                        "mut_aa": mut_aa,
                        "sequence_hash": sequence_sha256(mutated),
                        "full_sequence": mutated,
                    }
                )
    frame = pd.DataFrame(rows, columns=FIXED_PROBE_COLUMNS)
    frame["position"] = frame["position"].astype("int32")
    for column in set(FIXED_PROBE_COLUMNS) - {"position"}:
        frame[column] = frame[column].astype("string")
    if frame.duplicated(["protein_id", "position", "mut_aa"]).any():
        raise FixedProbeError("duplicate_probe_identity", "Probe identities repeat")
    return frame


def render_protein_manifest(manifest: dict[str, Any]) -> bytes:
    payload = (
        json.dumps(
            manifest,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if b"/home/" in payload or b"/mnt/" in payload:
        raise FixedProbeError("nonportable_path", "Manifest contains an absolute path")
    return payload


def render_fixed_probe_parquet(frame: pd.DataFrame) -> bytes:
    if tuple(frame.columns) != FIXED_PROBE_COLUMNS:
        raise FixedProbeError("invalid_probe_schema", "Probe columns differ")
    schema = pa.schema(
        [
            pa.field("protein_id", pa.string(), nullable=False),
            pa.field("uniprot_accession", pa.string(), nullable=False),
            pa.field("position", pa.int32(), nullable=False),
            pa.field("wt_aa", pa.string(), nullable=False),
            pa.field("mut_aa", pa.string(), nullable=False),
            pa.field("sequence_hash", pa.string(), nullable=False),
            pa.field("full_sequence", pa.string(), nullable=False),
        ]
    )
    table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
    buffer = BytesIO()
    pq.write_table(
        table,
        buffer,
        compression="snappy",
        version="2.6",
        data_page_version="1.0",
        use_dictionary=False,
        write_statistics=True,
        row_group_size=len(frame),
    )
    return buffer.getvalue()


def materialize_fixed_probes(config_path: Path, project_root: Path) -> dict[str, Any]:
    """Validate all inputs before immutably writing the two Stage-0 outputs."""
    root = project_root.resolve()
    manifest = build_protein_manifest(config_path, root)
    probes = build_fixed_probe_candidates(manifest)
    manifest_bytes = render_protein_manifest(manifest)
    parquet_bytes = render_fixed_probe_parquet(probes)
    output_dir = root / "experiments/p2_design_baseline/stage0"
    outputs = {
        output_dir / "protein_manifest.json": manifest_bytes,
        output_dir / "fixed_probe_candidates.parquet": parquet_bytes,
    }
    conflicts = [path for path, payload in outputs.items() if path.exists() and path.read_bytes() != payload]
    if conflicts:
        raise FixedProbeError("immutable_output_conflict", conflicts[0].name)
    statuses = {
        path.relative_to(root).as_posix(): write_immutable_release(path, payload)
        for path, payload in outputs.items()
    }
    return {
        "protein_manifest_sha256": sha256_bytes(manifest_bytes),
        "protein_manifest_bytes": len(manifest_bytes),
        "fixed_probe_candidates_sha256": sha256_bytes(parquet_bytes),
        "fixed_probe_candidates_bytes": len(parquet_bytes),
        "protein_count": len(manifest["proteins"]),
        "mask_position_count": manifest["summary"]["total_mask_positions"],
        "probe_count": len(probes),
        "write_status": statuses,
    }
