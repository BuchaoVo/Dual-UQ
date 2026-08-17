"""Derive, execute, resume, or consolidate frozen formal scoring requests."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from dual_uq.core.atomic_io import atomic_write_new_bytes
from dual_uq.core.hashing import sha256_canonical, sha256_file
from dual_uq.core.paths import ProjectPathError, ProjectPaths
from dual_uq.inference.formal import (
    FormalInventoryError,
    HistoricalReuseStatus,
    MaterializationState,
    derive_and_materialize_inventory,
    derive_execution_inventory,
    materialize_historical_reuse_set,
    partition_formal_work,
    select_deterministic_preflight_pair,
)
from dual_uq.inference.materialization import (
    FormalMaterializationError,
    consolidate_formal_measurements,
    formal_shard_binding_record,
    materialize_formal_artifact,
    validate_formal_shard_payload,
)
from dual_uq.models.proteinmpnn import (
    AUTHORIZED_CHECKPOINT_SHA256,
    AUTHORIZED_IMPLEMENTATION_COMMIT,
    ProteinMPNNScoringError,
    ProteinMPNNStructureInput,
    make_decoding_realization,
    sequence_sha256,
    validate_structure_input,
)
from dual_uq.models.scoring import ScoreRequest
from dual_uq.workflows.final_confirmatory_protocol import (
    FinalConfirmatoryProtocolError,
    build_final_confirmatory_projection_resolver,
    iter_final_confirmatory_historical_score_records,
    load_final_confirmatory_formal_bundle,
)

_CODE_PATHS = (
    "src/dual_uq/inference/formal.py",
    "src/dual_uq/inference/materialization.py",
    "src/dual_uq/workflows/final_confirmatory_protocol.py",
    "src/dual_uq/models/scoring.py",
    "src/dual_uq/models/proteinmpnn.py",
    "src/dual_uq/structure/conditions.py",
    "scripts/dataset/proteinmpnn_g2_worker.py",
    "scripts/dataset/run_formal_scoring.py",
)
_FROZEN_BATCH_SIZE = 128


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--inventory-root", type=Path)
    parser.add_argument(
        "--mode",
        choices=("inventory", "preflight", "worker", "consolidate"),
        default="inventory",
    )
    parser.add_argument("--device")
    parser.add_argument("--model-python", type=Path)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=_FROZEN_BATCH_SIZE)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--staging-root", type=Path)
    return parser.parse_args(argv)


def _runtime_path(value: Path | None, *, default: Path, root: Path) -> Path:
    if value is None:
        return default.resolve()
    return value.expanduser().resolve() if value.is_absolute() else (root / value).resolve()


def _code_provenance(root: Path) -> dict[str, object]:
    try:
        head = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FormalInventoryError(
            "execution_code_identity_unavailable",
            "Unable to resolve execution code Git identity",
        ) from exc
    root = root.resolve()
    relative_paths = set(_CODE_PATHS)
    for name, loaded in tuple(sys.modules.items()):
        if name != "dual_uq" and not name.startswith("dual_uq."):
            continue
        source = getattr(loaded, "__file__", None)
        if not source:
            continue
        path = Path(source).resolve()
        if path.suffix == ".pyc":
            path = path.with_suffix(".py")
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if path.is_file():
            relative_paths.add(relative.as_posix())
    source_files = [
        {"path": relative, "sha256": sha256_file(root / relative)}
        for relative in sorted(relative_paths)
    ]
    effective_identity = sha256_canonical(
        {"git_head": head, "source_files": source_files}
    )
    return {
        "git_head": head,
        "effective_execution_code_sha256": effective_identity,
        "source_files": source_files,
    }


def validate_execution_arguments(args: argparse.Namespace) -> None:
    if args.mode in {"preflight", "worker"}:
        if args.batch_size != _FROZEN_BATCH_SIZE:
            raise FormalInventoryError(
                "runtime_configuration_mismatch",
                f"Production batch size must remain {_FROZEN_BATCH_SIZE}",
            )
        if not args.device:
            raise FormalInventoryError(
                "runtime_configuration_missing", "Execution device is required"
            )
        if args.model_python is None:
            raise FormalInventoryError(
                "runtime_configuration_missing", "model Python is required"
            )
    if (
        args.worker_count <= 0
        or args.worker_index < 0
        or args.worker_index >= args.worker_count
    ):
        raise FormalInventoryError(
            "invalid_worker_partition", "Worker index/count are invalid"
        )


def _blocked_inventory(snapshot) -> None:
    counts = snapshot.state_counts
    invalid = counts.get(MaterializationState.INVALID_EXISTING, 0)
    conflicts = counts.get(MaterializationState.CONFLICT, 0)
    reuse_pending = counts.get(MaterializationState.HISTORICAL_REUSE_ACCEPTED, 0)
    if invalid or conflicts or reuse_pending:
        raise FormalInventoryError(
            "formal_live_state_blocked",
            "Live inventory contains invalid, conflicting, or unmaterialized reuse state",
        )


def _runtime_provenance(*, root: Path, args, bundle) -> dict[str, object]:
    return {
        "materialization_source": "FRESH_EXECUTION",
        "execution_code_identity": _code_provenance(root),
        "frozen_request_aggregate_sha256": bundle.frozen_request_aggregate_sha256,
        "control_python_version": platform.python_version(),
        "device": str(args.device),
        "worker_count": args.worker_count,
        "worker_index": args.worker_index,
        "batch_size": args.batch_size,
        "precision_policy": "torch_float32_no_autocast",
        "process_boundary": "formal_score_request_worker_v1",
    }


def _build_formal_worker_request(
    selected,
    *,
    context_definitions,
    projection_resolver: Callable[[ScoreRequest], ProteinMPNNStructureInput],
) -> dict[str, object]:
    """Adapt normalized R5 definitions to the existing R4 model boundary."""
    selected = tuple(sorted(selected, key=lambda row: row.orchestration_id))
    if not selected:
        raise FormalInventoryError("empty_worker_group", "Worker group is empty")
    first = selected[0]
    request = first.request
    protein_id = request.protein_id
    collection = request.candidate_collection
    positions = request.canonical_positions
    score_contract = request.score_contract_id
    if any(
        row.request.protein_id != protein_id
        or row.request.candidate_collection != collection
        or row.request.canonical_positions != positions
        or row.request.scoring_domain_id != request.scoring_domain_id
        or row.request.score_contract_id != score_contract
        or row.scorer_binding != first.scorer_binding
        for row in selected
    ):
        raise FormalInventoryError(
            "formal_worker_group_mismatch",
            "Worker group does not share one frozen protein/scoring domain",
        )
    binding = first.scorer_binding
    if (
        binding.implementation_id != AUTHORIZED_IMPLEMENTATION_COMMIT
        or binding.checkpoint_id != AUTHORIZED_CHECKPOINT_SHA256
    ):
        raise FormalInventoryError(
            "scorer_binding_mismatch", "Worker group model identity differs"
        )
    context = tuple(
        row for row in context_definitions if row.request.protein_id == protein_id
    )
    representatives = {}
    for condition in ("PDB", "AFDB"):
        representatives[condition] = next(
            (
                row
                for row in context
                if row.request.condition.condition_id == condition
            ),
            None,
        )
        if representatives[condition] is None:
            raise FormalInventoryError(
                "formal_worker_context_missing",
                f"Worker context lacks {condition} for {protein_id}",
            )
    projections = {
        condition: projection_resolver(row.request)
        for condition, row in representatives.items()
    }
    projected_wt = "".join(collection.wt.sequence[position - 1] for position in positions)
    for condition, projection in projections.items():
        validate_structure_input(projection)
        if (
            projection.protein_id != protein_id
            or projection.backbone_condition != condition
            or projection.structure_sha256
            != representatives[condition].request.condition.structure_sha256
            or projection.uniprot_positions != positions
            or projection.wt_sequence_projection != projected_wt
        ):
            raise FormalInventoryError(
                "formal_worker_projection_mismatch",
                f"Worker {condition} projection differs from the frozen request",
            )
    projected_candidates = tuple(
        "".join(probe.sequence[position - 1] for position in positions)
        for probe in collection.probes
    )
    tasks = []
    for row in selected:
        normalized = row.request
        realization = make_decoding_realization(
            protein_id=protein_id,
            mask_length=len(positions),
            repeat_index=normalized.repeat_index,
            seed=normalized.seed,
            protocol_version=score_contract,
        )
        if (
            realization.fingerprint != normalized.realization_id
            or realization.algorithm != normalized.realization_algorithm
        ):
            raise FormalInventoryError(
                "decoding_realization_mismatch",
                "Worker realization differs from the frozen request",
            )
        tasks.append(
            {
                "binding": formal_shard_binding_record(
                    request=normalized,
                    scorer_binding=row.scorer_binding,
                    scientific_fingerprint=row.scientific_fingerprint,
                    orchestration_id=row.orchestration_id,
                ),
                "order": list(realization.order),
                "output_filename": f"{row.orchestration_id}.json",
            }
        )
    return {
        "schema_version": "stage0_formal_protein_request_v2",
        "model_identity": {
            "implementation_path": "third_party/ProteinMPNN",
            "implementation_commit": AUTHORIZED_IMPLEMENTATION_COMMIT,
            "checkpoint_path": (
                "third_party/ProteinMPNN/vanilla_model_weights/v_48_020.pt"
            ),
            "checkpoint_sha256": AUTHORIZED_CHECKPOINT_SHA256,
        },
        "scoring_protocol": score_contract,
        "protein_id": protein_id,
        "uniprot_positions": list(positions),
        "wt_sequence": projected_wt,
        "canonical_wt_sequence": collection.wt.sequence,
        "canonical_sequence_sha256": collection.wt.sequence_hash,
        "common_mask_binding": request.scoring_domain_id,
        "candidate_sequence_hashes": [
            probe.sequence_hash for probe in collection.probes
        ],
        "candidate_projection_sha256": [
            sequence_sha256(sequence) for sequence in projected_candidates
        ],
        "candidate_sequences": list(projected_candidates),
        "candidate_records": [
            {
                "sequence_hash": probe.sequence_hash,
                "full_sequence": probe.sequence,
                "position": probe.position,
                "wt_aa": probe.wt_aa,
                "mut_aa": probe.mut_aa,
            }
            for probe in collection.probes
        ],
        "pdb_coordinates": projections["PDB"].coordinates.tolist(),
        "afdb_coordinates": projections["AFDB"].coordinates.tolist(),
        "tasks": tasks,
    }


def _write_immutable_runtime_input(path: Path, payload: dict[str, object]) -> None:
    rendered = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if path.exists():
        if path.read_bytes() != rendered:
            raise FormalInventoryError(
                "formal_runtime_request_conflict",
                f"Existing worker request differs: {path.name}",
            )
        return
    atomic_write_new_bytes(path, rendered)


def _execute_via_model_worker(
    selected,
    *,
    context_definitions,
    project_root: Path,
    artifact_root: Path,
    staging_root: Path,
    model_python: Path,
    device: str,
    batch_size: int,
    execution_provenance: dict[str, object],
) -> dict[str, object]:
    """Execute protein groups through the existing data/model JSON boundary."""
    model_python = model_python.expanduser().resolve()
    worker_path = project_root / "scripts/dataset/proteinmpnn_g2_worker.py"
    if not model_python.is_file() or not worker_path.is_file():
        raise FormalInventoryError(
            "model_worker_unavailable", "Model Python or worker script is missing"
        )
    resolver = build_final_confirmatory_projection_resolver(project_root)
    grouped = defaultdict(list)
    for definition in selected:
        grouped[definition.request.protein_id].append(definition)
    counts = {
        "selected": len(tuple(selected)),
        "executed": 0,
        "protein_groups": len(grouped),
        "reused_identical": 0,
    }
    environments = []
    for protein_id in sorted(grouped):
        definitions = tuple(grouped[protein_id])
        payload = _build_formal_worker_request(
            definitions,
            context_definitions=context_definitions,
            projection_resolver=resolver,
        )
        group_id = sha256_canonical(
            {
                "protein_id": protein_id,
                "orchestration_ids": sorted(
                    row.orchestration_id for row in definitions
                ),
            }
        )
        directory = staging_root / group_id
        shard_directory = directory / "shards"
        request_path = directory / "request.json"
        response_path = directory / "response.json"
        directory.mkdir(parents=True, exist_ok=True)
        shard_directory.mkdir(parents=True, exist_ok=True)
        _write_immutable_runtime_input(request_path, payload)
        environment = os.environ.copy()
        source_root = project_root / "src"
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(source_root)
            if not existing_pythonpath
            else f"{source_root}{os.pathsep}{existing_pythonpath}"
        )
        completed = subprocess.run(
            [
                str(model_python),
                str(worker_path),
                "--formal-request",
                str(request_path),
                "--response",
                str(response_path),
                "--shard-directory",
                str(shard_directory),
                "--project-root",
                str(project_root),
                "--device",
                device,
                "--batch-size",
                str(batch_size),
            ],
            cwd=project_root,
            env=environment,
            check=False,
        )
        if completed.returncode != 0 or not response_path.is_file():
            raise FormalInventoryError(
                "model_worker_failed", f"Model worker failed for {protein_id}"
            )
        try:
            response = json.loads(response_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise FormalInventoryError(
                "invalid_model_worker_response",
                f"Model worker response is unreadable for {protein_id}",
            ) from exc
        if (
            response.get("schema_version")
            != "stage0_formal_runtime_response_v1"
            or response.get("status") != "complete"
            or response.get("request_sha256") != sha256_file(request_path)
            or response.get("completed_shards") != len(definitions)
        ):
            raise FormalInventoryError(
                "invalid_model_worker_response",
                f"Model worker response binding differs for {protein_id}",
            )
        observed_environment = response.get("execution_environment")
        if isinstance(observed_environment, dict):
            environments.append(observed_environment)
        by_id = {row.orchestration_id: row for row in definitions}
        for orchestration_id, definition in by_id.items():
            temporary = shard_directory / f"{orchestration_id}.json"
            try:
                shard = json.loads(temporary.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise FormalInventoryError(
                    "invalid_model_worker_shard",
                    f"Worker shard is unreadable: {orchestration_id}",
                ) from exc
            worker_environment = shard.get("execution_environment")
            if not isinstance(worker_environment, dict):
                raise FormalInventoryError(
                    "invalid_model_worker_shard",
                    f"Worker provenance is absent: {orchestration_id}",
                )
            shard["execution_environment"] = {
                **worker_environment,
                "control_plane_provenance": execution_provenance,
            }
            validate_formal_shard_payload(
                shard,
                request=definition.request,
                scorer_binding=definition.scorer_binding,
                scientific_fingerprint=definition.scientific_fingerprint,
                orchestration_id=definition.orchestration_id,
            )
            status = materialize_formal_artifact(
                root=artifact_root,
                artifact=definition.artifact,
                payload=shard,
                request=definition.request,
                scorer_binding=definition.scorer_binding,
                scientific_fingerprint=definition.scientific_fingerprint,
            )
            counts["executed"] += 1
            counts["reused_identical"] += int(status == "reused_identical")
            print(
                json.dumps(
                    {
                        "event": "canonical_request_complete",
                        "orchestration_id": orchestration_id,
                        "protein_id": protein_id,
                        "write_status": status,
                    }
                ),
                flush=True,
            )
    return {**counts, "execution_environments": environments}


def _inventory_result(snapshot) -> dict[str, object]:
    return {
        "total_requests": snapshot.total_requests,
        "expected_wt_rows": snapshot.expected_wt_rows,
        "expected_probe_rows": snapshot.expected_probe_rows,
        "states": {
            state.value: count for state, count in snapshot.state_counts.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_execution_arguments(args)
        paths = ProjectPaths.discover(project_root=args.project_root)
        artifact_root = _runtime_path(
            args.artifact_root,
            default=paths.runs_root / "dataset/confirmatory/formal",
            root=paths.repository_root,
        )
        inventory_root = _runtime_path(
            args.inventory_root,
            default=paths.runs_root / "dataset/confirmatory/formal/inventory_snapshots",
            root=paths.repository_root,
        )
        output_root = _runtime_path(
            args.output_root,
            default=artifact_root / "consolidated",
            root=paths.repository_root,
        )
        staging_root = _runtime_path(
            args.staging_root,
            default=paths.repository_root / "tmp/dataset_confirmatory/worker_staging",
            root=paths.repository_root,
        )
        bundle = load_final_confirmatory_formal_bundle(paths.repository_root)
        scorer_bindings = sorted(
            {
                (
                    definition.scorer_binding.scorer_id,
                    definition.scorer_binding.implementation_id,
                    definition.scorer_binding.checkpoint_id,
                    definition.scorer_binding.score_contract_id,
                )
                for definition in bundle.definitions
            }
        )
        provenance = {
            "workflow_id": "scale1b_v2_formal_scoring_inventory_v1",
            "frozen_request_aggregate_sha256": (
                bundle.frozen_request_aggregate_sha256
            ),
            "frozen_artifacts": [dict(record) for record in bundle.frozen_artifacts],
            "historical_reuse_policy": {
                "status": bundle.reuse_policy.status.value,
                "identity": bundle.reuse_policy.policy_identity,
                "candidates_reconstructed": (
                    bundle.historical_reuse_candidates_reconstructed
                ),
                "accepted": bundle.historical_reuse_accepted,
                "rejected": bundle.historical_reuse_rejected,
            },
            "scorer_bindings": [
                {
                    "scorer_id": row[0],
                    "implementation_id": row[1],
                    "checkpoint_id": row[2],
                    "score_contract_id": row[3],
                }
                for row in scorer_bindings
            ],
            "execution_code_identity": _code_provenance(paths.repository_root),
            "proteinmpnn_forward_executions": 0,
        }
        if args.mode == "inventory":
            reuse_materialization = materialize_historical_reuse_set(
                iter_final_confirmatory_historical_score_records(
                    bundle, paths.repository_root
                ),
                artifact_root=artifact_root,
            )
            result = derive_and_materialize_inventory(
                bundle.definitions,
                artifact_root=artifact_root,
                output_root=inventory_root,
                provenance=provenance,
            )
            result["historical_reuse_materialization"] = reuse_materialization
        else:
            snapshot = derive_execution_inventory(
                bundle.definitions, artifact_root=artifact_root
            )
            _blocked_inventory(snapshot)
            if args.mode == "preflight":
                selected = select_deterministic_preflight_pair(
                    snapshot, bundle.definitions
                )
                if not selected:
                    result = {
                        "status": "NO_PENDING_PREFLIGHT",
                        "inventory": _inventory_result(snapshot),
                        "proteinmpnn_forward_executions": 0,
                    }
                else:
                    execution_provenance = _runtime_provenance(
                        root=paths.repository_root, args=args, bundle=bundle
                    )
                    execution = _execute_via_model_worker(
                        selected,
                        context_definitions=bundle.definitions,
                        project_root=paths.repository_root,
                        artifact_root=artifact_root,
                        staging_root=staging_root,
                        model_python=args.model_python,
                        device=args.device,
                        batch_size=args.batch_size,
                        execution_provenance=execution_provenance,
                    )
                    refreshed = derive_execution_inventory(
                        bundle.definitions, artifact_root=artifact_root
                    )
                    result = {
                        "status": "PREFLIGHT_COMPLETE",
                        "selected": [row.workflow_request_id for row in selected],
                        "execution": execution,
                        "inventory": _inventory_result(refreshed),
                        "execution_provenance": execution_provenance,
                        "proteinmpnn_forward_executions": execution["executed"],
                    }
            elif args.mode == "worker":
                fresh_disposition = tuple(
                    definition
                    for definition in bundle.definitions
                    if definition.historical_reuse.status
                    is not HistoricalReuseStatus.ACCEPTED
                )
                partition = partition_formal_work(
                    fresh_disposition,
                    worker_count=args.worker_count,
                    worker_index=args.worker_index,
                )
                pending = {
                    entry.orchestration_id
                    for entry in snapshot.entries
                    if entry.state is MaterializationState.FRESH_EXECUTION_REQUIRED
                }
                selected = tuple(
                    definition
                    for definition in partition
                    if definition.orchestration_id in pending
                )
                if not selected:
                    result = {
                        "status": "WORKER_NOTHING_PENDING",
                        "worker_count": args.worker_count,
                        "worker_index": args.worker_index,
                        "proteinmpnn_forward_executions": 0,
                    }
                else:
                    execution_provenance = _runtime_provenance(
                        root=paths.repository_root, args=args, bundle=bundle
                    )
                    execution = _execute_via_model_worker(
                        selected,
                        context_definitions=bundle.definitions,
                        project_root=paths.repository_root,
                        artifact_root=artifact_root,
                        staging_root=staging_root,
                        model_python=args.model_python,
                        device=args.device,
                        batch_size=args.batch_size,
                        execution_provenance=execution_provenance,
                    )
                    result = {
                        "status": "WORKER_COMPLETE",
                        "worker_count": args.worker_count,
                        "worker_index": args.worker_index,
                        "execution": execution,
                        "execution_provenance": execution_provenance,
                        "proteinmpnn_forward_executions": execution["executed"],
                    }
            else:
                if snapshot.state_counts != {
                    MaterializationState.VALID_CANONICAL_COMPLETE: len(
                        bundle.definitions
                    )
                }:
                    raise FormalInventoryError(
                        "consolidation_requires_complete_inventory",
                        "All formal requests must validate before consolidation",
                    )
                result = consolidate_formal_measurements(
                    bundle.definitions,
                    artifact_root=artifact_root,
                    output_root=output_root,
                    provenance={
                        "workflow_id": "scale1b_v2_confirmatory_scoring_v1",
                        "frozen_request_aggregate_sha256": (
                            bundle.frozen_request_aggregate_sha256
                        ),
                        "execution_code_identity": _code_provenance(
                            paths.repository_root
                        ),
                    },
                )
    except (
        FinalConfirmatoryProtocolError,
        FormalInventoryError,
        FormalMaterializationError,
        ProteinMPNNScoringError,
        ProjectPathError,
    ) as exc:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_FORMAL_SCORING",
                    "failure_code": getattr(exc, "code", type(exc).__name__),
                    "message": str(exc),
                    "proteinmpnn_forward_executions": 0,
                },
                sort_keys=True,
            )
        )
        return 2
    output = dict(result)
    for key in ("inventory_path", "manifest_path", "wt_path", "probe_path"):
        if key in output:
            output[key] = str(output[key])
    output.setdefault("proteinmpnn_forward_executions", 0)
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
