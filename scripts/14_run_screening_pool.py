from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from dual_uq.screening_runner import (
    GEOMETRY,
    PAIR,
    ROBUST,
    SEGMENT_CONTEXT,
    STAGES,
    append_status_record,
    build_stage_plan,
    derive_stage_seed,
    execute_stage,
    make_planned_status_record,
    reduce_status_history,
    validate_stage_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the screening pool with preflight gates and stage-level resume."
    )
    parser.add_argument("--config", default="configs/screening_pool.yaml")
    parser.add_argument("--pool", default="data/manifests/screening_pool.tsv")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--only", default=None)
    parser.add_argument("--include-warnings", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--force-stage",
        action="append",
        choices=STAGES,
        default=[],
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--status-path", default=None)
    return parser.parse_args()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _read_status(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            print(f"Warning: ignoring invalid status JSONL line {line_number}.")
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _attempt(records: list[dict[str, Any]], index: int, stage: str) -> int:
    state = reduce_status_history(records).get((index, stage))
    return 1 if state is None else state.attempt + 1


def _commands(
    *,
    root: Path,
    candidate: dict[str, Any],
    batch: dict[str, Any],
    base_seed: int,
) -> dict[str, list[str]]:
    python = sys.executable
    pair_dir = root / "data/processed/pairs" / candidate["pair_name"]
    return {
        PAIR: [
            python,
            "scripts/03_build_pair.py",
            "--pdb-id",
            candidate["pdb_id"],
            "--chain-id",
            candidate["chain_id"],
            "--uniprot-id",
            candidate["uniprot_id"],
            "--project-root",
            str(root),
        ],
        GEOMETRY: [
            python,
            "scripts/06_analyze_pair_geometry.py",
            "--pair-report",
            str(pair_dir / "pair_qc.json"),
            "--pair-min-separation",
            str(batch["pair_min_separation"]),
        ],
        ROBUST: [
            python,
            "scripts/09_robust_pair_diagnostics.py",
            "--pair-dir",
            str(pair_dir),
            "--local-permutations",
            str(batch["local_permutations"]),
            "--pair-permutations",
            str(batch["pair_permutations"]),
            "--min-separation",
            str(batch["pair_min_separation"]),
            "--seed",
            str(
                derive_stage_seed(
                    base_seed,
                    int(candidate["screening_index"]),
                    ROBUST,
                )
            ),
        ],
        SEGMENT_CONTEXT: [
            python,
            "scripts/11_characterize_disagreement_segments.py",
            "--pair-dir",
            str(pair_dir),
            "--threshold",
            str(batch["segment_threshold"]),
            "--min-length",
            str(batch["segment_min_length"]),
            "--flank-size",
            str(batch["segment_flank_size"]),
        ],
    }


def _print_plan(index: int, candidate: dict[str, Any], plan: list[Any]) -> None:
    print(
        f"[{index:03d}] {candidate['pdb_id']}:{candidate['chain_id']} "
        f"↔ {candidate['uniprot_id']}"
    )
    for item in plan:
        print(f"  {item.stage:<18} {item.action:<18} {item.reason}")


def _candidate_from_row(row: Any, batch: dict[str, Any]) -> dict[str, Any]:
    index = int(row.screening_index)
    pdb_id = str(row.pdb_id).lower()
    chain_id = str(row.chain_id)
    uniprot_id = str(row.uniprot_id).upper()
    return {
        "screening_index": index,
        "pdb_id": pdb_id,
        "chain_id": chain_id,
        "uniprot_id": uniprot_id,
        "pair_name": f"{pdb_id}_{chain_id}__{uniprot_id}",
        "provisional_stratum": getattr(row, "provisional_stratum", None),
        "preflight_status": str(getattr(row, "preflight_status", "not_run")),
        "segment_threshold": float(batch["segment_threshold"]),
        "segment_min_length": int(batch["segment_min_length"]),
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = Path(args.project_root or config["project_root"]).expanduser().resolve()
    batch = config["batch"]
    base_seed = int(
        args.seed
        if args.seed is not None
        else batch.get(
            "seed",
            config.get("discovery", {}).get("random_seed", 20260730),
        )
    )

    pool_path = _resolve(root, args.pool)
    preflight_path = _resolve(
        root,
        config.get(
            "preflight_path",
            "data/manifests/screening_pool_preflight.tsv",
        ),
    )
    pool = pd.read_csv(pool_path, sep="\t")
    if not preflight_path.exists():
        raise FileNotFoundError(f"Preflight manifest not found: {preflight_path}")
    preflight = pd.read_csv(preflight_path, sep="\t")
    preflight = preflight.drop_duplicates("screening_index", keep="last")
    pool = pool.drop(columns=["preflight_status"], errors="ignore").merge(
        preflight[["screening_index", "preflight_status"]],
        on="screening_index",
        how="left",
        validate="one_to_one",
    )
    pool["preflight_status"] = pool["preflight_status"].fillna("not_run")

    if args.only:
        requested = {int(value) for value in args.only.split(",") if value.strip()}
        pool = pool.loc[pool["screening_index"].astype(int).isin(requested)]
    else:
        pool = pool.loc[pool["screening_index"].astype(int) >= args.start_index]
        if args.limit is not None:
            pool = pool.head(args.limit)

    status_path = (
        _resolve(root, args.status_path)
        if args.status_path
        else root / "reports/screening_pool_status.jsonl"
    )
    records = _read_status(status_path)
    run_id = uuid.uuid4().hex
    requested_force = set(args.force_stage)
    if args.force or not args.resume:
        requested_force = set(STAGES)
    elif requested_force:
        first = min(STAGES.index(stage) for stage in requested_force)
        requested_force = set(STAGES[first:])

    stop_after_failure = False
    for row in pool.sort_values("screening_index").itertuples(index=False):
        candidate = _candidate_from_row(row, batch)
        index = int(candidate["screening_index"])
        candidate["commands"] = _commands(
            root=root,
            candidate=candidate,
            batch=batch,
            base_seed=base_seed,
        )
        remaining_force = set(requested_force)

        output_state = {
            stage: validate_stage_outputs(stage, candidate, root) for stage in STAGES
        }
        plan = build_stage_plan(
            candidate,
            records,
            output_state,
            resume=True,
            force_stages=remaining_force,
            base_seed=base_seed,
            include_warnings=args.include_warnings,
        )
        _print_plan(index, candidate, plan)
        if args.plan_only:
            continue

        if all(item.action in {"skip_preflight", "skip_unsupported"} for item in plan):
            for item in plan:
                status = (
                    "unsupported_afdb_fragment"
                    if item.action == "skip_unsupported"
                    else "skipped_preflight"
                )
                record = make_planned_status_record(
                    item,
                    run_id=run_id,
                    attempt=_attempt(records, index, item.stage),
                    status=status,
                )
                append_status_record(status_path, record)
                records.append(record.as_dict())
            continue

        while True:
            output_state = {
                stage: validate_stage_outputs(stage, candidate, root)
                for stage in STAGES
            }
            plan = build_stage_plan(
                candidate,
                records,
                output_state,
                resume=True,
                force_stages=remaining_force,
                base_seed=base_seed,
                include_warnings=args.include_warnings,
            )
            runnable = [item for item in plan if item.action == "run"]
            if not runnable:
                current_states = reduce_status_history(records)
                for item in plan:
                    if item.action != "skip_complete":
                        continue
                    status = (
                        "successful_no_segments"
                        if item.reason == "successful_no_segments"
                        else "skipped_complete"
                    )
                    current = current_states.get((index, item.stage))
                    if current is not None and (
                        status != "successful_no_segments"
                        or current.status
                        in {
                            "complete",
                            "skipped_complete",
                            "successful_no_segments",
                        }
                    ):
                        continue
                    record = make_planned_status_record(
                        item,
                        run_id=run_id,
                        attempt=_attempt(records, index, item.stage),
                        status=status,
                        validated_outputs=output_state[item.stage].checked_paths,
                    )
                    append_status_record(status_path, record)
                    records.append(record.as_dict())
                break

            item = runnable[0]
            downstream = STAGES[STAGES.index(item.stage) + 1 :]
            for stage in downstream:
                blocked = next(
                    plan_item for plan_item in plan if plan_item.stage == stage
                )
                record = make_planned_status_record(
                    blocked,
                    run_id=run_id,
                    attempt=_attempt(records, index, stage),
                    status="blocked_upstream",
                )
                append_status_record(status_path, record)
                records.append(record.as_dict())
            log_path = (
                root
                / "reports/screening_logs"
                / f"{index:03d}_{candidate['pair_name']}_{item.stage}.log"
            )
            terminal = execute_stage(
                item,
                project_root=root,
                log_path=log_path,
                status_path=status_path,
                run_id=run_id,
                attempt=_attempt(records, index, item.stage),
            )
            records.append(terminal.as_dict())
            remaining_force.discard(item.stage)
            print(f"  {item.stage:<18} {terminal.status}")
            if terminal.status.startswith("failed_"):
                stop_after_failure = not bool(batch["continue_on_error"])
                break
        if stop_after_failure:
            raise SystemExit(1)

    print(f"\nStatus log: {status_path}")


if __name__ == "__main__":
    main()
