"""Prepare or score one shard of the StructCal cross-model sensitivity study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from dual_uq.core.artifacts import write_immutable_json, write_immutable_parquet
from dual_uq.evaluation.cross_model_representation_cases import (
    build_controlled_cross_model_cases,
    build_exact_se3_cases,
    build_identical_input_cases,
    load_full_structcal_cohort,
)
from dual_uq.evaluation.structcal_local_response_cases import build_structcal_local_cases
from dual_uq.workflows.structcal_cross_model_representation_sensitivity import (
    build_structcal_scorer,
    DEFAULT_RUN_ROOT,
    MODEL_SPECS,
    REGIMES,
    filter_dynamicmpnn_contiguous_cases,
    load_structcal_model,
    response_shard_path,
    safe_output_path,
    score_cases_with_callback,
)

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "score"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--model", choices=tuple(MODEL_SPECS))
    parser.add_argument("--regime", choices=REGIMES)
    parser.add_argument("--checkpoint", default="default")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-pairs", type=int)
    return parser.parse_args(argv)


def prepare_cases(project_root: Path, output_root: Path) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    for atom_key, atoms in (("nca", ("N", "CA", "C")), ("ncao", ("N", "CA", "C", "O"))):
        controlled_cohort = load_full_structcal_cohort(project_root, cohort="controlled")
        controlled, controlled_exclusions = build_controlled_cross_model_cases(
            project_root, controlled_cohort, atom_names=atoms
        )
        operational_cohort = load_full_structcal_cohort(project_root, cohort="track_i")
        operational, operational_exclusions = build_structcal_local_cases(
            project_root, operational_cohort, atom_names=atoms
        )
        exact = build_exact_se3_cases(controlled)
        identical = build_identical_input_cases(controlled)
        for regime, cases, exclusions in (
            ("controlled", controlled, controlled_exclusions),
            ("operational_pdb_afdb", operational, operational_exclusions),
            ("exact_se3", exact, pd.DataFrame(columns=controlled_exclusions.columns)),
            (
                "identical",
                identical,
                pd.DataFrame(columns=controlled_exclusions.columns),
            ),
        ):
            write_immutable_parquet(
                safe_output_path(output_root, f"cases/{atom_key}/{regime}.parquet"), cases
            )
            write_immutable_parquet(
                safe_output_path(output_root, f"cases/{atom_key}/{regime}_exclusions.parquet"),
                exclusions,
            )
            counts[f"{atom_key}/{regime}"] = {
                "pairs": int(cases["pair_id"].nunique()),
                "proteins": int(cases["protein_id"].nunique()),
                "residue_rows": len(cases),
                "exclusions": len(exclusions),
            }
    summary = {"status": "COMPLETE", "counts": counts}
    path = safe_output_path(output_root, "case_projection_summary.json")
    write_immutable_json(path, summary)
    return summary


def score_shard(args: argparse.Namespace) -> dict[str, Any]:
    if args.model is None or args.regime is None:
        raise ValueError("score requires --model and --regime")
    spec = MODEL_SPECS[args.model]
    atom_key = "ncao" if len(spec["atoms"]) == 4 else "nca"
    cases = pd.read_parquet(
        safe_output_path(args.output_root, f"cases/{atom_key}/{args.regime}.parquet")
    )
    if args.limit_pairs is not None:
        pair_ids = sorted(cases["pair_id"].astype(str).unique())[: args.limit_pairs]
        cases = cases.loc[cases["pair_id"].astype(str).isin(pair_ids)].copy()
    model_exclusions = pd.DataFrame()
    if args.model == "dynamicmpnn":
        cases, model_exclusions = filter_dynamicmpnn_contiguous_cases(
            cases, regime=args.regime
        )
    adapter, checkpoint_id = load_structcal_model(
        args.model, args.checkpoint, args.project_root, args.output_root, args.device
    )
    residue, pair, quality = score_cases_with_callback(
        cases,
        build_structcal_scorer(args.model, adapter),
        model_id=args.model,
        checkpoint_id=checkpoint_id,
        semantic_class=str(spec["semantic_class"]),
        regime=args.regime,
    )
    shard = response_shard_path(
        args.output_root,
        model=args.model,
        checkpoint=checkpoint_id,
        regime=args.regime,
    )
    suffix = "_preflight" if args.limit_pairs is not None else ""
    write_immutable_parquet(shard / f"residue_response{suffix}.parquet", residue)
    write_immutable_parquet(shard / f"pair_response{suffix}.parquet", pair)
    write_immutable_parquet(shard / f"standard_quality{suffix}.parquet", quality)
    if args.model == "dynamicmpnn":
        write_immutable_parquet(shard / f"exclusions{suffix}.parquet", model_exclusions)
    summary = {
        "status": "PREFLIGHT_PASS" if args.limit_pairs is not None else "COMPLETE",
        "model": args.model,
        "checkpoint": checkpoint_id,
        "regime": args.regime,
        "pairs": len(pair),
        "proteins": int(pair["protein_id"].nunique()),
        "residues": len(residue),
        "exclusions": len(model_exclusions),
        "binding": adapter.binding() if hasattr(adapter, "binding") else {},
    }
    write_immutable_json(shard / f"summary{suffix}.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.project_root = args.project_root.resolve()
    args.output_root = args.output_root.resolve()
    result = (
        prepare_cases(args.project_root, args.output_root)
        if args.command == "prepare"
        else score_shard(args)
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
