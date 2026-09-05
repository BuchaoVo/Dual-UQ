"""Prepare or score one shard of the StructCal cross-model sensitivity study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dual_uq.evaluation.apo_holo_local_response import build_decoding_realizations
from dual_uq.evaluation.cross_model_representation_cases import (
    build_controlled_cross_model_cases,
    build_exact_se3_cases,
    build_identical_input_cases,
    load_full_structcal_cohort,
    make_degenerate_dynamic_case,
)
from dual_uq.evaluation.structcal_local_response_cases import build_structcal_local_cases
from dual_uq.models.dynamicmpnn import DynamicMPNNAdapter
from dual_uq.models.esm_if1 import load_esm_if1
from dual_uq.models.kwdesign import KWDesignStructureInput, load_authorized_kwdesign_adapter
from dual_uq.models.pifold import PiFoldStructureInput, load_authorized_pifold_adapter
from dual_uq.models.proteinmpnn import (
    ProteinMPNNStructureInput,
    load_authorized_proteinmpnn_adapter,
)
from dual_uq.workflows.structcal_cross_model_representation_sensitivity import (
    MODEL_SPECS,
    filter_dynamicmpnn_contiguous_cases,
    response_shard_path,
    safe_output_path,
    score_cases_with_callback,
)

DEFAULT_OUTPUT = Path("runs/structcal_cross_model_representation_sensitivity")
ESM_IF1_REVISION = "2b369911bb5b4b0dda914521b9475cad1656b2ac"
ESM_IF1_SHA256 = "be4ba36edec22a9bfaa4946ff6b2815f1f19d8a3d7e0eada8b796d5a0eae9fd4"
ESM_IF1_CHECKPOINT = Path("/home/zbc/.cache/torch/hub/checkpoints/esm_if1_gvp4_t16_142M_UR50.pt")
PIFOLD_SOURCE = Path("/mnt/data/users/zbc/.cache/structcal_v1/sources/PiFold")
PIFOLD_CHECKPOINT = Path("/mnt/data/users/zbc/.cache/structcal_v1/checkpoints/pifold/checkpoint.pth")
KWD_SOURCE = Path("/mnt/data/users/zbc/.cache/structcal_cross_model/sources/ProteinInvBench")
KWD_ROOT = Path("/mnt/data/users/zbc/.cache/structcal_cross_model/checkpoints/kwdesign/extracted")
KWD_BASE = Path(
    "/mnt/data/users/zbc/.cache/structcal_cross_model/checkpoints/"
    "proteininvbench_pifold/extracted/checkpoint.pth"
)
KWD_ESM2 = Path(
    "/mnt/data/users/zbc/.cache/structcal_cross_model/huggingface/"
    "models--facebook--esm2_t33_650M_UR50D/snapshots/08e4846e537177426273712802403f7ba8261b6c"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "score"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", choices=tuple(MODEL_SPECS))
    parser.add_argument(
        "--regime",
        choices=("identical", "exact_se3", "controlled", "operational_pdb_afdb"),
    )
    parser.add_argument("--checkpoint", default="default")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-pairs", type=int)
    return parser.parse_args(argv)


def _write_once(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not pd.read_parquet(path).equals(frame):
            raise RuntimeError(f"refusing to overwrite non-identical artifact: {path}")
        return
    frame.to_parquet(path, index=False)


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
            _write_once(safe_output_path(output_root, f"cases/{atom_key}/{regime}.parquet"), cases)
            _write_once(
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _load_model(
    model_name: str,
    checkpoint: str,
    project_root: Path,
    output_root: Path,
    device: str,
) -> tuple[Any, str]:
    if model_name == "proteinmpnn":
        checkpoint_name = checkpoint if checkpoint.endswith(".pt") else f"{checkpoint}.pt"
        path = project_root / "third_party/ProteinMPNN/vanilla_model_weights" / checkpoint_name
        return (
            load_authorized_proteinmpnn_adapter(
                implementation_path=project_root / "third_party/ProteinMPNN",
                checkpoint_path=path,
                device_name=device,
                backbone_noise=0.0,
            ),
            checkpoint_name.removesuffix(".pt"),
        )
    if model_name == "pifold":
        return (
            load_authorized_pifold_adapter(
                implementation_path=PIFOLD_SOURCE,
                checkpoint_path=PIFOLD_CHECKPOINT,
                device_name=device,
            ),
            "official_checkpoint_pth",
        )
    if model_name == "esm_if1":
        return (
            load_esm_if1(
                project_root / "third_party/esm",
                ESM_IF1_CHECKPOINT,
                expected_revision=ESM_IF1_REVISION,
                expected_checkpoint_sha256=ESM_IF1_SHA256,
                device=device,
            ),
            "esm_if1_gvp4_t16_142M_UR50",
        )
    if model_name == "dynamicmpnn":
        return DynamicMPNNAdapter(repository_root=project_root, device=device), "single_chain_k2"
    if model_name == "kwdesign":
        return (
            load_authorized_kwdesign_adapter(
                implementation_path=KWD_SOURCE,
                model_parameters_path=KWD_ROOT / "model_param.json",
                base_checkpoint=KWD_BASE,
                tuner_checkpoint=KWD_ROOT / "checkpoint.pth",
                esm2_model_path=KWD_ESM2,
                esm_if1_checkpoint=ESM_IF1_CHECKPOINT,
                runtime_root=safe_output_path(output_root, "runtime/kwdesign"),
                device_name=device,
            ),
            "release_checkpoint_pth",
        )
    raise ValueError(f"unknown model: {model_name}")


def _scorer(model_name: str, adapter: Any):
    def score(**kwargs: Any) -> np.ndarray:
        coordinates = np.asarray(kwargs["coordinates"], dtype=np.float32)
        sequence = str(kwargs["sequence"])
        positions = tuple(kwargs["positions"])
        if model_name == "pifold":
            return adapter.probability_distributions(
                PiFoldStructureInput(coordinates=coordinates, sequence_length=len(sequence))
            )
        if model_name == "esm_if1":
            return adapter.score_teacher_forced(sequence, coordinates[:, :3])
        if model_name == "proteinmpnn":
            structure = ProteinMPNNStructureInput(
                protein_id=str(kwargs["protein_id"]),
                backbone_condition=str(kwargs["condition"]),
                uniprot_positions=positions,
                wt_sequence_projection=sequence,
                coordinates=coordinates,
                structure_sha256=None,
            )
            realization = build_decoding_realizations(
                str(kwargs["protein_id"]), len(sequence), count=1
            )[0]
            return adapter.probability_distributions(
                structure, (sequence,), realization, batch_size=1
            )[0]
        if model_name == "dynamicmpnn":
            case = make_degenerate_dynamic_case(
                protein_id=str(kwargs["protein_id"]),
                pair_id=f"{kwargs['pair_id']}::{kwargs['condition']}",
                canonical_positions=positions,
                sequence=sequence,
                coordinates=coordinates[:, :3],
                chain_id="A",
            )
            result = adapter.teacher_forced(case, sequence=sequence, inference=True)
            return result.logits.softmax(dim=-1).detach().cpu().numpy()
        if model_name == "kwdesign":
            return adapter.probability_distributions(
                KWDesignStructureInput(
                    coordinates=coordinates,
                    sequence=sequence,
                    title=f"{kwargs['pair_id']}::{kwargs['condition']}",
                )
            )
        raise ValueError(f"unknown model: {model_name}")

    return score


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
    adapter, checkpoint_id = _load_model(
        args.model, args.checkpoint, args.project_root, args.output_root, args.device
    )
    residue, pair, quality = score_cases_with_callback(
        cases,
        _scorer(args.model, adapter),
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
    _write_once(shard / f"residue_response{suffix}.parquet", residue)
    _write_once(shard / f"pair_response{suffix}.parquet", pair)
    _write_once(shard / f"standard_quality{suffix}.parquet", quality)
    if args.model == "dynamicmpnn":
        _write_once(shard / f"exclusions{suffix}.parquet", model_exclusions)
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
    (shard / f"summary{suffix}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n"
    )
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
