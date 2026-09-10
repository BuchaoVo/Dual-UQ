"""Build or execute the matched apo/holo local-response measurement layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dual_uq.evaluation.apo_holo_local_response import (
    ApoHoloLocalResponseError,
    build_apo_holo_cases,
    materialize_model_local_response,
    run_model_local_response,
)
from dual_uq.models.esm_if1 import load_esm_if1
from dual_uq.models.proteinmpnn import load_authorized_proteinmpnn_adapter

ESM_IF1_REVISION = "2b369911bb5b4b0dda914521b9475cad1656b2ac"
ESM_IF1_CHECKPOINT_SHA256 = "be4ba36edec22a9bfaa4946ff6b2815f1f19d8a3d7e0eada8b796d5a0eae9fd4"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--model", choices=("ProteinMPNN", "ESM-IF1"), required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        proteinmpnn_cases, esm_if1_cases, exclusions = build_apo_holo_cases(
            args.project_root, args.release_root, workers=args.workers
        )
        payload = {
            "status": "READY_FOR_MODEL_BINDING" if not args.execute else "EXECUTING",
            "model": args.model,
            "proteinmpnn_case_rows": len(proteinmpnn_cases),
            "esm_if1_case_rows": len(esm_if1_cases),
            "proteinmpnn_pair_count": int(proteinmpnn_cases["pair_id"].nunique()) if not proteinmpnn_cases.empty else 0,
            "esm_if1_pair_count": int(esm_if1_cases["pair_id"].nunique()) if not esm_if1_cases.empty else 0,
            "exclusions": exclusions.to_dict("records"),
        }
        if args.execute:
            if args.output_root is None:
                raise ApoHoloLocalResponseError("--output-root is required with --execute")
            if args.model == "ProteinMPNN":
                checkpoint = args.checkpoint_path or (
                    args.project_root / "third_party/ProteinMPNN/vanilla_model_weights/v_48_020.pt"
                )
                adapter = load_authorized_proteinmpnn_adapter(
                    implementation_path=args.project_root / "third_party/ProteinMPNN",
                    checkpoint_path=checkpoint,
                    device_name=args.device,
                    backbone_noise=0.2,
                )
            else:
                if args.checkpoint_path is None:
                    raise ApoHoloLocalResponseError(
                        "--checkpoint-path is required for ESM-IF1 execution"
                    )
                adapter = load_esm_if1(
                    args.project_root / "third_party/esm",
                    args.checkpoint_path,
                    expected_revision=ESM_IF1_REVISION,
                    expected_checkpoint_sha256=ESM_IF1_CHECKPOINT_SHA256,
                    device=args.device,
                )
            cases = proteinmpnn_cases if args.model == "ProteinMPNN" else esm_if1_cases
            position, protein, metadata = run_model_local_response(
                adapter,
                cases,
                args.model,
                condition_order=("APO", "HOLO"),
            )
            result = materialize_model_local_response(
                position, protein, metadata, args.output_root
            )
            payload["materialization"] = result
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except (ApoHoloLocalResponseError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "BLOCKED", "message": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
