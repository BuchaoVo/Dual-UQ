"""Generate the official DynamicMPNN two-state baseline on frozen inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dual_uq.models.dynamicmpnn import (
    DynamicMPNNAdapter,
    DynamicMPNNInputCase,
    DynamicMPNNInputError,
    validate_dynamic_input,
)

DEFAULT_CASES = "runs/analysis/multistate_baseline/cases_esm.pkl"
DEFAULT_COHORT = "experiments/comparisons/apo_holo/multi_state_baseline/protein_summary.parquet"
DEFAULT_PRIMARY_PAIRS = (
    "experiments/interventions/biological_states/apo_holo_selective_admission_release/primary_pairs.parquet"
)
DEFAULT_OUTPUT = "runs/analysis/dynamicmpnn_baseline"
ROOT_SEED = 20260817


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _stable_seed(protein_id: str, root_seed: int = ROOT_SEED) -> int:
    digest = hashlib.sha256(f"dynamicmpnn:{root_seed}:{protein_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _first_state_rows(cases: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    rows: dict[tuple[str, str], pd.Series] = {}
    for _, row in cases.drop_duplicates(["protein_id", "condition"]).iterrows():
        key = (str(row["protein_id"]), str(row["condition"]).upper())
        if key in rows:
            raise ValueError(f"duplicate frozen case state: {key}")
        rows[key] = row
    return rows


def load_frozen_cases(
    repository_root: Path,
    *,
    cases_path: Path,
    cohort_path: Path,
    primary_pairs_path: Path,
) -> tuple[list[DynamicMPNNInputCase], list[dict[str, Any]]]:
    cases = pd.read_pickle(cases_path)
    cohort = pd.read_parquet(cohort_path)
    primary = pd.read_parquet(primary_pairs_path).set_index("protein_id")
    state_rows = _first_state_rows(cases)
    expected_ids = tuple(str(value) for value in cohort["protein_id"].tolist())
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("frozen DynamicMPNN cohort contains duplicate proteins")

    evaluable: list[DynamicMPNNInputCase] = []
    unavailable: list[dict[str, Any]] = []
    for protein_id in expected_ids:
        try:
            apo = state_rows[(protein_id, "APO")]
            holo = state_rows[(protein_id, "HOLO")]
            if protein_id not in primary.index:
                raise DynamicMPNNInputError("missing_primary_pair", "primary pair is absent")
            pair = primary.loc[protein_id]
            if str(apo["pair_id"]) != str(holo["pair_id"]):
                raise DynamicMPNNInputError("pair_identity_mismatch", "APO/HOLO pair IDs differ")
            positions = tuple(int(value) for value in apo["canonical_positions"])
            if positions != tuple(int(value) for value in holo["canonical_positions"]):
                raise DynamicMPNNInputError("canonical_axis_mismatch", "APO/HOLO axes differ")
            coordinates = np.stack(
                [np.asarray(apo["coordinates"], dtype=np.float32), np.asarray(holo["coordinates"], dtype=np.float32)],
                axis=1,
            )
            present = np.isfinite(coordinates).all(axis=(2, 3))
            case = DynamicMPNNInputCase(
                protein_id=protein_id,
                pair_id=str(apo["pair_id"]),
                canonical_positions=positions,
                sequences=(str(apo["wt_sequence_projection"]), str(holo["wt_sequence_projection"])),
                coordinates=coordinates,
                coordinate_present=present,
                chain_ids=(str(pair["apo_chain_id"]), str(pair["holo_chain_id"])),
                structure_sha256=(str(apo["structure_sha256"]), str(holo["structure_sha256"])),
            )
            validate_dynamic_input(case)
            evaluable.append(case)
        except (KeyError, IndexError, TypeError, ValueError, DynamicMPNNInputError) as exc:
            reason = getattr(exc, "reason", "malformed_frozen_dynamic_input")
            unavailable.append(
                {
                    "protein_id": protein_id,
                    "pair_id": str(state_rows.get((protein_id, "APO"), {}).get("pair_id", ""))
                    if isinstance(state_rows.get((protein_id, "APO")), dict)
                    else str(getattr(state_rows.get((protein_id, "APO")), "pair_id", "")),
                    "status": "DYNAMICMPNN_UNAVAILABLE",
                    "reason": reason,
                }
            )
    if len(evaluable) + len(unavailable) != len(expected_ids):
        raise RuntimeError("DynamicMPNN inventory does not cover the frozen cohort")
    return evaluable, unavailable


def run_generation(
    *,
    repository_root: Path,
    cases_path: Path,
    cohort_path: Path,
    primary_pairs_path: Path,
    output_dir: Path,
    device: str,
    n_samples: int,
    limit: int | None = None,
    start: int = 0,
) -> dict[str, Any]:
    evaluable, unavailable = load_frozen_cases(
        repository_root,
        cases_path=cases_path,
        cohort_path=cohort_path,
        primary_pairs_path=primary_pairs_path,
    )
    if start < 0:
        raise ValueError("start must be non-negative")
    evaluable = evaluable[start:]
    if limit is not None:
        evaluable = evaluable[:limit]
    adapter = DynamicMPNNAdapter(repository_root=repository_root, device=device)
    sequence_rows: list[dict[str, Any]] = []
    inventory = list(unavailable)
    generated_count = 0
    for index, case in enumerate(evaluable, start=1):
        seed = _stable_seed(case.protein_id)
        try:
            result = adapter.sample(case, n_samples=n_samples, seed=seed)
        except DynamicMPNNInputError as exc:
            inventory.append(
                {
                    "protein_id": case.protein_id,
                    "pair_id": case.pair_id,
                    "status": "DYNAMICMPNN_UNAVAILABLE",
                    "reason": exc.reason,
                    "n_samples": 0,
                    "seed": seed,
                    "residue_count": case.residue_count,
                    "missing_coordinate_count": validate_dynamic_input(case).missing_coordinate_count,
                    "checkpoint_sha256": adapter.checkpoint_sha256,
                    "source_commit": adapter.source_commit,
                }
            )
            print(f"unavailable {index}/{len(evaluable)} {case.protein_id} {exc.reason}", flush=True)
            continue
        shard = output_dir / "shards" / f"{case.protein_id}.json"
        records = [
            {
                "sample_index": sample_index,
                "sequence": sequence,
                "sequence_hash": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
            }
            for sample_index, sequence in enumerate(result.sequences)
        ]
        _atomic_json(
            shard,
            {
                "schema_version": "apo_holo_multistate_generation_v1",
                "protein_id": case.protein_id,
                "pair_id": case.pair_id,
                "seed": seed,
                "n_samples": n_samples,
                "residue_count": result.residue_count,
                "checkpoint_sha256": result.checkpoint_sha256,
                "source_commit": result.source_commit,
                "records": records,
                "nested_prefixes": [16, 32, 64] if n_samples >= 64 else [],
            },
        )
        inventory.append(
            {
                "protein_id": case.protein_id,
                "pair_id": case.pair_id,
                "status": "GENERATED",
                "reason": None,
                "n_samples": n_samples,
                "seed": seed,
                "residue_count": case.residue_count,
                "missing_coordinate_count": validate_dynamic_input(case).missing_coordinate_count,
                "checkpoint_sha256": adapter.checkpoint_sha256,
                "source_commit": adapter.source_commit,
            }
        )
        sequence_rows.extend(
            {
                "protein_id": case.protein_id,
                "pair_id": case.pair_id,
                "sample_index": sample_index,
                "sequence": sequence,
                "prefix_16": sample_index < 16,
                "prefix_32": sample_index < 32,
                "prefix_64": sample_index < 64,
                "seed": seed,
            }
            for sample_index, sequence in enumerate(result.sequences)
        )
        generated_count += 1
        print(f"generated {index}/{len(evaluable)} {case.protein_id}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(inventory).sort_values("protein_id").to_parquet(output_dir / "inventory.parquet", index=False)
    pd.DataFrame(sequence_rows).to_parquet(output_dir / "sequences.parquet", index=False)
    summary = {
        "cohort_count": len(evaluable) + len(unavailable),
        "evaluable_count": generated_count,
        "unavailable_count": len(inventory) - generated_count,
        "generated_sequence_count": len(sequence_rows),
        "n_samples": n_samples,
        "root_seed": ROOT_SEED,
        "checkpoint_sha256": adapter.checkpoint_sha256,
        "source_commit": adapter.source_commit,
        "device": str(next(adapter._model.parameters()).device) if adapter._model is not None else None,
    }
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-path", default=DEFAULT_CASES)
    parser.add_argument("--cohort-path", default=DEFAULT_COHORT)
    parser.add_argument("--primary-pairs-path", default=DEFAULT_PRIMARY_PAIRS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    summary = run_generation(
        repository_root=root,
        cases_path=root / args.cases_path,
        cohort_path=root / args.cohort_path,
        primary_pairs_path=root / args.primary_pairs_path,
        output_dir=root / args.output_dir,
        device=args.device,
        n_samples=args.n_samples,
        limit=args.limit,
        start=args.start,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
