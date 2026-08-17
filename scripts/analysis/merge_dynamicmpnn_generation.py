"""Merge independent DynamicMPNN generation workers without model execution."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as handle:
        temporary = Path(handle.name)
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def merge(workers: tuple[Path, ...], output: Path) -> dict[str, int]:
    inventories = [pd.read_parquet(worker / "inventory.parquet") for worker in workers]
    sequences = [pd.read_parquet(worker / "sequences.parquet") for worker in workers]
    inventory = pd.concat(inventories, ignore_index=True)
    inventory = inventory.sort_values(["protein_id", "status"], kind="mergesort")
    inventory = inventory.drop_duplicates("protein_id", keep="first")
    if inventory["protein_id"].duplicated().any():
        raise ValueError("merged DynamicMPNN inventory contains duplicate proteins")
    generated = pd.concat(sequences, ignore_index=True)
    if generated.duplicated(["protein_id", "sample_index"]).any():
        raise ValueError("merged DynamicMPNN sequences contain duplicate sample identities")
    for worker in workers:
        for shard in sorted((worker / "shards").glob("*.json")):
            target = output / "shards" / shard.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != shard.read_bytes():
                raise ValueError(f"immutable shard conflict: {target}")
            if not target.exists():
                shutil.copyfile(shard, target)
    _atomic_parquet(inventory, output / "inventory.parquet")
    _atomic_parquet(generated.sort_values(["protein_id", "sample_index"], kind="mergesort"), output / "sequences.parquet")
    summary = {
        "cohort_count": int(len(inventory)),
        "evaluable_count": int((inventory["status"] == "GENERATED").sum()),
        "unavailable_count": int((inventory["status"] != "GENERATED").sum()),
        "generated_sequence_count": int(len(generated)),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("workers", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps(merge(tuple(args.workers), args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
