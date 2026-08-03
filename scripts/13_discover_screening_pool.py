from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import yaml

from dual_uq.rcsb_discovery import (
    build_polymer_entity_query,
    discover_candidates,
    search_polymer_entities,
    select_screening_pool,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover and select a stratified PDB–UniProt–AFDB screening pool."
    )
    parser.add_argument(
        "--config",
        default="configs/legacy/a0_screening/screening_pool.yaml",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Overrides project_root in the YAML configuration.",
    )
    return parser.parse_args()


def stage(message: str) -> float:
    print(f"\n[{time.strftime('%H:%M:%S')}] {message}", flush=True)
    return time.perf_counter()


def finish(started: float, message: str) -> None:
    elapsed = time.perf_counter() - started
    print(
        f"[{time.strftime('%H:%M:%S')}] {message} "
        f"({elapsed:.2f}s)",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    print(f"Loading configuration: {config_path}", flush=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    root = Path(args.project_root or config["project_root"]).expanduser().resolve()
    discovery_cfg = config["discovery"]
    pool_cfg = config["screening_pool"]

    output_dir = root / "data/processed/discovery"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    grouping_value = discovery_cfg.get("sequence_identity_grouping")
    grouping = None
    if grouping_value not in (None, "", False, "null", "None"):
        grouping = int(grouping_value)

    started = stage(
        "Stage 1/3: building and submitting the RCSB Search API query "
        f"(rows={discovery_cfg['query_rows']}, grouping={grouping})"
    )
    query = build_polymer_entity_query(
        rows=int(discovery_cfg["query_rows"]),
        methods=list(discovery_cfg["experimental_methods"]),
        resolution_max=float(discovery_cfg["resolution_max"]),
        length_min=int(discovery_cfg["length_min"]),
        length_max=int(discovery_cfg["length_max"]),
        sequence_identity_grouping=grouping,
    )
    query_path = output_dir / "rcsb_screening_query.json"
    query_path.write_text(json.dumps(query, indent=2), encoding="utf-8")

    identifiers = search_polymer_entities(query)
    identifier_path = output_dir / "rcsb_polymer_entities.txt"
    identifier_path.write_text("\n".join(identifiers) + "\n", encoding="utf-8")
    finish(started, f"RCSB returned {len(identifiers)} polymer entities")

    probe_limit = min(
        int(discovery_cfg["afdb_metadata_probe_limit"]),
        len(identifiers),
    )
    batch_size = max(1, int(discovery_cfg.get("discovery_batch_size", 5)))
    selected_identifiers = identifiers[:probe_limit]

    started = stage(
        "Stage 2/3: probing RCSB Data API and AlphaFold DB metadata "
        f"for {probe_limit} candidates in batches of {batch_size}"
    )

    checkpoint_path = output_dir / "discovered_candidates.checkpoint.parquet"
    completed_ids: set[str] = set()
    chunks: list[pd.DataFrame] = []

    if checkpoint_path.exists():
        checkpoint = pd.read_parquet(checkpoint_path)
        chunks.append(checkpoint)
        if "polymer_entity_id" in checkpoint.columns:
            completed_ids = set(
                checkpoint["polymer_entity_id"].dropna().astype(str)
            )
        print(
            f"Resuming from checkpoint with {len(completed_ids)} attempted candidates.",
            flush=True,
        )

    remaining = [
        identifier
        for identifier in selected_identifiers
        if identifier not in completed_ids
    ]

    total_batches = (len(remaining) + batch_size - 1) // batch_size
    for batch_number, start in enumerate(
        range(0, len(remaining), batch_size),
        start=1,
    ):
        batch = remaining[start : start + batch_size]
        batch_started = time.perf_counter()
        print(
            f"[probe {batch_number}/{max(total_batches, 1)}] "
            f"{batch[0]} .. {batch[-1]}",
            flush=True,
        )

        result = discover_candidates(
            batch,
            require_single_uniprot=bool(
                discovery_cfg["require_single_uniprot"]
            ),
            afdb_probe_limit=len(batch),
            request_pause_seconds=float(
                discovery_cfg["request_pause_seconds"]
            ),
        )
        chunks.append(result)

        discovered = pd.concat(chunks, ignore_index=True, sort=False)
        if "polymer_entity_id" in discovered.columns:
            discovered = discovered.drop_duplicates(
                subset=["polymer_entity_id"],
                keep="last",
            )
        discovered.to_parquet(checkpoint_path, index=False)

        successes = int(
            (discovered.get("discovery_status") == "eligible").sum()
        ) if "discovery_status" in discovered.columns else 0
        failures = int(
            (discovered.get("discovery_status") == "failed").sum()
        ) if "discovery_status" in discovered.columns else 0
        print(
            f"  checkpoint: attempted={len(discovered)}, "
            f"eligible={successes}, failed={failures}, "
            f"batch_time={time.perf_counter() - batch_started:.2f}s",
            flush=True,
        )

    if chunks:
        discovered = pd.concat(chunks, ignore_index=True, sort=False)
        if "polymer_entity_id" in discovered.columns:
            discovered = discovered.drop_duplicates(
                subset=["polymer_entity_id"],
                keep="last",
            )
    else:
        discovered = pd.DataFrame()

    discovered_path = output_dir / "discovered_candidates.parquet"
    discovered.to_parquet(discovered_path, index=False)
    finish(started, f"Metadata probing completed for {len(discovered)} candidates")

    started = stage("Stage 3/3: applying diversity rules and provisional quotas")
    required_columns = {"discovery_status", "pdb_id", "uniprot_id"}
    if not required_columns.issubset(discovered.columns):
        raise RuntimeError(
            "Discovery produced no usable table. Inspect the checkpoint and logs: "
            f"{checkpoint_path}"
        )

    eligible = discovered[
        (discovered["discovery_status"] == "eligible")
        & discovered["pdb_id"].notna()
        & discovered["uniprot_id"].notna()
    ].copy()
    print(f"Eligible candidates: {len(eligible)}", flush=True)

    selected = select_screening_pool(
        eligible,
        quotas={
            key: int(value)
            for key, value in pool_cfg["quotas"].items()
        },
        definitions=pool_cfg["definitions"],
        length_bins=pool_cfg["diversity"]["length_bins"],
        unique_sequence_cluster=bool(
            pool_cfg["diversity"]["unique_sequence_cluster"]
        ),
        max_per_organism=int(
            pool_cfg["diversity"]["max_per_organism"]
        ),
        seed=int(discovery_cfg["random_seed"]),
        prefer_single_protein_chain=bool(
            discovery_cfg["prefer_single_protein_chain"]
        ),
    )

    selected_path = root / "data/manifests/screening_pool.tsv"
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(selected_path, sep="\t", index=False)

    summary_columns = [
        "screening_index",
        "pdb_id",
        "chain_id",
        "uniprot_id",
        "length",
        "experimental_method",
        "resolution",
        "organism",
        "afdb_global_plddt",
        "provisional_stratum",
        "length_bin",
        "single_chain_proxy",
        "selection_reason",
    ]
    available_columns = [
        column for column in summary_columns if column in selected.columns
    ]
    if available_columns:
        print(selected[available_columns].to_string(index=False), flush=True)

    counts = (
        selected["provisional_stratum"].value_counts(dropna=False).to_dict()
        if not selected.empty
        else {}
    )
    summary_path = report_dir / "screening_pool_discovery_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "returned_polymer_entities": len(identifiers),
                "probed_candidates": len(discovered),
                "eligible_candidates": len(eligible),
                "selected_candidates": len(selected),
                "provisional_stratum_counts": counts,
                "screening_pool_path": str(selected_path),
                "checkpoint_path": str(checkpoint_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    finish(started, f"Selected {len(selected)} screening candidates")
    print(f"\nSaved discovery table: {discovered_path}", flush=True)
    print(f"Saved screening pool: {selected_path}", flush=True)
    print(f"Saved summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
