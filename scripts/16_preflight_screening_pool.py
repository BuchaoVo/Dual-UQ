from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import yaml

from dual_uq.afdb import get_afdb_prediction_records
from dual_uq.pdb_archive import fetch_pdb_mmcif
from dual_uq.preflight import (
    compute_preflight_metrics,
    evaluate_afdb_fragment_support,
    finalize_preflight_status,
)
from dual_uq.sifts import fetch_sifts_xml, parse_sifts_residue_mapping
from dual_uq.structure_io import load_chain_ca_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run lightweight mapping and construct QC before geometry analysis."
    )
    parser.add_argument("--config", default="configs/screening_preflight.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--only", default=None, help="Comma-separated screening indices.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = Path(config["project_root"]).expanduser().resolve()

    pool_path = Path(config["pool_path"])
    if not pool_path.is_absolute():
        pool_path = root / pool_path
    pool = pd.read_csv(pool_path, sep="\t")

    if args.only:
        requested = {int(v) for v in args.only.split(",") if v.strip()}
        pool = pool[pool["screening_index"].astype(int).isin(requested)]
    else:
        pool = pool[pool["screening_index"].astype(int) >= args.start_index]
        if args.limit is not None:
            pool = pool.head(args.limit)

    output_dir = root / "data/processed/preflight"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "screening_preflight.checkpoint.parquet"

    existing = pd.read_parquet(checkpoint_path) if checkpoint_path.exists() else pd.DataFrame()
    completed = (
        set(existing["screening_index"].astype(int))
        if not existing.empty and "screening_index" in existing.columns
        else set()
    )
    rows = existing.to_dict("records") if not existing.empty else []

    thresholds = config["thresholds"]
    pause = float(config["runtime"]["request_pause_seconds"])

    for item in pool.itertuples(index=False):
        index = int(item.screening_index)
        if index in completed and not args.force:
            print(f"[{index:02d}] already preflighted", flush=True)
            continue

        started = time.perf_counter()
        pdb_id = str(item.pdb_id).lower()
        chain_id = str(item.chain_id)
        uniprot_id = str(item.uniprot_id).upper()
        print(f"[{index:02d}] {pdb_id}:{chain_id} ↔ {uniprot_id} ...", flush=True)

        result = {
            "screening_index": index,
            "pdb_id": pdb_id,
            "chain_id": chain_id,
            "uniprot_id": uniprot_id,
            "provisional_stratum": getattr(item, "provisional_stratum", None),
            "discovery_length": int(float(item.length)),
        }

        try:
            pdb_path = fetch_pdb_mmcif(pdb_id, root / "data/raw/pdb")
            sifts_path = fetch_sifts_xml(pdb_id, root / "data/raw/mappings")
            mapping = parse_sifts_residue_mapping(
                sifts_path,
                chain_id=chain_id,
                uniprot_id=uniprot_id,
            )
            pdb_ca = load_chain_ca_table(pdb_path, chain_id)
            mapped_positions = (
                mapping["uniprot_residue_number"].dropna().astype(int).unique()
            )
            if len(mapped_positions) == 0:
                raise ValueError("SIFTS mapping contains no UniProt residue positions.")
            mapped_interval = (
                int(mapped_positions.min()),
                int(mapped_positions.max()),
            )
            prediction_records = get_afdb_prediction_records(uniprot_id)
            fragment_support = evaluate_afdb_fragment_support(
                prediction_records,
                mapped_interval,
            )

            metrics = compute_preflight_metrics(
                mapping,
                pdb_ca,
                uniprot_length=int(fragment_support["canonical_uniprot_length"]),
                pdb_entity_length=int(float(item.length)),
            )
            result.update(metrics)
            result.update(
                {
                    key: value
                    for key, value in fragment_support.items()
                    if key != "prediction"
                }
            )
            result.update(
                finalize_preflight_status(
                    fragment_support,
                    metrics,
                    thresholds,
                )
            )
            result.update(
                {
                    "runtime_seconds": time.perf_counter() - started,
                    "error": None,
                }
            )
        except (KeyError, LookupError, OSError, RuntimeError, TypeError, ValueError) as exc:
            result.update(
                {
                    "preflight_status": "failed_runtime",
                    "preflight_reason": "runtime_error",
                    "runtime_seconds": time.perf_counter() - started,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

        rows = [r for r in rows if int(r.get("screening_index", -1)) != index]
        rows.append(result)
        checkpoint = pd.DataFrame(rows).sort_values("screening_index")
        checkpoint.to_parquet(checkpoint_path, index=False)

        print(
            f"  {result['preflight_status']} | "
            f"coverage={result.get('full_length_mapping_coverage')} | "
            f"identity={result.get('sequence_identity')} | "
            f"{result.get('runtime_seconds', 0):.2f}s",
            flush=True,
        )
        if pause > 0:
            time.sleep(pause)

    final = pd.DataFrame(rows).sort_values("screening_index")
    final_path = root / "data/manifests/screening_pool_preflight.tsv"
    final.to_csv(final_path, sep="\t", index=False)

    summary = {
        "attempted": len(final),
        "status_counts": final["preflight_status"].value_counts(dropna=False).to_dict(),
        "pass_indices": final.loc[
            final["preflight_status"] == "pass_full_length", "screening_index"
        ].astype(int).tolist(),
        "warning_indices": final.loc[
            final["preflight_status"] == "warn_construct_difference", "screening_index"
        ].astype(int).tolist(),
        "failed_indices": final.loc[
            final["preflight_status"].isin(["fail_preflight", "failed_runtime"]),
            "screening_index",
        ].astype(int).tolist(),
        "unsupported_afdb_fragment_indices": final.loc[
            final["preflight_status"] == "unsupported_afdb_fragment",
            "screening_index",
        ].astype(int).tolist(),
        "output_path": str(final_path),
        "checkpoint_path": str(checkpoint_path),
    }
    summary_path = root / "reports/screening_preflight_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    show_cols = [
        "screening_index", "pdb_id", "chain_id", "uniprot_id",
        "provisional_stratum", "full_length_mapping_coverage",
        "entity_mapping_coverage", "sequence_identity",
        "observed_ca_fraction_of_mapped", "preflight_status",
        "preflight_reason",
    ]
    print("\n" + final[[c for c in show_cols if c in final.columns]].to_string(index=False))
    print(f"\nSaved: {final_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
