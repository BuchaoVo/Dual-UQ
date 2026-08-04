from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from dual_uq.afdb import canonical_uniprot_length, get_afdb_prediction_records
from dual_uq.pdb_archive import fetch_pdb_mmcif
from dual_uq.preflight import (
    compute_preflight_metrics,
    evaluate_afdb_fragment_support,
    finalize_preflight_status,
)
from dual_uq.replacements import (
    COMPOSITE_KEY,
    build_replacement_audit,
    build_replacement_shortlist,
    evaluate_full_length_proxy,
    merge_replacement_preflight,
)
from dual_uq.sifts import fetch_sifts_xml, parse_sifts_residue_mapping
from dual_uq.structure_io import load_chain_ca_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and independently preflight the full-length lower-global-"
            "confidence replacement pool."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/legacy/a0_screening/lower_conf_replacement.yaml",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse the independent metadata and preflight checkpoints (default).",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    *,
    source_name: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source_name} lacks required fields: {missing}")


def _write_checkpoint(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _normalise_cache(cache: pd.DataFrame) -> pd.DataFrame:
    if cache.empty:
        return cache
    result = cache.copy()
    result["uniprot_id"] = (
        result["uniprot_id"].astype(str).str.strip().str.upper()
    )
    return result.drop_duplicates("uniprot_id", keep="last")


def _candidate_accessions(
    discovered: pd.DataFrame,
    current_pool: pd.DataFrame,
    selection: dict[str, Any],
) -> list[str]:
    eligible = discovered.loc[
        discovered["discovery_status"].eq("eligible")
        & discovered["pdb_id"].notna()
        & discovered["chain_id"].notna()
        & discovered["uniprot_id"].notna()
    ].copy()
    lengths = pd.to_numeric(eligible["length"], errors="coerce")
    global_plddt = pd.to_numeric(
        eligible["afdb_global_plddt"],
        errors="coerce",
    )
    eligible = eligible.loc[
        lengths.between(
            float(selection["length_min"]),
            float(selection["length_max"]),
            inclusive="both",
        )
        & global_plddt.notna()
    ]
    used = set(
        current_pool["uniprot_id"].astype(str).str.strip().str.upper()
    )
    accessions = (
        eligible["uniprot_id"].astype(str).str.strip().str.upper()
    )
    return sorted(set(accessions) - used)


def _load_prediction_metadata(
    accessions: list[str],
    *,
    cache_path: Path,
    force: bool,
    pause_seconds: float,
) -> pd.DataFrame:
    cache = (
        _normalise_cache(pd.read_parquet(cache_path))
        if cache_path.exists()
        else pd.DataFrame()
    )
    cached = (
        set(cache["uniprot_id"])
        if not cache.empty and "uniprot_id" in cache
        else set()
    )
    rows = cache.to_dict("records") if not cache.empty else []

    for number, accession in enumerate(accessions, start=1):
        if accession in cached and not force:
            continue
        print(
            f"[metadata {number:03d}/{len(accessions):03d}] {accession}",
            flush=True,
        )
        started = time.perf_counter()
        result: dict[str, Any] = {
            "uniprot_id": accession,
            "canonical_uniprot_length": None,
            "prediction_records_json": None,
            "metadata_status": "failed_runtime",
            "metadata_error": None,
        }
        try:
            records = get_afdb_prediction_records(accession)
            result.update(
                {
                    "canonical_uniprot_length": canonical_uniprot_length(
                        records
                    ),
                    "prediction_records_json": json.dumps(
                        records,
                        separators=(",", ":"),
                    ),
                    "metadata_status": "complete",
                }
            )
        except (
            KeyError,
            LookupError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            result["metadata_error"] = f"{type(exc).__name__}: {exc}"
        result["runtime_seconds"] = time.perf_counter() - started
        rows = [
            row
            for row in rows
            if str(row.get("uniprot_id", "")).upper() != accession
        ]
        rows.append(result)
        cache = _normalise_cache(pd.DataFrame(rows))
        _write_checkpoint(cache, cache_path)
        if pause_seconds > 0:
            time.sleep(pause_seconds)

    if not rows:
        return pd.DataFrame(
            columns=[
                "uniprot_id",
                "canonical_uniprot_length",
                "prediction_records_json",
                "metadata_status",
                "metadata_error",
                "runtime_seconds",
            ]
        )
    return _normalise_cache(pd.DataFrame(rows))


def _build_shortlist(
    discovered: pd.DataFrame,
    current_pool: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    selection: dict[str, Any],
    confidence_ranking: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, int]]:
    source = discovered.copy()
    source["uniprot_id"] = (
        source["uniprot_id"].astype(str).str.strip().str.upper()
    )
    metadata_columns = [
        "uniprot_id",
        "canonical_uniprot_length",
        "prediction_records_json",
        "metadata_status",
        "metadata_error",
    ]
    source = source.merge(
        metadata[metadata_columns],
        on="uniprot_id",
        how="left",
        validate="many_to_one",
    )
    shortlist = build_replacement_shortlist(
        source,
        current_pool,
        target_count=int(selection["target_count"]),
        confidence_ranking=confidence_ranking,
        diversity_config=selection,
        seed=int(selection["seed"]),
    )
    counts = dict(shortlist.attrs["selection_audit"])
    shortlist.insert(
        0,
        "screening_index",
        range(101, 101 + len(shortlist)),
    )
    return shortlist, counts


def _checkpoint_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(int(row["screening_index"])),
        str(row["pdb_id"]).lower(),
        str(row["chain_id"]),
        str(row["uniprot_id"]).upper(),
    )


def _run_preflight(
    shortlist: pd.DataFrame,
    *,
    root: Path,
    checkpoint_path: Path,
    thresholds: dict[str, float],
    pause_seconds: float,
    force: bool,
) -> pd.DataFrame:
    empty_columns = [
        *COMPOSITE_KEY,
        "pdb_entity_length",
        "canonical_uniprot_length",
        "pdb_to_uniprot_length_ratio",
        "full_length_mapping_coverage",
        "entity_mapping_coverage",
        "sequence_identity",
        "observed_ca_fraction_of_mapped",
        "internal_unmapped_fraction",
        "preflight_status",
        "preflight_reason",
        "afdb_fragment_status",
        "afdb_model_entity_id",
        "afdb_version",
        "afdb_fragment_start",
        "afdb_fragment_end",
        "runtime_seconds",
        "error",
    ]
    if shortlist.empty:
        empty = pd.DataFrame(columns=empty_columns)
        _write_checkpoint(empty, checkpoint_path)
        return empty

    existing = (
        pd.read_parquet(checkpoint_path)
        if checkpoint_path.exists()
        else pd.DataFrame()
    )
    selected_keys = {
        _checkpoint_key(row)
        for row in shortlist[COMPOSITE_KEY].to_dict("records")
    }
    rows = [
        row
        for row in existing.to_dict("records")
        if _checkpoint_key(row) in selected_keys
    ]
    completed = {_checkpoint_key(row) for row in rows}

    for item in shortlist.itertuples(index=False):
        base = {
            "screening_index": int(item.screening_index),
            "pdb_id": str(item.pdb_id).lower(),
            "chain_id": str(item.chain_id),
            "uniprot_id": str(item.uniprot_id).upper(),
        }
        key = _checkpoint_key(base)
        if key in completed and not force:
            previous = next(row for row in rows if _checkpoint_key(row) == key)
            print(
                f"[{base['screening_index']}] {base['pdb_id']}:"
                f"{base['chain_id']} ↔ {base['uniprot_id']}\n"
                f"  preflight: cached {previous.get('preflight_status')}",
                flush=True,
            )
            continue

        print(
            f"[{base['screening_index']}] {base['pdb_id']}:"
            f"{base['chain_id']} ↔ {base['uniprot_id']}\n"
            "  discovery proxy: pass",
            flush=True,
        )
        started = time.perf_counter()
        result: dict[str, Any] = {
            **base,
            "pdb_entity_length": int(float(item.pdb_entity_length)),
            "canonical_uniprot_length": int(
                float(item.canonical_uniprot_length)
            ),
        }
        try:
            pdb_path = fetch_pdb_mmcif(
                base["pdb_id"],
                root / "data/raw/pdb",
            )
            sifts_path = fetch_sifts_xml(
                base["pdb_id"],
                root / "data/raw/mappings",
            )
            mapping = parse_sifts_residue_mapping(
                sifts_path,
                chain_id=base["chain_id"],
                uniprot_id=base["uniprot_id"],
            )
            pdb_ca = load_chain_ca_table(pdb_path, base["chain_id"])
            mapped_positions = (
                mapping["uniprot_residue_number"]
                .dropna()
                .astype(int)
                .unique()
            )
            if len(mapped_positions) == 0:
                raise ValueError(
                    "SIFTS mapping contains no UniProt residue positions."
                )
            mapped_interval = (
                int(mapped_positions.min()),
                int(mapped_positions.max()),
            )
            records = json.loads(str(item.prediction_records_json))
            fragment_support = evaluate_afdb_fragment_support(
                records,
                mapped_interval,
            )
            metrics = compute_preflight_metrics(
                mapping,
                pdb_ca,
                uniprot_length=int(
                    fragment_support["canonical_uniprot_length"]
                ),
                pdb_entity_length=int(float(item.pdb_entity_length)),
            )
            result.update(metrics)
            result.update(
                {
                    name: value
                    for name, value in fragment_support.items()
                    if name != "prediction"
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
        except (
            KeyError,
            LookupError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            result.update(
                {
                    "preflight_status": "failed_runtime",
                    "preflight_reason": "runtime_error",
                    "afdb_fragment_status": "not_evaluated",
                    "runtime_seconds": time.perf_counter() - started,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

        rows = [row for row in rows if _checkpoint_key(row) != key]
        rows.append(result)
        checkpoint = pd.DataFrame(rows).sort_values("screening_index")
        _write_checkpoint(checkpoint, checkpoint_path)
        print(
            f"  preflight: {result['preflight_status']}\n"
            f"  reasons: {result.get('preflight_reason')}",
            flush=True,
        )
        if pause_seconds > 0:
            time.sleep(pause_seconds)

    return pd.DataFrame(rows, columns=empty_columns).sort_values(
        "screening_index"
    )


def _finalise_output(
    shortlist: pd.DataFrame,
    preflight: pd.DataFrame,
    *,
    selection: dict[str, Any],
) -> pd.DataFrame:
    final = merge_replacement_preflight(shortlist, preflight)
    final["afdb_selection_status"] = final.get(
        "afdb_fragment_status",
        "not_evaluated",
    )
    if "afdb_model_entity_id" in final:
        final = final.rename(
            columns={
                "afdb_model_entity_id": "selected_afdb_model_entity_id"
            }
        )

    exclusion_reasons: list[str] = []
    for _, row in final.iterrows():
        reasons = list(
            evaluate_full_length_proxy(row, selection).exclusion_reasons
        )
        status = str(row.get("preflight_status") or "")
        if status == "warn_construct_difference":
            reasons.append("warn_construct_difference")
        elif status == "fail_preflight":
            reasons.append("fail_preflight")
        exclusion_reasons.append(";".join(dict.fromkeys(reasons)))
    final["exclusion_reasons"] = exclusion_reasons

    output_columns = [
        "screening_index",
        "pdb_id",
        "chain_id",
        "uniprot_id",
        "provisional_stratum",
        "selection_stage",
        "pdb_entity_length",
        "canonical_uniprot_length",
        "pdb_to_uniprot_length_ratio",
        "afdb_global_plddt",
        "resolution",
        "organism",
        "sequence_cluster",
        "experimental_method",
        "protein_chain_count",
        "full_length_mapping_coverage",
        "entity_mapping_coverage",
        "sequence_identity",
        "observed_ca_fraction_of_mapped",
        "internal_unmapped_fraction",
        "preflight_status",
        "preflight_reason",
        "afdb_selection_status",
        "selected_afdb_model_entity_id",
        "afdb_version",
        "afdb_fragment_start",
        "afdb_fragment_end",
        "discovery_rank",
        "selection_reason",
        "exclusion_reasons",
        "duplicate_with_original_pool",
        "error",
        "runtime_seconds",
    ]
    return final.reindex(columns=output_columns).copy()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = Path(config["project_root"]).expanduser().resolve()
    source = config["source"]
    selection = config["selection"]
    confidence_ranking = config["confidence_ranking"]
    runtime = config["runtime"]
    output = config["output"]

    discovered = pd.read_parquet(
        _resolve(root, source["discovered_candidates"])
    )
    current_pool = pd.read_csv(
        _resolve(root, source["current_pool"]),
        sep="\t",
    )
    _require_columns(
        discovered,
        {
            "pdb_id",
            "chain_id",
            "uniprot_id",
            "length",
            "afdb_global_plddt",
            "resolution",
            "organism",
            "protein_chain_count",
            "discovery_status",
            "sequence_cluster",
            "experimental_method",
        },
        source_name="discovered_candidates",
    )
    _require_columns(
        current_pool,
        {"pdb_id", "chain_id", "uniprot_id"},
        source_name="current_pool",
    )

    pause_seconds = float(runtime["request_pause_seconds"])
    accessions = _candidate_accessions(discovered, current_pool, selection)
    metadata = _load_prediction_metadata(
        accessions,
        cache_path=_resolve(root, output["canonical_length_cache"]),
        force=bool(args.force),
        pause_seconds=pause_seconds,
    )
    shortlist, source_counts = _build_shortlist(
        discovered,
        current_pool,
        metadata,
        selection=selection,
        confidence_ranking=confidence_ranking,
    )
    if args.limit is not None:
        shortlist = shortlist.head(args.limit).copy()
        source_counts["selected_shortlist_count"] = len(shortlist)
        source_counts["shortfall_count"] = max(
            int(selection["target_count"]) - len(shortlist),
            0,
        )

    display_columns = [
        "screening_index",
        "pdb_id",
        "chain_id",
        "uniprot_id",
        "pdb_to_uniprot_length_ratio",
        "afdb_global_plddt",
        "organism",
        "sequence_cluster",
        "selection_reason",
    ]
    print(
        "\n"
        + shortlist[
            [column for column in display_columns if column in shortlist]
        ].to_string(index=False)
    )
    print(
        "\nSelection audit:\n"
        + json.dumps(source_counts, indent=2, sort_keys=True)
    )
    if args.plan_only:
        print(
            "\nPlan only: no PDB/SIFTS preflight, geometry, PAE, or mapped-"
            "confidence analysis was run."
        )
        return

    preflight_thresholds = {
        **{
            name: float(selection[name])
            for name in [
                "min_full_length_mapping_coverage",
                "min_entity_mapping_coverage",
                "min_sequence_identity",
                "min_observed_ca_fraction",
            ]
        },
        **{
            name: float(config["preflight"][name])
            for name in [
                "warn_full_length_mapping_coverage",
                "max_internal_unmapped_fraction",
            ]
        },
    }
    preflight = _run_preflight(
        shortlist,
        root=root,
        checkpoint_path=_resolve(root, output["checkpoint"]),
        thresholds=preflight_thresholds,
        pause_seconds=pause_seconds,
        force=bool(args.force),
    )
    final = _finalise_output(
        shortlist,
        preflight,
        selection=selection,
    )
    replacement_path = _resolve(root, output["replacement_pool"])
    replacement_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(replacement_path, sep="\t", index=False)

    audit = build_replacement_audit(
        final,
        requested_count=int(selection["target_count"]),
        thresholds=selection,
        seed=int(selection["seed"]),
        source_counts=source_counts,
        ranking_config=confidence_ranking,
    )
    audit["inputs"] = {
        "discovered_candidates": str(
            _resolve(root, source["discovered_candidates"])
        ),
        "current_pool": str(_resolve(root, source["current_pool"])),
        "current_preflight_read_only": str(
            _resolve(root, source["current_preflight"])
        ),
    }
    audit["outputs"] = {
        "replacement_pool": str(replacement_path),
        "checkpoint": str(_resolve(root, output["checkpoint"])),
        "canonical_length_cache": str(
            _resolve(root, output["canonical_length_cache"])
        ),
    }
    audit_path = _resolve(root, output["audit"])
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"\nSaved: {replacement_path}")
    print(f"Saved: {audit_path}")


if __name__ == "__main__":
    main()
