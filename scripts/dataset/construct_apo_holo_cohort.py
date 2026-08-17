"""Construct and characterize the experimental apo/holo cohort."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dual_uq.dataset.apo_holo import (
    ApoHoloConfig,
    ApoHoloError,
    ApoHoloResult,
    acquire_candidate_assets,
    admit_pair,
    annotate_sequence_identity,
    build_candidate_pairs,
    build_candidate_structure_table,
    build_common_residue_mapping,
    characterize_pair_assets,
    identify_candidate_groups,
    load_assembly_context,
    load_candidate_mappings,
    load_ligand_annotations,
    render_apo_holo_release,
    run_expanded_discovery,
    select_primary_and_alternatives,
    summarize_apo_holo,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/interventions/biological_states/apo_holo"),
    )
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--census-workers", type=int, default=8)
    parser.add_argument(
        "--max-polymer-entities",
        type=int,
        default=500,
        help="Operational census cap; it is reported as truncation, not scientific exclusion.",
    )
    parser.add_argument("--min-common-fraction", type=float, default=0.90)
    parser.add_argument("--min-sequence-identity", type=float, default=0.95)
    parser.add_argument("--max-construct-mismatch", type=int, default=0)
    return parser


def run(args: argparse.Namespace) -> Path:
    root = args.project_root.expanduser().resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    config = ApoHoloConfig(
        project_root=root,
        census_page_size=args.page_size,
        census_workers=args.census_workers,
        max_polymer_entities=args.max_polymer_entities,
        min_common_fraction=args.min_common_fraction,
        min_sequence_identity=args.min_sequence_identity,
        max_construct_mismatch=args.max_construct_mismatch,
    )
    census, _anchor_summary = run_expanded_discovery(config)
    candidates = identify_candidate_groups(census)
    assembly = load_assembly_context(candidates, workers=config.census_workers)
    if not assembly.empty:
        candidates = candidates.merge(
            assembly,
            on="polymer_entity_id",
            how="left",
            validate="one_to_one",
        )
    ligand_annotations = load_ligand_annotations(candidates)
    assets = acquire_candidate_assets(config, candidates)
    structures = build_candidate_structure_table(candidates, assets, ligand_annotations)
    structures.attrs.update(census.attrs)
    mappings = load_candidate_mappings(structures, project_root=root)
    structures = annotate_sequence_identity(structures, mappings, project_root=root)
    pairs = build_candidate_pairs(structures, mappings, ligand_annotations)
    if not pairs.empty:
        decisions = pairs.apply(lambda row: admit_pair(row, config), axis=1, result_type="expand")
        for column in decisions.columns:
            pairs[column] = decisions[column]
    primary, alternatives = select_primary_and_alternatives(pairs)
    residue_tables: list[pd.DataFrame] = []
    mapping_tables: list[pd.DataFrame] = []
    pair_summaries: list[dict] = []
    for row in primary.to_dict(orient="records"):
        try:
            common_mapping = build_common_residue_mapping(
                mappings[str(row["apo_polymer_entity_id"])],
                mappings[str(row["holo_polymer_entity_id"])],
            )
            mapping_tables.append(
                common_mapping.assign(pair_id=row.get("pair_id"), protein_id=row.get("protein_id"))
            )
            residue, summary = characterize_pair_assets(
                row,
                mappings=mappings,
                project_root=root,
                ligand_annotations=ligand_annotations,
                config=config,
            )
            residue_tables.append(residue)
            pair_summaries.append(summary)
        except ApoHoloError as exc:
            pair_summaries.append(
                {
                    "pair_id": row.get("pair_id"),
                    "protein_id": row.get("protein_id"),
                    "apo_pdb_id": row.get("apo_pdb_id"),
                    "holo_pdb_id": row.get("holo_pdb_id"),
                    "geometry_status": exc.code,
                    "geometry_error": str(exc),
                }
            )
    residue_descriptors = pd.concat(residue_tables, ignore_index=True) if residue_tables else pd.DataFrame()
    pair_descriptors = pd.DataFrame(pair_summaries)
    if not residue_descriptors.empty:
        ligand_sites = residue_descriptors.loc[residue_descriptors["ligand_proximal"]].copy()
    else:
        ligand_sites = pd.DataFrame()
    summary = summarize_apo_holo(
        structures,
        pairs,
        primary,
        alternatives,
        pair_descriptors,
        residue_descriptors,
    )
    provenance = {
        "census": {
            "query": census.attrs.get("census_query"),
            "total_count": census.attrs.get("total_count"),
            "retrieved_count": census.attrs.get("retrieved_count"),
            "census_truncated": census.attrs.get("census_truncated"),
            "discovery_mode": census.attrs.get("discovery_mode"),
            "anchor_structure_count": census.attrs.get("anchor_structure_count"),
            "anchor_protein_count": census.attrs.get("anchor_protein_count"),
            "counterpart_structure_count": census.attrs.get("counterpart_structure_count"),
            "counterpart_protein_count": census.attrs.get("counterpart_protein_count"),
            "candidate_structure_count": len(structures),
        },
        "sources": ["RCSB Search/Data API", "wwPDB mmCIF", "PDBe/wwPDB SIFTS", "UniProtKB", "RCSB CCD"],
        "frozen_upstream_artifacts_modified": False,
        "inverse_folding_executed": False,
    }
    result = ApoHoloResult(
        census=structures,
        pairs=pairs,
        admission=pairs,
        primary_pairs=primary,
        alternative_pairs=alternatives,
        residue_mappings=pd.concat(mapping_tables, ignore_index=True)
        if mapping_tables
        else pd.DataFrame(),
        ligand_sites=ligand_sites,
        pair_descriptors=pair_descriptors,
        residue_descriptors=residue_descriptors,
        summary=summary,
        provenance=provenance,
        asset_ledger=assets,
    )
    render_apo_holo_release(result, output)
    return output


def main() -> int:
    args = _parser().parse_args()
    try:
        output = run(args)
    except (ApoHoloError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"APO_HOLO_CONSTRUCTION_FAILED: {exc}") from exc
    print(f"APO_HOLO_OUTPUT={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
