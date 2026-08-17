"""Run selective apo/holo raw-asset admission from the complete metadata census."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dual_uq.core.atomic_io import atomic_write_json, atomic_write_new_bytes
from dual_uq.core.hashing import sha256_file
from dual_uq.dataset.apo_holo import (
    ApoHoloConfig,
    ApoHoloError,
    ApoHoloResult,
    admit_pair,
    annotate_sequence_identity,
    build_candidate_pairs,
    build_candidate_structure_table,
    build_common_residue_mapping,
    characterize_pair_assets,
    load_assembly_context,
    load_candidate_mappings,
    render_apo_holo_release,
    select_primary_and_alternatives,
)
from dual_uq.dataset.apo_holo_selective import (
    assess_entity_asset_readiness,
    bind_shared_assets_to_entities,
    build_local_reuse_ledger,
    build_ligand_annotation_map,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--census",
        type=Path,
        default=Path("experiments/interventions/biological_states/apo_holo_full_census_complete/metadata_census.parquet"),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("experiments/interventions/biological_states/apo_holo_selective_admission"),
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--min-common-fraction", type=float, default=0.90)
    parser.add_argument("--min-sequence-identity", type=float, default=0.95)
    parser.add_argument("--max-construct-mismatch", type=int, default=0)
    parser.add_argument("--skip-acquisition", action="store_true")
    parser.add_argument("--offline-reuse", action="store_true", help="Use only existing local raw files; never contact remote sources.")
    return parser


def _path(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    payload = frame.to_parquet(index=False)
    if path.exists() and sha256_file(path) != __import__("hashlib").sha256(payload).hexdigest():
        raise ApoHoloError("immutable_conflict", f"refusing to overwrite {path}")
    if not path.exists():
        atomic_write_new_bytes(path, payload)


def run(args: argparse.Namespace) -> Path:
    root = args.project_root.expanduser().resolve()
    work = _path(root, args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    census = pd.read_parquet(_path(root, args.census))
    states = pd.read_parquet(work / "structure_state_resolution.parquet")
    components = pd.read_parquet(work / "component_inventory.parquet")
    reduced = pd.read_parquet(work / "non_dominated_pairs.parquet")
    structures = pd.read_parquet(work / "unique_structure_assets.parquet")

    config = ApoHoloConfig(
        root,
        min_common_fraction=args.min_common_fraction,
        min_sequence_identity=args.min_sequence_identity,
        max_construct_mismatch=args.max_construct_mismatch,
    )
    if args.offline_reuse:
        ledger = build_local_reuse_ledger(
            root,
            structures,
            components.loc[components["pdb_id"].isin(structures["pdb_id"].unique())],
        )
        _write_table(ledger, work / "selective_asset_ledger.parquet")
    elif args.skip_acquisition:
        ledger = pd.read_parquet(work / "selective_asset_ledger.parquet")
    else:
        from dual_uq.dataset.apo_holo_selective import acquire_selective_assets
        ledger = acquire_selective_assets(
            config,
            structures,
            components.loc[components["pdb_id"].isin(structures["pdb_id"].unique())],
            workers=args.workers,
        )
        _write_table(ledger, work / "selective_asset_ledger.parquet")

    bound_ledger = bind_shared_assets_to_entities(ledger, structures, components)
    readiness = assess_entity_asset_readiness(structures, bound_ledger, components)
    _write_table(readiness, work / "structure_asset_readiness.parquet")

    assembly_path = work / "assembly_context.parquet"
    if assembly_path.exists():
        assembly = pd.read_parquet(assembly_path)
    elif args.offline_reuse:
        assembly = pd.DataFrame(
            {
                "polymer_entity_id": readiness["polymer_entity_id"].astype(str),
                "assembly_oligomeric_details": None,
                "assembly_oligomeric_count": None,
                "assembly_details": None,
                "assembly_status": "unavailable_network",
            }
        )
        _write_table(assembly, assembly_path)
    else:
        assembly = load_assembly_context(readiness, workers=args.workers)
        _write_table(assembly, assembly_path)
    structures = readiness.merge(assembly, on="polymer_entity_id", how="left", validate="one_to_one")
    selected_components = components.loc[
        components["pdb_id"].isin(structures["pdb_id"].unique())
    ].copy()
    annotations = build_ligand_annotation_map(structures, selected_components)
    structure_table = build_candidate_structure_table(structures, bound_ledger, annotations)
    structure_table.attrs.update(census.attrs)
    # Mapping is only scientifically actionable for entities whose required raw
    # assets are locally valid. Keep the complete structure table for the
    # audit/funnel, but avoid reparsing unavailable entities; those remain
    # explicit UNRESOLVED_ASSET outcomes below.
    raw_valid_ids = set(
        readiness.loc[
            readiness["raw_asset_status"].eq("raw_valid"), "polymer_entity_id"
        ].astype(str)
    )
    mapping_inputs = structure_table.loc[
        structure_table["polymer_entity_id"].astype(str).isin(raw_valid_ids)
    ].copy()
    mappings = load_candidate_mappings(
        mapping_inputs, project_root=root, workers=max(1, args.workers)
    )
    structure_table = annotate_sequence_identity(structure_table, mappings, project_root=root)
    pairs = build_candidate_pairs(structure_table, mappings, annotations)
    pairs = pairs.loc[pairs["pair_id"].isin(set(reduced["pair_id"]))].copy()
    readiness_by_entity = readiness.set_index("polymer_entity_id")["raw_asset_status"].to_dict()
    decisions: list[dict[str, object]] = []
    for row in pairs.to_dict(orient="records"):
        entity_status = [
            readiness_by_entity.get(str(row["apo_polymer_entity_id"]), "unresolved_asset"),
            readiness_by_entity.get(str(row["holo_polymer_entity_id"]), "unresolved_asset"),
        ]
        if any(value != "raw_valid" for value in entity_status):
            decisions.append(
                {
                    "admitted": False,
                    "admission_status": "UNRESOLVED_ASSET",
                    "exclusion_reasons": "raw_asset_unresolved",
                }
            )
        else:
            decisions.append(admit_pair(row, config))
    decision_frame = pd.DataFrame(decisions, index=pairs.index)
    for column in decision_frame.columns:
        pairs[column] = decision_frame[column]
    admitted, alternatives = select_primary_and_alternatives(pairs)

    mapping_tables: list[pd.DataFrame] = []
    residue_tables: list[pd.DataFrame] = []
    pair_summaries: list[dict[str, object]] = []
    for row in admitted.to_dict(orient="records"):
        try:
            common = build_common_residue_mapping(
                mappings[str(row["apo_polymer_entity_id"])],
                mappings[str(row["holo_polymer_entity_id"])],
            )
            mapping_tables.append(common.assign(pair_id=row["pair_id"], protein_id=row["protein_id"]))
            residue, summary = characterize_pair_assets(
                row,
                mappings=mappings,
                project_root=root,
                ligand_annotations=annotations,
                config=config,
            )
            residue_tables.append(residue)
            pair_summaries.append(summary)
        except ApoHoloError as exc:
            pair_summaries.append({"pair_id": row["pair_id"], "protein_id": row["protein_id"], "geometry_status": exc.code, "geometry_error": str(exc)})

    residue_descriptors = pd.concat(residue_tables, ignore_index=True) if residue_tables else pd.DataFrame()
    pair_descriptors = pd.DataFrame(pair_summaries)
    ligand_sites = residue_descriptors.loc[residue_descriptors["ligand_proximal"]].copy() if not residue_descriptors.empty else pd.DataFrame()
    raw_valid_pairs = pairs.loc[pairs["admission_status"].ne("UNRESOLVED_ASSET")]
    summary = {
        "decision": "PASS" if len(admitted) else "LIMITED",
        "census_total_rows": len(census),
        "metadata_potential_protein_count": int(states["uniprot_id"].nunique()),
        "metadata_potential_protein_count_with_apo": int(states["uniprot_id"].nunique()),
        "state_resolved_structure_counts": states["state_label"].value_counts().to_dict(),
        "state_resolved_protein_count": int(states.loc[states["state_label"].ne("unresolved"), "uniprot_id"].nunique()),
        "proteins_with_resolved_apo": int(states.loc[states["state_label"].eq("apo"), "uniprot_id"].nunique()),
        "proteins_with_resolved_holo": int(states.loc[states["state_label"].eq("holo"), "uniprot_id"].nunique()),
        "proteins_with_both_states": int(reduced["protein_id"].nunique()),
        "metadata_candidate_pair_count": int(pd.read_parquet(work / "metadata_candidate_pairs.parquet").shape[0]),
        "non_dominated_pair_count": len(reduced),
        "unique_structure_entity_count": len(structures),
        "unique_pdb_entry_count": int(structures["pdb_id"].nunique()),
        "raw_valid_structure_count": int(readiness["raw_asset_status"].eq("raw_valid").sum()),
        "raw_unresolved_structure_count": int(readiness["raw_asset_status"].eq("unresolved_asset").sum()),
        "raw_valid_pair_count": len(raw_valid_pairs),
        "unresolved_asset_pair_count": int((pairs["admission_status"] == "UNRESOLVED_ASSET").sum()),
        "strictly_excluded_pair_count": int((pairs["admission_status"] == "EXCLUDED").sum()),
        "admitted_pair_count": len(admitted),
        "admitted_protein_count": int(admitted["protein_id"].nunique()) if not admitted.empty else 0,
        "alternative_pair_count": len(alternatives),
        "independent_30pct_identity_clusters": None,
        "upstream_descriptor_comparison": "unavailable: newly discovered apo/holo proteins do not overlap the frozen PDB/AFDB descriptor cohorts",
        "no_model_execution": True,
        "no_outcome_used_for_selection": True,
        "funnel_status": "raw selective admission completed; independent clustering remains unresolved without a frozen 30%-identity source",
    }
    provenance = {
        "census_path": _path(root, args.census).relative_to(root).as_posix(),
        "work_dir": work.relative_to(root).as_posix(),
        "sources": ["RCSB Search/Data/GraphQL API", "wwPDB mmCIF", "PDBe SIFTS", "UniProtKB", "RCSB CCD"],
        "inverse_folding_executed": False,
        "outcome_blind_state_and_pair_selection": True,
    }
    result = ApoHoloResult(
        census=structure_table,
        pairs=pairs,
        admission=pairs,
        primary_pairs=admitted,
        alternative_pairs=alternatives,
        residue_mappings=pd.concat(mapping_tables, ignore_index=True) if mapping_tables else pd.DataFrame(),
        ligand_sites=ligand_sites,
        pair_descriptors=pair_descriptors,
        residue_descriptors=residue_descriptors,
        summary=summary,
        provenance=provenance,
        asset_ledger=bound_ledger,
    )
    render_apo_holo_release(result, work)
    atomic_write_json(work / "selective_summary.json", summary)
    return work


def main() -> int:
    args = _parser().parse_args()
    try:
        output = run(args)
    except (ApoHoloError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"APO_HOLO_SELECTIVE_ADMISSION_FAILED: {exc}") from exc
    print(f"APO_HOLO_SELECTIVE_OUTPUT={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
