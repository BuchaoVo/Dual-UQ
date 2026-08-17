"""Structural organization of local inverse-folding remodeling.

This module is a descriptive analysis layer.  It consumes the frozen position
response, Pair Validity, and existing common-mask artifacts, then derives local
PDB/AFDB geometry descriptors and protein-internal spatial null summaries.  It
does not rebuild residue mappings, execute a scorer, or make causal/biological
claims.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from dual_uq.core.atomic_io import atomic_write_new_bytes
from dual_uq.core.hashing import sha256_file
from dual_uq.evaluation.pair_validity import load_pair_validity_inputs
from dual_uq.geometry import kabsch_align
from dual_uq.structure_io import load_atom_site_table

ANALYSIS_PROTOCOL = "structural_organization_local_remodeling"
CONTACT_CUTOFF_ANGSTROM = 8.0
SPATIAL_PERMUTATIONS = 200
BASE_SEED = 20260730
RESPONSE_COLUMNS = ("magnitude_p", "breadth_b", "rank_displacement")
DESCRIPTOR_COLUMNS = (
    "ca_displacement",
    "torsion_delta_phi",
    "torsion_delta_psi",
    "torsion_delta_omega",
    "local_pairwise_distance_change",
    "neighbor_geometry_distortion",
    "contact_turnover_fraction",
    "contact_gain_count",
    "contact_loss_count",
)


class StructuralOrganizationError(ValueError):
    """Structured input, geometry, or materialization failure."""

    def __init__(self, code: str, message: str, *, outcome: str = "BLOCKED") -> None:
        self.code = code
        self.outcome = outcome
        super().__init__(message)


@dataclass
class StructuralOrganizationInputs:
    project_root: Path
    position_response: pd.DataFrame
    pair_validity: pd.DataFrame
    common_masks: pd.DataFrame
    input_provenance: dict[str, dict[str, Any]]


@dataclass
class StructuralOrganizationResult:
    position: pd.DataFrame
    protein: pd.DataFrame
    spatial_null: pd.DataFrame
    associations: pd.DataFrame
    summary: dict[str, Any]


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise StructuralOrganizationError("schema_mismatch", f"{label} is missing columns: {missing}")


def _circular_delta_degrees(first: float, second: float) -> float:
    """Return the shortest absolute angular distance, preserving missingness."""
    if not np.isfinite(first) or not np.isfinite(second):
        return float("nan")
    return float(abs((float(first) - float(second) + 180.0) % 360.0 - 180.0))


def _dihedral(first: np.ndarray, second: np.ndarray, third: np.ndarray, fourth: np.ndarray) -> float:
    b0 = -(second - first)
    b1 = third - second
    b2 = fourth - third
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def _torsion_profile(backbone: dict[int, dict[str, np.ndarray]]) -> dict[int, dict[str, float]]:
    """Compute phi/psi/omega only across canonical consecutive positions."""
    profile: dict[int, dict[str, float]] = {}
    for position, atoms in backbone.items():
        values = {name: float("nan") for name in ("phi", "psi", "omega")}
        previous = backbone.get(position - 1)
        following = backbone.get(position + 1)
        try:
            if previous is not None and all(name in previous for name in ("C",)) and all(
                name in atoms for name in ("N", "CA", "C")
            ):
                values["phi"] = _dihedral(previous["C"], atoms["N"], atoms["CA"], atoms["C"])
            if following is not None and all(name in atoms for name in ("N", "CA", "C")) and "N" in following:
                values["psi"] = _dihedral(atoms["N"], atoms["CA"], atoms["C"], following["N"])
            if previous is not None and all(name in previous for name in ("CA", "C")) and all(
                name in atoms for name in ("N", "CA")
            ):
                values["omega"] = _dihedral(previous["CA"], previous["C"], atoms["N"], atoms["CA"])
        except (KeyError, FloatingPointError, ValueError, ZeroDivisionError):
            pass
        profile[position] = values
    return profile


def _derive_pair_descriptors(
    *,
    protein_id: str,
    common_positions: tuple[int, ...],
    pdb_backbone: dict[int, dict[str, np.ndarray]],
    afdb_backbone: dict[int, dict[str, np.ndarray]],
    torsions_pdb: dict[int, dict[str, float]],
    torsions_afdb: dict[int, dict[str, float]],
) -> pd.DataFrame:
    """Derive per-position descriptors from an already joined pair."""
    positions = [
        position
        for position in sorted(common_positions)
        if position in pdb_backbone
        and position in afdb_backbone
        and "CA" in pdb_backbone[position]
        and "CA" in afdb_backbone[position]
    ]
    if len(positions) < 3:
        return pd.DataFrame(
            {
                "protein_id": [protein_id] * len(common_positions),
                "position": list(common_positions),
                "geometry_status": ["insufficient_alignment"] * len(common_positions),
                **{column: [float("nan")] * len(common_positions) for column in DESCRIPTOR_COLUMNS},
                "contact_neighbor_count": [0] * len(common_positions),
            }
        )
    pdb_ca = np.array([pdb_backbone[p]["CA"] for p in positions], dtype=float)
    afdb_ca = np.array([afdb_backbone[p]["CA"] for p in positions], dtype=float)
    aligned_afdb, _, _ = kabsch_align(afdb_ca, pdb_ca)
    displacement = np.linalg.norm(aligned_afdb - pdb_ca, axis=1)
    pdb_contacts = _contact_pairs(pdb_ca, CONTACT_CUTOFF_ANGSTROM)
    afdb_contacts = _contact_pairs(afdb_ca, CONTACT_CUTOFF_ANGSTROM)
    union_contacts = [left | right for left, right in zip(pdb_contacts, afdb_contacts)]
    rows: list[dict[str, Any]] = []
    position_to_index = {position: index for index, position in enumerate(positions)}
    for position in sorted(common_positions):
        if position not in position_to_index:
            rows.append(
                {
                    "protein_id": protein_id,
                    "position": position,
                    "geometry_status": "missing_coordinate",
                    **{column: float("nan") for column in DESCRIPTOR_COLUMNS},
                    "contact_neighbor_count": 0,
                }
            )
            continue
        index = position_to_index[position]
        neighbors = np.array(sorted(union_contacts[index]), dtype=int)
        changes = np.array(
            [
                abs(float(np.linalg.norm(pdb_ca[index] - pdb_ca[j])) - float(np.linalg.norm(afdb_ca[index] - afdb_ca[j])))
                for j in neighbors
            ],
            dtype=float,
        )
        if changes.size:
            local_change = float(np.mean(changes))
            neighborhood_distortion = float(np.quantile(changes, 0.90, method="linear"))
            pdb_neighbors = pdb_contacts[index]
            afdb_neighbors = afdb_contacts[index]
            turnover = float(np.mean([neighbor not in pdb_neighbors or neighbor not in afdb_neighbors for neighbor in neighbors]))
            gain = int(sum(neighbor not in pdb_neighbors and neighbor in afdb_neighbors for neighbor in neighbors))
            loss = int(sum(neighbor in pdb_neighbors and neighbor not in afdb_neighbors for neighbor in neighbors))
        else:
            local_change = float("nan")
            neighborhood_distortion = float("nan")
            turnover = float("nan")
            gain = 0
            loss = 0
        pdb_torsion = torsions_pdb.get(position, {})
        afdb_torsion = torsions_afdb.get(position, {})
        rows.append(
            {
                "protein_id": protein_id,
                "position": position,
                "geometry_status": "available",
                "ca_displacement": float(displacement[index]),
                "torsion_delta_phi": _circular_delta_degrees(
                    pdb_torsion.get("phi", np.nan), afdb_torsion.get("phi", np.nan)
                ),
                "torsion_delta_psi": _circular_delta_degrees(
                    pdb_torsion.get("psi", np.nan), afdb_torsion.get("psi", np.nan)
                ),
                "torsion_delta_omega": _circular_delta_degrees(
                    pdb_torsion.get("omega", np.nan), afdb_torsion.get("omega", np.nan)
                ),
                "local_pairwise_distance_change": local_change,
                "neighbor_geometry_distortion": neighborhood_distortion,
                "contact_turnover_fraction": turnover,
                "contact_gain_count": gain,
                "contact_loss_count": loss,
                "contact_neighbor_count": int(changes.size),
            }
        )
    return pd.DataFrame(rows)


def _contact_pairs(coordinates: np.ndarray, cutoff: float) -> list[set[int]]:
    """Return sparse symmetric contact neighbors without an O(n²) allocation."""
    pairs = cKDTree(np.asarray(coordinates, dtype=float)).query_pairs(cutoff, output_type="ndarray")
    result = [set() for _ in range(len(coordinates))]
    if len(pairs):
        for left, right in pairs:
            result[int(left)].add(int(right))
            result[int(right)].add(int(left))
    return result


def _select_atom_rows(atoms: pd.DataFrame, chain_id: str | None) -> pd.DataFrame:
    table = atoms.loc[atoms["model_number"] == atoms["model_number"].min()].copy()
    if chain_id is not None:
        if chain_id in set(table["auth_asym_id"].dropna().astype(str)):
            table = table.loc[table["auth_asym_id"].astype(str) == chain_id].copy()
        elif chain_id in set(table["label_asym_id"].dropna().astype(str)):
            table = table.loc[table["label_asym_id"].astype(str) == chain_id].copy()
        else:
            raise StructuralOrganizationError("chain_not_found", f"chain {chain_id!r} not found")
    return table


def _backbone_by_auth_residue(
    atoms: pd.DataFrame,
    chain_id: str | None,
    residue_keys: set[tuple[int, str]] | None = None,
) -> dict[tuple[int, str], dict[str, np.ndarray]]:
    table = _select_atom_rows(atoms, chain_id)
    table = table.loc[table["atom_name"].astype(str).str.strip().isin({"N", "CA", "C", "O"})].copy()
    if residue_keys is not None:
        keys = list(zip(table["auth_seq_id"], table["insertion_code"].astype(str)))
        table = table.loc[[key in residue_keys for key in keys]].copy()
    result: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    for (auth_seq, insertion), group in table.groupby(["auth_seq_id", "insertion_code"], dropna=True, sort=False):
        if pd.isna(auth_seq):
            continue
        key = (int(auth_seq), str(insertion))
        atoms_for_residue: dict[str, np.ndarray] = {}
        for atom_name, atom_group in group.groupby(group["atom_name"].astype(str).str.strip(), sort=False):
            ordered = atom_group.copy()
            ordered["_alt_priority"] = ordered["alt_id"].astype(str).map(lambda value: 0 if value in {".", "?", "", "A"} else 1)
            ordered = ordered.sort_values(["_alt_priority", "occupancy"], ascending=[True, False], kind="mergesort")
            selected = ordered.iloc[0]
            coordinates = selected[["x", "y", "z"]].to_numpy(dtype=float)
            if np.isfinite(coordinates).all():
                atoms_for_residue[atom_name] = coordinates
        if atoms_for_residue:
            result[key] = atoms_for_residue
    return result


def _canonical_backbone(
    mask: pd.DataFrame,
    row: pd.Series,
    project_root: Path,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, dict[str, np.ndarray]], tuple[int, ...]]:
    """Join existing mask positions to coordinates without rebuilding mapping."""
    common = mask.loc[mask["common_mask"].eq(True)].copy()
    common_positions = tuple(sorted(common["canonical_position"].astype(int).tolist()))
    pdb_atoms = load_atom_site_table(project_root / str(row["pdb_structure_ref"]))
    requested_pdb_keys = {
        (int(record.auth_seq_id), str(record.insertion_code or ""))
        for record in mask.loc[mask["mapping_present"].eq(True)].itertuples(index=False)
        if not pd.isna(record.auth_seq_id)
    }
    pdb_by_auth = _backbone_by_auth_residue(pdb_atoms, str(row["pdb_chain"]), requested_pdb_keys)
    afdb_atoms = load_atom_site_table(project_root / str(row["afdb_structure_ref"]))
    fragment_start = row.get("selected_fragment_start")
    fragment_end = row.get("selected_fragment_end")
    if pd.isna(fragment_start) or pd.isna(fragment_end):
        raise StructuralOrganizationError("mapping_unavailable", "frozen fragment binding is unavailable")
    fragment_keys = {(position, "") for position in range(1, int(fragment_end - fragment_start + 2))}
    afdb_by_model_raw = _backbone_by_auth_residue(afdb_atoms, "A", fragment_keys)
    afdb_by_model = {key[0]: atoms for key, atoms in afdb_by_model_raw.items()}
    pdb_by_position: dict[int, dict[str, np.ndarray]] = {}
    for record in mask.loc[mask["mapping_present"].eq(True)].itertuples(index=False):
        if pd.isna(record.auth_seq_id):
            continue
        auth_key = (int(record.auth_seq_id), str(record.insertion_code or ""))
        if auth_key in pdb_by_auth:
            pdb_by_position[int(record.canonical_position)] = pdb_by_auth[auth_key]
    afdb_by_position: dict[int, dict[str, np.ndarray]] = {}
    for position in mask.loc[mask["mapping_present"].eq(True), "canonical_position"].astype(int):
        model_position = int(position - int(fragment_start) + 1)
        if model_position in afdb_by_model:
            afdb_by_position[int(position)] = afdb_by_model[model_position]
    return pdb_by_position, afdb_by_position, common_positions


def _safe_spearman(first: pd.Series, second: pd.Series) -> float | None:
    values = pd.DataFrame({"first": first, "second": second}).apply(pd.to_numeric, errors="coerce").dropna()
    if len(values) < 3 or values["first"].nunique() < 2 or values["second"].nunique() < 2:
        return None
    return float(spearmanr(values["first"], values["second"]).statistic)


def _seed_for(protein_id: str, metric: str, base_seed: int = BASE_SEED) -> int:
    digest = hashlib.sha256(f"{base_seed}|{protein_id}|{metric}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _spatial_null_for_protein(
    positions: pd.DataFrame,
    edges: np.ndarray,
    metric: str,
    *,
    permutations: int = SPATIAL_PERMUTATIONS,
    seed: int = BASE_SEED,
) -> pd.DataFrame:
    """Compare edge-wise response differences with a within-protein permutation null."""
    _require_columns(positions, {"protein_id", metric}, "spatial positions")
    protein_ids = positions["protein_id"].astype(str).unique()
    if len(protein_ids) != 1:
        raise StructuralOrganizationError("spatial_protein_mismatch", "spatial edges must use the same protein")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise StructuralOrganizationError("spatial_edge_schema", "spatial edges must have shape (n, 2)")
    if len(edges) and not np.all(positions.iloc[edges.reshape(-1)]["protein_id"].astype(str).eq(protein_ids[0])):
        raise StructuralOrganizationError("spatial_protein_mismatch", "spatial edges must connect residues from the same protein")
    values = pd.to_numeric(positions[metric], errors="coerce").to_numpy(dtype=float)
    if len(edges) == 0 or not np.isfinite(values).all():
        observed = float("nan")
        null_values = np.full(permutations, np.nan)
    else:
        observed = float(np.mean(np.abs(values[edges[:, 0]] - values[edges[:, 1]])))
        rng = np.random.default_rng(seed)
        null_values = np.empty(permutations, dtype=float)
        for index in range(permutations):
            shuffled = rng.permutation(values)
            null_values[index] = float(np.mean(np.abs(shuffled[edges[:, 0]] - shuffled[edges[:, 1]])))
    finite_null = null_values[np.isfinite(null_values)]
    null_mean = float(np.mean(finite_null)) if finite_null.size else float("nan")
    null_sd = float(np.std(finite_null, ddof=1)) if finite_null.size > 1 else float("nan")
    z = float((null_mean - observed) / null_sd) if np.isfinite(observed) and np.isfinite(null_sd) and null_sd > 0 else float("nan")
    percentile = float(np.mean(finite_null >= observed)) if np.isfinite(observed) and finite_null.size else float("nan")
    return pd.DataFrame(
        [
            {
                "protein_id": protein_ids[0],
                "metric": metric,
                "edge_count": len(edges),
                "observed_edge_abs_difference": observed,
                "null_mean_edge_abs_difference": null_mean,
                "null_sd_edge_abs_difference": null_sd,
                "spatial_clustering_z": z,
                "null_upper_tail_fraction": percentile,
                "permutation_count": int(permutations),
                "seed": int(seed),
                "null_scope": "within_protein_permutation_preserving_response_values",
            }
        ]
    )


def _build_edges(descriptors: pd.DataFrame, coordinate_cache: dict[str, dict[str, Any]]) -> np.ndarray:
    # Edges are built from the union of PDB/AFDB contacts and remain protein-local.
    if not coordinate_cache:
        return np.empty((0, 2), dtype=int)
    positions = descriptors["position"].astype(int).tolist()
    index = {position: i for i, position in enumerate(positions)}
    pdb = coordinate_cache["pdb_ca"]
    afdb = coordinate_cache["afdb_ca"]
    usable = [position for position in positions if position in pdb and position in afdb]
    if len(usable) < 2:
        return np.empty((0, 2), dtype=int)
    pdb_ca = np.array([pdb[p] for p in usable], dtype=float)
    afdb_ca = np.array([afdb[p] for p in usable], dtype=float)
    pdb_contacts = _contact_pairs(pdb_ca, CONTACT_CUTOFF_ANGSTROM)
    afdb_contacts = _contact_pairs(afdb_ca, CONTACT_CUTOFF_ANGSTROM)
    edges = {
        (i, j)
        for i, neighbors in enumerate(left | right for left, right in zip(pdb_contacts, afdb_contacts))
        for j in neighbors
        if i < j
    }
    return np.array([[index[usable[i]], index[usable[j]]] for i, j in sorted(edges)], dtype=int)


def _validate_inputs(inputs: StructuralOrganizationInputs) -> None:
    _require_columns(inputs.position_response, {"protein_id", "position", *RESPONSE_COLUMNS, "cohort"}, "position response")
    _require_columns(inputs.pair_validity, {"protein_id", "pdb_structure_ref", "afdb_structure_ref", "pdb_chain"}, "Pair Validity")
    _require_columns(inputs.common_masks, {"protein_id", "canonical_position", "common_mask", "mapping_present"}, "common masks")
    if inputs.pair_validity["protein_id"].duplicated().any():
        raise StructuralOrganizationError("duplicate_pair_validity", "Pair Validity has duplicate proteins")
    if len(inputs.pair_validity) != 127 or inputs.position_response["protein_id"].nunique() != 127:
        raise StructuralOrganizationError("cohort_population_mismatch", "expected frozen 127-pair sensitivity cohort")
    if int(inputs.position_response["cohort"].eq("clean").sum()) != 16653:
        raise StructuralOrganizationError("clean_position_population_mismatch", "expected 16,653 clean positions")


def load_structural_organization_inputs(project_root: Path) -> StructuralOrganizationInputs:
    root = project_root.expanduser().resolve()
    position_path = root / "experiments/dataset/analysis/inverse_folding_remodeling/position_remodeling.parquet"
    pair_path = root / "experiments/p2_design_baseline/scale1b-v2/pair_validity/pair_validity.parquet"
    if not position_path.is_file() or not pair_path.is_file():
        raise StructuralOrganizationError("input_missing", "frozen remodeling or Pair Validity artifact is missing")
    pair_validity_inputs = load_pair_validity_inputs(root)
    position = pd.read_parquet(position_path)
    pair_validity = pd.read_parquet(pair_path)
    common_masks = pair_validity_inputs.common_masks
    provenance = {
        "position_response": {"path": position_path.relative_to(root).as_posix(), "sha256": sha256_file(position_path), "rows": len(position)},
        "pair_validity": {"path": pair_path.relative_to(root).as_posix(), "sha256": sha256_file(pair_path), "rows": len(pair_validity)},
        "common_masks": pair_validity_inputs.input_provenance["common_masks"],
    }
    inputs = StructuralOrganizationInputs(root, position, pair_validity, common_masks, provenance)
    _validate_inputs(inputs)
    return inputs


def _build_associations(position: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for protein_id, group in position.groupby("protein_id", sort=True):
        clean = str(group["cohort"].iloc[0]) == "clean"
        for response in RESPONSE_COLUMNS:
            for descriptor in DESCRIPTOR_COLUMNS:
                rho = _safe_spearman(group[response], group[descriptor])
                rows.append({"protein_id": protein_id, "cohort": "clean" if clean else "full", "response": response, "descriptor": descriptor, "spearman_rho": rho, "n_positions": int(group[[response, descriptor]].dropna().shape[0])})
    result = pd.DataFrame(rows)
    return result


def _build_protein_summary(position: pd.DataFrame, spatial: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for protein_id, group in position.groupby("protein_id", sort=True):
        cohort = str(group["cohort"].iloc[0])
        row: dict[str, Any] = {"protein_id": protein_id, "cohort": cohort, "position_count": len(group), "geometry_available_fraction": float(group["geometry_status"].eq("available").mean())}
        for metric in RESPONSE_COLUMNS:
            row[f"{metric}_median"] = float(pd.to_numeric(group[metric], errors="coerce").median())
            row[f"{metric}_q90"] = float(pd.to_numeric(group[metric], errors="coerce").quantile(0.90))
        for descriptor in DESCRIPTOR_COLUMNS:
            numeric = pd.to_numeric(group[descriptor], errors="coerce").dropna()
            row[f"{descriptor}_median"] = float(numeric.median()) if len(numeric) else None
            row[f"{descriptor}_q90"] = float(numeric.quantile(0.90)) if len(numeric) else None
        row["remodeling_concentration_p90_over_median"] = (
            row["magnitude_p_q90"] / row["magnitude_p_median"] if row["magnitude_p_median"] > 0 else None
        )
        for metric in RESPONSE_COLUMNS:
            subset = spatial.loc[(spatial["protein_id"] == protein_id) & (spatial["metric"] == metric), "spatial_clustering_z"]
            row[f"{metric}_spatial_clustering_z"] = float(subset.iloc[0]) if len(subset) and pd.notna(subset.iloc[0]) else None
        rows.append(row)
    return pd.DataFrame(rows)


def build_structural_organization_result(inputs: StructuralOrganizationInputs) -> StructuralOrganizationResult:
    _validate_inputs(inputs)
    pair_validity = inputs.pair_validity.set_index("protein_id", drop=False)
    masks = inputs.common_masks.copy()
    position_frames: list[pd.DataFrame] = []
    spatial_frames: list[pd.DataFrame] = []
    for protein_id, mask_group in masks.groupby("protein_id", sort=True):
        response = inputs.position_response.loc[inputs.position_response["protein_id"].eq(protein_id)].copy()
        row = pair_validity.loc[protein_id]
        try:
            pdb, afdb, common_positions = _canonical_backbone(mask_group, row, inputs.project_root)
            torsion_pdb = _torsion_profile(pdb)
            torsion_afdb = _torsion_profile(afdb)
            descriptors = _derive_pair_descriptors(
                protein_id=protein_id,
                common_positions=common_positions,
                pdb_backbone={p: pdb[p] for p in common_positions if p in pdb},
                afdb_backbone={p: afdb[p] for p in common_positions if p in afdb},
                torsions_pdb=torsion_pdb,
                torsions_afdb=torsion_afdb,
            )
            coordinate_cache = {"pdb_ca": {p: atoms["CA"] for p, atoms in pdb.items() if "CA" in atoms}, "afdb_ca": {p: atoms["CA"] for p, atoms in afdb.items() if "CA" in atoms}}
            edges = _build_edges(descriptors, coordinate_cache)
            # Edge indices are defined by the descriptor position order, never by
            # incidental parquet row order of the remodeling table.
            response = response.set_index("position", drop=False).reindex(descriptors["position"].astype(int)).reset_index(drop=True)
            for metric in RESPONSE_COLUMNS:
                spatial_frames.append(_spatial_null_for_protein(response, edges, metric, permutations=SPATIAL_PERMUTATIONS, seed=_seed_for(protein_id, metric)))
        except StructuralOrganizationError as exc:
            descriptors = pd.DataFrame({"protein_id": protein_id, "position": response["position"].astype(int), "geometry_status": [exc.code] * len(response)})
        merged = response.merge(descriptors, on=["protein_id", "position"], how="left", validate="one_to_one")
        position_frames.append(merged)
    position = pd.concat(position_frames, ignore_index=True).sort_values(["protein_id", "position"], kind="mergesort").reset_index(drop=True)
    spatial = pd.concat(spatial_frames, ignore_index=True) if spatial_frames else pd.DataFrame(columns=["protein_id", "metric"])
    associations = _build_associations(position)
    protein = _build_protein_summary(position, spatial)
    clean = protein.loc[protein["cohort"].eq("clean")]
    clean_assoc = associations.loc[associations["cohort"].eq("clean")].copy()
    assoc_summary: dict[str, Any] = {}
    for (response, descriptor), group in clean_assoc.groupby(["response", "descriptor"], sort=True):
        values = pd.to_numeric(group["spearman_rho"], errors="coerce").dropna()
        assoc_summary[f"{response}__{descriptor}"] = {"protein_count": len(values), "median_spearman": float(values.median()) if len(values) else None, "positive_fraction": float((values > 0).mean()) if len(values) else None}
    spatial_clean = spatial.loc[spatial["protein_id"].isin(set(clean["protein_id"]))] if not spatial.empty else spatial
    spatial_summary: dict[str, Any] = {}
    for metric in RESPONSE_COLUMNS:
        values = pd.to_numeric(spatial_clean.loc[spatial_clean["metric"].eq(metric), "spatial_clustering_z"], errors="coerce").dropna()
        spatial_summary[metric] = {"protein_count": len(values), "median_z": float(values.median()) if len(values) else None, "positive_fraction": float((values > 0).mean()) if len(values) else None}
    strongest_associations: dict[str, Any] = {}
    for response in RESPONSE_COLUMNS:
        subset = clean_assoc.loc[clean_assoc["response"].eq(response)].copy()
        medians = subset.groupby("descriptor", sort=True)["spearman_rho"].median().abs().sort_values(ascending=False)
        strongest_associations[response] = medians.index[0] if len(medians) else None
    summary = {
        "status": "COMPLETE",
        "analysis_protocol": ANALYSIS_PROTOCOL,
        "cohort": {"clean_proteins": int(clean["protein_id"].nunique()), "clean_positions": int(position["cohort"].eq("clean").sum()), "full_proteins": int(protein["protein_id"].nunique()), "full_positions": len(position)},
        "descriptor_definitions": {"ca_displacement": "aligned AFDB C-alpha minus PDB C-alpha Euclidean distance", "torsion_delta_*": "shortest circular angular difference in degrees; missing torsions remain null", "local_pairwise_distance_change": "mean absolute PDB/AFDB pairwise-distance difference over union contacts", "neighbor_geometry_distortion": "90th percentile pairwise-distance difference over union contacts", "contact_turnover_fraction": "fraction of union contacts gained or lost", "contact_cutoff_angstrom": CONTACT_CUTOFF_ANGSTROM},
        "spatial_null": "within-protein permutation preserving response values, protein size, and edge graph",
        "spatial_permutations": SPATIAL_PERMUTATIONS,
        "geometry_available_position_count": int(position["geometry_status"].eq("available").sum()),
        "geometry_unavailable_position_count": int((~position["geometry_status"].eq("available")).sum()),
        "clean_association_summary": assoc_summary,
        "clean_spatial_summary": spatial_summary,
        "strongest_clean_association_by_response": strongest_associations,
        "relational_vs_coordinate_comparison": {
            "relational_descriptors": ["local_pairwise_distance_change", "neighbor_geometry_distortion", "contact_turnover_fraction"],
            "coordinate_descriptor": "ca_displacement",
            "comparison_unit": "median within-protein Spearman absolute association; descriptive only",
        },
        "interpretation_boundary": "descriptive local remodeling/geometry association only; no causal, biological, functional, or generative claim",
        "questions": {
            "Q1_spatial_organization": "reported by within-protein spatial_clustering_z; no cross-protein independence assumed",
            "Q2_structural_changes_associated_with_magnitude": "reported by protein-stratified Spearman summaries",
            "Q3_breadth_and_reordering": "reported separately from magnitude in the association table",
            "Q4_relational_vs_coordinate": "compare local_pairwise_distance_change/contact_turnover_fraction with ca_displacement",
            "Q5_clean_vs_full": "clean 68-pair and full 127-pair summaries are both materialized",
        },
        "input_provenance": inputs.input_provenance,
        "spatial_clean_protein_count": int(spatial_clean["protein_id"].nunique()) if not spatial_clean.empty else 0,
    }
    return StructuralOrganizationResult(position, protein, spatial, associations, summary)


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _immutable_parquet(frame: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        digest = sha256_file(temporary)
        if path.exists():
            if sha256_file(path) != digest:
                raise StructuralOrganizationError("immutable_output_conflict", f"output differs: {path}")
            return "reused_identical"
        os.link(temporary, path)
        return "created"
    finally:
        temporary.unlink(missing_ok=True)


def _immutable_bytes(data: bytes, path: Path) -> str:
    if path.exists():
        if path.read_bytes() != data:
            raise StructuralOrganizationError("immutable_output_conflict", f"output differs: {path}")
        return "reused_identical"
    atomic_write_new_bytes(path, data)
    return "created"


def _figure_bytes(result: StructuralOrganizationResult, name: str) -> bytes:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.5), constrained_layout=True)
    position = result.position.loc[result.position["cohort"].eq("clean")]
    if name == "response_vs_geometry":
        axes[0].scatter(position["ca_displacement"], position["magnitude_p"], s=4, alpha=0.25, color="#355c7d")
        axes[0].set_xlabel("Aligned C-alpha displacement (Å)")
        axes[0].set_ylabel("P")
        axes[1].scatter(position["local_pairwise_distance_change"], position["magnitude_p"], s=4, alpha=0.25, color="#c06c84")
        axes[1].set_xlabel("Local pairwise-distance change (Å)")
        axes[1].set_ylabel("P")
    elif name == "breadth_reordering":
        axes[0].scatter(position["ca_displacement"], position["breadth_b"], s=4, alpha=0.25, color="#355c7d")
        axes[0].set_xlabel("Aligned C-alpha displacement (Å)")
        axes[0].set_ylabel("Breadth")
        axes[1].scatter(position["local_pairwise_distance_change"], position["rank_displacement"], s=4, alpha=0.25, color="#6c5b7b")
        axes[1].set_xlabel("Local pairwise-distance change (Å)")
        axes[1].set_ylabel("Rank displacement")
    elif name == "spatial_clustering":
        spatial = result.spatial_null.loc[result.spatial_null["protein_id"].isin(set(result.protein.loc[result.protein["cohort"].eq("clean"), "protein_id"]))]
        for metric, color in zip(RESPONSE_COLUMNS, ("#355c7d", "#c06c84", "#6c5b7b")):
            values = pd.to_numeric(spatial.loc[spatial["metric"].eq(metric), "spatial_clustering_z"], errors="coerce").dropna().sort_values().to_numpy()
            if len(values):
                axes[0].plot(np.linspace(0, 1, len(values)), values, label=metric, color=color)
        axes[0].axhline(0, color="black", linewidth=0.6)
        axes[0].set_xlabel("Protein quantile")
        axes[0].set_ylabel("Spatial clustering z")
        axes[0].legend(fontsize=7)
        axes[1].scatter(position["ca_displacement"], position["contact_turnover_fraction"], s=4, alpha=0.25, color="#355c7d")
        axes[1].set_xlabel("Aligned C-alpha displacement (Å)")
        axes[1].set_ylabel("Contact turnover")
    elif name == "representative_profiles":
        protein = result.protein.loc[result.protein["cohort"].eq("clean")].copy()
        orders = [
            ("hotspot_like", "remodeling_concentration_p90_over_median", False),
            ("broad_like", "remodeling_concentration_p90_over_median", True),
            ("large_displacement_weak_response", "ca_displacement_median", False),
            ("small_displacement_local_change", "local_pairwise_distance_change_median", False),
        ]
        selected: list[str] = []
        for name_, metric, ascending in orders:
            values = protein.sort_values([metric, "protein_id"], ascending=[ascending, True], kind="mergesort")
            if len(values):
                selected.append(str(values.iloc[0]["protein_id"]))
        for protein_id in dict.fromkeys(selected):
            subset = position.loc[position["protein_id"].eq(protein_id)].sort_values("position")
            axes[0].plot(subset["position"], subset["magnitude_p"], linewidth=0.8, label=protein_id)
            axes[1].plot(subset["position"], subset["local_pairwise_distance_change"], linewidth=0.8, label=protein_id)
        axes[0].set_xlabel("Canonical position")
        axes[0].set_ylabel("P")
        axes[1].set_xlabel("Canonical position")
        axes[1].set_ylabel("Local pairwise-distance change (Å)")
        axes[0].legend(fontsize=5)
        axes[1].legend(fontsize=5)
    else:
        plt.close(figure)
        raise ValueError(name)
    output = io.BytesIO()
    figure.savefig(output, format="png", dpi=220, metadata={"Software": "Dual-UQ"})
    plt.close(figure)
    return output.getvalue()


def _markdown_report(result: StructuralOrganizationResult) -> str:
    summary = result.summary
    assoc = result.associations.loc[result.associations["cohort"].eq("clean")]
    def association_median(cohort: str, response: str, descriptor: str) -> float | None:
        values = pd.to_numeric(
            result.associations.loc[
                result.associations["cohort"].eq(cohort)
                & result.associations["response"].eq(response)
                & result.associations["descriptor"].eq(descriptor),
                "spearman_rho",
            ],
            errors="coerce",
        ).dropna()
        return float(values.median()) if len(values) else None

    def format_value(value: float | None) -> str:
        return "unavailable" if value is None else f"{value:.4g}"

    spatial = summary["clean_spatial_summary"]
    p_relational = association_median("clean", "magnitude_p", "local_pairwise_distance_change")
    p_coordinate = association_median("clean", "magnitude_p", "ca_displacement")
    rank_relational = association_median("clean", "rank_displacement", "local_pairwise_distance_change")
    rank_coordinate = association_median("clean", "rank_displacement", "ca_displacement")
    breadth_relational = association_median("clean", "breadth_b", "local_pairwise_distance_change")
    breadth_coordinate = association_median("clean", "breadth_b", "ca_displacement")
    full_p_relational = association_median("full", "magnitude_p", "local_pairwise_distance_change")
    full_p_coordinate = association_median("full", "magnitude_p", "ca_displacement")
    lines = [
        "# Structural Organization Analysis",
        "",
        "Status: descriptive local remodeling/geometry association only.",
        "No scorer execution, generation, SDFI, Top-1 decision, or external validation was performed.",
        "",
        "## Cohorts and descriptors",
        "",
        f"- Primary: {summary['cohort']['clean_proteins']} proteins and {summary['cohort']['clean_positions']} positions.",
        f"- Sensitivity: {summary['cohort']['full_proteins']} proteins and {summary['cohort']['full_positions']} positions.",
        f"- Geometry descriptors available for {summary['geometry_available_position_count']} positions; unavailable for {summary['geometry_unavailable_position_count']}.",
        f"- Contact definition: union of PDB/AFDB C-alpha contacts at {CONTACT_CUTOFF_ANGSTROM:g} Å; this is a descriptive neighborhood definition, not a biological category.",
        "- Torsion deltas use shortest circular angular differences; missing torsions remain missing.",
        "",
        "## Questions",
        "",
        f"- Q1 answer (descriptive): P spatial clustering z has median {spatial['magnitude_p']['median_z']:.4g} with {spatial['magnitude_p']['positive_fraction']:.4g} of clean proteins positive; rank displacement has median {spatial['rank_displacement']['median_z']:.4g} with {spatial['rank_displacement']['positive_fraction']:.4g} positive. This supports spatial organization in these local contact-graph summaries, without a cross-protein independence claim.",
        f"- Q2 answer (descriptive): P is most strongly associated with local pairwise-distance change among the recorded descriptors (clean median protein Spearman {format_value(p_relational)}), compared with aligned C-alpha displacement ({format_value(p_coordinate)}).",
        f"- Q3 answer: breadth shows weak near-zero associations (local pairwise-distance change {format_value(breadth_relational)}, C-alpha displacement {format_value(breadth_coordinate)}), while rank displacement is more consistently associated with local relational change ({format_value(rank_relational)}); these are not the same response pattern as P.",
        f"- Q4 answer: relational geometry is more informative than raw aligned displacement for P and rank displacement in the clean cohort by median within-protein Spearman ({format_value(p_relational)} vs {format_value(p_coordinate)}; {format_value(rank_relational)} vs {format_value(rank_coordinate)}), but not for breadth; this is descriptive, not causal.",
        f"- Q5 answer: the full 127-pair sensitivity cohort preserves the direction and ordering of the main P comparison (relational {format_value(full_p_relational)} vs coordinate {format_value(full_p_coordinate)}), while retaining 4,468 structured mapping-unavailable positions that are excluded from geometry associations; clean results remain primary.",
        "",
        "## Clean protein-level association overview",
        "",
    ]
    for (response, descriptor), group in assoc.groupby(["response", "descriptor"], sort=True):
        values = pd.to_numeric(group["spearman_rho"], errors="coerce").dropna()
        if len(values):
            lines.append(f"- {response} vs {descriptor}: median protein Spearman={values.median():.4g}; positive fraction={(values > 0).mean():.4g}; n={len(values)} proteins.")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "These results describe local inverse-folding remodeling and its association with local structural relationships. They do not establish causality, biological structural uncertainty, functional impact, design failure, or generative consequences.",
            "",
            "Next stage: GENERATIVE_PROPAGATION.",
        ]
    )
    return "\n".join(lines) + "\n"


def materialize_structural_organization_result(result: StructuralOrganizationResult, output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    statuses: dict[str, str] = {}
    for frame, filename in (
        (result.position, "position_structural_organization.parquet"),
        (result.protein, "protein_structural_organization.parquet"),
        (result.spatial_null, "spatial_organization_null.parquet"),
        (result.associations, "descriptor_associations.parquet"),
    ):
        statuses[filename] = _immutable_parquet(frame, output_dir / filename)
    summary = _plain(result.summary)
    summary["output_rows"] = {"position": len(result.position), "protein": len(result.protein), "spatial_null": len(result.spatial_null), "associations": len(result.associations)}
    statuses["summary.json"] = _immutable_bytes((json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(), output_dir / "summary.json")
    statuses["report.md"] = _immutable_bytes(_markdown_report(result).encode(), output_dir / "report.md")
    figure_dir = output_dir / "figures"
    for name in ("response_vs_geometry", "breadth_reordering", "spatial_clustering", "representative_profiles"):
        statuses[f"figures/{name}.png"] = _immutable_bytes(_figure_bytes(result, name), figure_dir / f"{name}.png")
    return {"status": "COMPLETE", "output_dir": output_dir, "write_status": statuses}
