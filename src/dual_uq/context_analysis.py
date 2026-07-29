from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser, NeighborSearch, ShrakeRupley

from .geometry import kabsch_align, rmsd


@dataclass(frozen=True)
class SegmentClassification:
    label: str
    explanation: str


def _normalise_residue_number(value: Any) -> str:
    return str(value).strip()


def _ca_coordinates(table: pd.DataFrame, prefix: str) -> np.ndarray:
    return table[[f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"]].to_numpy(dtype=float)


def residue_sasa_table(cif_path: str | Path, chain_id: str) -> pd.DataFrame:
    path = Path(cif_path)
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(path.stem, str(path))
    model = next(structure.get_models())
    if chain_id not in model:
        available = [chain.id for chain in model.get_chains()]
        raise KeyError(f"Chain {chain_id!r} not found in {path}; available={available}")

    ShrakeRupley().compute(model, level="R")
    rows: list[dict[str, Any]] = []
    for residue in model[chain_id]:
        if "CA" not in residue:
            continue
        _, seqnum, insertion_code = residue.id
        residue_number = str(seqnum)
        insertion = str(insertion_code).strip()
        if insertion:
            residue_number += insertion
        rows.append(
            {
                "pdb_residue_number_norm": residue_number,
                "pdb_residue_sasa": float(getattr(residue, "sasa", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def nearest_nonwater_hetero(
    cif_path: str | Path,
    chain_id: str,
    segment_residue_numbers: set[str],
) -> dict[str, Any]:
    path = Path(cif_path)
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(path.stem, str(path))
    model = next(structure.get_models())

    segment_atoms = []
    hetero_atoms = []
    hetero_meta: dict[int, dict[str, Any]] = {}

    for chain in model:
        for residue in chain:
            hetero_flag, seqnum, insertion_code = residue.id
            residue_number = str(seqnum)
            insertion = str(insertion_code).strip()
            if insertion:
                residue_number += insertion

            if chain.id == chain_id and residue_number in segment_residue_numbers:
                segment_atoms.extend(list(residue.get_atoms()))

            is_hetero = str(hetero_flag).strip() not in ("", "W")
            is_water = residue.resname.upper() in {"HOH", "WAT", "DOD"}
            if is_hetero and not is_water:
                for atom in residue.get_atoms():
                    hetero_atoms.append(atom)
                    hetero_meta[id(atom)] = {
                        "chain_id": chain.id,
                        "residue_name": residue.resname,
                        "residue_number": residue_number,
                        "atom_name": atom.name,
                    }

    if not segment_atoms or not hetero_atoms:
        return {
            "nearest_hetero_distance": None,
            "nearest_hetero_chain": None,
            "nearest_hetero_residue_name": None,
            "nearest_hetero_residue_number": None,
            "nearest_hetero_atom_name": None,
        }

    search = NeighborSearch(hetero_atoms)
    best_distance = float("inf")
    best_atom = None
    for atom in segment_atoms:
        neighbours = search.search(atom.coord, 20.0, level="A")
        for other in neighbours:
            distance = float(np.linalg.norm(atom.coord - other.coord))
            if distance < best_distance:
                best_distance = distance
                best_atom = other

    if best_atom is None:
        return {
            "nearest_hetero_distance": None,
            "nearest_hetero_chain": None,
            "nearest_hetero_residue_name": None,
            "nearest_hetero_residue_number": None,
            "nearest_hetero_atom_name": None,
        }

    meta = hetero_meta[id(best_atom)]
    return {
        "nearest_hetero_distance": best_distance,
        "nearest_hetero_chain": meta["chain_id"],
        "nearest_hetero_residue_name": meta["residue_name"],
        "nearest_hetero_residue_number": meta["residue_number"],
        "nearest_hetero_atom_name": meta["atom_name"],
    }


def classify_segment(
    *,
    local_fit_rmsd: float,
    flank_fit_segment_rmsd: float,
    internal_distance_mae: float,
) -> SegmentClassification:
    if local_fit_rmsd <= 0.75 and flank_fit_segment_rmsd >= 1.50:
        return SegmentClassification(
            "rigid_or_state_shift",
            "The segment is internally similar after local superposition but displaced "
            "relative to the surrounding structure.",
        )
    if local_fit_rmsd >= 1.00 or internal_distance_mae >= 0.75:
        return SegmentClassification(
            "local_deformation",
            "The segment retains substantial disagreement even after local superposition.",
        )
    return SegmentClassification(
        "mixed_or_small_change",
        "The segment shows neither a clear rigid displacement nor a strong local deformation.",
    )


def characterize_segment(
    residue_geometry: pd.DataFrame,
    pairwise_data: Any,
    *,
    start_position: int,
    end_position: int,
    flank_size: int = 10,
) -> dict[str, Any]:
    table = residue_geometry.sort_values("uniprot_residue_number").reset_index(drop=True)
    positions = table["uniprot_residue_number"].astype(int).to_numpy()
    segment_mask = (positions >= start_position) & (positions <= end_position)
    segment_indices = np.where(segment_mask)[0]
    if len(segment_indices) < 3:
        raise ValueError(
            f"Segment {start_position}-{end_position} has fewer than 3 mapped residues."
        )

    pdb = _ca_coordinates(table, "pdb")
    afdb = _ca_coordinates(table, "afdb")

    aligned_local, _, _ = kabsch_align(afdb[segment_indices], pdb[segment_indices])
    local_fit = rmsd(aligned_local, pdb[segment_indices])

    flank_mask = (
        ((positions >= start_position - flank_size) & (positions < start_position))
        | ((positions > end_position) & (positions <= end_position + flank_size))
    )
    flank_indices = np.where(flank_mask)[0]
    if len(flank_indices) < 3:
        flank_indices = np.where(~segment_mask)[0]

    _, rotation, translation = kabsch_align(afdb[flank_indices], pdb[flank_indices])
    afdb_flank_aligned = afdb @ rotation + translation
    flank_fit_segment = rmsd(
        afdb_flank_aligned[segment_indices],
        pdb[segment_indices],
    )

    pdb_segment = pdb[segment_indices]
    afdb_segment = afdb[segment_indices]
    pdb_delta = pdb_segment[:, None, :] - pdb_segment[None, :, :]
    afdb_delta = afdb_segment[:, None, :] - afdb_segment[None, :, :]
    pdb_distance = np.sqrt(np.sum(pdb_delta**2, axis=-1))
    afdb_distance = np.sqrt(np.sum(afdb_delta**2, axis=-1))
    upper = np.triu(np.ones_like(pdb_distance, dtype=bool), k=1)
    internal_mae = float(np.mean(np.abs(pdb_distance[upper] - afdb_distance[upper])))

    plddt = table.loc[segment_indices, "plddt"].to_numpy(dtype=float)
    global_disagreement = table.loc[
        segment_indices, "aligned_ca_distance"
    ].to_numpy(dtype=float)

    matrix_positions = np.asarray(pairwise_data["uniprot_positions"], dtype=int)
    matrix_lookup = {int(position): index for index, position in enumerate(matrix_positions)}
    matrix_segment_indices = np.array(
        [matrix_lookup[int(position)] for position in positions[segment_indices]],
        dtype=int,
    )
    matrix_rest_indices = np.array(
        [matrix_lookup[int(position)] for position in positions[~segment_mask]],
        dtype=int,
    )

    symmetric_pae = np.asarray(pairwise_data["symmetric_pae"], dtype=float)
    pair_error = np.asarray(pairwise_data["absolute_pairwise_error"], dtype=float)
    segment_to_rest_pae = symmetric_pae[
        np.ix_(matrix_segment_indices, matrix_rest_indices)
    ]
    segment_to_rest_error = pair_error[
        np.ix_(matrix_segment_indices, matrix_rest_indices)
    ]

    classification = classify_segment(
        local_fit_rmsd=local_fit,
        flank_fit_segment_rmsd=flank_fit_segment,
        internal_distance_mae=internal_mae,
    )

    return {
        "start_position": int(start_position),
        "end_position": int(end_position),
        "residue_count": int(len(segment_indices)),
        "median_plddt": float(np.median(plddt)),
        "min_plddt": float(np.min(plddt)),
        "median_global_disagreement": float(np.median(global_disagreement)),
        "max_global_disagreement": float(np.max(global_disagreement)),
        "local_fit_rmsd": float(local_fit),
        "flank_fit_segment_rmsd": float(flank_fit_segment),
        "internal_distance_mae": float(internal_mae),
        "segment_to_rest_pae_median": float(np.nanmedian(segment_to_rest_pae)),
        "segment_to_rest_pae_q90": float(np.nanquantile(segment_to_rest_pae, 0.90)),
        "segment_to_rest_pair_error_median": float(
            np.nanmedian(segment_to_rest_error)
        ),
        "segment_classification": classification.label,
        "classification_explanation": classification.explanation,
    }
