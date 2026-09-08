"""Load the frozen 68-protein PDB/AFDB operational cohort.

This is the sole runtime adapter for the compact pre-StructCal cohort release.
It consumes frozen membership/masks and never reruns the retired Stage0/Scale1
construction or scoring workflows.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from dual_uq.core.hashing import sha256_file
from dual_uq.structure_io import load_atom_site_table, residue_name_to_one_letter

COHORT_PATH = Path("experiments/dataset/releases/confirmatory/primary_cohort.parquet")
MASK_PATH = Path("experiments/dataset/releases/confirmatory/primary_common_masks.parquet")
VALIDITY_PATH = Path("experiments/dataset/analysis/pair_validity/pair_validity.parquet")
BACKBONE_ATOMS = ("N", "CA", "C", "O")


class OperationalPairError(ValueError):
    """Raised when the compact frozen operational release is inconsistent."""


@dataclass(frozen=True, slots=True)
class OperationalCondition:
    protein_id: str
    condition: str
    source_path: Path
    source_sha256: str
    source_chain_id: str
    source_id: str
    canonical_positions: tuple[int, ...]
    wt_sequence_projection: str
    coordinates: np.ndarray
    auth_keys: tuple[tuple[int, str], ...]


def frozen_operational_paths(project_root: Path) -> dict[str, Path]:
    root = Path(project_root).expanduser().resolve()
    return {
        "cohort": root / COHORT_PATH,
        "masks": root / MASK_PATH,
        "validity": root / VALIDITY_PATH,
    }


@lru_cache(maxsize=256)
def _backbone(path: str, chain_id: str) -> dict[tuple[int, str], tuple[str, np.ndarray]]:
    atoms = load_atom_site_table(Path(path))
    if atoms.empty:
        raise OperationalPairError(f"structure has no atoms: {path}")
    atoms = atoms.loc[atoms["model_number"].eq(int(atoms["model_number"].min()))]
    atoms = atoms.loc[atoms["auth_asym_id"].astype(str).eq(chain_id)].copy()
    atoms = atoms.loc[atoms["atom_name"].astype(str).str.upper().isin(BACKBONE_ATOMS)]
    atoms["_alt_rank"] = atoms["alt_id"].map(
        lambda value: 0 if str(value).strip().upper() in {"", ".", "?", "A"} else 1
    )
    atoms = atoms.sort_values(
        ["_alt_rank", "occupancy"], ascending=[True, False], kind="mergesort"
    )
    grouped: dict[tuple[int, str], dict[str, object]] = {}
    for row in atoms.itertuples(index=False):
        if pd.isna(row.auth_seq_id):
            continue
        key = (int(row.auth_seq_id), str(row.insertion_code or "").strip().upper())
        residue = grouped.setdefault(key, {"aa": residue_name_to_one_letter(str(row.residue_name))})
        atom = str(row.atom_name).strip().upper()
        residue.setdefault(atom, np.asarray([row.x, row.y, row.z], dtype=np.float32))
    result: dict[tuple[int, str], tuple[str, np.ndarray]] = {}
    for key, residue in grouped.items():
        if all(atom in residue for atom in BACKBONE_ATOMS):
            coordinates = np.asarray([residue[atom] for atom in BACKBONE_ATOMS], dtype=np.float32)
            if np.isfinite(coordinates).all():
                result[key] = (str(residue["aa"]), coordinates)
    return result


def find_true_gap_proteins(
    masks: pd.DataFrame, protein_ids: Iterable[str] | None = None
) -> tuple[str, ...]:
    required = {"protein_id", "canonical_position", "common_mask"}
    missing = sorted(required.difference(masks.columns))
    if missing:
        raise OperationalPairError(f"common mask missing columns: {missing}")
    selected = None if protein_ids is None else {str(value) for value in protein_ids}
    gaps: list[str] = []
    for protein_id, group in masks.loc[masks["common_mask"].eq(True)].groupby(
        "protein_id", sort=True
    ):
        if selected is not None and str(protein_id) not in selected:
            continue
        positions = tuple(sorted(group["canonical_position"].astype(int)))
        if any(right != left + 1 for left, right in pairwise(positions)):
            gaps.append(str(protein_id))
    return tuple(gaps)


def load_operational_conditions(
    project_root: Path,
    *,
    protein_ids: Iterable[str] | None = None,
    subset: int | None = None,
    exclude_true_gaps: bool = False,
) -> tuple[OperationalCondition, ...]:
    """Materialize frozen common-mask backbones without legacy workflow code."""

    root = Path(project_root).expanduser().resolve()
    paths = frozen_operational_paths(root)
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise OperationalPairError(f"frozen operational artifacts are missing: {missing}")
    cohort = pd.read_parquet(paths["cohort"])
    masks = pd.read_parquet(paths["masks"])
    validity = pd.read_parquet(paths["validity"])
    clean_ids = set(
        validity.loc[validity["high_comparability_eligible"].eq(True), "protein_id"].astype(str)
    )
    if len(clean_ids) != 68:
        raise OperationalPairError(f"frozen clean cohort must contain 68 proteins, got {len(clean_ids)}")
    cohort = cohort.loc[cohort["protein_id"].astype(str).isin(clean_ids)].copy()
    if len(cohort) != 68 or cohort["protein_id"].duplicated().any():
        raise OperationalPairError("clean cohort and compact primary cohort differ")
    requested = None if protein_ids is None else tuple(dict.fromkeys(map(str, protein_ids)))
    if requested is not None:
        unknown = sorted(set(requested).difference(clean_ids))
        if unknown:
            raise OperationalPairError(f"proteins are absent from the clean cohort: {unknown}")
        cohort = cohort.loc[cohort["protein_id"].astype(str).isin(requested)]
    if exclude_true_gaps:
        gap_ids = set(find_true_gap_proteins(masks, cohort["protein_id"]))
        cohort = cohort.loc[~cohort["protein_id"].astype(str).isin(gap_ids)]
    cohort = cohort.sort_values("protein_id", kind="mergesort").reset_index(drop=True)
    if subset is not None:
        if type(subset) is not int or not 0 < subset <= len(cohort):
            raise OperationalPairError(f"subset must be between 1 and {len(cohort)}")
        cohort = cohort.iloc[:subset]

    result: list[OperationalCondition] = []
    for source in cohort.itertuples(index=False):
        protein_id = str(source.protein_id)
        selected = masks.loc[
            masks["protein_id"].astype(str).eq(protein_id) & masks["common_mask"].eq(True)
        ].sort_values("canonical_position", kind="mergesort")
        if selected.empty or selected["canonical_position"].duplicated().any():
            raise OperationalPairError(f"common mask is empty or duplicated: {protein_id}")
        positions = tuple(selected["canonical_position"].astype(int))
        sequence = str(source.canonical_sequence)
        projection = "".join(sequence[position - 1] for position in positions)
        for condition in ("PDB", "AFDB"):
            prefix = condition.lower()
            path = root / str(getattr(source, f"{prefix}_structure_ref"))
            expected_sha = str(getattr(source, f"{prefix}_structure_sha256"))
            if not path.is_file() or sha256_file(path) != expected_sha:
                raise OperationalPairError(f"frozen structure hash mismatch: {protein_id}/{condition}")
            chain = str(source.pdb_chain) if condition == "PDB" else "A"
            backbone = _backbone(str(path.resolve()), chain)
            auth_keys = tuple(
                (
                    int(row.auth_seq_id),
                    "" if pd.isna(row.insertion_code) else str(row.insertion_code).strip().upper(),
                )
                if condition == "PDB"
                else (int(row.canonical_position), "")
                for row in selected.itertuples(index=False)
            )
            coordinates: list[np.ndarray] = []
            for position, expected_aa, key in zip(
                positions, selected["canonical_aa"].astype(str), auth_keys, strict=True
            ):
                residue = backbone.get(key)
                if residue is None:
                    raise OperationalPairError(
                        f"mapped backbone residue is absent: {protein_id}/{condition}/{position}"
                    )
                if expected_aa != sequence[position - 1] or residue[0] != expected_aa:
                    raise OperationalPairError(
                        f"residue identity mismatch: {protein_id}/{condition}/{position}"
                    )
                coordinates.append(residue[1])
            result.append(
                OperationalCondition(
                    protein_id=protein_id,
                    condition=condition,
                    source_path=path,
                    source_sha256=expected_sha,
                    source_chain_id=chain,
                    source_id=(str(source.pdb_id) if condition == "PDB" else str(source.afdb_model_id)),
                    canonical_positions=positions,
                    wt_sequence_projection=projection,
                    coordinates=np.asarray(coordinates, dtype=np.float32),
                    auth_keys=auth_keys,
                )
            )
    return tuple(result)
