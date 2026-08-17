"""ESM-IF1 cross-model generalization analysis.

This module owns only ESM-IF1-native local-response and generation summaries.
ProteinMPNN artifacts are read as frozen descriptors for protein-level comparison;
their score semantics are never reimplemented here.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from dual_uq.core.atomic_io import atomic_write_new_bytes
from dual_uq.core.hashing import sha256_bytes
from dual_uq.models.esm_if1 import STANDARD_AMINO_ACIDS_TUPLE, ESMIF1ContractError

ESM_IF1_ANALYSIS_PROTOCOL = "esm_if1_cross_model_generalization_v1"
GENERATION_TEMPERATURE = 0.1
DEFAULT_INDEPENDENT_SAMPLES = 128
GENERATION_BATCH_SIZE = 16
TRUE_GAP_SENSITIVITY_PROTOCOL = "esm_if1_gap_preserving_sensitivity_v1"


def _load_structure_rows(
    path: Path, chain: str
) -> tuple[np.ndarray, list[tuple[int, str]], list[str]]:
    """Load official ESM backbone atoms and residue identifiers lazily."""
    try:
        from Bio.Data.PDBData import protein_letters_3to1_extended
        from esm.inverse_folding.util import load_structure
    except ImportError as exc:  # pragma: no cover - isolated runtime only
        raise ESMIF1ContractError("ESM-IF1 structure dependencies are unavailable") from exc
    try:
        structure = load_structure(str(path), chain)
        residue_keys = list(dict.fromkeys(zip(structure.res_id.tolist(), structure.ins_code.tolist(), strict=True)))
        coords = np.full((len(residue_keys), 3, 3), np.nan, dtype=np.float32)
        residue_names: list[str] = []
        for residue_index, (residue_id, insertion_code) in enumerate(residue_keys):
            residue_mask = (structure.res_id == residue_id) & (structure.ins_code == insertion_code)
            residue_atoms = structure[residue_mask]
            residue_name = str(residue_atoms.res_name[0]).upper()
            residue_names.append(
                residue_name
                if len(residue_name) == 1
                else protein_letters_3to1_extended.get(residue_name, "X")
            )
            for atom_index, atom_name in enumerate(("N", "CA", "C")):
                atom_rows = residue_atoms[residue_atoms.atom_name == atom_name]
                if len(atom_rows) > 1:
                    raise ESMIF1ContractError(
                        f"backbone contains duplicate {atom_name} atom: {path}"
                    )
                if len(atom_rows) == 1:
                    coords[residue_index, atom_index] = atom_rows.coord[0]
    except (OSError, RuntimeError, ValueError) as exc:
        raise ESMIF1ContractError(f"cannot parse backbone structure: {path}") from exc
    return (
        np.asarray(coords, dtype=np.float32),
        [(int(residue_id), str(insertion_code)) for residue_id, insertion_code in residue_keys],
        residue_names,
    )


def find_true_gap_proteins(
    masks: pd.DataFrame,
    protein_ids: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Return proteins whose frozen common-mask axis contains an internal gap."""
    required = {"protein_id", "canonical_position", "common_mask"}
    missing = sorted(required - set(masks.columns))
    if missing:
        raise ESMIF1ContractError(f"common mask missing columns: {missing}")
    selected = {str(value) for value in protein_ids} if protein_ids is not None else None
    gap_proteins: list[str] = []
    for protein_id, group in masks.loc[masks["common_mask"].eq(True)].groupby(
        "protein_id", sort=True
    ):
        if selected is not None and str(protein_id) not in selected:
            continue
        positions = tuple(sorted(int(value) for value in group["canonical_position"]))
        if any(right != left + 1 for left, right in pairwise(positions)):
            gap_proteins.append(str(protein_id))
    return tuple(gap_proteins)


def build_frozen_common_cases(
    project_root: Path,
    pair_validity_path: Path,
    common_mask_path: Path,
    *,
    subset: int | None = None,
    exclude_true_gap_proteins: bool = False,
) -> pd.DataFrame:
    """Build exactly the frozen clean PDB/AFDB common-mask input table."""
    project_root = Path(project_root)
    pairs = pd.read_parquet(pair_validity_path)
    masks = pd.read_parquet(common_mask_path)
    clean = pairs.loc[pairs["high_comparability_eligible"].eq(True)].copy()
    if len(clean) != 68:
        raise ESMIF1ContractError(f"frozen clean cohort must contain 68 proteins, got {len(clean)}")
    clean = clean.sort_values("protein_id", kind="mergesort").reset_index(drop=True)
    if exclude_true_gap_proteins:
        gap_proteins = set(find_true_gap_proteins(masks, clean["protein_id"]))
        clean = clean.loc[~clean["protein_id"].astype(str).isin(gap_proteins)].reset_index(drop=True)
    if subset is not None:
        maximum = len(clean)
        if type(subset) is not int or not 0 < subset <= maximum:
            raise ESMIF1ContractError(f"subset must be between 1 and {maximum}")
        clean = clean.iloc[:subset].copy()
    rows: list[dict[str, Any]] = []
    for source in clean.itertuples(index=False):
        protein_masks = masks.loc[
            masks["protein_id"].eq(source.protein_id) & masks["common_mask"].eq(True)
        ].sort_values("canonical_position", kind="mergesort")
        if protein_masks.empty:
            raise ESMIF1ContractError(f"common mask is empty: {source.protein_id}")
        canonical_positions = tuple(int(value) for value in protein_masks["canonical_position"])
        validate_contiguous_common_positions(canonical_positions)
        wt_sequence = "".join(
            str(source.canonical_sequence)[position - 1] for position in canonical_positions
        )
        pdb_path = project_root / str(source.pdb_structure_ref)
        afdb_path = project_root / str(source.afdb_structure_ref)
        pdb_coords, pdb_ids, pdb_names = _load_structure_rows(pdb_path, str(source.pdb_chain))
        afdb_coords, afdb_ids, afdb_names = _load_structure_rows(afdb_path, "A")
        pdb_lookup = {(int(residue_id), code): index for index, (residue_id, code) in enumerate(pdb_ids)}
        afdb_lookup = {(int(residue_id), code): index for index, (residue_id, code) in enumerate(afdb_ids)}
        selected_pdb: list[np.ndarray] = []
        selected_afdb: list[np.ndarray] = []
        for mask_row in protein_masks.itertuples(index=False):
            insertion_code = "" if pd.isna(mask_row.insertion_code) else str(mask_row.insertion_code)
            pdb_key = (int(mask_row.auth_seq_id), insertion_code)
            # AFDB model author numbering is the frozen canonical residue
            # namespace; the mask's label_seq_id belongs to the PDB mmCIF
            # namespace and is never used as an AFDB offset heuristic.
            afdb_key = (int(mask_row.canonical_position), "")
            if pdb_key not in pdb_lookup or afdb_key not in afdb_lookup:
                raise ESMIF1ContractError(f"common mask residue is absent from backbone: {source.protein_id}")
            pdb_index = pdb_lookup[pdb_key]
            afdb_index = afdb_lookup[afdb_key]
            expected_aa = str(mask_row.canonical_aa)
            if pdb_names[pdb_index] != expected_aa or afdb_names[afdb_index] != expected_aa:
                raise ESMIF1ContractError(
                    f"backbone residue identity differs from the frozen common mask: {source.protein_id}"
                )
            selected_pdb.append(pdb_coords[pdb_index])
            selected_afdb.append(afdb_coords[afdb_index])
        selected_pdb_array = np.asarray(selected_pdb, dtype=np.float32)
        selected_afdb_array = np.asarray(selected_afdb, dtype=np.float32)
        if not np.isfinite(selected_pdb_array).all() or not np.isfinite(selected_afdb_array).all():
            raise ESMIF1ContractError(f"common-mask backbone contains missing atoms: {source.protein_id}")
        for condition, coordinates, structure_sha256 in (
            ("PDB", selected_pdb_array, str(source.pdb_structure_sha256)),
            ("AFDB", selected_afdb_array, str(source.afdb_structure_sha256)),
        ):
            for position in canonical_positions:
                rows.append(
                    {
                        "protein_id": str(source.protein_id),
                        "condition": condition,
                        "position": position,
                        "wt_sequence": wt_sequence,
                        "coordinates": coordinates,
                        "canonical_positions": canonical_positions,
                        "structure_sha256": structure_sha256,
                        "common_mask_sha256": None,
                    }
                )
    cases = pd.DataFrame(rows)
    validate_common_backbone(cases)
    return cases


def build_gap_preserving_cases(
    project_root: Path,
    pair_validity_path: Path,
    common_mask_path: Path,
) -> pd.DataFrame:
    """Build canonical-length cases with NaN rows for true UniProt gaps.

    The six gap proteins are kept separate from the primary cohort.  Every
    canonical sequence position remains on the model axis; positions absent
    from the frozen mapping are represented by fully-NaN N/CA/C coordinates
    for both conditions.  Non-gap coordinate non-observability is retained as
    condition-specific NaN data and is never converted into a sequence gap.
    """
    project_root = Path(project_root)
    pairs = pd.read_parquet(pair_validity_path)
    masks = pd.read_parquet(common_mask_path)
    clean = pairs.loc[pairs["high_comparability_eligible"].eq(True)].copy()
    if len(clean) != 68:
        raise ESMIF1ContractError(f"frozen clean cohort must contain 68 proteins, got {len(clean)}")
    gap_ids = find_true_gap_proteins(masks, clean["protein_id"])
    if len(gap_ids) != 6:
        raise ESMIF1ContractError(f"expected six true-gap proteins, got {len(gap_ids)}")
    rows: list[dict[str, Any]] = []
    for source in clean.loc[clean["protein_id"].astype(str).isin(gap_ids)].sort_values(
        "protein_id", kind="mergesort"
    ).itertuples(index=False):
        protein_id = str(source.protein_id)
        canonical_sequence = str(source.canonical_sequence)
        canonical_positions = tuple(range(1, len(canonical_sequence) + 1))
        mask_rows = masks.loc[masks["protein_id"].eq(protein_id)].copy()
        if mask_rows["canonical_position"].duplicated().any():
            raise ESMIF1ContractError(f"duplicate canonical mapping rows: {protein_id}")
        mask_by_position = {
            int(row.canonical_position): row for row in mask_rows.itertuples(index=False)
        }
        pdb_path = project_root / str(source.pdb_structure_ref)
        afdb_path = project_root / str(source.afdb_structure_ref)
        pdb_coords, pdb_ids, pdb_names = _load_structure_rows(pdb_path, str(source.pdb_chain))
        afdb_coords, afdb_ids, afdb_names = _load_structure_rows(afdb_path, "A")
        pdb_lookup = {
            (int(residue_id), code): index
            for index, (residue_id, code) in enumerate(pdb_ids)
        }
        afdb_lookup = {
            (int(residue_id), code): index
            for index, (residue_id, code) in enumerate(afdb_ids)
        }
        condition_arrays: dict[str, np.ndarray] = {
            "PDB": np.full((len(canonical_sequence), 3, 3), np.nan, dtype=np.float32),
            "AFDB": np.full((len(canonical_sequence), 3, 3), np.nan, dtype=np.float32),
        }
        mapped_positions = set(mask_by_position)
        if mapped_positions:
            first_mapped = min(mapped_positions)
            last_mapped = max(mapped_positions)
            true_gap_positions = {
                position
                for position in range(first_mapped, last_mapped + 1)
                if position not in mapped_positions
            }
        else:
            true_gap_positions = set()
        comparable_positions: list[int] = []
        for position in canonical_positions:
            mask_row = mask_by_position.get(position)
            if mask_row is None:
                continue
            expected_aa = str(mask_row.canonical_aa)
            if expected_aa != canonical_sequence[position - 1]:
                raise ESMIF1ContractError(f"canonical sequence mismatch: {protein_id}/{position}")
            if bool(mask_row.common_mask):
                comparable_positions.append(position)
            insertion_code = "" if pd.isna(mask_row.insertion_code) else str(mask_row.insertion_code)
            if bool(mask_row.pdb_backbone_complete) and bool(mask_row.pdb_mapped):
                pdb_key = (int(mask_row.auth_seq_id), insertion_code)
                if pdb_key not in pdb_lookup:
                    raise ESMIF1ContractError(f"PDB mapped residue absent: {protein_id}/{position}")
                pdb_index = pdb_lookup[pdb_key]
                if pdb_names[pdb_index] != expected_aa:
                    raise ESMIF1ContractError(f"PDB residue identity mismatch: {protein_id}/{position}")
                condition_arrays["PDB"][position - 1] = pdb_coords[pdb_index]
            if bool(mask_row.afdb_backbone_complete) and bool(mask_row.afdb_mapped):
                afdb_key = (position, "")
                if afdb_key not in afdb_lookup:
                    raise ESMIF1ContractError(f"AFDB mapped residue absent: {protein_id}/{position}")
                afdb_index = afdb_lookup[afdb_key]
                if afdb_names[afdb_index] != expected_aa:
                    raise ESMIF1ContractError(f"AFDB residue identity mismatch: {protein_id}/{position}")
                condition_arrays["AFDB"][position - 1] = afdb_coords[afdb_index]
        for condition, coordinates, structure_sha256 in (
            ("PDB", condition_arrays["PDB"], str(source.pdb_structure_sha256)),
            ("AFDB", condition_arrays["AFDB"], str(source.afdb_structure_sha256)),
        ):
            for position in canonical_positions:
                rows.append(
                    {
                        "protein_id": protein_id,
                        "condition": condition,
                        "position": position,
                        "wt_sequence": canonical_sequence,
                        "coordinates": coordinates,
                        "canonical_positions": canonical_positions,
                        "comparable": position in comparable_positions,
                        "true_gap": position in true_gap_positions,
                        "structure_sha256": structure_sha256,
                        "common_mask_sha256": None,
                    }
                )
    cases = pd.DataFrame(rows)
    validate_gap_preserving_backbone(cases)
    return cases


def jensen_shannon_bits(left: Iterable[float], right: Iterable[float]) -> float:
    """Return a zero-safe Jensen-Shannon divergence in bits."""
    p = np.asarray(tuple(left), dtype=float)
    q = np.asarray(tuple(right), dtype=float)
    if p.shape != q.shape or p.ndim != 1 or not len(p):
        raise ESMIF1ContractError("JS inputs must be equal non-empty vectors")
    if not np.isfinite(p).all() or not np.isfinite(q).all() or (p < 0).any() or (q < 0).any():
        raise ESMIF1ContractError("JS inputs must be finite non-negative vectors")
    if p.sum() <= 0 or q.sum() <= 0:
        raise ESMIF1ContractError("JS inputs must have positive mass")
    p = p / p.sum()
    q = q / q.sum()
    midpoint = 0.5 * (p + q)
    terms = np.zeros_like(p)
    for distribution in (p, q):
        mask = distribution > 0
        terms[mask] += 0.5 * distribution[mask] * np.log2(
            distribution[mask] / midpoint[mask]
        )
    return float(terms.sum())


def _required_case_columns(cases: pd.DataFrame) -> None:
    required = {"protein_id", "condition", "position", "wt_sequence", "coordinates"}
    missing = sorted(required - set(cases.columns))
    if missing:
        raise ESMIF1ContractError(f"backbone cases missing columns: {missing}")


def validate_contiguous_common_positions(positions: Iterable[int]) -> None:
    """Reject true UniProt gaps that a single-chain ESM-IF1 array would compress."""
    ordered = tuple(int(value) for value in positions)
    if not ordered:
        raise ESMIF1ContractError("common positions are empty")
    if any(right <= left for left, right in pairwise(ordered)):
        raise ESMIF1ContractError("common positions must be strictly increasing")
    if any(right != left + 1 for left, right in pairwise(ordered)):
        raise ESMIF1ContractError(
            "true UniProt residue gaps cannot be compressed into an ESM-IF1 chain"
        )


def validate_common_backbone(cases: pd.DataFrame) -> None:
    """Require exact one-to-one PDB/AFDB common-position rows per protein."""
    _required_case_columns(cases)
    if cases.empty:
        raise ESMIF1ContractError("common backbone cases are empty")
    if set(cases["condition"].astype(str)) != {"PDB", "AFDB"}:
        raise ESMIF1ContractError("common backbone cases require PDB and AFDB conditions")
    key = ["protein_id", "condition", "position"]
    if cases.duplicated(key).any():
        raise ESMIF1ContractError("backbone rows must pair one-to-one")
    for protein_id, group in cases.groupby("protein_id", sort=False):
        positions = {
            condition: tuple(group.loc[group["condition"] == condition, "position"])
            for condition in ("PDB", "AFDB")
        }
        if positions["PDB"] != positions["AFDB"]:
            raise ESMIF1ContractError(f"common positions differ for {protein_id}")
        for condition, condition_group in group.groupby("condition", sort=False):
            sequences = condition_group["wt_sequence"].astype(str).unique()
            if len(sequences) != 1 or len(sequences[0]) != len(condition_group):
                raise ESMIF1ContractError(f"WT context differs for {protein_id}/{condition}")
            if tuple(condition_group["position"]) != tuple(sorted(condition_group["position"])):
                raise ESMIF1ContractError(f"positions are not ordered for {protein_id}/{condition}")
            coordinates = _coordinates_for_group(condition_group)
            if coordinates.shape[0] != len(condition_group):
                raise ESMIF1ContractError(f"coordinate length differs for {protein_id}/{condition}")


def validate_gap_preserving_backbone(cases: pd.DataFrame) -> None:
    """Validate canonical axes and explicit true-gap missingness for six proteins."""
    _required_case_columns(cases)
    required = {"canonical_positions", "comparable", "true_gap"}
    missing = sorted(required - set(cases.columns))
    if missing:
        raise ESMIF1ContractError(f"gap-preserving cases missing columns: {missing}")
    if cases.empty or set(cases["condition"].astype(str)) != {"PDB", "AFDB"}:
        raise ESMIF1ContractError("gap-preserving cases require PDB and AFDB conditions")
    for protein_id, group in cases.groupby("protein_id", sort=False):
        canonical_values = list(group["canonical_positions"])
        if not canonical_values:
            raise ESMIF1ContractError(f"canonical axis is empty: {protein_id}")
        canonical_positions = tuple(int(value) for value in canonical_values[0])
        if canonical_positions != tuple(range(1, len(canonical_positions) + 1)):
            raise ESMIF1ContractError(f"canonical axis is not contiguous: {protein_id}")
        if len(group) != 2 * len(canonical_positions):
            raise ESMIF1ContractError(f"canonical positions are not paired: {protein_id}")
        per_condition = {}
        for condition, condition_group in group.groupby("condition", sort=False):
            if tuple(condition_group.sort_values("position")["position"]) != canonical_positions:
                raise ESMIF1ContractError(f"position axis differs for {protein_id}/{condition}")
            coordinates = _coordinates_for_group(condition_group, allow_missing=True)
            if coordinates.shape[0] != len(canonical_positions):
                raise ESMIF1ContractError(f"coordinate length differs for {protein_id}/{condition}")
            per_condition[condition] = condition_group.sort_values("position")
        for condition in ("PDB", "AFDB"):
            condition_rows = per_condition[condition]
            condition_coordinates = _coordinates_for_group(
                condition_rows, allow_missing=True
            )
            gaps = {
                int(row.position)
                for row in condition_rows.itertuples(index=False)
                if bool(row.true_gap)
            }
            if any(np.isfinite(condition_coordinates[position - 1]).any() for position in gaps):
                raise ESMIF1ContractError(f"true gap has coordinates: {protein_id}/{condition}")
            if condition == "PDB":
                pdb_gaps = gaps
            elif gaps != pdb_gaps:
                raise ESMIF1ContractError(f"gap semantics differ across conditions: {protein_id}")
        comparable = {
            int(row.position)
            for row in per_condition["PDB"].itertuples(index=False)
            if bool(row.comparable)
        }
        afdb_comparable = {
            int(row.position)
            for row in per_condition["AFDB"].itertuples(index=False)
            if bool(row.comparable)
        }
        if comparable != afdb_comparable:
            raise ESMIF1ContractError(f"comparable positions differ across conditions: {protein_id}")


def _coordinates_for_group(
    group: pd.DataFrame, *, allow_missing: bool = False
) -> np.ndarray:
    values = list(group["coordinates"])
    if not values:
        raise ESMIF1ContractError("backbone coordinate rows are empty")
    first = np.asarray(values[0], dtype=np.float32)
    if first.ndim != 3 or first.shape[1:] != (3, 3):
        raise ESMIF1ContractError("backbone coordinates must have shape (L, 3, 3)")
    if not allow_missing and not np.isfinite(first).all():
        raise ESMIF1ContractError("backbone coordinates must be finite")
    if allow_missing:
        finite_rows = np.isfinite(first).all(axis=(1, 2))
        missing_rows = np.isnan(first).all(axis=(1, 2))
        if not np.logical_or(finite_rows, missing_rows).all():
            raise ESMIF1ContractError("missing-coordinate rows must be fully NaN or finite")
    for value in values[1:]:
        candidate = np.asarray(value, dtype=np.float32)
        if candidate.shape != first.shape or not np.array_equal(candidate, first, equal_nan=True):
            raise ESMIF1ContractError("duplicate backbone rows carry different coordinates")
    return first


def teacher_forced_local_response(
    adapter: Any,
    cases: pd.DataFrame,
    *,
    allow_missing_coordinates: bool = False,
    comparable_only: bool = False,
) -> pd.DataFrame:
    """Score identical WT context on both backbones and pair native distributions."""
    if allow_missing_coordinates:
        validate_gap_preserving_backbone(cases)
    else:
        validate_common_backbone(cases)
    rows: list[dict[str, Any]] = []
    for (protein_id, condition), group in cases.groupby(["protein_id", "condition"], sort=False):
        group = group.sort_values("position", kind="mergesort")
        sequence = str(group["wt_sequence"].iloc[0])
        distributions = np.asarray(
            adapter.score_teacher_forced(
                sequence,
                _coordinates_for_group(group, allow_missing=allow_missing_coordinates),
                allow_missing_coordinates=allow_missing_coordinates,
            ),
            dtype=float,
        )
        if distributions.shape != (len(group), 20) or not np.isfinite(distributions).all():
            raise ESMIF1ContractError("ESM-IF1 distribution shape or finiteness is invalid")
        if not np.allclose(distributions.sum(axis=1), 1.0, atol=1e-6):
            raise ESMIF1ContractError("ESM-IF1 distributions must sum to one")
        for index, (_, source) in enumerate(group.iterrows()):
            if comparable_only and not bool(source.get("comparable", True)):
                continue
            rows.append(
                {
                    "protein_id": str(protein_id),
                    "condition": str(condition),
                    "position": int(source["position"]),
                    "wt_aa": sequence[index],
                    "distribution": tuple(float(value) for value in distributions[index]),
                    "primary_metric": "js_bits",
                }
            )
    result = pd.DataFrame(rows)
    paired = result.pivot(index=["protein_id", "position"], columns="condition", values="distribution")
    paired["js_bits"] = [jensen_shannon_bits(row["PDB"], row["AFDB"]) for _, row in paired.iterrows()]
    result = result.merge(paired["js_bits"].reset_index(), on=["protein_id", "position"], how="left")
    return result.sort_values(["protein_id", "condition", "position"], kind="mergesort").reset_index(drop=True)


def summarize_local_response(position_table: pd.DataFrame) -> pd.DataFrame:
    required = {"protein_id", "position", "condition", "js_bits"}
    missing = sorted(required - set(position_table.columns))
    if missing:
        raise ESMIF1ContractError(f"local response missing columns: {missing}")
    summary = (
        position_table.groupby("protein_id", sort=True)
        .agg(
            esm_if1_local_burden=("js_bits", "mean"),
            esm_if1_local_median=("js_bits", "median"),
            esm_if1_position_count=("position", "nunique"),
        )
        .reset_index()
    )
    return summary


def compare_local_response_with_proteinmpnn(
    esm_protein: pd.DataFrame, proteinmpnn: pd.DataFrame
) -> pd.DataFrame:
    """Join local burdens at protein level and retain descriptive associations in attrs."""
    if "protein_id" not in esm_protein or "protein_id" not in proteinmpnn:
        raise ESMIF1ContractError("protein-level comparison requires protein_id")
    merged = esm_protein.merge(proteinmpnn, on="protein_id", how="inner", validate="one_to_one")
    if merged.empty:
        raise ESMIF1ContractError("ESM-IF1 and ProteinMPNN protein sets do not overlap")
    association: dict[str, float | None] = {}
    for column in ("magnitude_p", "rank_displacement", "breadth_b"):
        if column not in merged:
            continue
        valid = merged[["esm_if1_local_burden", column]].dropna()
        association[column] = (
            float(spearmanr(valid.iloc[:, 0], valid.iloc[:, 1]).statistic)
            if len(valid) >= 3
            else None
        )
    merged.attrs["spearman"] = association
    return merged


def _stable_seed(namespace: str, protein_id: str, condition: str, sample_index: int) -> int:
    payload = f"{namespace}|{protein_id}|{condition}|{sample_index}".encode()
    value = int(hashlib.sha256(payload).hexdigest()[:16], 16) % (2**32 - 1)
    return max(1, value)


def generate_esm_if1_ensemble(
    adapter: Any,
    cases: pd.DataFrame,
    *,
    n_samples: int = DEFAULT_INDEPENDENT_SAMPLES,
    temperature: float = GENERATION_TEMPERATURE,
    seed_namespace: str,
    allow_missing_coordinates: bool = False,
) -> pd.DataFrame:
    """Generate independent per-condition ensembles with disjoint seed domains."""
    if allow_missing_coordinates:
        validate_gap_preserving_backbone(cases)
    else:
        validate_common_backbone(cases)
    if type(n_samples) is not int or n_samples <= 0:
        raise ESMIF1ContractError("n_samples must be a positive integer")
    if float(temperature) != GENERATION_TEMPERATURE:
        raise ESMIF1ContractError("ESM-IF1 generation temperature must be exactly 0.1")
    rows: list[dict[str, Any]] = []
    for (protein_id, condition), group in cases.groupby(["protein_id", "condition"], sort=False):
        group = group.sort_values("position", kind="mergesort")
        coordinates = _coordinates_for_group(group, allow_missing=allow_missing_coordinates)
        canonical_positions = tuple(int(value) for value in group["position"])
        comparable_canonical_positions = tuple(
            int(value)
            for value, comparable in zip(
                canonical_positions,
                group["comparable"] if "comparable" in group else [True] * len(group),
                strict=True,
            )
            if bool(comparable)
        )
        comparable_positions = tuple(
            index
            for index, comparable in enumerate(
                group["comparable"] if "comparable" in group else [True] * len(group),
                start=1,
            )
            if bool(comparable)
        )
        sample_indices = list(range(n_samples))
        for start in range(0, n_samples, GENERATION_BATCH_SIZE):
            chunk_indices = sample_indices[start : start + GENERATION_BATCH_SIZE]
            seeds = tuple(
                _stable_seed(seed_namespace, str(protein_id), str(condition), sample_index)
                for sample_index in chunk_indices
            )
            if hasattr(adapter, "sample_batch"):
                sequences = adapter.sample_batch(
                    np.repeat(coordinates[None, ...], len(chunk_indices), axis=0),
                    tuple(float(temperature) for _ in chunk_indices),
                    seeds,
                    allow_missing_coordinates=allow_missing_coordinates,
                )
            else:
                sequences = tuple(
                    adapter.sample(
                        coordinates,
                        float(temperature),
                        seed,
                        allow_missing_coordinates=allow_missing_coordinates,
                    )
                    for seed in seeds
                )
            for sample_index, seed, sequence in zip(chunk_indices, seeds, sequences, strict=True):
                sequence = str(sequence)
                if len(sequence) != len(group) or set(sequence) - set(STANDARD_AMINO_ACIDS_TUPLE):
                    raise ESMIF1ContractError("ESM-IF1 generated sequence is incompatible with the mask")
                binding = dict(adapter.binding())
                rows.append(
                    {
                        "protein_id": str(protein_id),
                        "condition": str(condition),
                        "sample_index": sample_index,
                        "seed": seed,
                        "sequence": sequence,
                        "sequence_hash": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
                        "temperature": float(temperature),
                        "n_residues": len(group),
                        "canonical_positions": canonical_positions,
                        "comparable_positions": comparable_positions,
                        "comparable_canonical_positions": comparable_canonical_positions,
                        "allow_missing_coordinates": allow_missing_coordinates,
                        "model_name": binding.get("model_name", "esm_if1_gvp4_t16_142M_UR50"),
                        "implementation_revision": binding.get("implementation_revision"),
                        "checkpoint_sha256": binding.get("checkpoint_sha256"),
                    }
                )
    result = pd.DataFrame(rows).sort_values(
        ["protein_id", "condition", "sample_index"], kind="mergesort"
    ).reset_index(drop=True)
    return result


def _normalized_hamming(
    left: str, right: str, positions: tuple[int, ...] | None = None
) -> float:
    if len(left) != len(right):
        raise ESMIF1ContractError("generated sequences in a protein have different lengths")
    indices = positions or tuple(range(1, len(left) + 1))
    if not indices or any(position < 1 or position > len(left) for position in indices):
        raise ESMIF1ContractError("generated comparable positions are invalid")
    return float(
        sum(left[position - 1] != right[position - 1] for position in indices) / len(indices)
    )


def _within_diversity(
    sequences: list[str], positions: tuple[int, ...] | None = None
) -> float:
    if len(sequences) < 2:
        return 0.0
    values = [
        _normalized_hamming(sequences[i], sequences[j], positions)
        for i in range(len(sequences))
        for j in range(i + 1, len(sequences))
    ]
    return float(np.mean(values))


def _between_diversity(
    left: list[str], right: list[str], positions: tuple[int, ...] | None = None
) -> float:
    return float(np.mean([_normalized_hamming(a, b, positions) for a in left for b in right]))


def _position_js(
    left: list[str],
    right: list[str],
    positions: tuple[int, ...] | None = None,
) -> list[float]:
    if not left or not right or len(left[0]) != len(right[0]):
        raise ESMIF1ContractError("generated condition sequence domains do not match")
    result = []
    indices = positions or tuple(range(1, len(left[0]) + 1))
    for position in indices:
        index = position - 1
        left_counts = np.asarray(
            [sum(sequence[index] == aa for sequence in left) for aa in STANDARD_AMINO_ACIDS_TUPLE],
            dtype=float,
        )
        right_counts = np.asarray(
            [sum(sequence[index] == aa for sequence in right) for aa in STANDARD_AMINO_ACIDS_TUPLE],
            dtype=float,
        )
        result.append(jensen_shannon_bits(left_counts, right_counts))
    return result


def summarize_generation_propagation(
    records: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Derive D_PP, D_AA, D_PA, excess and position-wise JS from generated records."""
    required = {"protein_id", "condition", "sample_index", "sequence"}
    missing = sorted(required - set(records.columns))
    if missing:
        raise ESMIF1ContractError(f"generated records missing columns: {missing}")
    if records.duplicated(["protein_id", "condition", "sample_index"]).any():
        raise ESMIF1ContractError("generated records contain duplicate sample keys")
    protein_rows: list[dict[str, object]] = []
    position_rows: list[dict[str, object]] = []
    for protein_id, group in records.groupby("protein_id", sort=True):
        by_condition = {condition: list(values["sequence"]) for condition, values in group.groupby("condition", sort=False)}
        if set(by_condition) != {"PDB", "AFDB"}:
            raise ESMIF1ContractError(f"generation condition pair is incomplete: {protein_id}")
        if "comparable_positions" in group:
            comparable_values = list(group["comparable_positions"])
            positions = tuple(int(value) for value in comparable_values[0])
            if any(tuple(int(value) for value in item) != positions for item in comparable_values[1:]):
                raise ESMIF1ContractError(f"comparable position axis differs: {protein_id}")
        else:
            positions = None
        if "comparable_canonical_positions" in group:
            canonical_values = list(group["comparable_canonical_positions"])
            output_positions = tuple(int(value) for value in canonical_values[0])
            if any(tuple(int(value) for value in item) != output_positions for item in canonical_values[1:]):
                raise ESMIF1ContractError(f"canonical comparable axis differs: {protein_id}")
        else:
            output_positions = positions
        d_pp = _within_diversity(by_condition["PDB"], positions)
        d_aa = _within_diversity(by_condition["AFDB"], positions)
        d_pa = _between_diversity(by_condition["PDB"], by_condition["AFDB"], positions)
        d_excess = d_pa - 0.5 * (d_pp + d_aa)
        js_values = _position_js(by_condition["PDB"], by_condition["AFDB"], positions)
        protein_rows.append(
            {
                "protein_id": protein_id,
                "d_pp": d_pp,
                "d_aa": d_aa,
                "d_pa": d_pa,
                "d_excess": d_excess,
                "js_burden": float(np.mean(js_values)),
                "sample_count_pdb": len(by_condition["PDB"]),
                "sample_count_afdb": len(by_condition["AFDB"]),
                "comparable_position_count": len(positions) if positions is not None else len(js_values),
            }
        )
        for position, js_value in zip(
            output_positions or tuple(range(1, len(js_values) + 1)),
            js_values,
            strict=True,
        ):
            position_rows.append({"protein_id": protein_id, "position": position, "js_gen": js_value})
    protein = pd.DataFrame(protein_rows)
    position = pd.DataFrame(position_rows)
    summary = {
        "analysis_protocol": ESM_IF1_ANALYSIS_PROTOCOL,
        "protein_unit": "protein",
        "protein_count": len(protein),
        "generated_record_count": len(records),
        "positive_d_excess_proteins": int((protein["d_excess"] > 0).sum()),
        "positive_js_burden_proteins": int((protein["js_burden"] > 0).sum()),
    }
    return protein, position, summary


def materialize_esm_if1_release(result: dict[str, pd.DataFrame | dict[str, object]], output_root: Path) -> None:
    """Write a small immutable release from canonical dataframes and summary."""
    protein = result["protein_summary"]
    position = result["position_summary"]
    generated = result["generated_records"]
    summary = result["summary"]
    if not isinstance(protein, pd.DataFrame) or not isinstance(position, pd.DataFrame) or not isinstance(generated, pd.DataFrame):
        raise ESMIF1ContractError("release result tables are invalid")
    if int(summary.get("protein_count", -1)) != 68:
        raise ESMIF1ContractError("ESM-IF1 release requires 68 proteins")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    tables = {"generated_records": generated, "position_summary": position, "protein_summary": protein}
    hashes: dict[str, str] = {}
    for name, table in tables.items():
        path = output_root / f"{name}.parquet"
        buffer = io.BytesIO()
        table.to_parquet(buffer, index=False)
        atomic_write_new_bytes(path, buffer.getvalue())
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary_payload = dict(summary)
    summary_payload["artifacts"] = hashes
    summary_bytes = json.dumps(summary_payload, sort_keys=True, indent=2).encode("utf-8")
    atomic_write_new_bytes(output_root / "summary.json", summary_bytes)
    manifest = {
        "analysis_protocol": ESM_IF1_ANALYSIS_PROTOCOL,
        "protein_count": 68,
        "artifacts": hashes,
        "summary_sha256": sha256_bytes(summary_bytes),
    }
    atomic_write_new_bytes(
        output_root / "manifest.json", json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8")
    )
