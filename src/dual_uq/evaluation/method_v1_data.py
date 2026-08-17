"""Frozen-cohort inputs for the paired-state V1 training experiment."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dual_uq.models.proteinmpnn import STANDARD_AMINO_ACIDS
from dual_uq.structure_io import load_chain_ca_table


class V1CohortError(ValueError):
    """Raised when the frozen V1 cohort cannot be resolved safely."""


@dataclass(frozen=True, slots=True)
class CohortResolution:
    """Assigned and model-evaluable members for TRAIN/VALIDATION only."""

    assigned: pd.DataFrame
    evaluable: pd.DataFrame
    unavailable: pd.DataFrame
    split_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProteinStateExample:
    """One protein with aligned state-local features and canonical targets."""

    protein_id: str
    pair_id: str
    split: str
    sequence: str
    state_features: np.ndarray
    observability: np.ndarray
    disagreement: np.ndarray
    targets: np.ndarray


_AA_TO_INDEX = {aa: index for index, aa in enumerate(STANDARD_AMINO_ACIDS)}
_REQUIRED_SPLIT_COLUMNS = ("protein_id", "pair_id", "uniprot_id", "split", "sequence")


def _load_sequence(path: Path, accession: str) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        returned = payload["primaryAccession"]
        sequence = payload["sequence"]["value"]
        length = int(payload["sequence"]["length"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise V1CohortError(f"invalid canonical sequence record: {path}") from exc
    if returned != accession or type(sequence) is not str or len(sequence) != length:
        raise V1CohortError(f"canonical sequence identity mismatch: {path}")
    return sequence


def resolve_v1_cohort(
    split_path: Path,
    candidates_path: Path,
    evaluability_path: Path,
    sequence_dir: Path,
    *,
    split_names: tuple[str, ...] = ("TRAIN", "VALIDATION"),
) -> CohortResolution:
    """Resolve frozen split membership without touching LOCKED_TEST outcomes."""

    if "LOCKED_TEST" in split_names:
        raise V1CohortError("LOCKED_TEST is inaccessible during V1 training")
    if not split_names or any(name not in {"TRAIN", "VALIDATION"} for name in split_names):
        raise V1CohortError(f"unsupported V1 split names: {split_names}")
    split = pd.read_parquet(split_path)
    missing = sorted(set(_REQUIRED_SPLIT_COLUMNS).difference(split.columns))
    if missing:
        raise V1CohortError(f"identity split lacks required columns: {missing}")
    assigned = split.loc[split["split"].isin(split_names)].copy()
    if assigned.empty:
        raise V1CohortError("requested V1 split has no assigned proteins")
    if assigned["protein_id"].duplicated().any():
        raise V1CohortError("V1 split contains duplicate protein identities")
    candidates = pd.read_parquet(candidates_path)
    required_candidates = {"protein_id", "pair_id", "uniprot_id", "apo_mmcif_relative_path", "holo_mmcif_relative_path", "apo_chain_id", "holo_chain_id"}
    missing = sorted(required_candidates.difference(candidates.columns))
    if missing:
        raise V1CohortError(f"primary-pair table lacks required columns: {missing}")
    candidates = candidates.drop_duplicates("protein_id", keep=False)
    assigned = assigned.merge(
        candidates[list(required_candidates)],
        on=("protein_id", "pair_id", "uniprot_id"),
        how="left",
        validate="one_to_one",
    )
    if assigned["apo_mmcif_relative_path"].isna().any():
        raise V1CohortError("V1 split member lacks a frozen primary pair")
    evaluability = pd.read_parquet(evaluability_path)
    required_eval = {"protein_id", "model_evaluable", "model_failure_reason"}
    missing = sorted(required_eval.difference(evaluability.columns))
    if missing:
        raise V1CohortError(f"model-evaluability table lacks required columns: {missing}")
    assigned = assigned.merge(
        evaluability[list(required_eval)],
        on="protein_id",
        how="left",
        validate="one_to_one",
    )
    if assigned["model_evaluable"].isna().any():
        raise V1CohortError("V1 split member lacks model-evaluability status")
    sequence_root = Path(sequence_dir)
    reasons: list[str | None] = []
    for row in assigned.itertuples(index=False):
        sequence = _load_sequence(sequence_root / f"{row.uniprot_id}.json", str(row.uniprot_id))
        if sequence != str(row.sequence):
            raise V1CohortError(f"frozen split sequence differs from canonical source: {row.protein_id}")
        invalid = sorted(set(sequence).difference(STANDARD_AMINO_ACIDS))
        if invalid:
            reasons.append("nonstandard_canonical_residue")
        elif not bool(row.model_evaluable):
            reasons.append(str(row.model_failure_reason) if row.model_failure_reason else "model_unavailable")
        else:
            reasons.append(None)
    assigned["v1_unavailable_reason"] = reasons
    unavailable = assigned.loc[assigned["v1_unavailable_reason"].notna()].copy()
    evaluable = assigned.loc[assigned["v1_unavailable_reason"].isna()].copy()
    evaluable = evaluable.sort_values(["split", "protein_id"], kind="mergesort").reset_index(drop=True)
    unavailable = unavailable.sort_values(["split", "protein_id"], kind="mergesort").reset_index(drop=True)
    assigned = assigned.sort_values(["split", "protein_id"], kind="mergesort").reset_index(drop=True)
    return CohortResolution(assigned=assigned, evaluable=evaluable, unavailable=unavailable, split_names=tuple(split_names))


def _coordinate_array(ca: Any) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    keys = [str(value) for value in ca["pdb_residue_number"]]
    xyz = np.column_stack([np.asarray(ca[column], dtype=float) for column in ("x", "y", "z")])
    bfactor = np.asarray(ca["bfactor"], dtype=float) if "bfactor" in getattr(ca, "columns", ()) else np.zeros(len(ca), dtype=float)
    coordinates: dict[str, np.ndarray] = {}
    bfactors: dict[str, float] = {}
    for key, value, bvalue in zip(keys, xyz, bfactor, strict=True):
        if np.isfinite(value).all():
            coordinates[key] = value.astype(np.float32)
            bfactors[key] = float(bvalue) if np.isfinite(bvalue) else 0.0
    return coordinates, bfactors


def _state_features(
    positions: list[str | None],
    coordinates: dict[str, np.ndarray],
    bfactors: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    length = len(positions)
    observed = np.asarray([key is not None and key in coordinates for key in positions], dtype=bool)
    raw = np.full((length, 3), np.nan, dtype=np.float32)
    bvalues = np.zeros(length, dtype=np.float32)
    for index, key in enumerate(positions):
        if observed[index]:
            raw[index] = coordinates[str(key)]
            bvalues[index] = bfactors.get(str(key), 0.0)
    if observed.any():
        center = raw[observed].mean(axis=0)
        scale = float(raw[observed].std()) or 1.0
        xyz = np.where(observed[:, None], (raw - center) / scale, 0.0)
        bcenter = float(bvalues[observed].mean())
        bscale = float(bvalues[observed].std()) or 1.0
        bnorm = np.where(observed, (bvalues - bcenter) / bscale, 0.0)
    else:
        xyz = np.zeros((length, 3), dtype=np.float32)
        bnorm = np.zeros(length, dtype=np.float32)
    left = np.zeros(length, dtype=np.float32)
    right = np.zeros(length, dtype=np.float32)
    contacts = np.zeros(length, dtype=np.float32)
    for index in range(length):
        if not observed[index]:
            continue
        if index > 0 and observed[index - 1]:
            left[index] = np.linalg.norm(raw[index] - raw[index - 1]) / 10.0
        if index + 1 < length and observed[index + 1]:
            right[index] = np.linalg.norm(raw[index] - raw[index + 1]) / 10.0
        distances = np.linalg.norm(raw[observed] - raw[index], axis=1)
        contacts[index] = float(np.count_nonzero((distances > 0) & (distances <= 8.0))) / 32.0
    position = np.linspace(0.0, 1.0, length, dtype=np.float32) if length > 1 else np.zeros(1, dtype=np.float32)
    features = np.column_stack([position, xyz, left, right, contacts, bnorm]).astype(np.float32)
    neighbor = left + right
    return features, observed, neighbor


def _mapping_positions(mapping: pd.DataFrame, length: int, column: str) -> list[str | None]:
    values = mapping.set_index("canonical_position")[column].to_dict()
    positions: list[str | None] = []
    for index in range(1, length + 1):
        value = values.get(index)
        if value is None or pd.isna(value) or str(value).lower() in {"null", "nan", "none"}:
            positions.append(None)
        else:
            positions.append(str(value))
    return positions


def load_v1_examples(
    resolution: CohortResolution,
    mappings_path: Path,
    residue_descriptors_path: Path,
    structure_root: Path,
    cases_path: Path | None = None,
) -> tuple[ProteinStateExample, ...]:
    """Load pre-model geometry and exact sequence targets for evaluable members."""

    descriptors = pd.read_parquet(residue_descriptors_path)
    required_descriptor = {"protein_id", "pair_id", "canonical_position", "local_pairwise_distance_change", "aligned_ca_displacement", "contact_turnover_fraction"}
    if not required_descriptor.issubset(descriptors.columns):
        raise V1CohortError("residue descriptor table lacks V1 geometry fields")
    if cases_path is not None:
        return _load_examples_from_cases(resolution, descriptors, Path(cases_path))
    mappings = pd.read_parquet(mappings_path)
    required_mapping = {"protein_id", "pair_id", "canonical_position", "pdb_residue_number_apo", "pdb_residue_number_holo"}
    if not required_mapping.issubset(mappings.columns):
        raise V1CohortError("residue mapping table lacks V1 canonical-position fields")
    examples: list[ProteinStateExample] = []
    for row in resolution.evaluable.itertuples(index=False):
        sequence = str(row.sequence)
        length = len(sequence)
        mapping = mappings.loc[mappings["pair_id"].eq(row.pair_id)].drop_duplicates("canonical_position", keep="first")
        if mapping.empty:
            raise V1CohortError(f"missing residue mapping for {row.protein_id}")
        descriptor = descriptors.loc[descriptors["pair_id"].eq(row.pair_id)].set_index("canonical_position")
        state_features: list[np.ndarray] = []
        state_masks: list[np.ndarray] = []
        neighbors: list[np.ndarray] = []
        for state, path_column, chain_column, mapping_column in (
            ("apo", "apo_mmcif_relative_path", "apo_chain_id", "pdb_residue_number_apo"),
            ("holo", "holo_mmcif_relative_path", "holo_chain_id", "pdb_residue_number_holo"),
        ):
            ca = load_chain_ca_table(Path(structure_root) / str(getattr(row, path_column)), str(getattr(row, chain_column)))
            coordinates, bfactors = _coordinate_array(ca)
            positions = _mapping_positions(mapping, length, mapping_column)
            features, observed, neighbor = _state_features(positions, coordinates, bfactors)
            state_features.append(features)
            state_masks.append(observed)
            neighbors.append(neighbor)
        disagreement = np.zeros((length, 6), dtype=np.float32)
        for index in range(1, length + 1):
            if index in descriptor.index:
                values = descriptor.loc[index]
                disagreement[index - 1, 0] = float(pd.to_numeric(values["local_pairwise_distance_change"], errors="coerce") or 0.0)
                disagreement[index - 1, 2] = float(pd.to_numeric(values["contact_turnover_fraction"], errors="coerce") or 0.0)
                disagreement[index - 1, 3] = float(pd.to_numeric(values["aligned_ca_displacement"], errors="coerce") or 0.0)
        disagreement[:, 1] = np.abs(neighbors[0] - neighbors[1])
        disagreement[:, 4] = (~state_masks[0]).astype(np.float32)
        disagreement[:, 5] = (~state_masks[1]).astype(np.float32)
        targets = np.asarray([_AA_TO_INDEX[aa] for aa in sequence], dtype=np.int64)
        examples.append(
            ProteinStateExample(
                protein_id=str(row.protein_id),
                pair_id=str(row.pair_id),
                split=str(row.split),
                sequence=sequence,
                state_features=np.stack(state_features),
                observability=np.stack(state_masks),
                disagreement=disagreement,
                targets=targets,
            )
        )
    return tuple(examples)


def _read_case_cache(path: Path) -> pd.DataFrame:
    """Read the existing cross-model case cache across NumPy pickle spellings."""

    try:
        return pd.read_pickle(path)
    except ModuleNotFoundError as exc:
        if "numpy._core" not in str(exc):
            raise V1CohortError(f"cannot read V1 case cache: {path}") from exc
        import sys

        import numpy

        sys.modules.setdefault("numpy._core", numpy.core)
        sys.modules.setdefault("numpy._core.numeric", numpy.core.numeric)
        return pd.read_pickle(path)


def _load_examples_from_cases(
    resolution: CohortResolution,
    descriptors: pd.DataFrame,
    cases_path: Path,
) -> tuple[ProteinStateExample, ...]:
    """Use the frozen case cache to avoid reparsing each mmCIF during training."""

    cases = _read_case_cache(cases_path)
    required = {"protein_id", "condition", "canonical_positions", "wt_sequence", "coordinates"}
    if not required.issubset(cases.columns):
        raise V1CohortError("V1 case cache lacks common-position structure fields")
    allowed = set(resolution.evaluable["protein_id"].astype(str))
    cases = cases.loc[cases["protein_id"].astype(str).isin(allowed)]
    by_protein: dict[str, dict[str, pd.DataFrame]] = {}
    for (protein_id, condition), group in cases.groupby(["protein_id", "condition"], sort=False):
        by_protein.setdefault(str(protein_id), {})[str(condition).upper()] = group
    examples: list[ProteinStateExample] = []
    for row in resolution.evaluable.itertuples(index=False):
        states = by_protein.get(str(row.protein_id), {})
        if set(states) != {"APO", "HOLO"}:
            raise V1CohortError(f"case cache lacks matched APO/HOLO states for {row.protein_id}")
        sequence = str(states["APO"]["wt_sequence"].iloc[0])
        if sequence != str(states["HOLO"]["wt_sequence"].iloc[0]):
            raise V1CohortError(f"APO/HOLO WT sequence mismatch for {row.protein_id}")
        length = len(sequence)
        state_features: list[np.ndarray] = []
        state_masks: list[np.ndarray] = []
        neighbors: list[np.ndarray] = []
        for state in ("APO", "HOLO"):
            group = states[state]
            positions = tuple(int(value) for value in group["canonical_positions"].iloc[0])
            coordinate_values = np.asarray(group["coordinates"].iloc[0], dtype=np.float32)
            if coordinate_values.ndim != 3 or coordinate_values.shape[0] != len(positions):
                raise V1CohortError(f"invalid case-cache coordinates for {row.protein_id} {state}")
            ca = coordinate_values[:, 1, :] if coordinate_values.shape[1] >= 2 else coordinate_values[:, 0, :]
            coordinates = {str(position): value for position, value in zip(positions, ca, strict=True) if np.isfinite(value).all()}
            positions_full: list[str | None] = [None] * length
            for position in positions:
                if 1 <= position <= length:
                    positions_full[position - 1] = str(position)
            features, observed, neighbor = _state_features(positions_full, coordinates, {})
            state_features.append(features)
            state_masks.append(observed)
            neighbors.append(neighbor)
        descriptor = descriptors.loc[descriptors["pair_id"].eq(row.pair_id)].set_index("canonical_position")
        disagreement = np.zeros((length, 6), dtype=np.float32)
        for index in range(1, length + 1):
            if index in descriptor.index:
                values = descriptor.loc[index]
                for target, source in ((0, "local_pairwise_distance_change"), (2, "contact_turnover_fraction"), (3, "aligned_ca_displacement")):
                    numeric = pd.to_numeric(values[source], errors="coerce")
                    disagreement[index - 1, target] = float(numeric) if pd.notna(numeric) else 0.0
        disagreement[:, 1] = np.abs(neighbors[0] - neighbors[1])
        disagreement[:, 4] = (~state_masks[0]).astype(np.float32)
        disagreement[:, 5] = (~state_masks[1]).astype(np.float32)
        try:
            targets = np.asarray([_AA_TO_INDEX[aa] for aa in sequence], dtype=np.int64)
        except KeyError as exc:
            raise V1CohortError(f"nonstandard canonical sequence remains in V1 examples: {row.protein_id}") from exc
        examples.append(
            ProteinStateExample(
                protein_id=str(row.protein_id),
                pair_id=str(row.pair_id),
                split=str(row.split),
                sequence=sequence,
                state_features=np.stack(state_features),
                observability=np.stack(state_masks),
                disagreement=disagreement,
                targets=targets,
            )
        )
    return tuple(examples)
