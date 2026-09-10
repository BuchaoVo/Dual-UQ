"""Read-only projection of frozen StructCal v1 pairs into model cases."""

from __future__ import annotations

import re
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from dual_uq.structure_io import load_atom_site_table, residue_name_to_one_letter

from .structcal_local_response import STANDARD_AMINO_ACIDS, StructCalLocalResponseError


class StructCalLocalResponseCasesError(StructCalLocalResponseError):
    """Raised for a frozen release schema, mapping, or identity violation."""


class StructCalLocalResponsePairDataFailure(Exception):
    """Expected pair-local source-data failure that can become UNRESOLVED."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


COHORTS = ("track_i", "track_ii", "controlled")
SPLITS = ("TRAIN", "VALIDATION", "LOCKED_TEST")
_RESIDUE_ID = re.compile(r"^(?P<chain>[^:]+):(?P<number>-?\d+)(?P<insertion>[A-Za-z]?)$")
CASE_COLUMNS = (
    "protein_id",
    "pair_id",
    "condition",
    "condition_label",
    "canonical_position",
    "canonical_positions",
    "wt_sequence",
    "wt_sequence_projection",
    "coordinates",
    "structure_sha256",
    "identity_cluster_id",
    "split",
    "track_or_diagnostic",
    "state_family",
    "n_canonical_positions",
)
EXCLUSION_COLUMNS = ("pair_id", "protein_id", "split", "status", "reason", "detail")


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise StructCalLocalResponseCasesError(f"{label} missing required columns: {missing}")


def _optional_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _release_root(project_root: Path) -> Path:
    root = Path(project_root) / "artifacts/releases/structcal_v1"
    if not root.is_dir():
        raise StructCalLocalResponseCasesError(f"StructCal v1 release root is missing: {root}")
    return root


def load_structcal_local_cohort(
    project_root: Path,
    *,
    cohort: Literal["track_i", "track_ii", "controlled"],
    split: Literal["TRAIN", "VALIDATION", "LOCKED_TEST"],
) -> pd.DataFrame:
    """Select a frozen local-response cohort without changing membership."""

    if cohort not in COHORTS or split not in SPLITS:
        raise StructCalLocalResponseCasesError("unknown StructCal cohort or split")
    release = _release_root(Path(project_root))
    pairs = pd.read_parquet(release / "core/condition_pairs.parquet")
    structures = pd.read_parquet(release / "core/structures.parquet")
    splits = pd.read_parquet(release / "core/splits.parquet")
    _require_columns(
        pairs,
        {
            "pair_id",
            "protein_id",
            "arm",
            "condition_1_structure_id",
            "condition_2_structure_id",
            "condition_1_label",
            "condition_2_label",
        },
        "condition_pairs",
    )
    _require_columns(structures, {"structure_id", "protein_id", "chain_id", "source_file_ref"}, "structures")
    _require_columns(splits, {"protein_id", "identity_cluster_id", "split"}, "splits")
    if pairs["pair_id"].duplicated().any() or splits["protein_id"].duplicated().any():
        raise StructCalLocalResponseCasesError("frozen release keys are not unique")
    split_lookup = splits.set_index("protein_id")
    if cohort == "track_i":
        selected = pd.read_parquet(release / "tracks/track_i_invariance.parquet")
        track_name = "TRACK_I"
    elif cohort == "track_ii":
        selected = pd.read_parquet(release / "tracks/track_ii_sensitivity.parquet")
        track_name = "TRACK_II"
    else:
        descriptors = pd.read_parquet(release / "annotations/perturbation_descriptors.parquet")
        _require_columns(descriptors, {"pair_id", "perturbation_family", "requested_dose"}, "perturbation_descriptors")
        selected = pairs.loc[pairs["arm"].astype(str).eq("controlled_perturbation")].copy()
        selected = selected.merge(descriptors, on="pair_id", how="inner", validate="one_to_one")
        track_name = "CONTROLLED_DIAGNOSTIC"
    _require_columns(selected, {"pair_id", "protein_id"}, f"{cohort} membership")
    if selected["pair_id"].duplicated().any():
        raise StructCalLocalResponseCasesError(f"{cohort} membership contains duplicate pairs")
    pair_lookup = pairs.set_index("pair_id")
    missing_pairs = sorted(set(selected["pair_id"].astype(str)).difference(pair_lookup.index.astype(str)))
    if missing_pairs:
        raise StructCalLocalResponseCasesError(f"{cohort} references pairs outside Core: {missing_pairs[:5]}")
    rows: list[dict[str, Any]] = []
    for selected_row in selected.sort_values("pair_id", kind="mergesort").itertuples(index=False):
        pair_id = str(selected_row.pair_id)
        pair = pair_lookup.loc[pair_id]
        protein_id = str(pair.protein_id)
        if protein_id not in split_lookup.index:
            raise StructCalLocalResponseCasesError(f"pair protein is absent from frozen splits: {protein_id}")
        release_split = str(split_lookup.loc[protein_id, "split"])
        selected_split = _optional_text(getattr(selected_row, "split", None))
        if selected_split is not None and selected_split != release_split:
            raise StructCalLocalResponseCasesError(f"frozen split inheritance mismatch: {pair.pair_id}")
        if release_split != split:
            continue
        state_family = _optional_text(pair.get("state_family"))
        if state_family is None and str(pair.arm) == "ligand_state":
            state_family = "APO_HOLO"
        record = {
            "pair_id": pair_id,
            "protein_id": protein_id,
            "identity_cluster_id": str(split_lookup.loc[protein_id, "identity_cluster_id"]),
            "split": release_split,
            "track_or_diagnostic": track_name,
            "arm": str(pair.arm),
            "condition_1_structure_id": str(pair.condition_1_structure_id),
            "condition_2_structure_id": str(pair.condition_2_structure_id),
            "condition_1_label": str(pair.condition_1_label),
            "condition_2_label": str(pair.condition_2_label),
            "state_family": state_family,
        }
        for field in ("sequence_comparable", "construct_comparable", "assembly_comparable", "mapping_comparable"):
            if field in pair.index:
                record[field] = bool(pair[field])
        for field in ("perturbation_family", "requested_dose", "requested_dose_unit", "realized_ca_rmsd", "realized_pairwise_distance_change", "realized_contact_change"):
            if hasattr(selected_row, field):
                record[field] = getattr(selected_row, field)
        rows.append(record)
    return pd.DataFrame(rows).sort_values("pair_id", kind="mergesort").reset_index(drop=True)


def _resolve_asset(project_root: Path, reference: Any) -> Path:
    text = _optional_text(reference)
    if text is None:
        raise StructCalLocalResponsePairDataFailure("STRUCTURAL_ASSET_MISSING", "empty source_file_ref")
    path = Path(text)
    resolved = path if path.is_absolute() else Path(project_root) / path
    if not resolved.is_file():
        raise StructCalLocalResponsePairDataFailure("STRUCTURAL_ASSET_MISSING", f"structure asset is missing: {text}")
    return resolved


@lru_cache(maxsize=512)
def _load_backbone(path: str, chain_id: str, atom_names: tuple[str, ...]) -> dict[tuple[int, str], dict[str, tuple[str, np.ndarray]]]:
    try:
        atoms = load_atom_site_table(Path(path))
    except (OSError, ValueError) as exc:
        raise StructCalLocalResponsePairDataFailure("STRUCTURAL_PARSE_FAILURE", f"cannot parse structure asset: {path}") from exc
    if atoms.empty:
        raise StructCalLocalResponsePairDataFailure("STRUCTURAL_PARSE_FAILURE", f"structure has no atoms: {path}")
    atoms = atoms.loc[atoms["model_number"].eq(int(atoms["model_number"].min()))]
    atoms = atoms.loc[atoms["auth_asym_id"].astype(str).eq(str(chain_id))].copy()
    if atoms.empty:
        raise StructCalLocalResponsePairDataFailure("STRUCTURAL_PARSE_FAILURE", f"chain {chain_id} is absent: {path}")
    atoms = atoms.loc[atoms["atom_name"].astype(str).str.upper().isin(atom_names)].copy()
    atoms["_alt_rank"] = atoms["alt_id"].map(lambda value: 0 if str(value).strip().upper() in {"", ".", "?", "A"} else 1)
    atoms = atoms.sort_values(["_alt_rank", "occupancy"], ascending=[True, False], kind="mergesort")
    result: dict[tuple[int, str], dict[str, tuple[str, np.ndarray]]] = {}
    for row in atoms.itertuples(index=False):
        if pd.isna(row.auth_seq_id):
            continue
        key = (int(row.auth_seq_id), str(row.insertion_code or "").strip().upper())
        name = str(row.atom_name).strip().upper()
        residue = result.setdefault(key, {})
        if name in residue:
            continue
        coordinate = np.asarray([row.x, row.y, row.z], dtype=np.float32)
        if not np.isfinite(coordinate).all():
            raise StructCalLocalResponsePairDataFailure("NONFINITE_COORDINATE", f"nonfinite coordinate in {path}")
        residue[name] = (residue_name_to_one_letter(str(row.residue_name)), coordinate)
    return result


def _parse_residue_id(value: Any) -> tuple[str, tuple[int, str]] | None:
    text = _optional_text(value)
    if text is None or text.lower() in {"null", "none", "nan"} or text.endswith(":null"):
        return None
    match = _RESIDUE_ID.fullmatch(text)
    if match is None:
        raise StructCalLocalResponseCasesError(f"malformed persisted residue identity: {value}")
    return match.group("chain"), (int(match.group("number")), match.group("insertion").upper())


def _case_rows(pair: dict[str, Any], condition: str, label: str, positions: tuple[int, ...], sequence: str, coordinates: np.ndarray, total_positions: int) -> list[dict[str, Any]]:
    projection = "".join(sequence[position - 1] for position in positions)
    gap = any(right != left + 1 for left, right in pairwise(positions))
    return [
        {
            **pair,
            "condition": condition,
            "condition_label": label,
            "canonical_position": int(position),
            "canonical_positions": positions,
            "wt_sequence": sequence,
            "wt_sequence_projection": projection,
            "coordinates": coordinates.tolist() if index == 0 else None,
            "structure_sha256": None,
            "true_uniprot_gap": gap,
            "n_canonical_positions": total_positions,
        }
        for index, position in enumerate(positions)
    ]


def build_structcal_local_cases(
    project_root: Path,
    cohort: pd.DataFrame,
    *,
    atom_names: tuple[str, ...],
    allow_controlled_pair_consensus: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Project frozen mappings/assets to complete model input rows."""

    if not atom_names or any(atom not in {"N", "CA", "C", "O"} for atom in atom_names):
        raise StructCalLocalResponseCasesError("atom_names must be a non-empty backbone subset")
    _require_columns(
        cohort,
        {
            "pair_id", "protein_id", "identity_cluster_id", "split", "track_or_diagnostic",
            "condition_1_structure_id", "condition_2_structure_id", "condition_1_label", "condition_2_label",
        },
        "local-response cohort",
    )
    if allow_controlled_pair_consensus and not cohort["track_or_diagnostic"].astype(str).eq(
        "CONTROLLED_DIAGNOSTIC"
    ).all():
        raise StructCalLocalResponseCasesError(
            "pair-consensus identity is restricted to controlled diagnostics"
        )
    release = _release_root(Path(project_root))
    proteins = pd.read_parquet(release / "core/proteins.parquet").set_index("protein_id")
    structures = pd.read_parquet(release / "core/structures.parquet").set_index("structure_id")
    mappings = pd.read_parquet(release / "core/residue_mappings.parquet")
    _require_columns(proteins.reset_index(), {"protein_id", "canonical_sequence"}, "proteins")
    _require_columns(structures.reset_index(), {"structure_id", "protein_id", "chain_id", "source_file_ref"}, "structures")
    _require_columns(mappings, {"pair_id", "protein_id", "canonical_position", "canonical_aa", "condition_1_residue_id", "condition_2_residue_id", "condition_1_aa", "condition_2_aa", "condition_1_mapped", "condition_2_mapped", "common_mapped", "common_coordinate_visible"}, "residue_mappings")
    case_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for cohort_row in cohort.sort_values("pair_id", kind="mergesort").to_dict("records"):
        pair_id = str(cohort_row["pair_id"])
        protein_id = str(cohort_row["protein_id"])
        try:
            if "sequence_comparable" in cohort_row and not bool(cohort_row["sequence_comparable"]):
                raise StructCalLocalResponsePairDataFailure(
                    "SEQUENCE_NOT_COMPARABLE",
                    "frozen pair is not sequence-comparable",
                )
            if protein_id not in proteins.index:
                raise StructCalLocalResponseCasesError(f"protein foreign key is absent: {protein_id}")
            sequence = str(proteins.loc[protein_id, "canonical_sequence"]).strip().upper()
            if not sequence or any(aa not in STANDARD_AMINO_ACIDS for aa in sequence):
                raise StructCalLocalResponseCasesError(f"canonical sequence is invalid: {protein_id}")
            structure_ids = (str(cohort_row["condition_1_structure_id"]), str(cohort_row["condition_2_structure_id"]))
            if any(structure_id not in structures.index for structure_id in structure_ids):
                raise StructCalLocalResponseCasesError(f"structure foreign key is absent: {pair_id}")
            structure_rows = [structures.loc[structure_id] for structure_id in structure_ids]
            if any(str(row["protein_id"]) != protein_id for row in structure_rows):
                raise StructCalLocalResponseCasesError(f"structure protein identity mismatch: {pair_id}")
            mapping = mappings.loc[mappings["pair_id"].astype(str).eq(pair_id)].copy()
            mapping = mapping.loc[mapping["common_coordinate_visible"].astype(bool)]
            if mapping.empty:
                raise StructCalLocalResponsePairDataFailure("MAPPING_UNAVAILABLE", "no persisted common visible mapping")
            if mapping["canonical_position"].duplicated().any():
                raise StructCalLocalResponseCasesError(f"mapping canonical positions are not unique: {pair_id}")
            mapping["canonical_position"] = pd.to_numeric(mapping["canonical_position"], errors="coerce")
            if mapping["canonical_position"].isna().any():
                raise StructCalLocalResponseCasesError(f"mapping canonical position is invalid: {pair_id}")
            mapping["canonical_position"] = mapping["canonical_position"].astype(int)
            total_positions = len(mapping)
            paths = tuple(_resolve_asset(Path(project_root), row["source_file_ref"]) for row in structure_rows)
            backbones = tuple(_load_backbone(str(path.resolve()), str(row["chain_id"]), atom_names) for path, row in zip(paths, structure_rows, strict=True))
            projected: list[dict[int, np.ndarray]] = [{}, {}]
            for record in mapping.sort_values("canonical_position", kind="mergesort").itertuples(index=False):
                position = int(record.canonical_position)
                if not 1 <= position <= len(sequence) or str(record.canonical_aa).upper() != sequence[position - 1]:
                    raise StructCalLocalResponseCasesError(f"canonical sequence disagrees with mapping: {pair_id}/{position}")
                condition_aas = tuple(
                    str(getattr(record, f"condition_{side}_aa")).strip().upper()
                    for side in (1, 2)
                )
                canonical_mismatch = any(aa != sequence[position - 1] for aa in condition_aas)
                if canonical_mismatch and (
                    not allow_controlled_pair_consensus
                    or condition_aas[0] != condition_aas[1]
                ):
                    raise StructCalLocalResponsePairDataFailure(
                        "RESIDUE_IDENTITY_MISMATCH",
                        f"persisted condition residue identity disagrees with canonical sequence: {pair_id}/{position}",
                    )
                coordinates: list[np.ndarray] = []
                usable = True
                for side, (structure_row, backbone) in enumerate(zip(structure_rows, backbones, strict=True), start=1):
                    residue_id = _parse_residue_id(getattr(record, f"condition_{side}_residue_id"))
                    if residue_id is None:
                        usable = False
                        break
                    mapped_chain, key = residue_id
                    if mapped_chain != str(structure_row["chain_id"]):
                        raise StructCalLocalResponseCasesError(f"persisted residue chain disagrees with structure: {pair_id}/{position}")
                    expected_aa = str(getattr(record, f"condition_{side}_aa")).strip().upper()
                    residue = backbone.get(key)
                    if residue is None or any(atom not in residue for atom in atom_names):
                        usable = False
                        break
                    observed_aa = residue["CA"][0].upper()
                    if observed_aa == "X":
                        usable = False
                        break
                    if observed_aa != expected_aa:
                        raise StructCalLocalResponsePairDataFailure(
                            "STRUCTURE_IDENTITY_MISMATCH",
                            f"structure residue identity disagrees with persisted mapping: {pair_id}/{position}",
                        )
                    coordinates.append(np.asarray([residue[atom][1] for atom in atom_names], dtype=np.float32))
                if usable:
                    projected[0][position] = coordinates[0]
                    projected[1][position] = coordinates[1]
            positions = tuple(sorted(set(projected[0]).intersection(projected[1])))
            if len(positions) < 3:
                raise StructCalLocalResponsePairDataFailure("INSUFFICIENT_COMMON_COORDINATES", f"fewer than three complete common positions: {pair_id}")
            pair_metadata = {
                **cohort_row,
                "protein_id": protein_id,
                "pair_id": pair_id,
                "identity_cluster_id": str(cohort_row["identity_cluster_id"]),
                "split": str(cohort_row["split"]),
                "state_family": _optional_text(cohort_row.get("state_family")),
            }
            for side, condition in enumerate(("CONDITION_1", "CONDITION_2")):
                matrix = np.asarray([projected[side][position] for position in positions], dtype=np.float32)
                case_rows.extend(_case_rows(pair_metadata, condition, str(cohort_row[f"condition_{side + 1}_label"]), positions, sequence, matrix, total_positions))
        except StructCalLocalResponsePairDataFailure as failure:
            exclusions.append({"pair_id": pair_id, "protein_id": protein_id, "split": str(cohort_row["split"]), "status": "UNRESOLVED", "reason": failure.reason, "detail": failure.detail})
    cases = pd.DataFrame(case_rows, columns=CASE_COLUMNS)
    exclusions_frame = pd.DataFrame(exclusions, columns=EXCLUSION_COLUMNS)
    return cases, exclusions_frame
