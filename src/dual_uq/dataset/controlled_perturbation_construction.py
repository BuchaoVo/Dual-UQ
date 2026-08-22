"""Outcome-blind selection and persistence of controlled-perturbation parents."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFIO, MMCIFParser, PDBParser
from Bio.PDB.PDBExceptions import PDBException
from Bio.PDB.Polypeptide import is_aa

from dual_uq.benchmark.ids import canonical_id
from dual_uq.construction.annotations import annotate_aligned_pair
from dual_uq.construction.controlled_perturbation import (
    SUPPORTED_PERTURBATION_DOSES,
    SUPPORTED_PERTURBATION_FAMILIES,
    ControlledPerturbationSpec,
    CoordinateStructure,
    apply_perturbation,
    inherit_mapping,
)


@dataclass(frozen=True, slots=True)
class ParentSelectionConfig:
    """Bounds for a deterministic parent panel."""

    target_count: int = 200
    minimum_count: int = 150
    maximum_count: int = 250
    root_seed: int = 20260819


_PROTEIN_COLUMNS = {"protein_id", "canonical_length", "canonical_sequence_status"}
_STRUCTURE_COLUMNS = {
    "structure_id",
    "protein_id",
    "source_structure_id",
    "source_file_ref",
    "experimental_method",
    "mapping_coverage",
    "coordinate_coverage",
}
_CLUSTER_COLUMNS = {"protein_id", "identity_cluster_id", "cluster_status", "split"}
_SUPPORTED_CANONICAL_SEQUENCE_STATUSES = {"STANDARD_20AA", "NONSTANDARD"}
_PANEL_COLUMNS = [
    "parent_structure_id",
    "protein_id",
    "source_structure_id",
    "source_file_ref",
    "canonical_length",
    "length_bin",
    "identity_cluster_id",
    "cluster_status",
    "split",
    "mapping_coverage",
    "coordinate_coverage",
    "experimental_method",
    "selection_rank",
]
_PARENT_ELIGIBILITY_COLUMNS = [
    "parent_structure_id",
    "protein_id",
    "source_structure_id",
    "inheritance_ready",
    "inheritance_reason",
    "inheritance_pair_id",
    "canonical_length",
    "canonical_position_count",
    "mapped_residue_count",
    "identity_complete_count",
    "coordinate_visible_count",
    "mapping_coverage",
    "coordinate_coverage",
    "identity_consistency",
]

_PILOT_PANEL_COLUMNS = {
    "parent_structure_id",
    "protein_id",
    "source_structure_id",
    "source_file_ref",
}
_PILOT_INSTANCE_COLUMNS = [
    "pilot_instance_id",
    "pair_id",
    "protein_id",
    "parent_structure_id",
    "source_structure_id",
    "perturbation_family",
    "requested_dose",
    "requested_dose_unit",
    "realization",
    "random_seed",
    "perturbation_method",
    "perturbation_version",
    "realized_ca_rmsd",
    "realized_pairwise_distance_change",
    "realized_contact_change",
    "reference_source_file_ref",
    "perturbed_source_file_ref",
    "status",
    "reason",
]
_PAIR_DESCRIPTOR_COLUMNS = [
    "pair_id",
    "aligned_ca_rmsd",
    "median_residue_displacement",
    "upper_tail_residue_displacement",
    "global_contact_change",
    "descriptor_method",
    "descriptor_version",
]
_RESIDUE_DESCRIPTOR_COLUMNS = [
    "pair_id",
    "canonical_position",
    "ca_displacement",
    "local_pairwise_distance_change",
    "neighborhood_geometry_change",
    "contact_gain",
    "contact_loss",
    "descriptor_method",
    "descriptor_version",
]


class _ExpectedPilotDataFailure(Exception):
    """An instance-local source-data or structure failure with a public reason."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class InheritedParentMappingDataFailure(Exception):
    """Persisted parent mapping data are unavailable for one pilot pair."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class _CoreInputs:
    proteins: pd.DataFrame
    structures: pd.DataFrame
    condition_pairs: pd.DataFrame
    residue_mappings_path: Path


@dataclass(frozen=True, slots=True)
class _ParsedChain:
    coordinate_structure: CoordinateStructure
    atoms_by_residue: tuple[tuple[object, ...], ...]
    residue_indices: dict[str, int]


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _validate_config(config: ParentSelectionConfig) -> None:
    if not 0 < config.minimum_count <= config.target_count <= config.maximum_count:
        raise ValueError("parent selection counts must satisfy 0 < minimum <= target <= maximum")


def _length_bin(lengths: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(lengths, errors="coerce")
    return pd.Series(
        np.select(
            [numeric.lt(300), numeric.lt(600)],
            ["short", "medium"],
            default="long",
        ),
        index=lengths.index,
        dtype="object",
    )


def _method_rank(methods: pd.Series) -> pd.Series:
    normalized = methods.fillna("").astype(str).str.upper().str.strip()
    return pd.Series(
        np.select(
            [
                normalized.str.contains("X-RAY", regex=False),
                normalized.isin({"EM", "ELECTRON MICROSCOPY", "CRYO-EM", "CRYO EM"}),
                normalized.str.contains("NMR", regex=False),
            ],
            [0, 1, 2],
            default=3,
        ),
        index=methods.index,
        dtype="int64",
    )


def _prepared_candidates(
    proteins: pd.DataFrame,
    structures: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(proteins, _PROTEIN_COLUMNS, "proteins")
    _require_columns(structures, _STRUCTURE_COLUMNS, "structures")
    _require_columns(cluster_assignments, _CLUSTER_COLUMNS, "cluster assignments")
    if proteins["protein_id"].astype(str).duplicated().any():
        raise ValueError("proteins contains duplicate protein_id values")
    if cluster_assignments["protein_id"].astype(str).duplicated().any():
        raise ValueError("cluster assignments contains duplicate protein_id values")

    sequence_status = proteins["canonical_sequence_status"].astype("string")
    unsupported_status = sequence_status.isna() | ~sequence_status.isin(
        _SUPPORTED_CANONICAL_SEQUENCE_STATUSES
    )
    if unsupported_status.any():
        invalid_values = sorted(sequence_status.loc[unsupported_status].dropna().unique().tolist())
        raise ValueError(f"unsupported canonical_sequence_status values: {invalid_values}")

    protein_columns = ["protein_id", "canonical_length", "canonical_sequence_status"]
    protein_rows = proteins.loc[:, protein_columns].copy()
    protein_rows["protein_id"] = protein_rows["protein_id"].astype(str)
    cluster_columns = ["protein_id", "identity_cluster_id", "cluster_status", "split"]
    clusters = cluster_assignments.loc[:, cluster_columns].copy()
    clusters["protein_id"] = clusters["protein_id"].astype(str)
    work = structures.copy()
    work["protein_id"] = work["protein_id"].astype(str)
    work = work.merge(protein_rows, on="protein_id", how="inner", validate="many_to_one")
    work = work.merge(clusters, on="protein_id", how="inner", validate="many_to_one")
    work = work.loc[work["canonical_sequence_status"].eq("STANDARD_20AA")].copy()
    work["mapping_coverage"] = pd.to_numeric(work["mapping_coverage"], errors="coerce")
    work["coordinate_coverage"] = pd.to_numeric(work["coordinate_coverage"], errors="coerce")
    work["canonical_length"] = pd.to_numeric(work["canonical_length"], errors="coerce")
    work = work.loc[
        work["mapping_coverage"].gt(0)
        & work["coordinate_coverage"].gt(0)
        & work["canonical_length"].gt(0)
        & work["source_structure_id"].notna()
        & work["source_file_ref"].notna()
    ].copy()
    if work.empty:
        return work
    work["parent_structure_id"] = work["structure_id"].astype(str)
    work["source_structure_id"] = work["source_structure_id"].astype(str)
    work["source_file_ref"] = work["source_file_ref"].astype(str)
    work["experimental_method"] = work["experimental_method"].fillna("UNRESOLVED").astype(str)
    work["identity_cluster_id"] = work["identity_cluster_id"].astype(str)
    work["cluster_status"] = work["cluster_status"].astype("string")
    work["split"] = work["split"].astype("string")
    work["length_bin"] = _length_bin(work["canonical_length"])
    work["_method_rank"] = _method_rank(work["experimental_method"])
    resolution = (
        work["resolution"]
        if "resolution" in work
        else pd.Series(np.inf, index=work.index)
    )
    work["_resolution_rank"] = pd.to_numeric(resolution, errors="coerce").fillna(np.inf)
    return work.sort_values(
        [
            "mapping_coverage",
            "coordinate_coverage",
            "_method_rank",
            "_resolution_rank",
            "canonical_length",
            "identity_cluster_id",
            "parent_structure_id",
            "source_structure_id",
        ],
        ascending=[False, False, True, True, True, True, True, True],
        kind="mergesort",
    )


def select_parent_panel(
    *,
    proteins: pd.DataFrame,
    structures: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
    config: ParentSelectionConfig,
) -> pd.DataFrame:
    """Select bounded parents using only canonical, mapping, and provenance facts."""

    _validate_config(config)
    candidates = _prepared_candidates(proteins, structures, cluster_assignments)
    if candidates.empty:
        raise ValueError("parent selection has no eligible candidates")

    # Quality is the primary rank.  Taking the first candidate in each identity
    # cluster preserves a broad cluster panel without using any downstream result.
    per_cluster = candidates.drop_duplicates("identity_cluster_id", keep="first")
    ordered = per_cluster.sort_values(
        [
            "mapping_coverage",
            "coordinate_coverage",
            "_method_rank",
            "_resolution_rank",
            "length_bin",
            "identity_cluster_id",
            "parent_structure_id",
        ],
        ascending=[False, False, True, True, True, True, True],
        kind="mergesort",
    )
    leading_length_bin = str(ordered.iloc[0]["length_bin"])
    length_bin_order = [
        leading_length_bin,
        *sorted(set(ordered["length_bin"].astype(str)) - {leading_length_bin}),
    ]
    remaining_indices = {
        length_bin: ordered.index[ordered["length_bin"].eq(length_bin)].tolist()
        for length_bin in length_bin_order
    }
    selected_indices: list[int] = []
    while len(selected_indices) < config.target_count:
        selected_this_round = False
        for length_bin in length_bin_order:
            if remaining_indices[length_bin]:
                selected_indices.append(remaining_indices[length_bin].pop(0))
                selected_this_round = True
                if len(selected_indices) == config.target_count:
                    break
        if not selected_this_round:
            break
    selected = ordered.loc[selected_indices].copy()
    if len(selected) < config.minimum_count:
        raise ValueError(
            "parent selection has insufficient eligible identity clusters: "
            f"eligible={len(selected)} minimum={config.minimum_count}"
        )
    selected["selection_rank"] = range(1, len(selected) + 1)
    selected["canonical_length"] = selected["canonical_length"].astype(int)
    for column in _PANEL_COLUMNS:
        if column not in {"canonical_length", "mapping_coverage", "coordinate_coverage", "selection_rank"}:
            selected[column] = selected[column].astype("string")
    return selected.loc[:, _PANEL_COLUMNS].reset_index(drop=True)


def write_parent_panel(panel: pd.DataFrame, destination: Path) -> None:
    """Persist a panel atomically as Parquet."""

    _require_columns(panel, set(_PANEL_COLUMNS), "parent panel")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    panel.loc[:, _PANEL_COLUMNS].to_parquet(temporary, index=False)
    temporary.replace(destination)


def write_parent_eligibility(eligibility: pd.DataFrame, destination: Path) -> None:
    """Persist construction-only parent eligibility without changing the panel."""

    _require_columns(
        eligibility,
        set(_PARENT_ELIGIBILITY_COLUMNS),
        "parent eligibility",
    )
    if eligibility["parent_structure_id"].astype(str).duplicated().any():
        raise ValueError("parent eligibility contains duplicate parent_structure_id values")
    if not eligibility["inheritance_ready"].map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).all():
        raise ValueError("parent eligibility inheritance_ready must be boolean")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    eligibility.loc[:, _PARENT_ELIGIBILITY_COLUMNS].to_parquet(temporary, index=False)
    temporary.replace(destination)


def load_parent_panel(path: Path) -> pd.DataFrame:
    """Load a persisted parent panel after checking its required fields."""

    panel = pd.read_parquet(path)
    _require_columns(panel, set(_PANEL_COLUMNS), "parent panel")
    return panel.loc[:, _PANEL_COLUMNS]


def _parent_candidate_mapping(
    *,
    core: _CoreInputs,
    pair_id: str,
    side: str,
) -> pd.DataFrame:
    fields = [
        "canonical_position",
        "canonical_aa",
        f"{side}_residue_id",
        f"{side}_aa",
        f"{side}_mapped",
        f"{side}_coordinate_visible",
    ]
    source = pd.read_parquet(
        core.residue_mappings_path,
        filters=[("pair_id", "==", pair_id)],
        columns=["pair_id", *fields],
    )
    _require_columns(source, {"pair_id", *fields}, "Core residue mappings")
    return source.rename(
        columns={
            f"{side}_residue_id": "residue_id",
            f"{side}_aa": "aa",
            f"{side}_mapped": "mapped",
            f"{side}_coordinate_visible": "coordinate_visible",
        }
    )


def _parent_mapping_metrics(
    frame: pd.DataFrame,
    *,
    canonical_length: int,
) -> dict[str, object]:
    positions = pd.to_numeric(frame["canonical_position"], errors="coerce")
    if positions.isna().any() or positions.duplicated().any():
        return {
            "valid": False,
            "reason": "MAPPING_SCHEMA_INVALID",
            "position_count": 0,
            "mapped_count": 0,
            "identity_count": 0,
            "visible_count": 0,
            "mapping_coverage": 0.0,
            "coordinate_coverage": 0.0,
        }
    mapped = frame["mapped"]
    visible = frame["coordinate_visible"]
    if mapped.isna().any() or not mapped.map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).all():
        return {
            "valid": False,
            "reason": "MAPPING_SCHEMA_INVALID",
            "position_count": int(positions.nunique()),
            "mapped_count": 0,
            "identity_count": 0,
            "visible_count": 0,
            "mapping_coverage": float(positions.nunique()) / canonical_length,
            "coordinate_coverage": 0.0,
        }
    if visible.isna().any() or not visible.map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).all():
        return {
            "valid": False,
            "reason": "MAPPING_SCHEMA_INVALID",
            "position_count": int(positions.nunique()),
            "mapped_count": int(mapped.astype(bool).sum()),
            "identity_count": 0,
            "visible_count": 0,
            "mapping_coverage": float(positions.nunique()) / canonical_length,
            "coordinate_coverage": 0.0,
        }
    mapped_bool = mapped.astype(bool)
    visible_bool = visible.astype(bool)
    if (visible_bool & ~mapped_bool).any():
        reason = "MAPPING_SEMANTICS_INVALID"
    elif mapped_bool.any() and (
        frame.loc[mapped_bool, ["residue_id", "aa"]].isna().any().any()
        or any(
            not str(value).strip()
            for value in frame.loc[mapped_bool, ["residue_id", "aa"]].to_numpy().ravel()
        )
    ):
        reason = "MAPPED_RESIDUE_IDENTITY_INCOMPLETE"
    else:
        reason = None
    identity_complete = mapped_bool & frame["residue_id"].notna() & frame["aa"].notna()
    identity_complete &= frame["residue_id"].astype(str).str.strip().ne("")
    identity_complete &= frame["aa"].astype(str).str.strip().ne("")
    position_count = int(positions.nunique())
    mapped_count = int(mapped_bool.sum())
    identity_count = int(identity_complete.sum())
    visible_count = int(visible_bool.sum())
    full_positions = set(positions.astype(int)) == set(range(1, canonical_length + 1))
    if reason is None and not full_positions:
        reason = "CANONICAL_MAPPING_COVERAGE_INCOMPLETE"
    if reason is None and identity_count != canonical_length:
        reason = "MAPPED_RESIDUE_IDENTITY_INCOMPLETE"
    return {
        "valid": reason is None,
        "reason": reason,
        "position_count": position_count,
        "mapped_count": mapped_count,
        "identity_count": identity_count,
        "visible_count": visible_count,
        "mapping_coverage": position_count / canonical_length,
        "coordinate_coverage": visible_count / canonical_length,
    }


def assess_parent_inheritance_eligibility(
    parent_panel: pd.DataFrame,
    core: _CoreInputs,
) -> pd.DataFrame:
    """Assess construction-only inheritance readiness from persisted Core facts.

    This function does not alter the scientific parent panel.  It reads only
    persisted Core structures, canonical sequences, condition pairs, and
    residue mappings.  It never fills missing identity from canonical fields.
    """

    _require_columns(parent_panel, set(_PANEL_COLUMNS), "parent panel")
    _require_columns(
        core.proteins,
        {"protein_id", "canonical_sequence"},
        "Core proteins",
    )
    _require_columns(
        core.structures,
        {"structure_id", "protein_id", "source_structure_id"},
        "Core structures",
    )
    _require_columns(
        core.condition_pairs,
        {"pair_id", "condition_1_structure_id", "condition_2_structure_id"},
        "Core condition pairs",
    )
    protein_lengths = (
        core.proteins.assign(protein_id=core.proteins["protein_id"].astype(str))
        .set_index("protein_id")["canonical_sequence"]
        .map(lambda value: len(value) if isinstance(value, str) else 0)
        .to_dict()
    )
    structure_lookup = core.structures.assign(
        structure_id=core.structures["structure_id"].astype(str)
    ).set_index("structure_id")
    pairs = core.condition_pairs.copy()
    pairs["pair_id"] = pairs["pair_id"].astype(str)
    rows: list[dict[str, object]] = []
    for parent in parent_panel.to_dict(orient="records"):
        structure_id = str(parent["parent_structure_id"])
        protein_id = str(parent["protein_id"])
        canonical_length = int(protein_lengths.get(protein_id, 0))
        base = {
            "parent_structure_id": structure_id,
            "protein_id": protein_id,
            "source_structure_id": str(parent["source_structure_id"]),
            "inheritance_ready": False,
            "inheritance_reason": "PARENT_MAPPING_UNAVAILABLE",
            "inheritance_pair_id": None,
            "canonical_length": canonical_length,
            "canonical_position_count": 0,
            "mapped_residue_count": 0,
            "identity_complete_count": 0,
            "coordinate_visible_count": 0,
            "mapping_coverage": 0.0,
            "coordinate_coverage": 0.0,
            "identity_consistency": False,
        }
        if canonical_length <= 0 or structure_id not in structure_lookup.index:
            base["inheritance_reason"] = "CORE_PARENT_IDENTITY_INVALID"
            rows.append(base)
            continue
        candidates: list[tuple[str, str, pd.DataFrame, dict[str, object]]] = []
        for side in ("condition_1", "condition_2"):
            matches = pairs.loc[
                pairs[f"{side}_structure_id"].astype(str).eq(structure_id)
            ]
            for pair_id in matches["pair_id"].tolist():
                frame = _parent_candidate_mapping(
                    core=core, pair_id=pair_id, side=side
                )
                metrics = _parent_mapping_metrics(
                    frame, canonical_length=canonical_length
                )
                candidates.append((pair_id, side, frame, metrics))
        if not candidates:
            rows.append(base)
            continue
        identity_facts: list[pd.DataFrame] = []
        for pair_id, side, frame, metrics in candidates:
            mapped = frame["mapped"].map(
                lambda value: isinstance(value, (bool, np.bool_)) and bool(value)
            )
            facts = frame.loc[mapped, ["canonical_position", "residue_id", "aa"]].copy()
            facts["pair_id"] = pair_id
            identity_facts.append(facts)
        identity_consistency = True
        if identity_facts:
            joined = pd.concat(identity_facts, ignore_index=True)
            joined["identity"] = (
                joined["residue_id"].astype(str)
                + "\x1f"
                + joined["aa"].astype(str)
            )
            conflicts = (
                joined.groupby("canonical_position", dropna=False)["identity"]
                .nunique()
                .gt(1)
            )
            identity_consistency = not bool(conflicts.any())
        if not identity_consistency:
            base["inheritance_reason"] = "STRUCTURE_POSITION_IDENTITY_INCONSISTENT"
        complete = [item for item in candidates if bool(item[3]["valid"])]
        if identity_consistency and complete:
            pair_id, _, _, metrics = min(complete, key=lambda item: item[0])
            base.update(
                {
                    "inheritance_ready": True,
                    "inheritance_reason": None,
                    "inheritance_pair_id": pair_id,
                    "identity_consistency": True,
                }
            )
        else:
            best = max(
                candidates,
                key=lambda item: (
                    int(item[3]["identity_count"]),
                    int(item[3]["position_count"]),
                    item[0],
                ),
            )
            metrics = best[3]
            if identity_consistency and base["inheritance_reason"] == "PARENT_MAPPING_UNAVAILABLE":
                base["inheritance_reason"] = str(metrics["reason"])
            base["identity_consistency"] = identity_consistency
        base.update(
            {
                "canonical_position_count": int(metrics["position_count"]),
                "mapped_residue_count": int(metrics["mapped_count"]),
                "identity_complete_count": int(metrics["identity_count"]),
                "coordinate_visible_count": int(metrics["visible_count"]),
                "mapping_coverage": float(metrics["mapping_coverage"]),
                "coordinate_coverage": float(metrics["coordinate_coverage"]),
            }
        )
        rows.append(base)
    return pd.DataFrame(rows, columns=_PARENT_ELIGIBILITY_COLUMNS).sort_values(
        ["parent_structure_id"], kind="mergesort"
    ).reset_index(drop=True)


def select_inheritance_ready_replacements(
    parent_panel: pd.DataFrame,
    eligibility: pd.DataFrame,
    *,
    original_parent_ids: set[str],
    replacement_count: int,
) -> pd.DataFrame:
    """Keep ready original parents and fill their profile with panel replacements."""

    _require_columns(parent_panel, set(_PANEL_COLUMNS), "parent panel")
    _require_columns(
        eligibility,
        {"parent_structure_id", "inheritance_ready"},
        "parent eligibility",
    )
    if replacement_count < 1:
        raise ValueError("replacement_count must be positive")
    panel = parent_panel.copy()
    panel["parent_structure_id"] = panel["parent_structure_id"].astype(str)
    eligibility_index = eligibility.set_index("parent_structure_id")
    if eligibility_index.index.duplicated().any():
        raise ValueError("parent eligibility contains duplicate parent_structure_id values")
    missing = sorted(set(panel["parent_structure_id"]) - set(eligibility_index.index))
    if missing:
        raise ValueError(f"parent eligibility is missing panel rows: {missing[:5]}")
    original_parent_ids = {str(value) for value in original_parent_ids}
    if not original_parent_ids <= set(panel["parent_structure_id"]):
        raise ValueError("original parent IDs are not all present in parent panel")
    original = panel.loc[panel["parent_structure_id"].isin(original_parent_ids)].copy()
    ready_original = original.loc[
        original["parent_structure_id"].map(eligibility_index["inheritance_ready"].astype(bool))
    ]
    required = len(original) - len(ready_original)
    if replacement_count != required:
        raise ValueError(
            f"replacement_count must replace every non-ready original parent: {required}"
        )
    candidates = panel.loc[~panel["parent_structure_id"].isin(original_parent_ids)].copy()
    candidates = candidates.loc[
        candidates["parent_structure_id"].map(eligibility_index["inheritance_ready"].astype(bool))
    ]
    if candidates.empty:
        raise ValueError("parent panel has no inheritance-ready replacement candidates")
    target_counts = original["length_bin"].value_counts().sub(
        ready_original["length_bin"].value_counts(), fill_value=0
    )
    target_counts = target_counts.loc[target_counts.gt(0)].astype(int)
    if int(target_counts.sum()) != replacement_count:
        raise ValueError("replacement profile does not match replacement_count")
    used_clusters = set(ready_original["identity_cluster_id"].astype(str))
    chosen: list[pd.DataFrame] = []
    for length_bin, count in target_counts.sort_index().items():
        available = candidates.loc[candidates["length_bin"].eq(length_bin)].copy()
        available["_cluster_seen"] = available["identity_cluster_id"].astype(str).isin(used_clusters)
        available = available.sort_values(
            ["_cluster_seen", "selection_rank", "protein_id", "parent_structure_id", "source_structure_id"],
            ascending=[True, True, True, True, True],
            kind="mergesort",
        )
        if len(available) < count:
            raise ValueError(f"not enough ready replacement candidates in length bin: {length_bin}")
        selected_rows = available.head(int(count)).drop(columns="_cluster_seen")
        chosen.append(selected_rows)
        used_clusters.update(selected_rows["identity_cluster_id"].astype(str))
        candidates = candidates.loc[
            ~candidates["parent_structure_id"].isin(selected_rows["parent_structure_id"])
        ]
    replacements = pd.concat(chosen, ignore_index=True)
    selected = pd.concat([ready_original, replacements], ignore_index=True)
    if len(selected) != len(original):
        raise ValueError("replacement selection did not preserve parent count")
    return selected.sort_values(
        ["protein_id", "parent_structure_id", "source_structure_id"], kind="mergesort"
    ).reset_index(drop=True).loc[:, _PANEL_COLUMNS]


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _semantic_seed(
    *,
    root_seed: int,
    protein_id: str,
    parent_structure_id: str,
    source_structure_id: str,
    family: str,
    dose: float,
    realization: int,
) -> int:
    material = "|".join(
        (
            str(root_seed),
            protein_id,
            parent_structure_id,
            source_structure_id,
            family,
            f"{dose:.8f}",
            str(realization),
        )
    )
    return int.from_bytes(sha256(material.encode("utf-8")).digest()[:8], "big")


def _pilot_identity(
    *,
    protein_id: str,
    parent_structure_id: str,
    family: str,
    dose: float,
    realization: int,
) -> tuple[str, str]:
    fields = {
        "protein_id": protein_id,
        "parent_structure_id": parent_structure_id,
        "perturbation_family": family,
        "requested_dose": f"{dose:.8f}",
        "realization": realization,
    }
    return canonical_id("pilot_perturbation_pair", fields), canonical_id(
        "pilot_perturbation_instance", fields
    )


def _validate_pilot_request(
    *,
    parent_panel: pd.DataFrame,
    doses: tuple[float, ...],
    families: tuple[str, ...],
    realizations: int,
    root_seed: int,
    output_root: Path,
) -> tuple[tuple[float, ...], tuple[str, ...]]:
    if not isinstance(parent_panel, pd.DataFrame):
        raise TypeError("parent_panel must be a pandas DataFrame")
    _require_columns(parent_panel, _PILOT_PANEL_COLUMNS, "parent panel")
    if parent_panel.empty:
        raise ValueError("parent panel must contain at least one parent")
    if parent_panel["parent_structure_id"].isna().any() or parent_panel[
        "protein_id"
    ].isna().any():
        raise ValueError("parent panel contains empty semantic identities")
    if parent_panel["parent_structure_id"].astype(str).duplicated().any():
        raise ValueError("parent panel contains duplicate parent_structure_id values")
    if isinstance(realizations, bool) or not isinstance(realizations, int) or realizations < 1:
        raise ValueError("realizations must be a positive integer")
    if isinstance(root_seed, bool) or not isinstance(root_seed, int):
        raise TypeError("root_seed must be an integer")
    if not isinstance(output_root, Path):
        raise TypeError("output_root must be a pathlib.Path")

    normalized_doses: list[float] = []
    for dose in doses:
        if isinstance(dose, bool):
            raise TypeError("doses must be numeric")
        try:
            normalized = float(dose)
        except (TypeError, ValueError) as exc:
            raise TypeError("doses must be numeric") from exc
        if not np.isfinite(normalized) or normalized not in SUPPORTED_PERTURBATION_DOSES:
            raise ValueError(f"unsupported perturbation dose: {normalized}")
        normalized_doses.append(normalized)
    if not normalized_doses or len(set(normalized_doses)) != len(normalized_doses):
        raise ValueError("doses must be a non-empty collection of unique supported values")

    normalized_families = tuple(str(family).strip().upper() for family in families)
    if not normalized_families or len(set(normalized_families)) != len(normalized_families):
        raise ValueError("families must be a non-empty collection of unique supported values")
    unsupported_families = sorted(set(normalized_families).difference(SUPPORTED_PERTURBATION_FAMILIES))
    if unsupported_families:
        raise ValueError(f"unsupported perturbation families: {unsupported_families}")
    return tuple(sorted(normalized_doses)), tuple(sorted(normalized_families))


@lru_cache(maxsize=8)
def _load_core_inputs(repository_root: Path) -> _CoreInputs:
    core = repository_root / "benchmark/core"
    required_paths = {
        "proteins": core / "proteins.parquet",
        "structures": core / "structures.parquet",
        "condition pairs": core / "condition_pairs.parquet",
        "residue mappings": core / "residue_mappings.parquet",
    }
    missing = [label for label, path in required_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"controlled-perturbation Core artifacts are missing: {missing}")
    proteins = pd.read_parquet(required_paths["proteins"])
    structures = pd.read_parquet(required_paths["structures"])
    condition_pairs = pd.read_parquet(required_paths["condition pairs"])
    _require_columns(proteins, {"protein_id", "canonical_sequence"}, "Core proteins")
    _require_columns(
        structures,
        {"structure_id", "protein_id", "source_structure_id", "source_file_ref", "chain_id"},
        "Core structures",
    )
    _require_columns(
        condition_pairs,
        {"pair_id", "condition_1_structure_id", "condition_2_structure_id"},
        "Core condition pairs",
    )
    return _CoreInputs(
        proteins=proteins,
        structures=structures,
        condition_pairs=condition_pairs,
        residue_mappings_path=required_paths["residue mappings"],
    )


@lru_cache(maxsize=4096)
def _read_parent_mapping_rows(
    residue_mappings_path: Path,
    parent_pair_id: str,
    fields: tuple[str, ...],
) -> pd.DataFrame:
    """Read one immutable parent mapping slice once per process."""

    return pd.read_parquet(
        residue_mappings_path,
        filters=[("pair_id", "==", parent_pair_id)],
        columns=list(fields),
    )


def _canonical_residues(core: _CoreInputs, protein_id: str) -> pd.DataFrame:
    matches = core.proteins.loc[core.proteins["protein_id"].astype(str).eq(protein_id)]
    if len(matches) != 1:
        raise ValueError(f"Core proteins must contain exactly one row for {protein_id}")
    sequence = matches.iloc[0]["canonical_sequence"]
    if not isinstance(sequence, str) or not sequence.strip():
        raise _ExpectedPilotDataFailure("CANONICAL_SEQUENCE_UNAVAILABLE")
    residues = list(sequence.strip())
    if any(not residue.isalpha() for residue in residues):
        raise ValueError(f"Core canonical sequence is malformed for {protein_id}")
    return pd.DataFrame(
        {"canonical_position": range(1, len(residues) + 1), "canonical_aa": residues}
    )


def _core_structure(core: _CoreInputs, parent: pd.Series) -> pd.Series:
    parent_structure_id = str(parent["parent_structure_id"])
    matches = core.structures.loc[
        core.structures["structure_id"].astype(str).eq(parent_structure_id)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Core structures must contain exactly one row for {parent_structure_id}"
        )
    structure = matches.iloc[0]
    for field in ("protein_id", "source_structure_id", "source_file_ref"):
        if str(structure[field]) != str(parent[field]):
            raise ValueError(f"parent panel conflicts with Core structures for {parent_structure_id}: {field}")
    if pd.isna(structure["chain_id"]) or not str(structure["chain_id"]).strip():
        raise _ExpectedPilotDataFailure("STRUCTURAL_PARSE_FAILURE")
    return structure


def _read_parent_mapping(
    *,
    core: _CoreInputs,
    parent_structure_id: str,
    protein_id: str,
    canonical: pd.DataFrame,
    inherited_pair_id: str,
) -> pd.DataFrame:
    candidates = core.condition_pairs.loc[
        core.condition_pairs["condition_1_structure_id"].astype(str).eq(parent_structure_id)
        | core.condition_pairs["condition_2_structure_id"].astype(str).eq(parent_structure_id)
    ].copy()
    if candidates.empty:
        raise _ExpectedPilotDataFailure("PARENT_MAPPING_UNAVAILABLE")
    candidates["mapping_side"] = np.where(
        candidates["condition_1_structure_id"].astype(str).eq(parent_structure_id),
        "condition_1",
        "condition_2",
    )
    for candidate in candidates.sort_values(["pair_id", "mapping_side"], kind="mergesort").itertuples(
        index=False
    ):
        parent_pair_id = str(candidate.pair_id)
        side = str(candidate.mapping_side)
        fields = [
            "canonical_position",
            "canonical_aa",
            f"{side}_residue_id",
            f"{side}_aa",
            f"{side}_mapped",
            f"{side}_coordinate_visible",
            f"{side}_missing_reason",
        ]
        try:
            source = _read_parent_mapping_rows(
                core.residue_mappings_path,
                parent_pair_id,
                tuple(fields),
            )
        except (KeyError, ValueError) as exc:
            raise ValueError("Core residue mappings violate the required mapping schema") from exc
        if source.empty:
            continue
        required = set(fields)
        _require_columns(source, required, "Core residue mappings")
        renamed = source.rename(
            columns={
                f"{side}_residue_id": "residue_id",
                f"{side}_aa": "aa",
                f"{side}_mapped": "mapped",
                f"{side}_coordinate_visible": "coordinate_visible",
                f"{side}_missing_reason": "missing_reason",
            }
        )
        positions = pd.to_numeric(renamed["canonical_position"], errors="coerce")
        if positions.isna().any() or positions.duplicated().any():
            raise ValueError("Core residue mappings have invalid canonical positions")
        expected_positions = canonical["canonical_position"].tolist()
        if sorted(positions.astype(int).tolist()) != expected_positions:
            continue
        mapped = renamed["mapped"]
        if mapped.isna().any() or not mapped.map(lambda value: isinstance(value, (bool, np.bool_))).all():
            raise ValueError("Core residue mappings contain non-boolean mapped values")
        mapped_rows = renamed.loc[mapped.astype(bool)]
        if mapped_rows[["residue_id", "aa"]].isna().any().any() or any(
            not str(value).strip()
            for value in mapped_rows[["residue_id", "aa"]].to_numpy().ravel()
        ):
            continue
        parent_mapping = renamed.loc[
            :, ["canonical_position", "residue_id", "aa", "mapped", "coordinate_visible", "missing_reason"]
        ]
        return inherit_mapping(
            pair_id=inherited_pair_id,
            protein_id=protein_id,
            canonical=canonical,
            parent_mapping=parent_mapping,
        )
    raise _ExpectedPilotDataFailure("PARENT_MAPPING_INCOMPLETE")


def load_inherited_parent_mapping(
    *,
    repository_root: Path,
    parent_structure_id: str,
    protein_id: str,
    source_structure_id: str,
    inherited_pair_id: str,
) -> pd.DataFrame:
    """Load the canonical parent correspondence for a generated pair.

    This is a read-only public adapter around the existing construction owner.
    It never aligns or remaps generated coordinates.
    """

    core = _load_core_inputs(Path(repository_root))
    structure = core.structures.loc[
        core.structures["structure_id"].astype(str).eq(str(parent_structure_id))
    ]
    if len(structure) != 1:
        raise ValueError(
            f"Core structures must contain exactly one row for {parent_structure_id}"
        )
    structure_row = structure.iloc[0]
    parent = pd.Series(
        {
            "parent_structure_id": str(parent_structure_id),
            "protein_id": str(protein_id),
            "source_structure_id": str(source_structure_id),
            "source_file_ref": str(structure_row["source_file_ref"]),
        }
    )
    _core_structure(core, parent)
    canonical = _canonical_residues(core, str(protein_id))
    try:
        return _read_parent_mapping(
            core=core,
            parent_structure_id=str(parent_structure_id),
            protein_id=str(protein_id),
            canonical=canonical,
            inherited_pair_id=str(inherited_pair_id),
        )
    except _ExpectedPilotDataFailure as failure:
        raise InheritedParentMappingDataFailure(failure.reason) from failure


def _parse_selected_chain(
    *, source_path: Path, chain_id: str, protein_id: str, source_structure_id: str
) -> _ParsedChain:
    if not source_path.is_file():
        raise _ExpectedPilotDataFailure("STRUCTURAL_PARSE_FAILURE")
    parser = PDBParser(QUIET=True) if source_path.suffix.lower() in {".pdb", ".ent"} else MMCIFParser(QUIET=True)
    try:
        parsed = parser.get_structure("controlled_perturbation_parent", str(source_path))
        model = next(parsed.get_models())
        if chain_id not in model:
            raise _ExpectedPilotDataFailure("STRUCTURAL_PARSE_FAILURE")
        chain = model[chain_id]
        residue_positions: list[int] = []
        residue_names: list[str] = []
        atom_names: list[tuple[str, ...]] = []
        coordinates: list[np.ndarray] = []
        atoms_by_residue: list[tuple[object, ...]] = []
        residue_indices: dict[str, int] = {}
        for residue in chain:
            if not is_aa(residue, standard=False):
                continue
            hetero_flag, position, insertion_code = residue.id
            insertion = str(insertion_code).strip()
            if str(hetero_flag).strip() or insertion:
                raise _ExpectedPilotDataFailure("STRUCTURAL_PARSE_FAILURE")
            names = tuple(str(atom.get_name()).strip() for atom in residue.get_atoms())
            atom_objects = tuple(residue.get_atoms())
            if not names or len(set(names)) != len(names) or names.count("CA") != 1:
                raise _ExpectedPilotDataFailure("STRUCTURAL_PARSE_FAILURE")
            residue_coordinates = np.vstack([np.asarray(atom.coord, dtype=float) for atom in atom_objects])
            if not np.isfinite(residue_coordinates).all():
                raise _ExpectedPilotDataFailure("INVALID_PARENT_COORDINATES")
            residue_id = f"{chain_id}:{int(position)}"
            if residue_id in residue_indices:
                raise _ExpectedPilotDataFailure("STRUCTURAL_PARSE_FAILURE")
            residue_indices[residue_id] = len(residue_positions)
            residue_positions.append(int(position))
            residue_names.append(str(residue.resname).strip())
            atom_names.append(names)
            coordinates.append(residue_coordinates)
            atoms_by_residue.append(atom_objects)
        coordinate_structure = CoordinateStructure(
            protein_id=protein_id,
            source_structure_id=source_structure_id,
            residue_positions=tuple(residue_positions),
            residue_names=tuple(residue_names),
            atom_names=tuple(atom_names),
            atom_coordinates=tuple(coordinates),
        )
    except _ExpectedPilotDataFailure:
        raise
    except (OSError, PDBException, ValueError, StopIteration) as exc:
        raise _ExpectedPilotDataFailure("STRUCTURAL_PARSE_FAILURE") from exc
    return _ParsedChain(
        coordinate_structure=coordinate_structure,
        atoms_by_residue=tuple(atoms_by_residue),
        residue_indices=residue_indices,
    )


def _mapped_ca_coordinates(
    *,
    reference: CoordinateStructure,
    perturbed: CoordinateStructure,
    mapping: pd.DataFrame,
    parsed: _ParsedChain,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    visible = mapping.loc[mapping["common_coordinate_visible"].astype(bool)].sort_values(
        "canonical_position", kind="mergesort"
    )
    if len(visible) < 3:
        raise _ExpectedPilotDataFailure("MAPPED_COORDINATES_UNAVAILABLE")
    indices: list[int] = []
    positions: list[int] = []
    for row in visible.itertuples(index=False):
        residue_id = str(row.condition_1_residue_id)
        if residue_id not in parsed.residue_indices:
            raise ValueError("inherited mapping residue identity is absent from the selected chain")
        indices.append(parsed.residue_indices[residue_id])
        positions.append(int(row.canonical_position))
    reference_ca = np.vstack(
        [reference.atom_coordinates[index][reference.atom_names[index].index("CA")] for index in indices]
    )
    perturbed_ca = np.vstack(
        [perturbed.atom_coordinates[index][perturbed.atom_names[index].index("CA")] for index in indices]
    )
    return reference_ca, perturbed_ca, np.asarray(positions, dtype=int)


def _write_perturbed_chain(
    *, parsed: _ParsedChain, perturbed: CoordinateStructure, destination: Path
) -> None:
    if len(parsed.atoms_by_residue) != len(perturbed.atom_coordinates):
        raise RuntimeError("perturbation atom identity differs from parsed parent")
    for atoms, coordinates in zip(parsed.atoms_by_residue, perturbed.atom_coordinates, strict=True):
        if len(atoms) != len(coordinates):
            raise RuntimeError("perturbation atom count differs from parsed parent")
        for atom, coordinate in zip(atoms, coordinates, strict=True):
            atom.set_coord(np.asarray(coordinate, dtype=float))
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = MMCIFIO()
    writer.set_structure(next(iter(parsed.atoms_by_residue)).__getitem__(0).get_parent().get_parent().get_parent())
    writer.save(str(destination))


def _instance_row(
    *,
    pair_id: str,
    instance_id: str,
    parent: pd.Series,
    family: str,
    dose: float,
    realization: int,
    random_seed: int,
    status: str,
    reason: str | None,
    operator_metadata: dict[str, object] | None = None,
    realized_pairwise_distance_change: float | None = None,
    realized_contact_change: float | None = None,
    perturbed_source_file_ref: str | None = None,
) -> dict[str, object]:
    metadata = operator_metadata or {}
    return {
        "pilot_instance_id": instance_id,
        "pair_id": pair_id,
        "protein_id": str(parent["protein_id"]),
        "parent_structure_id": str(parent["parent_structure_id"]),
        "source_structure_id": str(parent["source_structure_id"]),
        "perturbation_family": family,
        "requested_dose": dose,
        "requested_dose_unit": "angstrom",
        "realization": realization,
        "random_seed": random_seed,
        "perturbation_method": metadata.get("operator_method"),
        "perturbation_version": "controlled_perturbation_v1",
        "realized_ca_rmsd": metadata.get("realized_ca_rmsd"),
        "realized_pairwise_distance_change": realized_pairwise_distance_change,
        "realized_contact_change": realized_contact_change,
        "reference_source_file_ref": str(parent["source_file_ref"]),
        "perturbed_source_file_ref": perturbed_source_file_ref,
        "status": status,
        "reason": reason,
    }


def build_pilot_instances(
    *,
    parent_panel: pd.DataFrame,
    doses: tuple[float, ...] = (0.25, 0.50, 1.00, 2.00),
    families: tuple[str, ...] = (
        "COORDINATE_NOISE",
        "SHEAR",
        "CONTACT_RELATIONAL_DEFORMATION",
    ),
    realizations: int = 1,
    root_seed: int = 20260819,
    output_root: Path,
) -> dict[str, pd.DataFrame]:
    """Build and persist a bounded, model-independent perturbation pilot.

    Only instance-local source/mapping/structure failures are represented as
    explicit statuses.  Core schema, identity, and mapping-contract violations
    raise so a caller cannot mistake an invalid scientific input for attrition.
    """

    normalized_doses, normalized_families = _validate_pilot_request(
        parent_panel=parent_panel,
        doses=doses,
        families=families,
        realizations=realizations,
        root_seed=root_seed,
        output_root=output_root,
    )
    pilot_root = output_root / "pilot"
    if pilot_root.exists():
        raise FileExistsError(f"pilot output already exists: {pilot_root}")
    core = _load_core_inputs(_repository_root())
    output_root.mkdir(parents=True, exist_ok=True)
    temporary_root = output_root / ".pilot-tmp"
    if temporary_root.exists():
        raise FileExistsError(f"pilot temporary output already exists: {temporary_root}")
    temporary_root.mkdir()

    instance_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    residue_frames: list[pd.DataFrame] = []
    try:
        parents = parent_panel.sort_values(
            ["protein_id", "parent_structure_id", "source_structure_id"], kind="mergesort"
        )
        for parent in parents.itertuples(index=False):
            parent_series = pd.Series(parent._asdict())
            protein_id = str(parent_series["protein_id"])
            parent_structure_id = str(parent_series["parent_structure_id"])
            source_structure_id = str(parent_series["source_structure_id"])
            for family in normalized_families:
                for dose in normalized_doses:
                    for realization in range(1, realizations + 1):
                        pair_id, instance_id = _pilot_identity(
                            protein_id=protein_id,
                            parent_structure_id=parent_structure_id,
                            family=family,
                            dose=dose,
                            realization=realization,
                        )
                        random_seed = _semantic_seed(
                            root_seed=root_seed,
                            protein_id=protein_id,
                            parent_structure_id=parent_structure_id,
                            source_structure_id=source_structure_id,
                            family=family,
                            dose=dose,
                            realization=realization,
                        )
                        try:
                            structure = _core_structure(core, parent_series)
                            canonical = _canonical_residues(core, protein_id)
                            mapping = _read_parent_mapping(
                                core=core,
                                parent_structure_id=parent_structure_id,
                                protein_id=protein_id,
                                canonical=canonical,
                                inherited_pair_id=pair_id,
                            )
                            source_ref = Path(str(structure["source_file_ref"]))
                            source_path = (
                                source_ref
                                if source_ref.is_absolute()
                                else _repository_root() / source_ref
                            )
                            parsed = _parse_selected_chain(
                                source_path=source_path,
                                chain_id=str(structure["chain_id"]),
                                protein_id=protein_id,
                                source_structure_id=source_structure_id,
                            )
                        except _ExpectedPilotDataFailure as failure:
                            instance_rows.append(
                                _instance_row(
                                    pair_id=pair_id,
                                    instance_id=instance_id,
                                    parent=parent_series,
                                    family=family,
                                    dose=dose,
                                    realization=realization,
                                    random_seed=random_seed,
                                    status="UNRESOLVED",
                                    reason=failure.reason,
                                )
                            )
                            continue

                        spec = ControlledPerturbationSpec(
                            perturbation_family=family,
                            requested_dose=dose,
                            requested_dose_unit="angstrom",
                            random_seed=random_seed,
                            perturbation_method="controlled_perturbation",
                            perturbation_version="controlled_perturbation_v1",
                        )
                        try:
                            operator_result = apply_perturbation(
                                parsed.coordinate_structure, spec
                            )
                        except ValueError as exc:
                            if str(exc) in {"SEVERITY_CALIBRATION_FAILURE"} or str(exc).startswith(
                                "CONTACT_RELATIONAL_DEFORMATION requires"
                            ):
                                instance_rows.append(
                                    _instance_row(
                                        pair_id=pair_id,
                                        instance_id=instance_id,
                                        parent=parent_series,
                                        family=family,
                                        dose=dose,
                                        realization=realization,
                                        random_seed=random_seed,
                                        status="EXCLUDED",
                                        reason="OPERATOR_FAILURE",
                                    )
                                )
                                continue
                            raise
                        generated_path: Path | None = None
                        try:
                            reference_ca, perturbed_ca, canonical_positions = _mapped_ca_coordinates(
                                reference=parsed.coordinate_structure,
                                perturbed=operator_result.perturbed,
                                mapping=mapping,
                                parsed=parsed,
                            )
                            pair_descriptor, residue_descriptor = annotate_aligned_pair(
                                pair_id=pair_id,
                                reference_ca=reference_ca,
                                perturbed_ca=perturbed_ca,
                            )
                            residue_descriptor["canonical_position"] = canonical_positions
                            generated_name = f"{instance_id}.cif"
                            generated_path = temporary_root / "structures" / generated_name
                            _write_perturbed_chain(
                                parsed=parsed,
                                perturbed=operator_result.perturbed,
                                destination=generated_path,
                            )
                            reloaded = _parse_selected_chain(
                                source_path=generated_path,
                                chain_id=str(structure["chain_id"]),
                                protein_id=protein_id,
                                source_structure_id=source_structure_id,
                            )
                            if (
                                reloaded.coordinate_structure.residue_positions
                                != parsed.coordinate_structure.residue_positions
                                or reloaded.coordinate_structure.residue_names
                                != parsed.coordinate_structure.residue_names
                                or reloaded.coordinate_structure.atom_names
                                != parsed.coordinate_structure.atom_names
                            ):
                                raise RuntimeError("serialized perturbation changed structural identity")
                        except _ExpectedPilotDataFailure as failure:
                            if generated_path is not None and generated_path.is_file():
                                generated_path.unlink()
                            instance_rows.append(
                                _instance_row(
                                    pair_id=pair_id,
                                    instance_id=instance_id,
                                    parent=parent_series,
                                    family=family,
                                    dose=dose,
                                    realization=realization,
                                    random_seed=random_seed,
                                    status="UNRESOLVED",
                                    reason=failure.reason,
                                    operator_metadata=operator_result.operator_metadata,
                                )
                            )
                            continue
                        projection = residue_descriptor.attrs["perturbation_projection"]
                        generated_ref = f"pilot/structures/{generated_name}"
                        instance_rows.append(
                            _instance_row(
                                pair_id=pair_id,
                                instance_id=instance_id,
                                parent=parent_series,
                                family=family,
                                dose=dose,
                                realization=realization,
                                random_seed=random_seed,
                                status="GENERATED",
                                reason=None,
                                operator_metadata=operator_result.operator_metadata,
                                realized_pairwise_distance_change=projection[
                                    "realized_pairwise_distance_change"
                                ],
                                realized_contact_change=pair_descriptor["global_contact_change"],
                                perturbed_source_file_ref=generated_ref,
                            )
                        )
                        pair_rows.append(pair_descriptor)
                        residue_frames.append(residue_descriptor.loc[:, _RESIDUE_DESCRIPTOR_COLUMNS])

        instances = pd.DataFrame(instance_rows, columns=_PILOT_INSTANCE_COLUMNS)
        pair_descriptors = pd.DataFrame(pair_rows, columns=_PAIR_DESCRIPTOR_COLUMNS)
        residue_descriptors = (
            pd.concat(residue_frames, ignore_index=True)
            if residue_frames
            else pd.DataFrame(columns=_RESIDUE_DESCRIPTOR_COLUMNS)
        )
        instances.to_parquet(temporary_root / "perturbation_instances.parquet", index=False)
        pair_descriptors.to_parquet(
            temporary_root / "pair_structural_descriptors.parquet", index=False
        )
        residue_descriptors.to_parquet(
            temporary_root / "residue_structural_descriptors.parquet", index=False
        )
        temporary_root.replace(pilot_root)
    except BaseException:
        if temporary_root.exists():
            import shutil

            shutil.rmtree(temporary_root)
        raise
    return {
        "instances": instances,
        "pair_descriptors": pair_descriptors,
        "residue_descriptors": residue_descriptors,
    }
