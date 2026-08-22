"""Read-only parent sources for the controlled relational-geometry path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd
from Bio.PDB.Polypeptide import protein_letters_3to1

from dual_uq.construction.controlled_perturbation import inherit_mapping
from dual_uq.dataset.controlled_perturbation_construction import (
    InheritedParentMappingDataFailure,
    _canonical_residues,
    _core_structure,
    _load_core_inputs,
    _read_parent_mapping,
)


@dataclass(frozen=True, slots=True)
class ParentSourceRecord:
    """One parent structure and its inherited canonical correspondence."""

    protein_id: str
    source_structure_id: str
    source_file_ref: str
    chain_id: str
    canonical_sequence: str
    mapping: pd.DataFrame


class ParentSource(Protocol):
    """Minimal lookup contract shared by geometry and local-response callers."""

    def resolve_parent(
        self, parent: pd.Series, *, inherited_pair_id: str
    ) -> ParentSourceRecord: ...


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


@dataclass(frozen=True, slots=True)
class CoreParentSource:
    """Adapter retaining the original benchmark/core construction owner."""

    repository_root: Path

    def resolve_parent(self, parent: pd.Series, *, inherited_pair_id: str) -> ParentSourceRecord:
        core = _load_core_inputs(Path(self.repository_root).resolve())
        required = {"parent_structure_id", "protein_id", "source_structure_id", "source_file_ref"}
        core_parent = parent.copy()
        if "source_file_ref" not in core_parent and "reference_source_file_ref" in core_parent:
            core_parent["source_file_ref"] = core_parent["reference_source_file_ref"]
        _require_columns(pd.DataFrame([core_parent]), required, "Core parent")
        parent_id = str(core_parent["parent_structure_id"])
        protein_id = str(core_parent["protein_id"])
        structure = _core_structure(core, core_parent)
        canonical = _canonical_residues(core, protein_id)
        mapping = _read_parent_mapping(
            core=core,
            parent_structure_id=parent_id,
            protein_id=protein_id,
            canonical=canonical,
            inherited_pair_id=str(inherited_pair_id),
        )
        return ParentSourceRecord(
            protein_id=protein_id,
            source_structure_id=str(parent["source_structure_id"]),
            source_file_ref=str(structure["source_file_ref"]),
            chain_id=str(structure["chain_id"]),
            canonical_sequence="".join(canonical["canonical_aa"].astype(str)),
            mapping=mapping,
        )


def _observed_one_letter(value: object) -> str:
    name = str(value).strip().upper()
    try:
        return str(protein_letters_3to1[name]).upper()
    except KeyError as exc:
        raise InheritedParentMappingDataFailure(
            f"MAPPED_RESIDUE_IDENTITY_INCOMPLETE: unsupported observed residue {name}"
        ) from exc


@dataclass
class ApoHoloParentSource:
    """Project persisted Apo/Holo SIFTS facts into the existing parent contract."""

    repository_root: Path
    parent_panel: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        root = Path(self.repository_root).resolve()
        release = (
            root
            / "experiments/interventions/biological_states/apo_holo_selective_admission_release"
        )
        object.__setattr__(self, "_release_root", release)
        object.__setattr__(
            self,
            "_pairs",
            pd.read_parquet(release / "primary_pairs.parquet").assign(
                pair_id=lambda frame: frame["pair_id"].astype(str),
                protein_id=lambda frame: frame["protein_id"].astype(str),
            ),
        )
        object.__setattr__(
            self,
            "_mapping",
            pd.read_parquet(
                release / "residue_mappings.parquet",
                columns=[
                    "pair_id",
                    "protein_id",
                    "canonical_position",
                    "pdb_chain_id_apo",
                    "pdb_residue_number_apo",
                    "pdb_residue_name_apo",
                    "pdb_chain_id_holo",
                    "pdb_residue_number_holo",
                    "pdb_residue_name_holo",
                ],
            ).assign(pair_id=lambda frame: frame["pair_id"].astype(str)),
        )
        clusters = pd.read_parquet(
            root
            / "experiments/comparisons/method_design/identity_split/cluster_assignments.parquet",
            columns=["protein_id", "sequence"],
        ).drop_duplicates("protein_id")
        object.__setattr__(
            self,
            "_sequences",
            clusters.set_index("protein_id")["sequence"].astype(str).to_dict(),
        )
        panel = self.parent_panel
        if panel is not None:
            _require_columns(panel, {"parent_structure_id"}, "Apo/Holo parent panel")
            object.__setattr__(
                self,
                "_parent_lookup",
                panel.assign(
                    parent_structure_id=panel["parent_structure_id"].astype(str)
                ).set_index("parent_structure_id"),
            )
        else:
            object.__setattr__(self, "_parent_lookup", None)

    def resolve_parent(self, parent: pd.Series, *, inherited_pair_id: str) -> ParentSourceRecord:
        parent_record = parent.copy()
        if "source_state" not in parent_record and self._parent_lookup is not None:
            parent_id = str(parent_record["parent_structure_id"])
            if parent_id not in self._parent_lookup.index:
                raise InheritedParentMappingDataFailure("PARENT_MAPPING_UNAVAILABLE")
            panel_record = self._parent_lookup.loc[parent_id]
            for field, value in panel_record.items():
                if field not in parent_record.index:
                    parent_record[field] = value
        required = {
            "protein_id",
            "source_structure_id",
            "source_file_ref",
            "source_state",
            "chain_id",
            "inheritance_pair_id",
        }
        _require_columns(pd.DataFrame([parent_record]), required, "Apo/Holo parent")
        protein_id = str(parent_record["protein_id"])
        pair_id = str(parent_record["inheritance_pair_id"])
        state = str(parent_record["source_state"]).strip().lower()
        if state not in {"apo", "holo"}:
            raise ValueError(f"unsupported Apo/Holo source state: {state}")
        pair = self._pairs.loc[
            self._pairs["pair_id"].eq(pair_id) & self._pairs["protein_id"].eq(protein_id)
        ]
        sequence = self._sequences.get(protein_id, "")
        if len(pair) != 1 or not sequence:
            raise InheritedParentMappingDataFailure("PARENT_MAPPING_UNAVAILABLE")
        mapping = self._mapping.loc[self._mapping["pair_id"].eq(pair_id)].copy()
        number = pd.to_numeric(mapping[f"pdb_residue_number_{state}"], errors="coerce")
        chain = mapping[f"pdb_chain_id_{state}"].astype("string").str.strip()
        names = mapping[f"pdb_residue_name_{state}"].astype("string").str.strip()
        positions = pd.to_numeric(mapping["canonical_position"], errors="coerce")
        expected = set(range(1, len(sequence) + 1))
        if (
            len(mapping) != len(sequence)
            or positions.isna().any()
            or positions.duplicated().any()
            or set(positions.astype(int)) != expected
            or number.isna().any()
            or chain.isna().any()
            or chain.eq("").any()
            or names.isna().any()
            or names.eq("").any()
            or chain.nunique() != 1
        ):
            raise InheritedParentMappingDataFailure("PARENT_MAPPING_INCOMPLETE")
        if str(chain.iloc[0]) != str(parent_record["chain_id"]):
            raise ValueError("Apo/Holo parent chain disagrees with persisted mapping")
        observed = [_observed_one_letter(value) for value in names]
        parent_mapping = pd.DataFrame(
            {
                "canonical_position": positions.astype(int),
                "residue_id": [
                    f"{chain_value!s}:{int(residue_number)}"
                    for chain_value, residue_number in zip(chain, number, strict=True)
                ],
                "aa": observed,
                "mapped": True,
                "coordinate_visible": True,
                "missing_reason": None,
            }
        )
        canonical = pd.DataFrame(
            {
                "canonical_position": range(1, len(sequence) + 1),
                "canonical_aa": list(sequence),
            }
        )
        inherited = inherit_mapping(
            pair_id=str(inherited_pair_id),
            protein_id=protein_id,
            canonical=canonical,
            parent_mapping=parent_mapping,
        )
        return ParentSourceRecord(
            protein_id=protein_id,
            source_structure_id=str(parent_record["source_structure_id"]),
            source_file_ref=str(parent_record["source_file_ref"]),
            chain_id=str(parent_record["chain_id"]),
            canonical_sequence=sequence,
            mapping=inherited,
        )


__all__ = [
    "ApoHoloParentSource",
    "CoreParentSource",
    "ParentSource",
    "ParentSourceRecord",
]
