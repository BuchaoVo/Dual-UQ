"""Matched apo/holo inverse-folding local-response analysis.

This module owns the read-only measurement layer for the released apo/holo
cohort.  Admission and residue mapping remain owned by the release; this
module only binds those records to model-native structure inputs and aggregates
paired model responses.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from dual_uq.core.atomic_io import atomic_write_new_bytes
from dual_uq.core.hashing import sha256_bytes
from dual_uq.models.proteinmpnn import (
    DecodingRealization,
    ProteinMPNNStructureInput,
    make_decoding_realization,
)
from dual_uq.structure_io import load_atom_site_table

LOCAL_RESPONSE_PROTOCOL = "apo_holo_matched_local_response_v1"
STANDARD_AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")


class ApoHoloLocalResponseError(ValueError):
    """Raised when a released apo/holo measurement input is inconsistent."""


def js_bits(left: Iterable[float], right: Iterable[float]) -> float:
    """Return the symmetric Jensen-Shannon divergence in bits."""
    p = np.asarray(tuple(left), dtype=float)
    q = np.asarray(tuple(right), dtype=float)
    if p.shape != q.shape or p.ndim != 1 or p.size == 0:
        raise ApoHoloLocalResponseError("JS inputs must be equal non-empty vectors")
    if not np.isfinite(p).all() or not np.isfinite(q).all() or (p < 0).any() or (q < 0).any():
        raise ApoHoloLocalResponseError("JS inputs must be finite and non-negative")
    if p.sum() <= 0 or q.sum() <= 0:
        raise ApoHoloLocalResponseError("JS inputs must have positive mass")
    p = p / p.sum()
    q = q / q.sum()
    midpoint = 0.5 * (p + q)
    result = 0.0
    for distribution in (p, q):
        positive = distribution > 0
        result += float(
            0.5
            * np.sum(
                distribution[positive]
                * np.log2(distribution[positive] / midpoint[positive])
            )
        )
    return result


def _read_json_sequence(project_root: Path, relative_path: str, expected: str) -> str:
    path = project_root / relative_path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        sequence = str((payload.get("sequence") or {}).get("value") or "").strip().upper()
    except (OSError, TypeError, ValueError) as exc:
        raise ApoHoloLocalResponseError(f"canonical sequence is unreadable: {path}") from exc
    if not sequence or any(aa not in {*STANDARD_AMINO_ACIDS, "X"} for aa in sequence):
        raise ApoHoloLocalResponseError(f"canonical sequence is invalid: {expected}")
    accession = str(payload.get("primaryAccession") or expected)
    if accession != expected:
        raise ApoHoloLocalResponseError(
            f"canonical sequence identity mismatch: expected {expected}, got {accession}"
        )
    return sequence


def _load_backbone_by_auth_key(path: Path, chain_id: str, atom_names: Sequence[str]) -> dict[tuple[int, str], dict[str, np.ndarray]]:
    """Select deterministic first-model backbone atoms in author namespace."""
    atoms = load_atom_site_table(path)
    if atoms.empty:
        raise ApoHoloLocalResponseError(f"structure contains no atoms: {path}")
    atoms = atoms.loc[atoms["model_number"].eq(int(atoms["model_number"].min()))].copy()
    atoms = atoms.loc[atoms["auth_asym_id"].astype(str).eq(str(chain_id))].copy()
    if atoms.empty:
        raise ApoHoloLocalResponseError(f"chain {chain_id!r} is absent: {path}")
    atoms = atoms.loc[atoms["atom_name"].astype(str).str.upper().isin(atom_names)].copy()
    atoms["_alt_rank"] = atoms["alt_id"].map(
        lambda value: 0 if str(value).strip().upper() in {"", ".", "?", "A"} else 1
    )
    atoms = atoms.sort_values(
        ["_alt_rank", "occupancy"], ascending=[True, False], kind="mergesort"
    )
    result: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    for row in atoms.itertuples(index=False):
        if pd.isna(row.auth_seq_id):
            continue
        key = (int(row.auth_seq_id), str(row.insertion_code or "").strip().upper())
        name = str(row.atom_name).strip().upper()
        residue = result.setdefault(key, {})
        if name not in residue:
            coordinate = np.asarray([row.x, row.y, row.z], dtype=np.float32)
            if np.isfinite(coordinate).all():
                residue[name] = coordinate
    return result


def _load_backbone_task(
    task: tuple[Path, str, tuple[str, ...]],
) -> tuple[tuple[str, str, tuple[str, ...]], dict[tuple[int, str], dict[str, np.ndarray]]]:
    path, chain_id, atom_names = task
    key = (str(path), str(chain_id), tuple(atom_names))
    return key, _load_backbone_by_auth_key(path, chain_id, atom_names)


def _asset_sha(ledger: pd.DataFrame, entity_id: str, path: Path) -> str:
    rows = ledger.loc[
        ledger["polymer_entity_id"].astype(str).eq(entity_id)
        & ledger["asset_type"].astype(str).eq("pdb_mmcif")
    ]
    if len(rows) != 1:
        raise ApoHoloLocalResponseError(
            f"expected one PDB asset ledger row for {entity_id}, got {len(rows)}"
        )
    value = str(rows.iloc[0].get("sha256") or "")
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ApoHoloLocalResponseError(f"invalid structure SHA for {entity_id}")
    if not path.is_file():
        raise ApoHoloLocalResponseError(f"structure asset is missing: {path}")
    return value


def _mapping_rows_for_pair(mappings: pd.DataFrame, pair_id: str) -> pd.DataFrame:
    rows = mappings.loc[mappings["pair_id"].astype(str).eq(pair_id)].copy()
    required = {
        "canonical_position",
        "apo_auth_seq_id",
        "holo_auth_seq_id",
        "insertion_code_apo",
        "insertion_code_holo",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ApoHoloLocalResponseError(f"residue mappings missing columns: {missing}")
    rows["canonical_position"] = pd.to_numeric(rows["canonical_position"], errors="coerce")
    rows = rows.dropna(subset=["canonical_position"]).copy()
    rows["canonical_position"] = rows["canonical_position"].astype(int)
    if rows.empty or rows["canonical_position"].duplicated().any():
        raise ApoHoloLocalResponseError(f"mapping axis is not unique: {pair_id}")
    return rows.sort_values("canonical_position", kind="mergesort").reset_index(drop=True)


def _condition_coordinates(
    mapping: pd.DataFrame,
    canonical_sequence: str,
    backbone: dict[tuple[int, str], dict[str, np.ndarray]],
    *,
    prefix: str,
    atom_names: Sequence[str],
    allow_missing: bool = False,
) -> tuple[tuple[int, ...], np.ndarray]:
    usable_positions: list[int] = []
    coordinate_rows: list[np.ndarray] = []
    for row in mapping.itertuples(index=False):
        position = int(row.canonical_position)
        if not 1 <= position <= len(canonical_sequence):
            continue
        expected_aa = canonical_sequence[position - 1]
        name_column = f"uniprot_residue_name_{prefix}"
        if (
            hasattr(row, name_column)
            and str(getattr(row, name_column)) not in {"", "nan", "None"}
            and str(getattr(row, name_column)).upper() != expected_aa
        ):
            raise ApoHoloLocalResponseError(f"canonical residue identity mismatch at {position}")
        seq = getattr(row, f"{prefix}_auth_seq_id")
        if pd.isna(seq):
            if allow_missing:
                usable_positions.append(position)
                coordinate_rows.append(np.full((len(atom_names), 3), np.nan, dtype=np.float32))
            continue
        insertion = str(getattr(row, f"insertion_code_{prefix}") or "").strip().upper()
        residue = backbone.get((int(seq), insertion), {})
        if not all(atom in residue for atom in atom_names):
            if allow_missing:
                usable_positions.append(position)
                coordinate_rows.append(np.full((len(atom_names), 3), np.nan, dtype=np.float32))
            continue
        usable_positions.append(position)
        coordinate_rows.append(np.asarray([residue[atom] for atom in atom_names], dtype=np.float32))
    if not usable_positions:
        raise ApoHoloLocalResponseError(f"no complete {prefix} backbone residues")
    return tuple(usable_positions), np.asarray(coordinate_rows, dtype=np.float32)


def _case_rows(
    *,
    protein_id: str,
    pair_id: str,
    condition: str,
    positions: tuple[int, ...],
    canonical_sequence: str,
    coordinates: np.ndarray,
    structure_sha256: str,
) -> list[dict[str, Any]]:
    projected = "".join(canonical_sequence[position - 1] for position in positions)
    return [
        {
            "protein_id": protein_id,
            "pair_id": pair_id,
            "condition": condition,
            "canonical_position": int(position),
            "canonical_positions": positions,
            "wt_sequence": canonical_sequence,
            "wt_sequence_projection": projected,
            "coordinates": coordinates,
            "structure_sha256": structure_sha256,
            "common_mask_sha256": None,
            "true_uniprot_gap": any(right != left + 1 for left, right in pairwise(positions)),
        }
        for position in positions
    ]


def build_apo_holo_cases(
    project_root: Path,
    release_root: Path,
    *,
    require_model_evaluable: bool = True,
    require_contiguous_esm_if1: bool = True,
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Bind released primary pairs to ProteinMPNN and ESM-IF1 input cases."""
    project_root = Path(project_root)
    release_root = Path(release_root)
    primary = pd.read_parquet(release_root / "primary_pairs.parquet")
    primary = primary.loc[primary["admitted"].eq(True)].copy()
    if primary.empty or primary["pair_id"].duplicated().any():
        raise ApoHoloLocalResponseError("primary release does not contain unique admitted pairs")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ApoHoloLocalResponseError("workers must be a positive integer")
    evaluations = pd.read_parquet(release_root / "model_evaluability.parquet")
    mappings = pd.read_parquet(release_root / "residue_mappings.parquet")
    structures = pd.read_parquet(release_root / "candidate_structures.parquet")
    ledger = pd.read_parquet(release_root / "asset_ledger.parquet")
    structure_by_entity = structures.set_index(structures["polymer_entity_id"].astype(str), drop=False)
    tasks: dict[tuple[str, str, tuple[str, ...]], tuple[Path, str, tuple[str, ...]]] = {}
    for pair in primary.itertuples(index=False):
        for prefix, chain_column, path_column in (
            ("apo", "apo_chain_id", "apo_mmcif_relative_path"),
            ("holo", "holo_chain_id", "holo_mmcif_relative_path"),
        ):
            path = project_root / str(getattr(pair, path_column))
            chain = str(getattr(pair, chain_column))
            key = (str(path), chain, ("N", "CA", "C", "O"))
            tasks[key] = (path, chain, ("N", "CA", "C", "O"))
    backbone_cache: dict[tuple[str, str, tuple[str, ...]], dict[tuple[int, str], dict[str, np.ndarray]]]
    if workers == 1:
        backbone_cache = {key: _load_backbone_task(task)[1] for key, task in tasks.items()}
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            backbone_cache = dict(executor.map(_load_backbone_task, tasks.values()))
    protein_rows: list[dict[str, Any]] = []
    esm_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for pair in primary.sort_values("pair_id", kind="mergesort").itertuples(index=False):
        pair_id = str(pair.pair_id)
        protein_id = str(pair.protein_id)
        evaluation = evaluations.loc[evaluations["pair_id"].astype(str).eq(pair_id)]
        model_ok = len(evaluation) == 1 and bool(evaluation.iloc[0].get("model_evaluable", False))
        if require_model_evaluable and not model_ok:
            exclusions.append({"protein_id": protein_id, "pair_id": pair_id, "reason": "model_not_evaluable", "detail": "release model_evaluability is false"})
            continue
        mapping = _mapping_rows_for_pair(mappings, pair_id)
        condition_specs = (
            ("APO", "apo", str(pair.apo_chain_id), str(pair.apo_polymer_entity_id), str(pair.apo_mmcif_relative_path)),
            ("HOLO", "holo", str(pair.holo_chain_id), str(pair.holo_polymer_entity_id), str(pair.holo_mmcif_relative_path)),
        )
        canonical_record = structure_by_entity.loc[str(pair.apo_polymer_entity_id)]
        sequence_path = str(canonical_record.get("uniprot_relative_path") or "")
        canonical_sequence = _read_json_sequence(project_root, sequence_path, str(pair.uniprot_id))
        condition_data: dict[str, tuple[tuple[int, ...], np.ndarray, str]] = {}
        esm_condition_data: dict[str, tuple[tuple[int, ...], np.ndarray, str]] = {}
        for condition, prefix, chain_id, entity_id, relative_path in condition_specs:
            structure_path = project_root / relative_path
            sha = _asset_sha(ledger, entity_id, structure_path)
            backbone = backbone_cache[(str(structure_path), chain_id, ("N", "CA", "C", "O"))]
            positions, coordinates = _condition_coordinates(
                mapping, canonical_sequence, backbone, prefix=prefix, atom_names=("N", "CA", "C", "O")
            )
            condition_data[condition] = (positions, coordinates, sha)
            esm_positions, esm_coordinates = _condition_coordinates(
                mapping,
                canonical_sequence,
                backbone,
                prefix=prefix,
                atom_names=("N", "CA", "C"),
                allow_missing=True,
            )
            esm_condition_data[condition] = (esm_positions, esm_coordinates, sha)
        common_positions = tuple(sorted(set(condition_data["APO"][0]).intersection(condition_data["HOLO"][0])))
        if len(common_positions) < 3:
            exclusions.append({"protein_id": protein_id, "pair_id": pair_id, "reason": "insufficient_common_model_backbone", "detail": "fewer than three complete paired residues"})
            continue
        for condition, (positions, coordinates, sha) in condition_data.items():
            index = [positions.index(position) for position in common_positions]
            selected_coordinates = coordinates[index]
            protein_rows.extend(
                _case_rows(
                    protein_id=protein_id,
                    pair_id=pair_id,
                    condition=condition,
                    positions=common_positions,
                    canonical_sequence=canonical_sequence,
                    coordinates=selected_coordinates,
                    structure_sha256=sha,
                )
            )
        mapping_positions = tuple(int(value) for value in mapping["canonical_position"])
        has_gap = any(right != left + 1 for left, right in pairwise(mapping_positions))
        if require_contiguous_esm_if1 and has_gap:
            exclusions.append({"protein_id": protein_id, "pair_id": pair_id, "reason": "true_uniprot_gap", "detail": "internal canonical position gap"})
        else:
            esm_positions = esm_condition_data["APO"][0]
            for condition, (positions, coordinates, sha) in esm_condition_data.items():
                if positions != esm_positions:
                    raise ApoHoloLocalResponseError(
                        f"apo/holo mapping axes differ for ESM-IF1: {pair_id}"
                    )
                esm_rows.extend(
                    _case_rows(
                        protein_id=protein_id,
                        pair_id=pair_id,
                        condition=condition,
                        positions=esm_positions,
                        canonical_sequence=canonical_sequence,
                        coordinates=coordinates,
                        structure_sha256=sha,
                    )
                )
    protein_table = pd.DataFrame(protein_rows)
    esm_table = pd.DataFrame(esm_rows)
    exclusion_table = pd.DataFrame(exclusions, columns=["protein_id", "pair_id", "reason", "detail"])
    return protein_table, esm_table, exclusion_table


def build_decoding_realizations(
    protein_id: str,
    length: int,
    count: int = 16,
) -> tuple[DecodingRealization, ...]:
    """Build deterministic matched ProteinMPNN realization metadata."""
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ApoHoloLocalResponseError("length must be a positive integer")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ApoHoloLocalResponseError("count must be a positive integer")
    result = []
    for repeat_index in range(count):
        digest = hashlib.sha256(f"{LOCAL_RESPONSE_PROTOCOL}|{protein_id}|{repeat_index}".encode()).digest()
        seed = int.from_bytes(digest[:8], "big") % (2**32 - 1)
        result.append(
            make_decoding_realization(
                protein_id=protein_id,
                mask_length=length,
                repeat_index=repeat_index,
                seed=max(seed, 1),
                protocol_version=LOCAL_RESPONSE_PROTOCOL,
            )
        )
    return tuple(result)


def _validate_case_axis(cases: pd.DataFrame) -> None:
    required = {
        "protein_id",
        "pair_id",
        "condition",
        "canonical_position",
        "canonical_positions",
        "wt_sequence",
        "wt_sequence_projection",
        "coordinates",
    }
    missing = sorted(required - set(cases.columns))
    if missing:
        raise ApoHoloLocalResponseError(f"model cases missing columns: {missing}")
    if cases.empty or set(cases["condition"].astype(str)) != {"APO", "HOLO"}:
        raise ApoHoloLocalResponseError("model cases require APO and HOLO conditions")
    key = ["protein_id", "pair_id", "condition", "canonical_position"]
    if cases.duplicated(key).any():
        raise ApoHoloLocalResponseError("model cases contain duplicate position rows")
    for (protein_id, pair_id), group in cases.groupby(["protein_id", "pair_id"], sort=False):
        axes: dict[str, tuple[int, ...]] = {}
        for condition, condition_rows in group.groupby("condition", sort=False):
            ordered = condition_rows.sort_values("canonical_position", kind="mergesort")
            positions = tuple(int(value) for value in ordered["canonical_position"])
            axes[str(condition)] = positions
            sequences = ordered["wt_sequence_projection"].astype(str).unique()
            if len(sequences) != 1 or len(sequences[0]) != len(positions):
                raise ApoHoloLocalResponseError(
                    f"WT projection axis differs for {protein_id}/{pair_id}/{condition}"
                )
            coordinate_values = list(ordered["coordinates"])
            if not coordinate_values:
                raise ApoHoloLocalResponseError("model case coordinates are empty")
            coordinates = np.asarray(coordinate_values[0], dtype=np.float32)
            if coordinates.ndim != 3 or coordinates.shape[0] != len(positions) or coordinates.shape[2] != 3:
                raise ApoHoloLocalResponseError(
                    f"coordinate shape differs for {protein_id}/{pair_id}/{condition}"
                )
            finite_rows = np.isfinite(coordinates).all(axis=(1, 2))
            missing_rows = np.isnan(coordinates).all(axis=(1, 2))
            if not np.logical_or(finite_rows, missing_rows).all():
                raise ApoHoloLocalResponseError(
                    "model case coordinate rows must be finite or fully missing"
                )
        if axes.get("APO") != axes.get("HOLO"):
            raise ApoHoloLocalResponseError(f"APO/HOLO position axes differ for {protein_id}/{pair_id}")


def _model_binding(adapter: Any) -> dict[str, Any]:
    binding = getattr(adapter, "binding", None)
    if callable(binding):
        value = binding()
        return dict(value) if isinstance(value, dict) else {"binding": str(value)}
    return {
        key: getattr(adapter, key)
        for key in ("implementation_id", "checkpoint_id")
        if hasattr(adapter, key)
    }


def run_model_local_response(
    adapter: Any,
    cases: pd.DataFrame,
    model_name: str,
    realization_count: int = 16,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Aggregate one model's paired local response without retaining raw tensors."""
    _validate_case_axis(cases)
    if model_name not in {"ProteinMPNN", "ESM-IF1"}:
        raise ApoHoloLocalResponseError(f"unsupported local-response model: {model_name}")
    if isinstance(realization_count, bool) or not isinstance(realization_count, int) or realization_count <= 0:
        raise ApoHoloLocalResponseError("realization_count must be a positive integer")
    position_rows: list[dict[str, Any]] = []
    protein_rows: list[dict[str, Any]] = []
    convergence: list[dict[str, Any]] = []
    for (protein_id, pair_id), group in cases.groupby(["protein_id", "pair_id"], sort=True):
        ordered_groups = {
            str(condition): value.sort_values("canonical_position", kind="mergesort")
            for condition, value in group.groupby("condition", sort=False)
        }
        positions = tuple(int(value) for value in ordered_groups["APO"]["canonical_position"])
        first = ordered_groups["APO"].iloc[0]
        sequence = str(first["wt_sequence_projection"])
        apo_coordinates = np.asarray(ordered_groups["APO"].iloc[0]["coordinates"], dtype=np.float32)
        holo_coordinates = np.asarray(ordered_groups["HOLO"].iloc[0]["coordinates"], dtype=np.float32)
        if model_name == "ProteinMPNN":
            realizations = build_decoding_realizations(str(protein_id), len(positions), realization_count)
            paired_distributions: list[tuple[np.ndarray, np.ndarray]] = []
            for realization in realizations:
                apo_input = ProteinMPNNStructureInput(
                    protein_id=str(protein_id),
                    backbone_condition="APO",
                    uniprot_positions=positions,
                    wt_sequence_projection=sequence,
                    coordinates=apo_coordinates,
                    structure_sha256=str(ordered_groups["APO"].iloc[0].get("structure_sha256") or "") or None,
                )
                holo_input = ProteinMPNNStructureInput(
                    protein_id=str(protein_id),
                    backbone_condition="HOLO",
                    uniprot_positions=positions,
                    wt_sequence_projection=sequence,
                    coordinates=holo_coordinates,
                    structure_sha256=str(ordered_groups["HOLO"].iloc[0].get("structure_sha256") or "") or None,
                )
                apo = np.asarray(
                    adapter.probability_distributions(apo_input, (sequence,), realization, batch_size=1)[0],
                    dtype=float,
                )
                holo = np.asarray(
                    adapter.probability_distributions(holo_input, (sequence,), realization, batch_size=1)[0],
                    dtype=float,
                )
                if apo.shape != (len(positions), 20) or holo.shape != apo.shape:
                    raise ApoHoloLocalResponseError("ProteinMPNN distribution shape is invalid")
                if not np.isfinite(apo).all() or not np.isfinite(holo).all():
                    raise ApoHoloLocalResponseError("ProteinMPNN distribution is non-finite")
                paired_distributions.append((apo, holo))
            js_by_repeat = np.asarray(
                [[js_bits(apo[index], holo[index]) for index in range(len(positions))] for apo, holo in paired_distributions],
                dtype=float,
            )
            n_realizations = realization_count
            distributions = (
                np.mean([item[0] for item in paired_distributions], axis=0),
                np.mean([item[1] for item in paired_distributions], axis=0),
            )
            for prefix in (4, 8, 16):
                if prefix <= realization_count:
                    convergence.append(
                        {
                            "protein_id": str(protein_id),
                            "pair_id": str(pair_id),
                            "model": model_name,
                            "realization_count": prefix,
                            "mean_js_bits": float(js_by_repeat[:prefix].mean()),
                        }
                    )
        else:
            apo_missing = not np.isfinite(apo_coordinates[:, :3]).all()
            holo_missing = not np.isfinite(holo_coordinates[:, :3]).all()
            apo = np.asarray(
                adapter.score_teacher_forced(
                    sequence,
                    apo_coordinates[:, :3],
                    allow_missing_coordinates=apo_missing,
                ),
                dtype=float,
            )
            holo = np.asarray(
                adapter.score_teacher_forced(
                    sequence,
                    holo_coordinates[:, :3],
                    allow_missing_coordinates=holo_missing,
                ),
                dtype=float,
            )
            if apo.shape != (len(positions), 20) or holo.shape != apo.shape:
                raise ApoHoloLocalResponseError("ESM-IF1 distribution shape is invalid")
            if not np.isfinite(apo).all() or not np.isfinite(holo).all():
                raise ApoHoloLocalResponseError("ESM-IF1 distribution is non-finite")
            js_by_repeat = np.asarray([[js_bits(apo[index], holo[index]) for index in range(len(positions))]], dtype=float)
            n_realizations = 1
            distributions = (apo, holo)
            convergence.append(
                {
                    "protein_id": str(protein_id),
                    "pair_id": str(pair_id),
                    "model": model_name,
                    "realization_count": 1,
                    "mean_js_bits": float(js_by_repeat.mean()),
                }
            )
        mean_js = js_by_repeat.mean(axis=0)
        median_js = np.median(js_by_repeat, axis=0)
        q90_js = np.quantile(js_by_repeat, 0.9, axis=0, method="linear")
        near_zero = np.mean(js_by_repeat <= 1e-12, axis=0)
        for condition, distribution in zip(("APO", "HOLO"), distributions, strict=True):
            for index, position in enumerate(positions):
                position_rows.append(
                    {
                        "protein_id": str(protein_id),
                        "pair_id": str(pair_id),
                        "model": model_name,
                        "condition": condition,
                        "canonical_position": position,
                        "wt_aa": sequence[index],
                        "distribution": tuple(float(value) for value in distribution[index]),
                        "js_bits_mean": float(mean_js[index]),
                        "js_bits_median": float(median_js[index]),
                        "js_bits_q90": float(q90_js[index]),
                        "near_zero_fraction": float(near_zero[index]),
                        "realization_count": n_realizations,
                    }
                )
        protein_rows.append(
            {
                "protein_id": str(protein_id),
                "pair_id": str(pair_id),
                "model": model_name,
                "position_count": len(positions),
                "measurable_position_fraction": 1.0,
                "local_burden_mean": float(mean_js.mean()),
                "local_burden_median": float(np.median(mean_js)),
                "local_burden_q90": float(np.quantile(mean_js, 0.9, method="linear")),
                "near_zero_position_fraction": float(np.mean(mean_js <= 1e-12)),
                "realization_count": n_realizations,
            }
        )
    position_table = pd.DataFrame(position_rows).sort_values(
        ["protein_id", "canonical_position", "condition"], kind="mergesort"
    ).reset_index(drop=True)
    protein_table = pd.DataFrame(protein_rows).sort_values("protein_id", kind="mergesort").reset_index(drop=True)
    metadata = {
        "model": model_name,
        "protocol": LOCAL_RESPONSE_PROTOCOL,
        "realization_count": realization_count if model_name == "ProteinMPNN" else 1,
        "convergence": convergence,
        "binding": _model_binding(adapter),
    }
    return position_table, protein_table, metadata


def summarize_cross_model(
    protein_mpnn: pd.DataFrame,
    protein_if1: pd.DataFrame,
    position_mpnn: pd.DataFrame,
    position_if1: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare model-native response burden at protein and residue levels."""
    required_protein = {"protein_id", "local_burden_mean"}
    if not required_protein.issubset(protein_mpnn) or not required_protein.issubset(protein_if1):
        raise ApoHoloLocalResponseError("protein response tables lack local burden columns")
    protein = protein_mpnn[["protein_id", "local_burden_mean"]].rename(
        columns={"local_burden_mean": "proteinmpnn_local_burden"}
    ).merge(
        protein_if1[["protein_id", "local_burden_mean"]].rename(
            columns={"local_burden_mean": "esm_if1_local_burden"}
        ),
        on="protein_id",
        how="inner",
        validate="one_to_one",
    )
    if protein.empty:
        raise ApoHoloLocalResponseError("model protein cohorts do not overlap")
    valid = protein[["proteinmpnn_local_burden", "esm_if1_local_burden"]].dropna()
    protein_spearman = (
        float(spearmanr(valid.iloc[:, 0], valid.iloc[:, 1]).statistic)
        if len(valid) >= 3
        else None
    )
    def _one_row_per_position(table: pd.DataFrame, name: str) -> pd.DataFrame:
        required = {"protein_id", "pair_id", "canonical_position", "condition", "js_bits_mean"}
        if not required.issubset(table):
            raise ApoHoloLocalResponseError(f"{name} position table lacks required keys")
        rows = table.loc[table["condition"].astype(str).eq("APO")].copy()
        key = ["protein_id", "pair_id", "canonical_position"]
        if rows.duplicated(key).any():
            raise ApoHoloLocalResponseError(f"{name} position keys are not unique")
        return rows[key + ["js_bits_mean"]].rename(columns={"js_bits_mean": f"{name}_js_bits"})
    mpnn_position = _one_row_per_position(position_mpnn, "proteinmpnn")
    if1_position = _one_row_per_position(position_if1, "esm_if1")
    position = mpnn_position.merge(
        if1_position,
        on=["protein_id", "pair_id", "canonical_position"],
        how="inner",
        validate="one_to_one",
    )
    residue_correlations: list[dict[str, Any]] = []
    for (protein_id, pair_id), group in position.groupby(["protein_id", "pair_id"], sort=True):
        valid_group = group[["proteinmpnn_js_bits", "esm_if1_js_bits"]].dropna()
        residue_correlations.append(
            {
                "protein_id": str(protein_id),
                "pair_id": str(pair_id),
                "position_count": len(valid_group),
                "within_protein_spearman": (
                    float(spearmanr(valid_group.iloc[:, 0], valid_group.iloc[:, 1]).statistic)
                    if len(valid_group) >= 3
                    else None
                ),
            }
        )
    metadata = {
        "protein_count": len(protein),
        "shared_position_count": len(position),
        "protein_level_spearman": protein_spearman,
        "within_protein_residue_correlations": residue_correlations,
    }
    return protein, metadata


def join_structural_associations(position_table: pd.DataFrame, release_root: Path) -> pd.DataFrame:
    """Join local response to released geometry and ligand annotations by exact keys."""
    required = {"protein_id", "pair_id", "canonical_position", "condition", "js_bits_mean"}
    if not required.issubset(position_table):
        raise ApoHoloLocalResponseError("position table lacks association keys")
    local = position_table.loc[position_table["condition"].astype(str).eq("APO")].copy()
    key = ["protein_id", "pair_id", "canonical_position"]
    if local.duplicated(key).any():
        raise ApoHoloLocalResponseError("local position association keys are not unique")
    root = Path(release_root)
    geometry = pd.read_parquet(root / "residue_structural_descriptors.parquet")
    pair_geometry = pd.read_parquet(root / "pair_structural_descriptors.parquet")
    geometry = geometry.loc[geometry["geometry_status"].astype(str).eq("available")].copy()
    geometry = geometry.drop_duplicates(key, keep=False)
    merged = local.merge(geometry, on=key, how="inner", validate="one_to_one")
    if merged.empty:
        return merged.assign(ligand_status=pd.Series(dtype="object"))
    pair_lookup = pair_geometry.set_index("pair_id")
    ligand_status: list[str] = []
    for row in merged.itertuples(index=False):
        pair = pair_lookup.loc[str(row.pair_id)] if str(row.pair_id) in pair_lookup.index else None
        if bool(getattr(row, "ligand_proximal", False)):
            ligand_status.append("proximal")
        elif pair is not None and float(getattr(pair, "ligand_proximal_count", 0) or 0) == 0:
            ligand_status.append("explicit_zero_proximal")
        else:
            ligand_status.append("distal_or_nonproximal")
    merged["ligand_status"] = ligand_status
    return merged.sort_values(key, kind="mergesort").reset_index(drop=True)


def summarize_associations(association_table: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive model-response contrasts by geometry/ligand labels."""
    required = {"protein_id", "js_bits_mean", "ligand_status", "local_pairwise_distance_change"}
    if not required.issubset(association_table):
        raise ApoHoloLocalResponseError("association table lacks descriptive fields")
    rows: list[dict[str, Any]] = []
    for status, group in association_table.groupby("ligand_status", sort=True):
        rows.append(
            {
                "group": str(status),
                "position_count": len(group),
                "protein_count": int(group["protein_id"].nunique()),
                "js_bits_mean": float(group["js_bits_mean"].mean()),
                "local_pairwise_distance_change_mean": float(
                    group["local_pairwise_distance_change"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def _immutable_file(path: Path, payload: bytes) -> str:
    if path.exists():
        if path.read_bytes() != payload:
            raise ApoHoloLocalResponseError(f"immutable output conflict: {path}")
        return "reused_identical"
    atomic_write_new_bytes(path, payload)
    return "created"


def materialize_model_local_response(
    position_table: pd.DataFrame,
    protein_table: pd.DataFrame,
    metadata: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    """Materialize compact model response outputs with immutable semantics."""
    output_root = Path(output_root)
    outputs = {
        "position": output_root / "position_local_response.parquet",
        "protein": output_root / "protein_local_response.parquet",
        "summary": output_root / "summary.json",
    }
    payloads = {
        "position": _parquet_bytes(position_table),
        "protein": _parquet_bytes(protein_table),
        "summary": (json.dumps(
            {
                **metadata,
                "position_rows": len(position_table),
                "protein_rows": len(protein_table),
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ) + "\n").encode("utf-8"),
    }
    statuses = {name: _immutable_file(outputs[name], payloads[name]) for name in outputs}
    manifest = {
        "schema_version": "dual-uq.apo-holo-local-response-model-manifest.v1",
        "model": metadata.get("model"),
        "protocol": LOCAL_RESPONSE_PROTOCOL,
        "outputs": {
            name: {"path": path.name, "sha256": sha256_bytes(payloads[name]), "rows": len(position_table if name == "position" else protein_table) if name != "summary" else None}
            for name, path in outputs.items()
        },
    }
    manifest_payload = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    statuses["manifest"] = _immutable_file(output_root / "manifest.json", manifest_payload)
    manifest["write_status"] = statuses
    manifest["manifest_sha256"] = sha256_bytes(manifest_payload)
    return manifest
