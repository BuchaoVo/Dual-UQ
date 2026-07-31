#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}
STRUCTURE_SUFFIXES = {".cif", ".mmcif", ".pdb"}
REQUIRED_BACKBONE_DEFAULT = ("N", "CA", "C", "O")


def json_walk(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from json_walk(child, child_prefix)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            yield from json_walk(child, child_prefix)
    else:
        yield prefix, value


def clean_token(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    return "" if text in {"", ".", "?", "nan", "None"} else text


def clean_int_token(value: Any) -> int | None:
    text = clean_token(value)
    if not text:
        return None
    try:
        number = float(text)
        if not math.isfinite(number) or int(number) != number:
            return None
        return int(number)
    except ValueError:
        match = re.fullmatch(r"(-?\d+)", text)
        return int(match.group(1)) if match else None


def parse_pair_name(pair_dir: Path) -> dict[str, str | None]:
    match = re.fullmatch(
        r"(?P<pdb>[0-9A-Za-z]{4})_(?P<chain>.+?)__(?P<uniprot>.+)",
        pair_dir.name,
    )
    if not match:
        return {"pdb_id": None, "pdb_chain": None, "uniprot_id": None}
    return {
        "pdb_id": match.group("pdb").lower(),
        "pdb_chain": match.group("chain"),
        "uniprot_id": match.group("uniprot"),
    }


def discover_structure_candidates(
    project_root: Path,
    pair_dir: Path,
    v2a_metrics: dict[str, Any],
) -> dict[str, list[Path]]:
    identity = parse_pair_name(pair_dir)
    pdb_id = identity["pdb_id"]
    uniprot = identity["uniprot_id"]
    candidates: set[Path] = set()

    for hit in v2a_metrics.get("source_path_hits", []):
        value = hit.get("value")
        if not isinstance(value, str):
            continue
        path = Path(value)
        if not path.is_absolute():
            path = project_root / path
        if path.is_file() and path.suffix.lower() in STRUCTURE_SUFFIXES:
            candidates.add(path.resolve())

    for filename in ("pair_qc.json", "pair_geometry_qc.json", "robust_pair_diagnostics.json"):
        path = pair_dir / filename
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for _, value in json_walk(data):
            if not isinstance(value, str):
                continue
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = project_root / candidate
            if candidate.is_file() and candidate.suffix.lower() in STRUCTURE_SUFFIXES:
                candidates.add(candidate.resolve())

    if pdb_id:
        raw_pdb = project_root / "data/raw/pdb"
        for suffix in (".cif", ".mmcif", ".pdb"):
            path = raw_pdb / f"{pdb_id}{suffix}"
            if path.is_file():
                candidates.add(path.resolve())
            path_upper = raw_pdb / f"{pdb_id.upper()}{suffix}"
            if path_upper.is_file():
                candidates.add(path_upper.resolve())

    if uniprot:
        afdb_root = project_root / "data/raw/afdb" / uniprot
        if afdb_root.is_dir():
            for path in afdb_root.rglob("*"):
                if path.is_file() and path.suffix.lower() in STRUCTURE_SUFFIXES:
                    candidates.add(path.resolve())

    pdb_candidates = []
    afdb_candidates = []
    unknown = []

    for path in sorted(candidates):
        lower = str(path).lower()
        filename = path.name.lower()
        if "/afdb/" in lower or filename.startswith("af-"):
            afdb_candidates.append(path)
        elif "/pdb/" in lower or (pdb_id and filename.startswith(pdb_id)):
            pdb_candidates.append(path)
        else:
            unknown.append(path)

    return {
        "pdb": pdb_candidates,
        "afdb": afdb_candidates,
        "unknown": unknown,
    }


def normalize_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def parse_mmcif(path: Path) -> dict[str, Any]:
    try:
        from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    except ImportError as exc:
        raise RuntimeError(
            "Biopython is required to parse mmCIF in V2B1."
        ) from exc

    raw = MMCIF2Dict(str(path))

    def get(key: str, default: str = "?") -> list[Any]:
        value = raw.get(key)
        if value is None:
            n = len(normalize_list(raw["_atom_site.group_PDB"]))
            return [default] * n
        return normalize_list(value)

    group = get("_atom_site.group_PDB")
    atom_name = get("_atom_site.label_atom_id")
    comp_id = get("_atom_site.label_comp_id")
    auth_chain = get("_atom_site.auth_asym_id")
    auth_seq = get("_atom_site.auth_seq_id")
    label_chain = get("_atom_site.label_asym_id")
    label_seq = get("_atom_site.label_seq_id")
    insertion = get("_atom_site.pdbx_PDB_ins_code")
    model_num = get("_atom_site.pdbx_PDB_model_num", "1")
    x = get("_atom_site.Cartn_x")
    y = get("_atom_site.Cartn_y")
    z = get("_atom_site.Cartn_z")
    occupancy = get("_atom_site.occupancy", "1.0")
    bfactor = get("_atom_site.B_iso_or_equiv", "0.0")

    n = len(group)
    lengths = {
        len(atom_name), len(comp_id), len(auth_chain), len(auth_seq),
        len(label_chain), len(label_seq), len(insertion), len(model_num),
        len(x), len(y), len(z), len(occupancy), len(bfactor),
    }
    if lengths != {n}:
        raise ValueError(f"Inconsistent _atom_site column lengths in {path}: {lengths}")

    atoms = []
    for index in range(n):
        if clean_token(model_num[index]) not in {"", "1"}:
            continue
        atoms.append({
            "group": clean_token(group[index]),
            "atom_name": clean_token(atom_name[index]).upper(),
            "comp_id": clean_token(comp_id[index]).upper(),
            "auth_chain": clean_token(auth_chain[index]),
            "auth_seq": clean_int_token(auth_seq[index]),
            "label_chain": clean_token(label_chain[index]),
            "label_seq": clean_int_token(label_seq[index]),
            "icode": clean_token(insertion[index]),
            "x": float(x[index]),
            "y": float(y[index]),
            "z": float(z[index]),
            "occupancy": float(occupancy[index]),
            "bfactor": float(bfactor[index]),
        })
    return build_structure_index(path, atoms)


def parse_pdb(path: Path) -> dict[str, Any]:
    atoms = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            altloc = line[16].strip()
            if altloc not in {"", "A"}:
                continue
            atom_name = line[12:16].strip().upper()
            comp_id = line[17:20].strip().upper()
            chain = line[21].strip()
            seq = clean_int_token(line[22:26].strip())
            icode = line[26].strip()
            atoms.append({
                "group": line[:6].strip(),
                "atom_name": atom_name,
                "comp_id": comp_id,
                "auth_chain": chain,
                "auth_seq": seq,
                "label_chain": chain,
                "label_seq": seq,
                "icode": icode,
                "x": float(line[30:38]),
                "y": float(line[38:46]),
                "z": float(line[46:54]),
                "occupancy": float(line[54:60] or 1.0),
                "bfactor": float(line[60:66] or 0.0),
            })
    return build_structure_index(path, atoms)


def build_structure_index(path: Path, atoms: list[dict[str, Any]]) -> dict[str, Any]:
    auth_residues: dict[tuple[str, int, str], dict[str, Any]] = {}
    label_residues: dict[tuple[str, int], dict[str, Any]] = {}

    for atom in atoms:
        if atom["auth_seq"] is not None:
            key = (atom["auth_chain"], atom["auth_seq"], atom["icode"])
            residue = auth_residues.setdefault(key, {
                "chain": atom["auth_chain"],
                "seq": atom["auth_seq"],
                "icode": atom["icode"],
                "comp_id": atom["comp_id"],
                "aa": AA3_TO_1.get(atom["comp_id"], "X"),
                "atoms": {},
            })
            residue["atoms"].setdefault(atom["atom_name"], atom)

        if atom["label_seq"] is not None:
            key = (atom["label_chain"], atom["label_seq"])
            residue = label_residues.setdefault(key, {
                "chain": atom["label_chain"],
                "seq": atom["label_seq"],
                "icode": "",
                "comp_id": atom["comp_id"],
                "aa": AA3_TO_1.get(atom["comp_id"], "X"),
                "atoms": {},
            })
            residue["atoms"].setdefault(atom["atom_name"], atom)

    return {
        "path": str(path),
        "format": path.suffix.lower(),
        "atom_count": len(atoms),
        "auth_residues": auth_residues,
        "label_residues": label_residues,
        "auth_chains": sorted({key[0] for key in auth_residues}),
        "label_chains": sorted({key[0] for key in label_residues}),
        "auth_residue_count": len(auth_residues),
        "label_residue_count": len(label_residues),
    }


def parse_structure(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".pdb":
        return parse_pdb(path)
    return parse_mmcif(path)


def candidate_columns(frame: pd.DataFrame, side: str, mode: str) -> dict[str, list[str]]:
    columns = [str(column) for column in frame.columns]

    def score_column(column: str, kind: str) -> int:
        lower = column.lower()
        score = 0
        if side == "pdb" and "pdb" in lower:
            score += 8
        if side == "afdb" and ("afdb" in lower or lower.startswith("af_")):
            score += 8
        if mode == "auth" and "auth" in lower:
            score += 5
        if mode == "label" and "label" in lower:
            score += 5
        if kind == "chain" and ("chain" in lower or "asym" in lower):
            score += 4
        if kind == "seq" and (
            "seq_id" in lower or "residue_number" in lower
            or "residue_id" in lower or "position" in lower
        ):
            score += 4
        if kind == "icode" and (
            "icode" in lower or "insertion" in lower or "ins_code" in lower
        ):
            score += 4
        if kind == "aa" and (
            "one_letter" in lower or lower.endswith("_aa")
            or "amino_acid" in lower
        ):
            score += 4
        if "uniprot" in lower:
            if side == "afdb" and kind == "seq":
                score += 3
            elif kind == "aa":
                score += 2
        return score

    result = {}
    for kind in ("chain", "seq", "icode", "aa"):
        ranked = sorted(
            (
                (score_column(column, kind), column)
                for column in columns
                if score_column(column, kind) > 0
            ),
            key=lambda item: (-item[0], item[1]),
        )
        result[kind] = [column for _, column in ranked]
    return result


def get_structure_residue(
    structure: dict[str, Any],
    mode: str,
    chain: str,
    seq: int,
    icode: str,
) -> dict[str, Any] | None:
    if mode == "auth":
        return structure["auth_residues"].get((chain, seq, icode))
    return structure["label_residues"].get((chain, seq))


def evaluate_representation(
    frame: pd.DataFrame,
    structure: dict[str, Any],
    side: str,
    mode: str,
    required_atoms: tuple[str, ...],
) -> list[dict[str, Any]]:
    columns = candidate_columns(frame, side, mode)
    residue_index = (
        structure["auth_residues"] if mode == "auth"
        else structure["label_residues"]
    )
    chains = (
        structure["auth_chains"] if mode == "auth"
        else structure["label_chains"]
    )
    if not residue_index:
        return []

    chain_options = columns["chain"][:8]
    if len(chains) == 1:
        chain_options = [None] + chain_options
    seq_options = columns["seq"][:12]
    icode_options = columns["icode"][:6] if mode == "auth" else [None]
    if mode == "auth" and not icode_options:
        icode_options = [None]
    aa_options = columns["aa"][:6] or [None]

    representations = []
    for chain_column in chain_options:
        for seq_column in seq_options:
            for icode_column in icode_options:
                for aa_column in aa_options:
                    matched = complete = aa_compared = aa_matched = duplicate_keys = 0
                    keys_seen = set()
                    for _, row in frame.iterrows():
                        seq = clean_int_token(row.get(seq_column))
                        if seq is None:
                            continue
                        if chain_column is None:
                            chain = chains[0] if len(chains) == 1 else ""
                        else:
                            chain = clean_token(row.get(chain_column))
                        icode = (
                            clean_token(row.get(icode_column))
                            if icode_column is not None else ""
                        )
                        key = (chain, seq, icode) if mode == "auth" else (chain, seq)
                        duplicate_keys += int(key in keys_seen)
                        keys_seen.add(key)
                        residue = get_structure_residue(
                            structure, mode, chain, seq, icode
                        )
                        if residue is None:
                            continue
                        matched += 1
                        complete += int(
                            all(atom in residue["atoms"] for atom in required_atoms)
                        )
                        if aa_column is not None:
                            expected = clean_token(row.get(aa_column)).upper()
                            if len(expected) == 1 and expected != "X":
                                aa_compared += 1
                                aa_matched += int(expected == residue["aa"])

                    if matched == 0:
                        continue
                    representations.append({
                        "side": side,
                        "structure_path": structure["path"],
                        "mode": mode,
                        "chain_column": chain_column,
                        "constant_chain": (
                            chains[0] if chain_column is None and len(chains) == 1
                            else None
                        ),
                        "seq_column": seq_column,
                        "icode_column": icode_column,
                        "aa_column": aa_column,
                        "matched_rows": matched,
                        "complete_backbone_rows": complete,
                        "aa_compared_rows": aa_compared,
                        "aa_matched_rows": aa_matched,
                        "aa_match_fraction": (
                            aa_matched / aa_compared if aa_compared else None
                        ),
                        "duplicate_mapping_keys": duplicate_keys,
                    })

    return sorted(
        representations,
        key=lambda row: (
            -row["complete_backbone_rows"],
            -row["matched_rows"],
            row["duplicate_mapping_keys"],
            -(
                row["aa_match_fraction"]
                if row["aa_match_fraction"] is not None else -1
            ),
            row["mode"],
            str(row["seq_column"]),
        ),
    )


def resolve_row(
    row: pd.Series,
    structure: dict[str, Any],
    representation: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    seq = clean_int_token(row.get(representation["seq_column"]))
    chain = (
        representation["constant_chain"]
        if representation["chain_column"] is None
        else clean_token(row.get(representation["chain_column"]))
    )
    icode = (
        clean_token(row.get(representation["icode_column"]))
        if representation["icode_column"] is not None else ""
    )
    residue = (
        None if seq is None
        else get_structure_residue(
            structure,
            representation["mode"],
            chain,
            seq,
            icode,
        )
    )
    return residue, {
        "chain": chain,
        "seq": seq,
        "icode": icode,
    }


def find_uniprot_column(frame: pd.DataFrame) -> str | None:
    preferred = [
        "uniprot_residue_number",
        "uniprot_position",
        "uniprot_seq_id",
        "uniprot_residue_id",
    ]
    lower_map = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in preferred:
        if candidate in lower_map:
            return lower_map[candidate]
    for column in frame.columns:
        lower = str(column).lower()
        if "uniprot" in lower and (
            "position" in lower or "residue" in lower or "seq" in lower
        ):
            return str(column)
    return None


def find_column(frame: pd.DataFrame, terms: tuple[str, ...]) -> str | None:
    for column in frame.columns:
        lower = str(column).lower()
        if all(term in lower for term in terms):
            return str(column)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit and build a generalized paired-backbone plan."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--screening-index", type=int, default=36)
    parser.add_argument("--min-common-residues", type=int, default=20)
    parser.add_argument("--min-common-fraction", type=float, default=0.90)
    parser.add_argument("--required-backbone-atoms", default="N,CA,C,O")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    v2_root = project_root / "experiments/a0_validation/v2_cross_mechanism"
    v2a_path = v2_root / f"metrics/index{args.screening_index}_high_pae_preflight.json"
    v2a = json.loads(v2a_path.read_text(encoding="utf-8"))
    if v2a.get("preflight_pass") is not True:
        raise RuntimeError("V2A preflight_pass is not true.")

    pair_dir = Path(v2a["resolved_pair_directory"])
    mapping_path = pair_dir / "residue_mapping.parquet"
    geometry_path = pair_dir / "residue_geometry.parquet"
    mapping = pd.read_parquet(mapping_path)
    geometry = pd.read_parquet(geometry_path)
    required_atoms = tuple(
        atom.strip().upper()
        for atom in args.required_backbone_atoms.split(",")
        if atom.strip()
    )

    candidates = discover_structure_candidates(
        project_root,
        pair_dir,
        v2a,
    )
    errors: list[str] = []
    structure_records: dict[str, dict[str, Any]] = {}
    parse_errors = []

    for side in ("pdb", "afdb"):
        for path in candidates[side]:
            try:
                structure_records[str(path)] = parse_structure(path)
            except Exception as exc:
                parse_errors.append({
                    "side": side,
                    "path": str(path),
                    "error": f"{type(exc).__name__}: {exc}",
                })

    all_representations = {"pdb": [], "afdb": []}
    for side in ("pdb", "afdb"):
        for path in candidates[side]:
            structure = structure_records.get(str(path))
            if structure is None:
                continue
            for mode in ("auth", "label"):
                all_representations[side].extend(
                    evaluate_representation(
                        mapping,
                        structure,
                        side,
                        mode,
                        required_atoms,
                    )
                )
        all_representations[side] = sorted(
            all_representations[side],
            key=lambda row: (
                -row["complete_backbone_rows"],
                -row["matched_rows"],
                row["duplicate_mapping_keys"],
                -(
                    row["aa_match_fraction"]
                    if row["aa_match_fraction"] is not None else -1
                ),
                row["structure_path"],
                row["mode"],
            ),
        )

    selected = {}
    for side in ("pdb", "afdb"):
        if not all_representations[side]:
            errors.append(f"No viable {side.upper()} mapping representation.")
            selected[side] = None
            continue
        selected[side] = all_representations[side][0]

        if len(all_representations[side]) > 1:
            first = all_representations[side][0]
            second = all_representations[side][1]
            first_key = (
                first["complete_backbone_rows"],
                first["matched_rows"],
                first["duplicate_mapping_keys"],
                first["aa_match_fraction"],
            )
            second_key = (
                second["complete_backbone_rows"],
                second["matched_rows"],
                second["duplicate_mapping_keys"],
                second["aa_match_fraction"],
            )
            if first_key == second_key and (
                first["structure_path"] != second["structure_path"]
                or first["mode"] != second["mode"]
                or first["seq_column"] != second["seq_column"]
            ):
                errors.append(
                    f"Ambiguous top {side.upper()} representation: "
                    f"{first} versus {second}"
                )

    uniprot_column = find_uniprot_column(mapping)
    if uniprot_column is None:
        errors.append("No UniProt position column found in residue_mapping.parquet.")

    plddt_column = find_column(geometry, ("plddt",))
    ca_distance_column = find_column(geometry, ("aligned", "ca", "distance"))
    geometry_uniprot_column = find_uniprot_column(geometry)

    geometry_lookup = {}
    if geometry_uniprot_column is not None:
        for _, row in geometry.iterrows():
            position = clean_int_token(row.get(geometry_uniprot_column))
            if position is not None:
                geometry_lookup[position] = row

    plan_rows = []
    sequence_mismatches = []
    if selected.get("pdb") and selected.get("afdb") and uniprot_column:
        pdb_structure = structure_records[selected["pdb"]["structure_path"]]
        afdb_structure = structure_records[selected["afdb"]["structure_path"]]

        for source_row_index, row in mapping.iterrows():
            uniprot_position = clean_int_token(row.get(uniprot_column))
            if uniprot_position is None:
                continue
            pdb_residue, pdb_key = resolve_row(
                row, pdb_structure, selected["pdb"]
            )
            afdb_residue, afdb_key = resolve_row(
                row, afdb_structure, selected["afdb"]
            )
            if pdb_residue is None or afdb_residue is None:
                continue
            pdb_complete = all(
                atom in pdb_residue["atoms"] for atom in required_atoms
            )
            afdb_complete = all(
                atom in afdb_residue["atoms"] for atom in required_atoms
            )
            if not (pdb_complete and afdb_complete):
                continue

            canonical_aa = None
            for column in (
                selected["afdb"].get("aa_column"),
                selected["pdb"].get("aa_column"),
            ):
                if column is None:
                    continue
                value = clean_token(row.get(column)).upper()
                if len(value) == 1:
                    canonical_aa = value
                    break
            if canonical_aa is None:
                canonical_aa = afdb_residue["aa"]

            mismatch = (
                pdb_residue["aa"] != afdb_residue["aa"]
                or pdb_residue["aa"] != canonical_aa
                or afdb_residue["aa"] != canonical_aa
            )
            if mismatch:
                sequence_mismatches.append({
                    "uniprot_position": uniprot_position,
                    "pdb_aa": pdb_residue["aa"],
                    "afdb_aa": afdb_residue["aa"],
                    "canonical_aa": canonical_aa,
                })

            geometry_row = geometry_lookup.get(uniprot_position)
            plan_rows.append({
                "source_mapping_row": source_row_index,
                "uniprot_position": uniprot_position,
                "canonical_aa": canonical_aa,
                "pdb_source_aa": pdb_residue["aa"],
                "afdb_source_aa": afdb_residue["aa"],
                "source_sequence_mismatch": mismatch,
                "pdb_mode": selected["pdb"]["mode"],
                "pdb_chain": pdb_key["chain"],
                "pdb_seq_id": pdb_key["seq"],
                "pdb_icode": pdb_key["icode"],
                "afdb_mode": selected["afdb"]["mode"],
                "afdb_chain": afdb_key["chain"],
                "afdb_seq_id": afdb_key["seq"],
                "afdb_icode": afdb_key["icode"],
                "plddt": (
                    float(geometry_row[plddt_column])
                    if geometry_row is not None
                    and plddt_column is not None
                    and pd.notna(geometry_row[plddt_column])
                    else None
                ),
                "aligned_ca_distance": (
                    float(geometry_row[ca_distance_column])
                    if geometry_row is not None
                    and ca_distance_column is not None
                    and pd.notna(geometry_row[ca_distance_column])
                    else None
                ),
            })

    plan = pd.DataFrame(plan_rows)
    if not plan.empty:
        plan = plan.sort_values(
            ["uniprot_position", "source_mapping_row"],
            kind="mergesort",
        ).reset_index(drop=True)
        plan.insert(0, "output_position", np.arange(1, len(plan) + 1))

    mapping_rows = len(mapping)
    common_rows = len(plan)
    common_fraction = common_rows / mapping_rows if mapping_rows else 0.0

    duplicate_uniprot = (
        int(plan["uniprot_position"].duplicated().sum())
        if not plan.empty else 0
    )
    discontinuities = []
    if len(plan) >= 2:
        positions = plan["uniprot_position"].astype(int).to_numpy()
        for left, right in zip(positions[:-1], positions[1:]):
            if right != left + 1:
                discontinuities.append({
                    "left": int(left),
                    "right": int(right),
                    "gap": int(right - left - 1),
                })

    if common_rows < args.min_common_residues:
        errors.append(
            f"Common complete residues={common_rows}, "
            f"minimum={args.min_common_residues}."
        )
    if common_fraction < args.min_common_fraction:
        errors.append(
            f"Common complete fraction={common_fraction:.6f}, "
            f"minimum={args.min_common_fraction:.6f}."
        )
    if duplicate_uniprot:
        errors.append(f"Duplicate UniProt positions in plan: {duplicate_uniprot}.")

    plan_path = (
        v2_root
        / f"manifests/index{args.screening_index}_paired_input_plan.tsv"
    )
    decision_path = (
        v2_root
        / f"manifests/index{args.screening_index}_paired_input_decision.json"
    )
    metrics_path = (
        v2_root
        / f"metrics/index{args.screening_index}_generalized_pair_audit.json"
    )
    markdown_path = (
        v2_root
        / f"V2B1_INDEX{args.screening_index}_GENERALIZED_PAIR_AUDIT.md"
    )

    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(plan_path, sep="\t", index=False)

    decision = {
        "screening_index": args.screening_index,
        "pair_directory": str(pair_dir),
        "pair_identity": parse_pair_name(pair_dir),
        "required_backbone_atoms": list(required_atoms),
        "selected_pdb_representation": selected.get("pdb"),
        "selected_afdb_representation": selected.get("afdb"),
        "mapping_rows": mapping_rows,
        "common_complete_residue_count": common_rows,
        "common_complete_fraction": common_fraction,
        "uniprot_interval": (
            [
                int(plan["uniprot_position"].min()),
                int(plan["uniprot_position"].max()),
            ]
            if not plan.empty else None
        ),
        "uniprot_discontinuities": discontinuities,
        "source_sequence_mismatches": sequence_mismatches,
        "source_sequence_exact_match": len(sequence_mismatches) == 0,
        "source_sequence_identity": (
            (common_rows - len(sequence_mismatches)) / common_rows
            if common_rows else None
        ),
        "duplicate_uniprot_positions": duplicate_uniprot,
        "paired_backbone_mapping_pass": not errors,
        "plan_path": str(plan_path),
    }
    decision_path.write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    audit = {
        **decision,
        "structure_candidates": {
            side: [str(path) for path in candidates[side]]
            for side in candidates
        },
        "structure_parse_errors": parse_errors,
        "structure_summaries": {
            path: {
                key: value
                for key, value in record.items()
                if key not in {"auth_residues", "label_residues"}
            }
            for path, record in structure_records.items()
        },
        "top_pdb_representations": all_representations["pdb"][:10],
        "top_afdb_representations": all_representations["afdb"][:10],
        "mapping_columns": [str(column) for column in mapping.columns],
        "geometry_columns": [str(column) for column in geometry.columns],
        "errors": errors,
        "audit_pass": not errors,
        "next_stage": (
            "V2B2_write_generalized_paired_pdb_inputs"
            if not errors
            else "review_source_model_or_mapping_representation"
        ),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        f"# A0-V2B1 Index {args.screening_index} Generalized Pair Audit",
        "",
        f"- Pair directory: `{pair_dir}`",
        f"- Mapping rows: `{mapping_rows}`",
        f"- Common complete residues: `{common_rows}`",
        f"- Common complete fraction: `{common_fraction}`",
        f"- UniProt interval: `{decision['uniprot_interval']}`",
        f"- UniProt discontinuities: `{len(discontinuities)}`",
        f"- Source-sequence mismatches: `{len(sequence_mismatches)}`",
        f"- Source-sequence identity: `{decision['source_sequence_identity']}`",
        f"- Audit pass: `{not errors}`",
        f"- Next stage: `{audit['next_stage']}`",
        "",
        "## Selected PDB representation",
        "",
        f"`{selected.get('pdb')}`",
        "",
        "## Selected AFDB representation",
        "",
        f"`{selected.get('afdb')}`",
        "",
        "## Sequence mismatches",
        "",
        f"`{sequence_mismatches}`",
        "",
        "## Structure candidates",
        "",
        f"- PDB: `{audit['structure_candidates']['pdb']}`",
        f"- AFDB: `{audit['structure_candidates']['afdb']}`",
        f"- Unknown: `{audit['structure_candidates']['unknown']}`",
        "",
    ]
    if errors:
        lines += ["## Errors", ""]
        lines.extend(f"- {error}" for error in errors)
        lines.append("")

    lines += [
        "## Interpretation boundary",
        "",
        "V2B1只冻结源结构、编号表示和共同主链残基。"
        "它尚未写出ProteinMPNN派生PDB，也不对high-PAE机制强度作新的统计推断。",
        "",
    ]
    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    print("Pair directory:", pair_dir)
    print("PDB candidates:", [str(path) for path in candidates["pdb"]])
    print("AFDB candidates:", [str(path) for path in candidates["afdb"]])
    print("Selected PDB:", selected.get("pdb"))
    print("Selected AFDB:", selected.get("afdb"))
    print("Mapping rows:", mapping_rows)
    print("Common complete residues:", common_rows)
    print("Common complete fraction:", common_fraction)
    print("UniProt interval:", decision["uniprot_interval"])
    print("UniProt discontinuities:", len(discontinuities))
    print("Source-sequence mismatches:", sequence_mismatches)
    print("Errors:", errors)
    print("Audit pass:", not errors)
    print("Next stage:", audit["next_stage"])
    print("Wrote:", plan_path)
    print("Wrote:", decision_path)
    print("Wrote:", metrics_path)
    print("Wrote:", markdown_path)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
