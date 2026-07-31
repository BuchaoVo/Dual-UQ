#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


AA1_TO_3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def clean_token(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    return "" if text in {"", ".", "?", "nan", "None"} else text


def clean_int(value: Any) -> int | None:
    text = clean_token(value)
    if not text:
        return None
    number = float(text)
    if not math.isfinite(number) or int(number) != number:
        return None
    return int(number)


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def altloc_rank(altloc: str) -> int:
    """Tie-break alternate locations after occupancy.

    Blank atoms are shared by all conformers and are preferred on ties,
    followed by conformer A and then other explicit conformers.
    """
    if altloc == "":
        return 2
    if altloc == "A":
        return 1
    return 0


def should_replace_atom(
    previous: dict[str, Any] | None,
    candidate: dict[str, Any],
) -> bool:
    if previous is None:
        return True
    previous_key = (
        float(previous["occupancy"]),
        altloc_rank(clean_token(previous.get("altloc"))),
        clean_token(previous.get("altloc")),
    )
    candidate_key = (
        float(candidate["occupancy"]),
        altloc_rank(clean_token(candidate.get("altloc"))),
        clean_token(candidate.get("altloc")),
    )
    return candidate_key > previous_key


def parse_mmcif(path: Path) -> dict[str, dict[tuple[Any, ...], dict[str, Any]]]:
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict

    raw = MMCIF2Dict(str(path))
    count = len(as_list(raw["_atom_site.group_PDB"]))

    def get(key: str, default: str = "?") -> list[Any]:
        values = as_list(raw[key]) if key in raw else [default] * count
        if len(values) != count:
            raise ValueError(f"{path}: {key} length mismatch.")
        return values

    columns = {
        "group": get("_atom_site.group_PDB"),
        "atom": get("_atom_site.label_atom_id"),
        "comp": get("_atom_site.label_comp_id"),
        "auth_chain": get("_atom_site.auth_asym_id"),
        "auth_seq": get("_atom_site.auth_seq_id"),
        "label_chain": get("_atom_site.label_asym_id"),
        "label_seq": get("_atom_site.label_seq_id"),
        "icode": get("_atom_site.pdbx_PDB_ins_code"),
        "model": get("_atom_site.pdbx_PDB_model_num", "1"),
        "alt": get("_atom_site.label_alt_id", "."),
        "x": get("_atom_site.Cartn_x"),
        "y": get("_atom_site.Cartn_y"),
        "z": get("_atom_site.Cartn_z"),
        "occ": get("_atom_site.occupancy", "1.0"),
        "bfactor": get("_atom_site.B_iso_or_equiv", "0.0"),
    }

    result = {"auth": {}, "label": {}}
    for index in range(count):
        if clean_token(columns["model"][index]) not in {"", "1"}:
            continue
        atom = {
            "name": clean_token(columns["atom"][index]).upper(),
            "altloc": clean_token(columns["alt"][index]),
            "comp": clean_token(columns["comp"][index]).upper(),
            "auth_chain": clean_token(columns["auth_chain"][index]),
            "auth_seq": clean_int(columns["auth_seq"][index]),
            "label_chain": clean_token(columns["label_chain"][index]),
            "label_seq": clean_int(columns["label_seq"][index]),
            "icode": clean_token(columns["icode"][index]),
            "coord": np.asarray([
                float(columns["x"][index]),
                float(columns["y"][index]),
                float(columns["z"][index]),
            ], dtype=float),
            "occupancy": float(columns["occ"][index]),
            "bfactor": float(columns["bfactor"][index]),
        }
        keys = []
        if atom["auth_seq"] is not None:
            keys.append(("auth", (atom["auth_chain"], atom["auth_seq"], atom["icode"])))
        if atom["label_seq"] is not None:
            keys.append(("label", (atom["label_chain"], atom["label_seq"])))
        for mode, key in keys:
            residue = result[mode].setdefault(key, {"atoms": {}})
            previous = residue["atoms"].get(atom["name"])
            if should_replace_atom(previous, atom):
                residue["atoms"][atom["name"]] = atom
    return result


def parse_pdb(path: Path) -> dict[str, dict[tuple[Any, ...], dict[str, Any]]]:
    result = {"auth": {}, "label": {}}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            chain = line[21].strip()
            seq = clean_int(line[22:26])
            if seq is None:
                continue
            atom = {
                "name": line[12:16].strip().upper(),
                "altloc": line[16].strip(),
                "comp": line[17:20].strip().upper(),
                "auth_chain": chain,
                "auth_seq": seq,
                "label_chain": chain,
                "label_seq": seq,
                "icode": line[26].strip(),
                "coord": np.asarray([
                    float(line[30:38]), float(line[38:46]), float(line[46:54])
                ], dtype=float),
                "occupancy": float(line[54:60].strip() or 1.0),
                "bfactor": float(line[60:66].strip() or 0.0),
            }
            for mode, key in (
                ("auth", (chain, seq, atom["icode"])),
                ("label", (chain, seq)),
            ):
                residue = result[mode].setdefault(key, {"atoms": {}})
                previous = residue["atoms"].get(atom["name"])
                if should_replace_atom(previous, atom):
                    residue["atoms"][atom["name"]] = atom
    return result


def parse_structure(path: Path) -> dict[str, dict[tuple[Any, ...], dict[str, Any]]]:
    return parse_pdb(path) if path.suffix.lower() == ".pdb" else parse_mmcif(path)


def source_key(row: pd.Series, side: str) -> tuple[Any, ...]:
    mode = clean_token(row[f"{side}_mode"])
    chain = clean_token(row[f"{side}_chain"])
    seq = clean_int(row[f"{side}_seq_id"])
    icode = clean_token(row.get(f"{side}_icode"))
    if seq is None:
        raise ValueError(f"{side}: missing source seq id.")
    if mode == "auth":
        return chain, seq, icode
    if mode == "label":
        return chain, seq
    raise ValueError(f"Unsupported mapping mode: {mode}")


def atom_line(
    serial: int,
    atom_name: str,
    residue_name: str,
    chain: str,
    residue_number: int,
    coord: np.ndarray,
    bfactor: float,
) -> str:
    atom_field = f" {atom_name:<3s}" if len(atom_name) < 4 else f"{atom_name:>4s}"
    return (
        f"ATOM  {serial:5d} {atom_field} {residue_name:>3s} "
        f"{chain}{residue_number:4d}    "
        f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
        f"{1.00:6.2f}{bfactor:6.2f}          {atom_name[0]:>2s}\n"
    )


def write_pdb(
    output: Path,
    plan: pd.DataFrame,
    source: dict[str, dict[tuple[Any, ...], dict[str, Any]]],
    side: str,
    chain: str,
    atoms: tuple[str, ...],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    lines, provenance, ca = [], [], []
    serial = 1
    for _, row in plan.iterrows():
        position = int(row["output_position"])
        aa = clean_token(row["canonical_aa"]).upper()
        if aa not in AA1_TO_3:
            raise ValueError(f"Unsupported canonical AA {aa!r} at {position}.")
        mode = clean_token(row[f"{side}_mode"])
        key = source_key(row, side)
        residue = source[mode].get(key)
        if residue is None:
            raise KeyError(f"{side}: source residue {key} not found.")
        missing = [name for name in atoms if name not in residue["atoms"]]
        if missing:
            raise ValueError(f"{side}: position {position} missing {missing}.")
        for atom_name in atoms:
            atom = residue["atoms"][atom_name]
            coord = np.asarray(atom["coord"], dtype=float)
            if not np.isfinite(coord).all():
                raise ValueError(f"{side}: non-finite coordinates at {position}.")
            lines.append(atom_line(
                serial, atom_name, AA1_TO_3[aa], chain, position,
                coord, float(atom["bfactor"])
            ))
            serial += 1
        ca.append(np.asarray(residue["atoms"]["CA"]["coord"], dtype=float))
        selected_altlocs = {
            atom_name: clean_token(
                residue["atoms"][atom_name].get("altloc")
            )
            for atom_name in atoms
        }
        provenance.append({
            "side": side,
            "output_position": position,
            "uniprot_position": int(row["uniprot_position"]),
            "canonical_aa": aa,
            "source_mode": mode,
            "source_chain": clean_token(row[f"{side}_chain"]),
            "source_seq_id": int(row[f"{side}_seq_id"]),
            "source_icode": clean_token(row.get(f"{side}_icode")),
            "source_key": repr(key),
            "source_bfactor_ca": float(residue["atoms"]["CA"]["bfactor"]),
            "source_altloc_N": selected_altlocs["N"],
            "source_altloc_CA": selected_altlocs["CA"],
            "source_altloc_C": selected_altlocs["C"],
            "source_altloc_O": selected_altlocs["O"],
            "source_nonblank_altlocs": ",".join(sorted({
                value for value in selected_altlocs.values() if value
            })),
        })
    lines.extend([f"TER   {serial:5d}\n", "END\n"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(lines), encoding="ascii")
    return provenance, np.asarray(ca, dtype=float)


def fit_distances(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mc, tc = mobile.mean(0), target.mean(0)
    x, y = mobile - mc, target - tc
    u, _, vt = np.linalg.svd(x.T @ y)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    translation = tc - mc @ rotation
    transformed = mobile @ rotation + translation
    return np.linalg.norm(transformed - target, axis=1), rotation, translation


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(values.min()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(values.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--screening-index", type=int, default=36)
    parser.add_argument("--output-chain", default="A")
    parser.add_argument("--required-backbone-atoms", default="N,CA,C,O")
    parser.add_argument("--high-confidence-threshold", type=float, default=70.0)
    args = parser.parse_args()

    project = args.project_root.resolve()
    v2 = project / "experiments/a0_validation/v2_cross_mechanism"
    plan_path = v2 / f"manifests/index{args.screening_index}_paired_input_plan.tsv"
    decision_path = v2 / f"manifests/index{args.screening_index}_paired_input_decision.json"
    plan = pd.read_csv(plan_path, sep="\t").sort_values("output_position")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("paired_backbone_mapping_pass") is not True:
        raise RuntimeError("V2B1 mapping decision did not pass.")

    expected = np.arange(1, len(plan) + 1)
    if not np.array_equal(plan["output_position"].astype(int).to_numpy(), expected):
        raise ValueError("Output positions are not continuous.")
    if plan["uniprot_position"].duplicated().any():
        raise ValueError("UniProt positions are duplicated.")

    atoms = tuple(x.strip().upper() for x in args.required_backbone_atoms.split(","))
    if atoms != ("N", "CA", "C", "O"):
        raise ValueError(f"Unexpected atom set: {atoms}")

    pdb_source_path = Path(decision["selected_pdb_representation"]["structure_path"])
    afdb_source_path = Path(decision["selected_afdb_representation"]["structure_path"])
    pdb_source = parse_structure(pdb_source_path)
    afdb_source = parse_structure(afdb_source_path)

    out_dir = v2 / f"inputs/index{args.screening_index}/paired_pdb"
    pdb_out = out_dir / f"index{args.screening_index}_pdb_mapped.pdb"
    afdb_out = out_dir / f"index{args.screening_index}_afdb_mapped.pdb"

    pdb_prov, pdb_ca = write_pdb(
        pdb_out, plan, pdb_source, "pdb", args.output_chain, atoms
    )
    afdb_prov, afdb_ca = write_pdb(
        afdb_out, plan, afdb_source, "afdb", args.output_chain, atoms
    )

    provenance_path = v2 / f"manifests/index{args.screening_index}_derived_backbone_residues.tsv"
    pd.DataFrame(pdb_prov + afdb_prov).to_csv(
        provenance_path, sep="\t", index=False
    )

    all_distances, _, _ = fit_distances(afdb_ca, pdb_ca)
    all_rmsd = float(np.sqrt(np.mean(np.square(all_distances))))

    plddt = pd.to_numeric(plan["plddt"], errors="coerce").to_numpy(float)
    high_mask = np.isfinite(plddt) & (plddt >= args.high_confidence_threshold)
    high_metrics = None
    core_fit_all = None
    if int(high_mask.sum()) >= 3:
        core_distances, rotation, translation = fit_distances(
            afdb_ca[high_mask], pdb_ca[high_mask]
        )
        core_fit_all = np.linalg.norm(
            afdb_ca @ rotation + translation - pdb_ca, axis=1
        )
        high_metrics = {
            "threshold": args.high_confidence_threshold,
            "residue_count": int(high_mask.sum()),
            "core_rmsd": float(np.sqrt(np.mean(np.square(core_distances)))),
            "all_residue_distance_under_core_fit": summarize(core_fit_all),
        }

    comparison = None
    planned = pd.to_numeric(
        plan["aligned_ca_distance"], errors="coerce"
    ).to_numpy(float)
    valid = np.isfinite(planned)
    if core_fit_all is not None and int(valid.sum()) >= 3:
        observed, expected_values = core_fit_all[valid], planned[valid]
        comparison = {
            "compared_residues": int(valid.sum()),
            "pearson_r": (
                float(np.corrcoef(observed, expected_values)[0, 1])
                if np.std(observed) > 0 and np.std(expected_values) > 0
                else None
            ),
            "median_abs_delta": float(np.median(np.abs(observed - expected_values))),
            "max_abs_delta": float(np.max(np.abs(observed - expected_values))),
        }

    sequence = "".join(plan["canonical_aa"].astype(str))
    result = {
        "screening_index": args.screening_index,
        "residue_count": len(plan),
        "atom_count_per_structure": len(plan) * len(atoms),
        "canonical_sequence": sequence,
        "canonical_sequence_length": len(sequence),
        "source_sequence_mismatch_count": int(
            plan["source_sequence_mismatch"].astype(bool).sum()
        ),
        "source_pdb": str(pdb_source_path),
        "source_afdb": str(afdb_source_path),
        "derived_pdb": str(pdb_out),
        "derived_afdb": str(afdb_out),
        "provenance_table": str(provenance_path),
        "all_residue_ca_kabsch_rmsd": all_rmsd,
        "all_residue_ca_distance": summarize(all_distances),
        "high_confidence_fit": high_metrics,
        "planned_aligned_distance_comparison": comparison,
        "build_pass": (
            len(sequence) == len(plan)
            and len(pdb_prov) == len(afdb_prov) == len(plan)
            and pdb_out.is_file() and afdb_out.is_file()
        ),
    }

    metrics = v2 / f"metrics/index{args.screening_index}_paired_pdb_build.json"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Residues:", len(plan))
    print("Atoms per structure:", result["atom_count_per_structure"])
    print("Source mismatches:", result["source_sequence_mismatch_count"])
    print("All-residue CA Kabsch RMSD:", all_rmsd)
    print("High-confidence fit:", high_metrics)
    print("Planned distance comparison:", comparison)
    print("Build pass:", result["build_pass"])
    print("Wrote:", pdb_out)
    print("Wrote:", afdb_out)
    print("Wrote:", provenance_path)
    print("Wrote:", metrics)
    return 0 if result["build_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
