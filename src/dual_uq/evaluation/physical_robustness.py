"""EvoEF2 physical sequence--structure compatibility evaluation.

This module is deliberately independent of ProteinMPNN.  It consumes the
frozen common-mask generation records and evaluates each requested sequence on
both paired structural representations using one explicit EvoEF2 protocol.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from dual_uq.dataset.models import BACKBONE_ATOM_NAMES, group_residue_records, select_backbone_atoms
from dual_uq.dataset.stages.derivation import _load_atom_records

_ONE_TO_THREE = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}
_THREE_TO_ONE = {value: key for key, value in _ONE_TO_THREE.items()}
_ENERGY_LINE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*([-+0-9.eE]+)\s*$")
EVOEF2_REVISION = "38df01d305ed728ef067c3e0d22072058f33e255"
EVOEF2_REPAIR_RUNS = 3
EVOEF2_ROTAMER_LIBRARY = "bbdep3per.lib"
EVOEF2_WEIGHT_FILE = "weight_EvoEF2.txt"
ESMFOLD_SAMPLE_INDICES = (0, 32, 64, 96, 128, 160, 192, 224)
EVOEF2_COMMAND_TIMEOUT_SECONDS = 120


class PhysicalEvaluationError(RuntimeError):
    """Raised when the physical protocol cannot be applied without fallback."""


@dataclass(frozen=True)
class EvoEF2Protocol:
    executable: Path
    source_revision: str = EVOEF2_REVISION
    repair_runs: int = EVOEF2_REPAIR_RUNS
    rotamer_library: str = EVOEF2_ROTAMER_LIBRARY
    weight_file: str = EVOEF2_WEIGHT_FILE

    def __post_init__(self) -> None:
        executable = Path(self.executable).expanduser().resolve()
        if not executable.is_file() or not executable.stat().st_mode & 0o111:
            raise PhysicalEvaluationError(f"EvoEF2 executable is unavailable: {executable}")
        if self.repair_runs <= 0:
            raise PhysicalEvaluationError("repair_runs must be positive")
        object.__setattr__(self, "executable", executable)

    @property
    def executable_sha256(self) -> str:
        digest = hashlib.sha256()
        with self.executable.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


@dataclass(frozen=True)
class PhysicalCondition:
    """One frozen common-mask structural input."""

    protein_id: str
    condition: str
    source_path: Path
    source_sha256: str
    source_chain_id: str
    projection_coordinates: Any
    wt_sequence: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_physical_conditions(project_root: Path, protein_ids: Iterable[str]) -> tuple[PhysicalCondition, ...]:
    """Load exact frozen PDB/AFDB projections without recomputing mappings."""
    from dual_uq.evaluation.operational_pairs import load_operational_conditions

    root = Path(project_root).expanduser().resolve()
    requested = tuple(dict.fromkeys(str(value) for value in protein_ids))
    if not requested:
        raise PhysicalEvaluationError("at least one protein is required")
    try:
        frozen = load_operational_conditions(root, protein_ids=requested)
    except ValueError as exc:
        raise PhysicalEvaluationError(str(exc)) from exc
    return tuple(
        PhysicalCondition(
            protein_id=item.protein_id,
            condition=item.condition,
            source_path=item.source_path,
            source_sha256=item.source_sha256,
            source_chain_id=item.source_chain_id,
            projection_coordinates=item.coordinates,
            wt_sequence=item.wt_sequence_projection,
        )
        for item in frozen
    )


def parse_energy_terms(stdout: str) -> dict[str, float]:
    """Parse EvoEF2's named energy terms, including ``Total``."""
    terms: dict[str, float] = {}
    for line in stdout.splitlines():
        match = _ENERGY_LINE.match(line)
        if match:
            terms[match.group(1)] = float(match.group(2))
    if "Total" not in terms:
        raise PhysicalEvaluationError("EvoEF2 output did not contain Total energy")
    if len(terms) < 2:
        raise PhysicalEvaluationError("EvoEF2 output did not contain energy components")
    return terms


def _coord_key(coordinates: Iterable[float]) -> bytes:
    import numpy as np

    return np.asarray(tuple(coordinates), dtype="<f4").tobytes()


@lru_cache(maxsize=256)
def _backbone_selections(source_path: str, source_id: str) -> tuple[tuple[bytes, Any], ...]:
    """Parse one source structure once per preparation process."""
    records = _load_atom_records(Path(source_path), source_id)
    by_coordinates: dict[bytes, Any] = {}
    for group in group_residue_records(records):
        selection = select_backbone_atoms(group.atoms)
        if selection.missing_atoms:
            continue
        key = _coord_key(selection.atom("CA").coordinates)
        if key in by_coordinates:
            raise PhysicalEvaluationError("source coordinate binding is ambiguous")
        by_coordinates[key] = selection
    return tuple(by_coordinates.items())


def _pdb_atom_line(serial: int, atom_name: str, residue_name: str, residue_number: int,
                   insertion_code: str, x: float, y: float, z: float, element: str) -> str:
    if not 1 <= residue_number <= 9999:
        raise PhysicalEvaluationError("mapped residue number cannot be represented in PDB")
    if not isinstance(insertion_code, str) or len(insertion_code) > 1:
        raise PhysicalEvaluationError("mapped insertion code cannot be represented in PDB")
    insertion_code = insertion_code or " "
    return (
        f"ATOM  {serial:5d} {atom_name:^4s} {residue_name:>3s} A"
        f"{residue_number:4d}{insertion_code}   {x:8.3f}{y:8.3f}{z:8.3f}"
        f"{1.00:6.2f}{0.00:6.2f}          {element:>2s}"
    )


def render_common_mask_pdb(
    source_path: Path,
    source_id: str,
    projection_coordinates: Any,
    target_sequence: str,
) -> bytes:
    """Render an exact common-mask backbone with the requested sequence.

    The source chain identifier is retained in provenance by the caller, while
    the single-chain PDB transport uses chain ``A`` (the legacy PDB format has
    one character for this field).  Residue numbering and insertion codes are
    not compressed; segment breaks are emitted as ``TER`` records.
    """
    if not isinstance(target_sequence, str) or not target_sequence:
        raise PhysicalEvaluationError("target sequence is required")
    if any(letter not in _ONE_TO_THREE for letter in target_sequence):
        raise PhysicalEvaluationError("target sequence contains unsupported amino acids")
    by_coordinates = dict(_backbone_selections(str(Path(source_path).resolve()), source_id))
    if len(projection_coordinates) != len(target_sequence):
        raise PhysicalEvaluationError("projection and target sequence lengths differ")
    selected = []
    for row in projection_coordinates:
        # Projection coordinates are four backbone atoms; bind by CA, then
        # verify every atom against the same source residue.
        try:
            ca_key = _coord_key(row[1])
        except (IndexError, TypeError):
            raise PhysicalEvaluationError("projection coordinates are malformed") from None
        selection = by_coordinates.get(ca_key)
        if selection is None:
            raise PhysicalEvaluationError("projection coordinate is absent from source")
        selected.append(selection)
    lines: list[str] = []
    serial = 1
    previous_number: int | None = None
    for residue_index, (selection, letter) in enumerate(zip(selected, target_sequence, strict=True)):
        residue_number = selection.residue.key.auth_seq_id
        if previous_number is not None and residue_number - previous_number > 1:
            lines.append("TER")
        for atom_name in BACKBONE_ATOM_NAMES:
            atom = selection.atom(atom_name)
            lines.append(_pdb_atom_line(
                serial, atom_name, _ONE_TO_THREE[letter], residue_number,
                selection.residue.key.insertion_code,
                atom.x, atom.y, atom.z, (atom.element or atom_name[0]).upper(),
            ))
            serial += 1
        previous_number = residue_number
    lines.extend(("TER", "END"))
    return ("\n".join(lines) + "\n").encode("ascii")


def _repaired_sequence(path: Path) -> str:
    residues: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise PhysicalEvaluationError("unable to read EvoEF2 repaired PDB") from exc
    for line in lines:
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 27:
            continue
        residue = (line[21:22], line[22:26].strip(), line[26:27])
        if residue in seen:
            continue
        seen.add(residue)
        code = _THREE_TO_ONE.get(line[17:20].strip().upper())
        if code is None:
            raise PhysicalEvaluationError("EvoEF2 produced an unsupported residue")
        residues.append((residue[0], residue[1], code))
    if not residues:
        raise PhysicalEvaluationError("EvoEF2 repaired PDB has no protein residues")
    return "".join(item[2] for item in residues)


def _pdb_residue_records(path: Path) -> list[tuple[str, str, str, str]]:
    """Return ordered chain/author-residue/icode/reference-AA records."""
    records: list[tuple[str, str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise PhysicalEvaluationError("unable to read EvoEF2 mutation input") from exc
    for line in lines:
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 27:
            continue
        key = (line[21:22], line[22:26].strip(), line[26:27])
        if key in seen:
            continue
        seen.add(key)
        code = _THREE_TO_ONE.get(line[17:20].strip().upper())
        if code is None:
            raise PhysicalEvaluationError("mutation input contains an unsupported residue")
        records.append((key[0], key[1], key[2], code))
    if not records:
        raise PhysicalEvaluationError("mutation input has no protein residues")
    return records


def _mutation_spec_from_pdb(path: Path, target_sequence: str) -> str:
    """Build EvoEF2's exact multi-mutant line from a WT PDB and target sequence."""
    records = _pdb_residue_records(path)
    if len(records) != len(target_sequence):
        raise PhysicalEvaluationError("WT and target sequence lengths differ for BuildMutant")
    mutations: list[str] = []
    for (chain, number, insertion, reference), target in zip(records, target_sequence, strict=True):
        if target not in _ONE_TO_THREE:
            raise PhysicalEvaluationError("target sequence contains unsupported amino acids")
        if target == reference:
            continue
        if insertion.strip():
            raise PhysicalEvaluationError(
                "BuildMutant cannot represent an insertion-code residue without fallback"
            )
        mutations.append(f"{reference}{chain}{number}{target}")
    if not mutations:
        raise PhysicalEvaluationError("BuildMutant has no sequence changes to apply")
    return ",".join(mutations) + ";"


def build_mutant(
    protocol: EvoEF2Protocol,
    wt_input_pdb: Path,
    target_sequence: str,
    workdir: Path,
) -> Path:
    """Build one complete target sequence with EvoEF2's BuildMutant command."""
    workdir.mkdir(parents=True, exist_ok=True)
    source = Path(wt_input_pdb).resolve()
    local_input = workdir / source.name
    local_input.write_bytes(source.read_bytes())
    mutation_file = workdir / "mutants.txt"
    mutation_file.write_text(_mutation_spec_from_pdb(local_input, target_sequence) + "\n", encoding="ascii")
    try:
        result = subprocess.run(
            [
                str(protocol.executable), "--command=BuildMutant",
                "--pdb", local_input.name, "--mutant_file", mutation_file.name,
            ],
            cwd=workdir, capture_output=True, text=True, check=False,
            timeout=EVOEF2_COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise PhysicalEvaluationError("EvoEF2 BuildMutant timed out") from exc
    (workdir / "build_mutant.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise PhysicalEvaluationError(f"EvoEF2 BuildMutant failed: {result.stderr[-500:]}")
    built = workdir / f"{local_input.stem}_Model_0001.pdb"
    if not built.is_file():
        raise PhysicalEvaluationError("EvoEF2 did not produce the expected BuildMutant model")
    if _repaired_sequence(built) != target_sequence:
        raise PhysicalEvaluationError("BuildMutant did not reproduce the requested sequence")
    return built


def repair_structure(
    protocol: EvoEF2Protocol,
    input_pdb: Path,
    workdir: Path,
) -> Path:
    """Repair one input and return the immutable repaired structure."""
    workdir.mkdir(parents=True, exist_ok=True)
    input_pdb = Path(input_pdb).resolve()
    local_input = workdir / input_pdb.name
    local_input.write_bytes(input_pdb.read_bytes())
    try:
        repair = subprocess.run(
            [str(protocol.executable), "--command=RepairStructure", "--pdb", local_input.name],
            cwd=workdir, capture_output=True, text=True, check=False,
            timeout=EVOEF2_COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise PhysicalEvaluationError("EvoEF2 RepairStructure timed out") from exc
    (workdir / "repair.log").write_text(repair.stdout + repair.stderr, encoding="utf-8")
    if repair.returncode != 0:
        raise PhysicalEvaluationError(f"EvoEF2 RepairStructure failed: {repair.stderr[-500:]}")
    repaired = workdir / f"{local_input.stem}_Repair.pdb"
    if not repaired.is_file():
        raise PhysicalEvaluationError("EvoEF2 did not produce a repaired PDB")
    return repaired


def evaluate_structure(
    protocol: EvoEF2Protocol,
    input_pdb: Path,
    workdir: Path,
) -> tuple[dict[str, float], Path]:
    """Repair one input then score the immutable repaired output."""
    repaired = repair_structure(protocol, input_pdb, workdir)
    try:
        stability = subprocess.run(
            [str(protocol.executable), "--command=ComputeStability", "--pdb", repaired.name],
            cwd=workdir, capture_output=True, text=True, check=False,
            timeout=EVOEF2_COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise PhysicalEvaluationError("EvoEF2 ComputeStability timed out") from exc
    (workdir / "compute_stability.log").write_text(
        stability.stdout + stability.stderr, encoding="utf-8"
    )
    if stability.returncode != 0:
        raise PhysicalEvaluationError(f"EvoEF2 ComputeStability failed: {stability.stderr[-500:]}")
    return parse_energy_terms(stability.stdout), repaired


def evaluate_records(
    *,
    project_root: Path,
    protocol: EvoEF2Protocol,
    protein_ids: Iterable[str],
    sample_indices: Iterable[int],
    output_root: Path,
) -> pd.DataFrame:
    """Evaluate generated and WT sequences on both paired structures."""
    from dual_uq.models.proteinmpnn import sequence_sha256

    root = Path(project_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    requested_ids = tuple(dict.fromkeys(str(value) for value in protein_ids))
    indices = tuple(dict.fromkeys(int(value) for value in sample_indices))
    if any(value < 0 for value in indices):
        raise PhysicalEvaluationError("sample indices must be non-negative")
    conditions = load_physical_conditions(root, requested_ids)
    by_key = {(item.protein_id, item.condition): item for item in conditions}
    generation_path = root / "experiments/dataset/analysis/generative_propagation/generated_sequences.parquet"
    generated = pd.read_parquet(generation_path)
    generated = generated[
        generated["protein_id"].astype(str).isin(requested_ids)
        & generated["sample_index"].isin(indices)
    ]
    expected = len(requested_ids) * 2 * len(indices)
    if len(generated) != expected:
        raise PhysicalEvaluationError(f"generated subset cardinality differs: {len(generated)} != {expected}")
    if set(generated["backbone_condition"].astype(str)) != {"PDB", "AFDB"}:
        raise PhysicalEvaluationError("generated subset lacks both conditions")
    rows: list[dict[str, Any]] = []
    ordinal = 0
    for _, generated_row in generated.sort_values(["protein_id", "backbone_condition", "sample_index"]).iterrows():
        protein_id = str(generated_row["protein_id"])
        generated_condition = str(generated_row["backbone_condition"])
        sequence = str(generated_row["sequence"])
        sequence_hash = str(generated_row["sequence_hash"])
        if sequence_sha256(sequence) != sequence_hash:
            raise PhysicalEvaluationError("generated sequence hash mismatch")
        for evaluated_condition in ("PDB", "AFDB"):
            condition = by_key[(protein_id, evaluated_condition)]
            case = output / protein_id / f"generated_{generated_condition.lower()}_{int(generated_row['sample_index']):03d}_{evaluated_condition.lower()}"
            input_path = case / "input.pdb"
            input_path.parent.mkdir(parents=True, exist_ok=True)
            input_path.write_bytes(render_common_mask_pdb(
                condition.source_path, f"{protein_id}:{evaluated_condition}",
                condition.projection_coordinates, sequence,
            ))
            terms, repaired = evaluate_structure(protocol, input_path, case)
            if _repaired_sequence(repaired) != sequence:
                raise PhysicalEvaluationError(f"EvoEF2 repair changed requested sequence: {protein_id}")
            rows.append({
                "protein_id": protein_id,
                "generated_condition": generated_condition,
                "evaluated_condition": evaluated_condition,
                "sample_index": int(generated_row["sample_index"]),
                "sample_class": str(generated_row["sample_class"]),
                "sequence_hash": sequence_hash,
                "structure_sha256": condition.source_sha256,
                "source_chain_id": condition.source_chain_id,
                "repair_runs": protocol.repair_runs,
                "rotamer_library": protocol.rotamer_library,
                "energy_terms": json.dumps(terms, sort_keys=True),
                "total_energy": terms["Total"],
                "case_ordinal": ordinal,
            })
            ordinal += 1
    for protein_id in requested_ids:
        for evaluated_condition in ("PDB", "AFDB"):
            condition = by_key[(protein_id, evaluated_condition)]
            sequence = condition.wt_sequence
            case = output / protein_id / f"wt_{evaluated_condition.lower()}"
            input_path = case / "input.pdb"
            input_path.parent.mkdir(parents=True, exist_ok=True)
            input_path.write_bytes(render_common_mask_pdb(
                condition.source_path, f"{protein_id}:{evaluated_condition}",
                condition.projection_coordinates, sequence,
            ))
            terms, repaired = evaluate_structure(protocol, input_path, case)
            if _repaired_sequence(repaired) != sequence:
                raise PhysicalEvaluationError(f"EvoEF2 repair changed WT sequence: {protein_id}")
            rows.append({
                "protein_id": protein_id,
                "generated_condition": "WT",
                "evaluated_condition": evaluated_condition,
                "sample_index": -1,
                "sample_class": "reference",
                "sequence_hash": sequence_sha256(sequence),
                "structure_sha256": condition.source_sha256,
                "source_chain_id": condition.source_chain_id,
                "repair_runs": protocol.repair_runs,
                "rotamer_library": protocol.rotamer_library,
                "energy_terms": json.dumps(terms, sort_keys=True),
                "total_energy": terms["Total"],
                "case_ordinal": ordinal,
            })
            ordinal += 1
    result = pd.DataFrame(rows)
    if len(result) != expected * 2 + len(requested_ids) * 2:
        raise PhysicalEvaluationError("physical result cardinality differs")
    return result


def prepare_inputs(
    *, project_root: Path,
    protein_ids: Iterable[str],
    sample_indices: Iterable[int],
    output_root: Path,
) -> Path:
    """Materialize common-mask inputs, then let a fresh process score them.

    Projection loading validates large frozen tables.  Keeping preparation and
    EvoEF2 subprocess execution in separate CLI phases avoids forking a large
    pandas process and therefore avoids changing the physical protocol because
    of host memory pressure.
    """
    from dual_uq.models.proteinmpnn import sequence_sha256

    root = Path(project_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    requested_ids = tuple(dict.fromkeys(str(value) for value in protein_ids))
    indices = tuple(dict.fromkeys(int(value) for value in sample_indices))
    conditions = load_physical_conditions(root, requested_ids)
    by_key = {(item.protein_id, item.condition): item for item in conditions}
    generation_path = root / "experiments/dataset/analysis/generative_propagation/generated_sequences.parquet"
    generated = pd.read_parquet(generation_path)
    generated = generated[
        generated["protein_id"].astype(str).isin(requested_ids)
        & generated["sample_index"].isin(indices)
    ]
    expected = len(requested_ids) * 2 * len(indices)
    if len(generated) != expected:
        raise PhysicalEvaluationError(f"generated subset cardinality differs: {len(generated)} != {expected}")
    wt_input_paths: dict[tuple[str, str], Path] = {}
    for protein_id in requested_ids:
        for evaluated_condition in ("PDB", "AFDB"):
            condition = by_key[(protein_id, evaluated_condition)]
            case = output / protein_id / f"wt_{evaluated_condition.lower()}"
            input_path = case / "input.pdb"
            input_path.parent.mkdir(parents=True, exist_ok=True)
            input_path.write_bytes(render_common_mask_pdb(
                condition.source_path, f"{protein_id}:{evaluated_condition}",
                condition.projection_coordinates, condition.wt_sequence,
            ))
            wt_input_paths[(protein_id, evaluated_condition)] = input_path
    cases: list[dict[str, Any]] = []
    for _, generated_row in generated.sort_values(["protein_id", "backbone_condition", "sample_index"]).iterrows():
        protein_id = str(generated_row["protein_id"])
        generated_condition = str(generated_row["backbone_condition"])
        sequence = str(generated_row["sequence"])
        sequence_hash = str(generated_row["sequence_hash"])
        if sequence_sha256(sequence) != sequence_hash:
            raise PhysicalEvaluationError("generated sequence hash mismatch")
        sample_index = int(generated_row["sample_index"])
        for evaluated_condition in ("PDB", "AFDB"):
            condition = by_key[(protein_id, evaluated_condition)]
            case = output / protein_id / f"generated_{generated_condition.lower()}_{sample_index:03d}_{evaluated_condition.lower()}"
            input_path = case / "input.pdb"
            input_path.parent.mkdir(parents=True, exist_ok=True)
            input_path.write_bytes(render_common_mask_pdb(
                condition.source_path, f"{protein_id}:{evaluated_condition}",
                condition.projection_coordinates, sequence,
            ))
            cases.append({
                "protein_id": protein_id,
                "generated_condition": generated_condition,
                "evaluated_condition": evaluated_condition,
                "sample_index": sample_index,
                "sample_class": str(generated_row["sample_class"]),
                "sequence": sequence,
                "sequence_hash": sequence_hash,
                "structure_sha256": condition.source_sha256,
                "source_chain_id": condition.source_chain_id,
                "input_path": input_path.relative_to(output).as_posix(),
                "wt_input_path": wt_input_paths[(protein_id, evaluated_condition)].relative_to(output).as_posix(),
            })
    for protein_id in requested_ids:
        for evaluated_condition in ("PDB", "AFDB"):
            condition = by_key[(protein_id, evaluated_condition)]
            sequence = condition.wt_sequence
            input_path = wt_input_paths[(protein_id, evaluated_condition)]
            cases.append({
                "protein_id": protein_id,
                "generated_condition": "WT",
                "evaluated_condition": evaluated_condition,
                "sample_index": -1,
                "sample_class": "reference",
                "sequence": sequence,
                "sequence_hash": sequence_sha256(sequence),
                "structure_sha256": condition.source_sha256,
                "source_chain_id": condition.source_chain_id,
                "input_path": input_path.relative_to(output).as_posix(),
            })
    manifest = output / "input_manifest.json"
    payload = {
        "schema": "evoef2_physical_input_manifest_v1",
        "cohort": "pilot" if len(requested_ids) < 68 else "clean_68",
        "protein_ids": list(requested_ids),
        "sample_indices": list(indices),
        "cases": cases,
    }
    manifest.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return manifest


def score_prepared_inputs(
    protocol: EvoEF2Protocol, manifest_path: Path, input_root: Path | None = None
) -> pd.DataFrame:
    """Score one prepared manifest in a low-memory process."""
    from dual_uq.models.proteinmpnn import sequence_sha256

    manifest_path = Path(manifest_path).expanduser().resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PhysicalEvaluationError("physical input manifest is unreadable") from exc
    if payload.get("schema") != "evoef2_physical_input_manifest_v1":
        raise PhysicalEvaluationError("physical input manifest schema is invalid")
    output = manifest_path.parent
    input_base = Path(input_root).expanduser().resolve() if input_root is not None else output
    rows: list[dict[str, Any]] = []
    for ordinal, case in enumerate(payload.get("cases", [])):
        sequence = str(case["sequence"])
        if sequence_sha256(sequence) != str(case["sequence_hash"]):
            raise PhysicalEvaluationError("prepared sequence hash mismatch")
        input_path = input_base / str(case["input_path"])
        base = {
            key: case[key] for key in (
                "protein_id", "generated_condition", "evaluated_condition", "sample_index",
                "sample_class", "sequence_hash", "structure_sha256", "source_chain_id",
            )
        }
        # Keep EvoEF2's mutable repair/log outputs isolated per case.  This is
        # required when split manifests share WT input paths or run in parallel;
        # it does not alter the scientific input or scoring protocol.
        case_workdir = input_path.parent / f".evoef2_case_{ordinal:04d}"
        build_status = "not_applicable" if case["generated_condition"] == "WT" else "not_started"
        try:
            scoring_input = input_path
            if case["generated_condition"] != "WT":
                wt_path_value = case.get("wt_input_path")
                if not wt_path_value:
                    raise PhysicalEvaluationError("prepared generated case lacks WT input for BuildMutant")
                scoring_input = build_mutant(
                    protocol,
                    input_base / str(wt_path_value),
                    sequence,
                    case_workdir / "build",
                )
                build_status = "success"
            terms, repaired = evaluate_structure(protocol, scoring_input, case_workdir / "score")
            if _repaired_sequence(repaired) != sequence:
                raise PhysicalEvaluationError(
                    f"EvoEF2 repair changed requested sequence: {case['protein_id']}"
                )
        except PhysicalEvaluationError as exc:
            if case["generated_condition"] != "WT" and build_status == "not_started":
                build_status = "failed"
                failure_type = "BuildMutant"
            else:
                message = str(exc)
                failure_type = (
                    "RepairStructure" if "RepairStructure" in message or "repair changed" in message
                    else "ComputeStability" if "ComputeStability" in message
                    else type(exc).__name__
                )
            rows.append(base | {
                "status": "failed",
                "build_status": build_status,
                "evaluation_status": "not_run" if build_status != "success" else "failed",
                "failure_type": failure_type,
                "failure_message": str(exc),
                "repair_runs": protocol.repair_runs,
                "rotamer_library": protocol.rotamer_library,
                "energy_terms": None,
                "total_energy": None,
                "case_ordinal": ordinal,
            })
            continue
        rows.append(base | {
            "status": "success",
            "build_status": build_status,
            "evaluation_status": "success",
            "failure_type": None,
            "failure_message": None,
            "repair_runs": protocol.repair_runs,
            "rotamer_library": protocol.rotamer_library,
            "energy_terms": json.dumps(terms, sort_keys=True),
            "total_energy": terms["Total"],
            "case_ordinal": ordinal,
        })
    if payload.get("manifest_mode") == "chunk":
        expected = len(payload.get("cases", []))
    else:
        expected = len(payload.get("protein_ids", [])) * (
            2 * len(payload.get("sample_indices", [])) * 2 + 2
        )
    if len(rows) != expected:
        raise PhysicalEvaluationError(f"prepared result cardinality differs: {len(rows)} != {expected}")
    return pd.DataFrame(rows)


def screen_prepared_inputs(
    protocol: EvoEF2Protocol, manifest_path: Path, input_root: Path | None = None
) -> pd.DataFrame:
    """Outcome-blind exact-sequence evaluability screen without energy scoring."""
    from dual_uq.models.proteinmpnn import sequence_sha256

    manifest_path = Path(manifest_path).expanduser().resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PhysicalEvaluationError("physical input manifest is unreadable") from exc
    if payload.get("schema") != "evoef2_physical_input_manifest_v1":
        raise PhysicalEvaluationError("physical input manifest schema is invalid")
    output = manifest_path.parent
    input_base = Path(input_root).expanduser().resolve() if input_root is not None else output
    rows: list[dict[str, Any]] = []
    for ordinal, case in enumerate(payload.get("cases", [])):
        sequence = str(case["sequence"])
        if sequence_sha256(sequence) != str(case["sequence_hash"]):
            raise PhysicalEvaluationError("prepared sequence hash mismatch")
        input_path = input_base / str(case["input_path"])
        base = {
            key: case[key] for key in (
                "protein_id", "generated_condition", "evaluated_condition", "sample_index",
                "sample_class", "sequence_hash", "structure_sha256", "source_chain_id",
            )
        }
        build_status = "not_applicable" if case["generated_condition"] == "WT" else "not_started"
        try:
            if case["generated_condition"] != "WT":
                wt_value = case.get("wt_input_path")
                if not wt_value:
                    raise PhysicalEvaluationError("prepared generated case lacks WT input for BuildMutant")
                built = build_mutant(
                    protocol, input_base / str(wt_value), sequence,
                    input_path.parent / f".evoef2_screen_{ordinal:04d}" / "build",
                )
                build_status = "success"
                repair_input = built
            else:
                repair_input = input_path
            repaired = repair_structure(
                protocol, repair_input,
                input_path.parent / f".evoef2_screen_{ordinal:04d}" / "repair",
            )
            if _repaired_sequence(repaired) != sequence:
                raise PhysicalEvaluationError(
                    f"EvoEF2 repair changed requested sequence: {case['protein_id']}"
                )
        except PhysicalEvaluationError as exc:
            if case["generated_condition"] != "WT" and build_status == "not_started":
                build_status = "failed"
                failure_type = "BuildMutant"
            else:
                failure_type = "RepairStructure"
            rows.append(base | {
                "status": "failed",
                "build_status": build_status,
                "evaluation_status": "not_run",
                "failure_type": failure_type,
                "failure_message": str(exc),
                "case_ordinal": ordinal,
            })
            continue
        rows.append(base | {
            "status": "success",
            "build_status": build_status,
            "evaluation_status": "screened_exact",
            "failure_type": None,
            "failure_message": None,
            "case_ordinal": ordinal,
        })
    expected = len(payload.get("cases", []))
    if len(rows) != expected:
        raise PhysicalEvaluationError(f"evaluability screen cardinality differs: {len(rows)} != {expected}")
    return pd.DataFrame(rows)


def preference_effects(records: pd.DataFrame) -> pd.DataFrame:
    """Compute within-protein AFDB-minus-PDB energy and WT-adjusted effects."""
    required = {"protein_id", "generated_condition", "evaluated_condition", "total_energy"}
    missing = required.difference(records.columns)
    if missing:
        raise PhysicalEvaluationError(f"energy table lacks columns: {sorted(missing)}")
    usable = records if "status" not in records.columns else records[records["status"] == "success"]
    wt_records = usable[usable["generated_condition"] == "WT"]
    generated_records = usable[usable["generated_condition"] != "WT"]
    wt_pivot = wt_records.pivot_table(
        index=["protein_id"], columns="evaluated_condition", values="total_energy", aggfunc="first"
    )
    if wt_pivot.empty or wt_pivot[["PDB", "AFDB"]].isna().any().any():
        raise PhysicalEvaluationError("WT PDB/AFDB energy values are incomplete")
    wt_by_protein = (wt_pivot["AFDB"] - wt_pivot["PDB"])
    pivot = generated_records.pivot_table(
        index=["protein_id", "generated_condition", "sample_index"],
        columns="evaluated_condition", values="total_energy", aggfunc="first",
    ).reset_index()
    if pivot[["PDB", "AFDB"]].isna().any().any():
        raise PhysicalEvaluationError("paired PDB/AFDB energy values are incomplete")
    pivot["pdb_preference"] = pivot["AFDB"] - pivot["PDB"]
    summary = pivot.groupby(["protein_id", "generated_condition"], as_index=False)["pdb_preference"].median()
    summary = summary.rename(columns={"pdb_preference": "generated_preference_median"})
    summary["wt_preference"] = summary["protein_id"].map(wt_by_protein)
    summary["baseline_adjusted_effect"] = summary["generated_preference_median"] - summary["wt_preference"]
    return summary


def paired_sequence_effects(records: pd.DataFrame) -> pd.DataFrame:
    """Return one paired ``E_A - E_P`` contrast for each successful sequence."""
    required = {
        "protein_id", "generated_condition", "evaluated_condition",
        "sample_index", "sequence_hash", "total_energy",
    }
    missing = required.difference(records.columns)
    if missing:
        raise PhysicalEvaluationError(f"energy table lacks columns: {sorted(missing)}")
    usable = records if "status" not in records.columns else records[records["status"] == "success"]
    paired = usable.pivot_table(
        index=["protein_id", "generated_condition", "sample_index", "sequence_hash"],
        columns="evaluated_condition", values="total_energy", aggfunc="first",
    ).reset_index()
    if paired.empty or not {"PDB", "AFDB"}.issubset(paired.columns):
        raise PhysicalEvaluationError("successful sequence pairs are incomplete")
    paired = paired.dropna(subset=["PDB", "AFDB"]).copy()
    paired["sequence_id"] = paired["sequence_hash"]
    paired["sequence_source"] = paired["generated_condition"]
    paired["preference_e"] = paired["AFDB"] - paired["PDB"]
    return paired.rename(columns={"PDB": "E_P", "AFDB": "E_A"})[
        [
            "protein_id", "sequence_id", "sequence_source", "sample_index",
            "sequence_hash", "E_P", "E_A", "preference_e",
        ]
    ].sort_values(["protein_id", "sequence_source", "sample_index"]).reset_index(drop=True)


def summarize_protein_effects(records: pd.DataFrame) -> pd.DataFrame:
    """Aggregate directional and WT-adjusted EvoEF2 effects per protein."""
    required = {"protein_id", "generated_condition", "evaluated_condition", "total_energy"}
    missing = required.difference(records.columns)
    if missing:
        raise PhysicalEvaluationError(f"energy table lacks columns: {sorted(missing)}")
    usable = records if "status" not in records.columns else records[records["status"] == "success"]
    wt = usable[usable["generated_condition"] == "WT"]
    generated = usable[usable["generated_condition"] != "WT"]
    wt_pivot = wt.pivot_table(
        index="protein_id", columns="evaluated_condition", values="total_energy", aggfunc="first"
    )
    if wt_pivot.empty or not {"PDB", "AFDB"}.issubset(wt_pivot.columns):
        raise PhysicalEvaluationError("WT PDB/AFDB energy values are incomplete")
    wt_preference = wt_pivot["AFDB"] - wt_pivot["PDB"]
    generated_pivot = generated.pivot_table(
        index=["protein_id", "generated_condition", "sample_index"],
        columns="evaluated_condition", values="total_energy", aggfunc="first",
    )
    if generated_pivot.empty or not {"PDB", "AFDB"}.issubset(generated_pivot.columns):
        raise PhysicalEvaluationError("generated PDB/AFDB energy pairs are incomplete")
    generated_pivot = generated_pivot.dropna(subset=["PDB", "AFDB"]).reset_index()
    generated_pivot["preference_e"] = generated_pivot["AFDB"] - generated_pivot["PDB"]
    grouped = generated_pivot.groupby(["protein_id", "generated_condition"])["preference_e"]
    summary = grouped.agg(
        generated_preference_median="median",
        generated_preference_q10=lambda values: values.quantile(0.10),
        generated_preference_q90=lambda values: values.quantile(0.90),
        generated_preference_positive_fraction=lambda values: float((values > 0).mean()),
        generated_success_count="count",
    ).reset_index()
    summary["wt_preference"] = summary["protein_id"].map(wt_preference)
    summary["baseline_adjusted_preference_median"] = (
        summary["generated_preference_median"] - summary["wt_preference"]
    )
    wide = summary.pivot(index="protein_id", columns="generated_condition")
    wide.columns = [f"{condition.lower()}_{field}" for field, condition in wide.columns]
    wide = wide.reset_index()
    for condition in ("pdb", "afdb"):
        if f"{condition}_generated_preference_median" not in wide:
            raise PhysicalEvaluationError(f"generated condition is incomplete: {condition}")
    wide["generated_condition_difference"] = (
        wide["afdb_generated_preference_median"] - wide["pdb_generated_preference_median"]
    )
    wide["baseline_adjusted_condition_difference"] = (
        wide["afdb_baseline_adjusted_preference_median"]
        - wide["pdb_baseline_adjusted_preference_median"]
    )
    wide["wt_preference"] = wide["pdb_wt_preference"]
    return wide.drop(columns=["afdb_wt_preference"])


def manifest_payload(protocol: EvoEF2Protocol, *, cohort: str, row_count: int) -> dict[str, Any]:
    return {
        "schema": "evoef2_physical_robustness_v1",
        "evaluator": "EvoEF2",
        "source_revision": protocol.source_revision,
        "executable_sha256": protocol.executable_sha256,
        "repair_runs": protocol.repair_runs,
        "rotamer_library": protocol.rotamer_library,
        "weight_file": protocol.weight_file,
        "cohort": cohort,
        "row_count": row_count,
        "energy_preference_definition": "E_A_minus_E_P",
        "interpretation": "within-protein model compatibility only; absolute energies are not experimental stability",
    }


__all__ = [
    "ESMFOLD_SAMPLE_INDICES",
    "EvoEF2Protocol",
    "PhysicalEvaluationError",
    "_mutation_spec_from_pdb",
    "build_mutant",
    "evaluate_structure",
    "load_physical_conditions",
    "manifest_payload",
    "paired_sequence_effects",
    "parse_energy_terms",
    "preference_effects",
    "prepare_inputs",
    "render_common_mask_pdb",
    "repair_structure",
    "score_prepared_inputs",
    "screen_prepared_inputs",
    "summarize_protein_effects",
]
