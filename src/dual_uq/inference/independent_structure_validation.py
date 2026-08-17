"""Independent local ESMFold validation of generated protein sequences.

This module owns only execution and immutable prediction records.  Statistical
aggregation lives in :mod:`dual_uq.evaluation.independent_structure_validation`.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from dual_uq.core.atomic_io import atomic_write_new_bytes
from dual_uq.core.hashing import sha256_canonical
from dual_uq.models.proteinmpnn import sequence_sha256
from dual_uq.models.proteinmpnn_generation import GeneratedSequenceRecord

VALIDATION_SAMPLE_INDICES = (0, 32, 64, 96, 128, 160, 192, 224)
VALIDATION_PROTEIN_COUNT = 68
VALIDATION_SCHEMA = "independent_structure_validation_prediction_v1"
WT_VALIDATION_SCHEMA = "independent_structure_validation_wt_prediction_v1"


class IndependentStructureValidationError(ValueError):
    """A validation input, local model, or immutable result is invalid."""


class StructuralValidationAsymmetryError(IndependentStructureValidationError):
    """A WT baseline or asymmetry input violates its explicit contract."""


@dataclass(frozen=True, slots=True)
class ESMFoldPrediction:
    pdb_text: str
    plddt: float | None
    pae: float | None
    ptm: float | None


class StructurePredictionAdapter(Protocol):
    model_identity: Mapping[str, Any]

    def predict(self, sequence: str) -> ESMFoldPrediction: ...


def _finite(value: Any, label: str) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise IndependentStructureValidationError(f"{label} is not numeric") from exc
    if not np.isfinite(result):
        return None
    return result


class LocalESMFoldAdapter:
    """Load one explicitly supplied local ESMFold checkpoint, never download."""

    def __init__(self, model_path: Path, *, device: str = "cuda:0", chunk_size: int = 64) -> None:
        path = Path(model_path).expanduser().resolve()
        if not path.is_dir() or not (path / "config.json").is_file():
            raise IndependentStructureValidationError(
                f"local ESMFold model is unavailable: {path}"
            )
        if not any((path / name).is_file() for name in ("pytorch_model.bin", "model.safetensors")):
            raise IndependentStructureValidationError(
                f"local ESMFold weights are unavailable: {path}"
            )
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
            raise IndependentStructureValidationError("ESMFold chunk_size must be positive")
        try:
            import torch
            from transformers import AutoTokenizer, EsmForProteinFolding

            self._torch = torch
            self._tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
            self._model = EsmForProteinFolding.from_pretrained(
                str(path), local_files_only=True, low_cpu_mem_usage=True
            )
            self._model = self._model.eval().to(device)
            self._model.trunk.set_chunk_size(chunk_size)
        except Exception as exc:  # model loading is an operational boundary
            raise IndependentStructureValidationError(
                f"local ESMFold model could not be loaded: {path}"
            ) from exc
        self.model_identity = {
            "family": "ESMFold",
            "checkpoint_path": str(path),
            "checkpoint_config_sha256": hashlib.sha256((path / "config.json").read_bytes()).hexdigest(),
            "device": device,
            "chunk_size": chunk_size,
        }

    def predict(self, sequence: str) -> ESMFoldPrediction:
        try:
            with self._torch.no_grad():
                output = self._model.infer(sequence)
            pdb_text = self._model.output_to_pdb(output)[0]
            plddt = _finite(self._tensor_mean(output.get("plddt")), "pLDDT")
            pae = _finite(self._tensor_mean(output.get("predicted_aligned_error")), "PAE")
            ptm = _finite(self._tensor_mean(output.get("ptm")), "pTM")
            return ESMFoldPrediction(pdb_text, plddt, pae, ptm)
        except IndependentStructureValidationError:
            raise
        except Exception as exc:
            raise IndependentStructureValidationError("local ESMFold prediction failed") from exc

    def _tensor_mean(self, value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().numpy()
        return np.asarray(value, dtype=float).mean()


@dataclass(frozen=True, slots=True)
class IndependentStructurePrediction:
    protein_id: str
    source_condition: str
    sample_index: int
    sample_class: str
    seed: int
    sequence_hash: str
    sequence: str
    structure_sha256: str
    predicted_ca_coordinates: tuple[tuple[float, float, float], ...]
    target_pdb_ca_coordinates: tuple[tuple[float, float, float], ...]
    target_afdb_ca_coordinates: tuple[tuple[float, float, float], ...]
    plddt: float | None
    pae: float | None
    ptm: float | None
    status: str = "ok"
    predicted_structure_path: str | None = None
    predicted_structure_sha256: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class WTSequenceRequest:
    """One canonical WT sequence request shared by the paired structures."""

    protein_id: str
    sequence: str
    canonical_positions: tuple[int, ...]
    pdb_structure_sha256: str
    afdb_structure_sha256: str


@dataclass(frozen=True, slots=True)
class WTStructurePrediction:
    """One immutable WT prediction bound to both paired target structures."""

    protein_id: str
    sequence: str
    sequence_hash: str
    canonical_positions: tuple[int, ...]
    pdb_structure_sha256: str
    afdb_structure_sha256: str
    predicted_ca_coordinates: tuple[tuple[float, float, float], ...]
    target_pdb_ca_coordinates: tuple[tuple[float, float, float], ...]
    target_afdb_ca_coordinates: tuple[tuple[float, float, float], ...]
    plddt: float | None
    pae: float | None
    ptm: float | None
    status: str = "ok"
    predicted_structure_path: str | None = None
    predicted_structure_sha256: str | None = None
    error: str | None = None


def _projection_value(projection: Any, name: str, label: str) -> Any:
    value = getattr(projection, name, None)
    if value is None:
        raise StructuralValidationAsymmetryError(f"paired projection lacks {label}")
    return value


def select_wt_sequences(
    projections: Mapping[tuple[str, str], Any],
    *,
    expected_protein_count: int = VALIDATION_PROTEIN_COUNT,
) -> tuple[WTSequenceRequest, ...]:
    """Select exactly one WT request from each complete PDB/AFDB pair."""
    if not projections:
        raise StructuralValidationAsymmetryError("paired projection grid is empty")
    proteins = tuple(sorted({str(protein_id) for protein_id, _condition in projections}))
    if len(proteins) != expected_protein_count:
        raise StructuralValidationAsymmetryError("WT cohort cardinality differs")
    expected = {(protein, condition) for protein in proteins for condition in ("PDB", "AFDB")}
    if set(projections) != expected:
        raise StructuralValidationAsymmetryError("paired projection grid is incomplete")
    requests: list[WTSequenceRequest] = []
    for protein_id in proteins:
        pdb = projections[(protein_id, "PDB")]
        afdb = projections[(protein_id, "AFDB")]
        pdb_positions = tuple(_projection_value(pdb, "uniprot_positions", "positions"))
        afdb_positions = tuple(_projection_value(afdb, "uniprot_positions", "positions"))
        pdb_sequence = str(_projection_value(pdb, "wt_sequence_projection", "WT sequence"))
        afdb_sequence = str(_projection_value(afdb, "wt_sequence_projection", "WT sequence"))
        if pdb_positions != afdb_positions or pdb_sequence != afdb_sequence:
            raise StructuralValidationAsymmetryError(
                "paired projection WT sequence or positions differ"
            )
        requests.append(
            WTSequenceRequest(
                protein_id=protein_id,
                sequence=pdb_sequence,
                canonical_positions=pdb_positions,
                pdb_structure_sha256=str(_projection_value(pdb, "structure_sha256", "PDB identity")),
                afdb_structure_sha256=str(_projection_value(afdb, "structure_sha256", "AFDB identity")),
            )
        )
    return tuple(requests)


def _wt_prediction_path(root: Path, request: WTSequenceRequest) -> Path:
    identity = sha256_canonical({
        "protein_id": request.protein_id,
        "sequence": request.sequence,
        "pdb_structure_sha256": request.pdb_structure_sha256,
        "afdb_structure_sha256": request.afdb_structure_sha256,
    })
    return root / "shards" / identity[:2] / f"{identity}.json"


def _wt_json_value(prediction: WTStructurePrediction) -> dict[str, Any]:
    value = asdict(prediction)
    for field in (
        "predicted_ca_coordinates",
        "target_pdb_ca_coordinates",
        "target_afdb_ca_coordinates",
        "canonical_positions",
    ):
        value[field] = [list(row) for row in value[field]] if field != "canonical_positions" else list(value[field])
    return {"schema_version": WT_VALIDATION_SCHEMA, "prediction": value}


def _wt_from_payload(payload: Mapping[str, Any]) -> WTStructurePrediction:
    if payload.get("schema_version") != WT_VALIDATION_SCHEMA:
        raise StructuralValidationAsymmetryError("malformed WT prediction schema")
    try:
        value = dict(payload["prediction"])
        value["canonical_positions"] = tuple(value["canonical_positions"])
        for field in (
            "predicted_ca_coordinates",
            "target_pdb_ca_coordinates",
            "target_afdb_ca_coordinates",
        ):
            value[field] = tuple(tuple(float(x) for x in row) for row in value[field])
        row = WTStructurePrediction(**value)
    except (KeyError, TypeError, ValueError) as exc:
        raise StructuralValidationAsymmetryError("malformed WT prediction record") from exc
    if row.status != "ok" or row.error is not None:
        raise StructuralValidationAsymmetryError("WT prediction is not complete")
    for label, coords in (
        ("predicted", row.predicted_ca_coordinates),
        ("PDB target", row.target_pdb_ca_coordinates),
        ("AFDB target", row.target_afdb_ca_coordinates),
    ):
        array = np.asarray(coords, dtype=float)
        if array.shape != (len(row.canonical_positions), 3) or not np.isfinite(array).all():
            raise StructuralValidationAsymmetryError(f"WT {label} coordinates are invalid")
    return row


def predict_wt_sequences(
    requests: Iterable[WTSequenceRequest],
    projections: Mapping[tuple[str, str], Any],
    adapter: StructurePredictionAdapter,
    *,
    output_root: Path,
    resume: bool = True,
) -> tuple[WTStructurePrediction, ...]:
    """Predict and immutably materialize one WT structure per protein."""
    root = Path(output_root).expanduser().resolve()
    rows: list[WTStructurePrediction] = []
    for request in requests:
        pdb = projections.get((request.protein_id, "PDB"))
        afdb = projections.get((request.protein_id, "AFDB"))
        if pdb is None or afdb is None:
            raise StructuralValidationAsymmetryError("WT paired projection is incomplete")
        if (
            str(getattr(pdb, "structure_sha256", "")) != request.pdb_structure_sha256
            or str(getattr(afdb, "structure_sha256", "")) != request.afdb_structure_sha256
        ):
            raise StructuralValidationAsymmetryError("WT projection identity mismatch")
        target_pdb = np.asarray(pdb.coordinates, dtype=float)[:, 1, :]
        target_afdb = np.asarray(afdb.coordinates, dtype=float)[:, 1, :]
        if (
            target_pdb.shape != (len(request.canonical_positions), 3)
            or target_afdb.shape != target_pdb.shape
            or not np.isfinite(target_pdb).all()
            or not np.isfinite(target_afdb).all()
        ):
            raise StructuralValidationAsymmetryError("WT common mask coordinates are invalid")
        path = _wt_prediction_path(root, request)
        if path.is_file() and resume:
            try:
                rows.append(_wt_from_payload(json.loads(path.read_text(encoding="utf-8"))))
                continue
            except (OSError, json.JSONDecodeError) as exc:
                raise StructuralValidationAsymmetryError("invalid existing WT prediction") from exc
        prediction = adapter.predict(request.sequence)
        predicted = _ca_coordinates(prediction.pdb_text, len(request.canonical_positions))
        structure_path = root / "structures" / path.stem[:2] / f"{path.stem}.pdb"
        structure_bytes = prediction.pdb_text.encode("utf-8")
        try:
            atomic_write_new_bytes(structure_path, structure_bytes)
        except FileExistsError:
            if structure_path.read_bytes() != structure_bytes:
                raise StructuralValidationAsymmetryError("immutable WT structure conflict") from None
        item = WTStructurePrediction(
            protein_id=request.protein_id,
            sequence=request.sequence,
            sequence_hash=sequence_sha256(request.sequence),
            canonical_positions=request.canonical_positions,
            pdb_structure_sha256=request.pdb_structure_sha256,
            afdb_structure_sha256=request.afdb_structure_sha256,
            predicted_ca_coordinates=tuple(tuple(float(x) for x in row) for row in predicted),
            target_pdb_ca_coordinates=tuple(tuple(float(x) for x in row) for row in target_pdb),
            target_afdb_ca_coordinates=tuple(tuple(float(x) for x in row) for row in target_afdb),
            plddt=prediction.plddt,
            pae=prediction.pae,
            ptm=prediction.ptm,
            predicted_structure_path=structure_path.name,
            predicted_structure_sha256=hashlib.sha256(structure_bytes).hexdigest(),
        )
        payload = (json.dumps(_wt_json_value(item), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        try:
            atomic_write_new_bytes(path, payload)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise StructuralValidationAsymmetryError("immutable WT prediction conflict") from None
        rows.append(item)
    return tuple(sorted(rows, key=lambda row: row.protein_id))


def load_wt_predictions(root: Path, *, expected_protein_count: int = VALIDATION_PROTEIN_COUNT) -> tuple[WTStructurePrediction, ...]:
    """Load and validate the immutable WT prediction set."""
    paths = tuple(sorted(Path(root).expanduser().resolve().glob("shards/*/*.json")))
    if not paths:
        raise StructuralValidationAsymmetryError("no WT predictions were found")
    rows = []
    seen: set[str] = set()
    for path in paths:
        try:
            row = _wt_from_payload(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            raise StructuralValidationAsymmetryError("malformed WT prediction") from exc
        if row.protein_id in seen:
            raise StructuralValidationAsymmetryError("WT prediction is duplicated")
        seen.add(row.protein_id)
        rows.append(row)
    if len(rows) != expected_protein_count:
        raise StructuralValidationAsymmetryError("WT prediction cohort cardinality differs")
    return tuple(sorted(rows, key=lambda row: row.protein_id))


def select_validation_records(
    records: Iterable[GeneratedSequenceRecord],
    *,
    sample_indices: tuple[int, ...] = VALIDATION_SAMPLE_INDICES,
) -> tuple[GeneratedSequenceRecord, ...]:
    """Select the fixed, outcome-blind eight-sample subset from the 256 grid."""
    rows = tuple(records)
    keys = {(r.request.protein_id, r.request.backbone_condition, r.request.sample_index) for r in rows}
    proteins = {r.request.protein_id for r in rows}
    expected = {
        (protein, condition, index)
        for protein in proteins
        for condition in ("PDB", "AFDB")
        for index in range(256)
    }
    if keys != expected or len(rows) != len(expected):
        raise IndependentStructureValidationError("generation grid is incomplete or duplicated")
    if len(proteins) != VALIDATION_PROTEIN_COUNT:
        raise IndependentStructureValidationError("generation grid does not contain 68 proteins")
    selected = tuple(
        sorted(
            (r for r in rows if r.request.sample_index in set(sample_indices)),
            key=lambda r: (r.request.protein_id, 0 if r.request.backbone_condition == "PDB" else 1, r.request.sample_index),
        )
    )
    if len(selected) != VALIDATION_PROTEIN_COUNT * 2 * len(sample_indices):
        raise IndependentStructureValidationError("selected validation grid is incomplete")
    return selected


def _ca_coordinates(pdb_text: str, expected_count: int) -> np.ndarray:
    try:
        from Bio.PDB import PDBParser

        structure = PDBParser(QUIET=True).get_structure("prediction", io.StringIO(pdb_text))
        residues = [residue for residue in next(structure.get_models()).get_residues() if residue.id[0] == " "]
        coordinates = np.asarray([residue["CA"].coord for residue in residues], dtype=float)
    except Exception as exc:
        raise IndependentStructureValidationError("ESMFold PDB has no valid CA coordinates") from exc
    if coordinates.shape != (expected_count, 3) or not np.isfinite(coordinates).all():
        raise IndependentStructureValidationError("ESMFold CA coordinate count does not match common mask")
    return coordinates


def _prediction_path(root: Path, record: GeneratedSequenceRecord) -> Path:
    identity = sha256_canonical({
        "protein_id": record.request.protein_id,
        "source_condition": record.request.backbone_condition,
        "sample_index": record.request.sample_index,
        "sequence_hash": record.sequence_hash,
    })
    return root / "shards" / identity[:2] / f"{identity}.json"


def _json_value(prediction: IndependentStructurePrediction) -> dict[str, Any]:
    value = asdict(prediction)
    value["predicted_ca_coordinates"] = [list(row) for row in prediction.predicted_ca_coordinates]
    value["target_pdb_ca_coordinates"] = [list(row) for row in prediction.target_pdb_ca_coordinates]
    value["target_afdb_ca_coordinates"] = [list(row) for row in prediction.target_afdb_ca_coordinates]
    return {"schema_version": VALIDATION_SCHEMA, "prediction": value}


def predict_selected_sequences(
    selected: Iterable[GeneratedSequenceRecord],
    projections: Mapping[tuple[str, str], Any],
    adapter: StructurePredictionAdapter,
    *,
    output_root: Path,
    resume: bool = True,
) -> tuple[IndependentStructurePrediction, ...]:
    """Predict and immutably materialize the fixed validation subset."""
    root = Path(output_root).expanduser().resolve()
    rows: list[IndependentStructurePrediction] = []
    for record in selected:
        key = (record.request.protein_id, record.request.backbone_condition)
        projection = projections.get(key)
        if projection is None or projection.structure_sha256 != record.request.structure_sha256:
            raise IndependentStructureValidationError("prediction projection identity mismatch")
        expected_count = len(record.request.canonical_positions)
        target_pdb = np.asarray(projections[(record.request.protein_id, "PDB")].coordinates, dtype=float)[:, 1, :]
        target_afdb = np.asarray(projections[(record.request.protein_id, "AFDB")].coordinates, dtype=float)[:, 1, :]
        path = _prediction_path(root, record)
        if path.is_file() and resume:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))["prediction"]
                rows.append(IndependentStructurePrediction(**value))
                continue
            except Exception as exc:
                raise IndependentStructureValidationError("invalid existing validation prediction") from exc
        prediction = adapter.predict(record.sequence)
        predicted = _ca_coordinates(prediction.pdb_text, expected_count)
        structure_path = root / "structures" / path.stem[:2] / f"{path.stem}.pdb"
        structure_bytes = prediction.pdb_text.encode("utf-8")
        try:
            atomic_write_new_bytes(structure_path, structure_bytes)
        except FileExistsError:
            if structure_path.read_bytes() != structure_bytes:
                raise IndependentStructureValidationError("immutable predicted structure conflict") from None
        item = IndependentStructurePrediction(
            protein_id=record.request.protein_id,
            source_condition=record.request.backbone_condition,
            sample_index=record.request.sample_index,
            sample_class=record.request.sample_class,
            seed=record.request.seed,
            sequence_hash=record.sequence_hash,
            sequence=record.sequence,
            structure_sha256=record.request.structure_sha256,
            predicted_ca_coordinates=tuple(tuple(float(x) for x in row) for row in predicted),
            target_pdb_ca_coordinates=tuple(tuple(float(x) for x in row) for row in target_pdb),
            target_afdb_ca_coordinates=tuple(tuple(float(x) for x in row) for row in target_afdb),
            plddt=prediction.plddt,
            pae=prediction.pae,
            ptm=prediction.ptm,
            predicted_structure_path=structure_path.name,
            predicted_structure_sha256=hashlib.sha256(structure_bytes).hexdigest(),
        )
        payload = (json.dumps(_json_value(item), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        try:
            atomic_write_new_bytes(path, payload)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise IndependentStructureValidationError("immutable validation prediction conflict") from None
        rows.append(item)
    return tuple(sorted(rows, key=lambda row: (row.protein_id, 0 if row.source_condition == "PDB" else 1, row.sample_index)))


def load_prediction_rows(root: Path) -> tuple[IndependentStructurePrediction, ...]:
    paths = tuple(sorted(Path(root).expanduser().resolve().glob("shards/*/*.json")))
    if not paths:
        raise IndependentStructureValidationError("no validation predictions were found")
    rows = []
    seen: set[tuple[str, str, int]] = set()
    for path in paths:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("schema_version") != VALIDATION_SCHEMA:
                raise ValueError
            row = IndependentStructurePrediction(**value["prediction"])
        except Exception as exc:
            raise IndependentStructureValidationError("malformed validation prediction") from exc
        key = (row.protein_id, row.source_condition, row.sample_index)
        if key in seen:
            raise IndependentStructureValidationError("validation prediction is duplicated")
        seen.add(key)
        rows.append(row)
    return tuple(sorted(rows, key=lambda row: (row.protein_id, 0 if row.source_condition == "PDB" else 1, row.sample_index)))
