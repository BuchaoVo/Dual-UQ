#!/usr/bin/env python3
"""Materialize baseline-adjusted structural-validation asymmetry analysis."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from dual_uq.core.hashing import sha256_canonical
from dual_uq.evaluation.structural_validation_asymmetry import (
    build_structural_validation_asymmetry,
    materialize_structural_validation_asymmetry,
)
from dual_uq.inference.independent_structure_validation import (
    StructuralValidationAsymmetryError,
    load_wt_predictions,
)


def _path(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_identity(model_path: Path | None) -> dict[str, str]:
    if model_path is None:
        return {}
    path = model_path.expanduser().resolve()
    config = path / "config.json"
    weights = path / "pytorch_model.bin"
    if not config.is_file() or not weights.is_file():
        raise StructuralValidationAsymmetryError("local ESMFold model identity is unavailable")
    return {"family": "ESMFold", "model_name": path.name, "config_sha256": _sha(config), "weights_sha256": _sha(weights)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--generated-comparison",
        type=Path,
        default=Path("experiments/dataset/analysis/independent_structure_validation/sequence_structural_comparison.parquet"),
    )
    parser.add_argument(
        "--wt-root",
        type=Path,
        default=Path("runs/analysis/independent_structure_validation/asymmetry/wt"),
    )
    parser.add_argument(
        "--generation-summary",
        type=Path,
        default=Path("experiments/dataset/analysis/generative_propagation/protein_generation_summary.parquet"),
    )
    parser.add_argument(
        "--remodeling-summary",
        type=Path,
        default=Path("experiments/dataset/analysis/inverse_folding_remodeling/protein_remodeling.parquet"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/dataset/analysis/independent_structure_validation/asymmetry"),
    )
    parser.add_argument("--model-path", type=Path, default=None)
    args = parser.parse_args()
    try:
        root = args.project_root.expanduser().resolve()
        generated_path = _path(root, args.generated_comparison)
        wt_root = _path(root, args.wt_root)
        generation_path = _path(root, args.generation_summary)
        remodeling_path = _path(root, args.remodeling_summary)
        rows = load_wt_predictions(wt_root)
        wt_frame = pd.DataFrame([asdict(row) for row in rows])
        generated = pd.read_parquet(generated_path)
        generation = pd.read_parquet(generation_path)
        remodeling = pd.read_parquet(remodeling_path)
        result = build_structural_validation_asymmetry(generated, wt_frame, generation, remodeling)
        wt_identity = sha256_canonical([
            {
                "protein_id": row.protein_id,
                "sequence_hash": row.sequence_hash,
                "pdb_structure_sha256": row.pdb_structure_sha256,
                "afdb_structure_sha256": row.afdb_structure_sha256,
                "predicted_structure_sha256": row.predicted_structure_sha256,
            }
            for row in rows
        ])
        manifest = materialize_structural_validation_asymmetry(
            result,
            output_root=_path(root, args.output_root),
            input_paths={
                "generated_comparison": generated_path,
                "generation_summary": generation_path,
                "remodeling_summary": remodeling_path,
            },
            wt_identity=wt_identity,
            model_identity=_model_identity(args.model_path),
        )
        print(f"COMPLETE proteins={manifest['row_counts']['wt_structural_baseline']} absolute_rows={manifest['row_counts']['absolute_structural_comparisons']}")
        return 0
    except (StructuralValidationAsymmetryError, OSError, ValueError, KeyError) as exc:
        print(f"BLOCKED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
