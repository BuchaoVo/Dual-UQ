#!/usr/bin/env python3
"""Aggregate independent local structural validation predictions."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from dual_uq.evaluation.independent_structure_validation import (
    build_independent_structure_analysis,
    materialize_independent_structure_analysis,
)
from dual_uq.inference.independent_structure_validation import (
    IndependentStructureValidationError,
    load_prediction_rows,
)


def _frame(rows: tuple[object, ...]) -> pd.DataFrame:
    return pd.DataFrame([row.__dict__ if hasattr(row, "__dict__") else {k: getattr(row, k) for k in row.__slots__} for row in rows])


def _path(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def _model_identity(model_path: Path | None) -> dict[str, str]:
    if model_path is None:
        return {}
    path = model_path.expanduser().resolve()
    config = path / "config.json"
    weights = path / "pytorch_model.bin"
    if not config.is_file() or not weights.is_file():
        raise IndependentStructureValidationError("local ESMFold model identity is unavailable")
    def digest(target: Path) -> str:
        h = hashlib.sha256()
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    return {
        "family": "ESMFold",
        "model_name": path.name,
        "config_sha256": digest(config),
        "weights_sha256": digest(weights),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, default=Path("runs/analysis/independent_structure_validation"))
    parser.add_argument("--generation-summary", type=Path, default=Path("experiments/dataset/analysis/generative_propagation/protein_generation_summary.parquet"))
    parser.add_argument("--compatibility-summary", type=Path, default=Path("experiments/dataset/analysis/cross_structure_compatibility/protein_cross_structure_summary.parquet"))
    parser.add_argument("--remodeling-summary", type=Path, default=Path("experiments/dataset/analysis/inverse_folding_remodeling/protein_remodeling.parquet"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/dataset/analysis/independent_structure_validation"))
    parser.add_argument("--model-path", type=Path, default=None)
    args = parser.parse_args()
    try:
        root = args.project_root.expanduser().resolve()
        prediction_root = _path(root, args.prediction_root)
        generation_path = _path(root, args.generation_summary)
        compatibility_path = _path(root, args.compatibility_summary)
        remodeling_path = _path(root, args.remodeling_summary)
        rows = _frame(load_prediction_rows(prediction_root))
        result = build_independent_structure_analysis(
            rows,
            pd.read_parquet(generation_path),
            pd.read_parquet(compatibility_path),
            pd.read_parquet(remodeling_path),
        )
        manifest = materialize_independent_structure_analysis(
            result,
            output_root=_path(root, args.output_root),
            input_paths={"generation_summary": generation_path, "compatibility_summary": compatibility_path, "remodeling_summary": remodeling_path},
            model_identity=_model_identity(args.model_path),
        )
        print(f"COMPLETE proteins={manifest['summary']['cohort_proteins']} rows={manifest['summary']['prediction_rows']}")
        return 0
    except (IndependentStructureValidationError, OSError, ValueError, KeyError) as exc:
        print(f"BLOCKED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
