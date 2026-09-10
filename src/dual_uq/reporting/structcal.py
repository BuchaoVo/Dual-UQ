"""Shared presentation labels for StructCal reports."""

import json
from typing import Any

import pandas as pd

MODEL_LABELS = {
    "dynamicmpnn": "DynamicMPNN",
    "esm_if1": "ESM-IF1",
    "pifold": "PiFold",
    "proteinmpnn": "ProteinMPNN",
}

DESCRIPTOR_LABELS = {
    "ca_displacement": "Cα displacement",
    "fragment_7_rmsd": "7-residue RMSD",
    "neighborhood_distance_deformation": "12 Å neighborhood",
    "torsion_phi_psi_change": "φ/ψ change",
}

SHORT_DESCRIPTOR_LABELS = {
    "ca_displacement": "Cα disp.",
    "fragment_7_rmsd": "7-res. RMSD",
    "neighborhood_distance_deformation": "12 Å neigh.",
    "torsion_phi_psi_change": "φ/ψ change",
}


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a report table to JSON-compatible records."""

    return json.loads(frame.to_json(orient="records"))


def format_decimal(value: float) -> str:
    """Apply the shared four-decimal precision used by StructCal reports."""

    return f"{value:.4f}"
