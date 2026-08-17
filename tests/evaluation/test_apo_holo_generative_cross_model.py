from __future__ import annotations

import pandas as pd

from dual_uq.evaluation.apo_holo_generative_cross_model import (
    build_cross_model_result,
)


def _generation(model_offset: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "protein_id": ["P1", "P2", "P3"],
            "d_apo_64": [0.1, 0.2, 0.3],
            "d_holo_64": [0.2, 0.3, 0.4],
            "d_ah_64": [0.4 + model_offset, 0.5 + model_offset, 0.6 + model_offset],
            "d_excess_64": [0.2 + model_offset, 0.3 + model_offset, 0.4 + model_offset],
            "js_bits_mean_64": [0.01 + model_offset, 0.02 + model_offset, 0.03 + model_offset],
            "js_bits_q90_64": [0.1, 0.2, 0.3],
        }
    )


def _local(model_offset: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "protein_id": ["P1", "P2", "P3"],
            "local_burden_mean": [0.1 + model_offset, 0.2 + model_offset, 0.3 + model_offset],
        }
    )


def test_cross_model_result_joins_shared_proteins_and_derives_associations() -> None:
    geometry = pd.DataFrame(
        {
            "protein_id": ["P1", "P2", "P3"],
            "geometry_status": ["available", "available", "available"],
            "aligned_ca_rmsd": [1.0, 2.0, 3.0],
            "median_residue_displacement": [0.1, 0.2, 0.3],
            "p90_residue_displacement": [0.2, 0.3, 0.4],
            "median_local_pairwise_distance_change": [0.3, 0.4, 0.5],
            "ligand_proximal_count": [2, 0, 1],
            "ligand_distal_count": [8, 10, 9],
        }
    )
    result = build_cross_model_result(
        proteinmpnn_generation=_generation(0.0),
        esm_if1_generation=_generation(0.1),
        proteinmpnn_local=_local(0.0),
        esm_if1_local=_local(0.1),
        pair_geometry=geometry,
        local_reference_spearman=0.783822,
    )

    assert len(result.protein_summary) == 3
    assert set(result.protein_summary["ligand_status"]) == {
        "ligand_proximal",
        "explicit_zero_proximal",
    }
    assert result.summary["shared_generation_protein_count"] == 3
    assert result.summary["local_response_reference_spearman"] == 0.783822
    cross = result.associations.loc[
        result.associations["association"] == "generative_d_excess_cross_model"
    ].iloc[0]
    assert cross["spearman"] == 1.0
    assert set(result.convergence["model"]) == {"ProteinMPNN", "ESM-IF1"}
