import pandas as pd

from dual_uq.preflight import classify_preflight, compute_preflight_metrics

THRESHOLDS = {
    "min_full_length_mapping_coverage": 0.90,
    "min_entity_mapping_coverage": 0.90,
    "min_sequence_identity": 0.95,
    "min_observed_ca_fraction": 0.90,
    "warn_full_length_mapping_coverage": 0.70,
    "max_internal_unmapped_fraction": 0.05,
}


def test_full_length_pair_passes() -> None:
    mapping = pd.DataFrame(
        {
            "pdb_residue_number": ["1", "2", "3", "4"],
            "pdb_residue_name": ["ALA", "CYS", "ASP", "GLU"],
            "uniprot_residue_number": [1, 2, 3, 4],
            "uniprot_residue_name": ["A", "C", "D", "E"],
        }
    )
    ca = pd.DataFrame({"pdb_residue_number": ["1", "2", "3", "4"]})
    metrics = compute_preflight_metrics(
        mapping, ca, uniprot_length=4, pdb_entity_length=4
    )
    status, _ = classify_preflight(metrics, THRESHOLDS)
    assert status == "pass_full_length"


def test_fragment_pair_fails() -> None:
    mapping = pd.DataFrame(
        {
            "pdb_residue_number": ["1", "2", "3"],
            "pdb_residue_name": ["ALA", "CYS", "ASP"],
            "uniprot_residue_number": [101, 102, 103],
            "uniprot_residue_name": ["A", "C", "D"],
        }
    )
    ca = pd.DataFrame({"pdb_residue_number": ["1", "2", "3"]})
    metrics = compute_preflight_metrics(
        mapping, ca, uniprot_length=500, pdb_entity_length=3
    )
    status, _ = classify_preflight(metrics, THRESHOLDS)
    assert status == "fail_preflight"
