from __future__ import annotations

from pathlib import Path

import pandas as pd

PROTEIN_COLUMNS = ["protein_id","uniprot_id","pdb_id","chain_id","sequence","length","sequence_cluster","domain_count","secondary_structure_class","split","mapping_coverage","sequence_identity","quality_flag"]
STRUCTURE_COLUMNS = ["structure_id","protein_id","source","parent_structure_id","coordinate_path","plddt_path","pae_path","perturbation_type","perturbation_strength","perturbed_residues","random_seed","mapping_quality","structure_valid"]
SEQUENCE_COLUMNS = ["sequence_id","protein_id","generation_structure_id","generator","checkpoint","temperature","sampling_seed","sequence","sequence_identity_to_native","generation_status"]
SCORE_COLUMNS = ["protein_id","structure_id","sequence_id","evaluator_id","property_id","score","score_mean","score_variance","ood_score","run_status","error_message","runtime"]

def initialize_manifests(output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    specs = {
        "protein_manifest.parquet": PROTEIN_COLUMNS,
        "structure_manifest.parquet": STRUCTURE_COLUMNS,
        "sequence_manifest.parquet": SEQUENCE_COLUMNS,
        "scores.parquet": SCORE_COLUMNS,
    }
    for filename, columns in specs.items():
        path = output / filename
        if not path.exists():
            pd.DataFrame(columns=columns).to_parquet(path, index=False)
