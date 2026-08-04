# Dataset-A Batch-1 Acquisition Plan v1

**Status:** offline acquisition plan; human authorization required.

The 48 candidates remain an acquisition panel, not a final inferential or formal mechanism-balanced panel. Acquisition success does not imply Dataset-A admission.

## Input bindings

- Discovery: `data/processed/discovery/discovered_candidates.parquet` — `dddb21ef1e41babb71827ba729eabe5c568ab6a7d6d6c5953f6da2e10bd98436`
- Inventory: `reports/dataset_a_scale/candidate_inventory_v1.json` — `05ee31ce8e13b699aa36ae0501a934f07b487ca9a5825e1c07a3c8c0e8f15fdf`
- Batch-1: `reports/dataset_a_scale/batch1_plan_v1.tsv` — `013ea4d4dd78a0269cf08afb7fe9294fb3ac1ba27bc131b875ff8ba43fec8f15`

## Repaired-prior invariant

- global pLDDT alone cannot establish `easy_control_prior`.
- Every non-unknown prior in the inventory has positive observable evidence.
- All 48 acquisition candidates are currently `uncertain_or_unclassified`.

## External acquisition counts

| Asset | Candidates |
|---|---:|
| `pdb_mmcif` | 10 |
| `sifts` | 10 |
| `afdb_metadata` | 38 |
| `afdb_structure` | 38 |
| `afdb_pae` | 38 |
| `afdb_confidence` | 38 |

## Local derivations after acquisition

| Derivation | Candidates |
|---|---:|
| `canonical_sequence_metadata_extraction` | 38 |
| `pair_qc` | 48 |
| `residue_mapping` | 48 |
| `fragment_resolution` | 48 |
| `p0_eligibility_inputs` | 48 |
| `mechanism_observability_features` | 48 |

## Primary acquisition classes

- `needs_afdb_side_acquisition`: 37
- `needs_both_structure_sides`: 1
- `needs_mapping_derivation_only`: 1
- `needs_pdb_side_acquisition`: 9

## Required post-acquisition flow

```text
acquire assets
→ verify hashes / identities
→ construct pair QC + mapping
→ evaluate fragment/model identity
→ recompute sampling priors
→ select final Round-2 census panel
→ only then run final P0/P1 protocol
```

Post-acquisition outputs must distinguish `still_unobservable`, `observed_no_positive_prior`, `easy_control_prior`, `low_confidence_local_prior`, `high_pae_long_range_prior`, and `state_disagreement_prior`. No mechanism prevalence claim may use the pre-acquisition distribution.

## H6 human-authorization package

Canonical sequence metadata requires no independent UniProt acquisition. For the 38 missing records it is extracted locally, without sequence-identity inference, from the already planned identity-validated AFDB metadata fields `uniprotAccession` and `uniprotSequence`/`sequence`.

AFDB confidence JSON is a direct external asset referenced by `metadata.plddtDocUrl`. Its raw bytes are stored under the normalized local filename `plddt.json`; no numeric transformation creates that file.

| External source category | Asset | Candidates | Destination / source binding | Purpose |
|---|---|---:|---|---|
| PDB archive | `pdb_mmcif` | 10 | `data/raw/pdb/{PDB}.cif`; PDB accession {PDB} -> deposited mmCIF raw bytes | experimental coordinates and residue provenance |
| PDBe SIFTS | `sifts` | 10 | `data/raw/mappings/{PDB}.xml.gz`; PDB accession {PDB} -> SIFTS XML raw bytes | PDB-to-UniProt residue mapping provenance |
| AlphaFold DB | `afdb_metadata` | 38 | `data/raw/afdb/{UniProt}/metadata.json`; AFDB prediction API record -> identity-validated metadata JSON | fragment/model identity and canonical sequence metadata |
| AlphaFold DB | `afdb_structure` | 38 | `data/raw/afdb/{UniProt}/{model_id}/model.cif`; AFDB metadata.cifUrl raw mmCIF bytes -> local model.cif | predicted structural condition |
| AlphaFold DB | `afdb_pae` | 38 | `data/raw/afdb/{UniProt}/{model_id}/pae.json`; AFDB metadata.paeDocUrl raw JSON bytes -> local pae.json | long-range uncertainty observability |
| AlphaFold DB | `afdb_confidence` | 38 | `data/raw/afdb/{UniProt}/{model_id}/plddt.json`; AFDB metadata.plddtDocUrl raw JSON bytes -> local plddt.json; no numeric transform | mapped local-confidence observability |

### Local derivations after authorized acquisition

| Local derivation | Candidates | Input dependency |
|---|---:|---|
| `canonical_sequence_metadata_extraction` | 38 | identity-validated AFDB metadata.uniprotAccession plus uniprotSequence/sequence |
| `pair_qc` | 48 | PDB mmCIF + SIFTS + selected AFDB metadata/model |
| `residue_mapping` | 48 | PDB mmCIF + SIFTS + canonical UniProt sequence metadata |
| `fragment_resolution` | 48 | mapped UniProt interval + complete AFDB prediction metadata |
| `p0_eligibility_inputs` | 48 | pair QC + residue mapping + verified raw-input identities |
| `mechanism_observability_features` | 48 | paired structures + residue mapping + AFDB PAE/confidence |

This package requests human authorization only for the listed external assets; it is not itself an authorization. It does not authorize executable model downloads, training, evaluator execution, ProteinMPNN generation, P0/P1/P2 execution, or candidate admission. AFDB structure files above are scientific data assets, not executable model weights.
