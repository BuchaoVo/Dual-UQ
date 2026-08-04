# Dataset-A Candidate Inventory v1

**Status:** metadata-only planning inventory; not candidate admission.

## Canonical universe

- Source: `data/processed/discovery/discovered_candidates.parquet`
- SHA-256: `dddb21ef1e41babb71827ba729eabe5c568ab6a7d6d6c5953f6da2e10bd98436`
- Total rows: 220
- Eligible rows: 213
- Indexing policy: `inventory-source-order-v1`
- Identity note: candidate_index is a 1-based inventory convenience ID in canonical eligible source order. It is not historical screening_index, does not represent admission, and is not a permanent protein identity.

## Evidence policy

sampling_stratum_prior is a non-inferential sampling aid; it is not a formal P6 mechanism label and does not change candidate admission. Missing local evidence remains null/unknown. No formal P6 mechanism label was recomputed.

## Inventory summary

- Candidate count: 213
- Local-data-complete: 26
- Mapping-ready: 26
- Fragment-ready: 60
- P0-likely-ready: 19
- Canonical UniProt length: n=89, min=125, q25=205, median=246, q75=314, max=7.1e+03
- Mapped length: n=26, min=125, q25=212, median=252, q75=308, max=499

### Proxy provenance

- `null_policy`: Unavailable evidence remains null; availability checks remain false.
- `identity_source_available`: True only when local pair-QC or Round-1 provides a numeric sequence-identity value; canonical discovery identity is tracked separately.
- `canonical_uniprot_length`: Historical preflight/lifecycle canonical length, then pair-QC length, then an explicit AFDB interval starting at 1 whose sequence length equals its end.
- `mapping_fields`: Existing pair-QC and residue_mapping.parquet only.
- `fragment_fields`: Existing local AFDB metadata and cached preflight intervals only.
- `PDB_CA_coverage_proxy`: Existing preflight/lifecycle value, then pair_geometry_qc.
- `global_pLDDT_proxy`: Canonical discovery-table afdb_global_plddt.
- `sampling_prior_positive_evidence`: A non-unknown sampling prior requires a complete, quality-passing historical A0 classification, its explicit positive label/evidence flag, and the required local PDB/AFDB/PAE-or-pLDDT/mapping provenance. Global pLDDT alone never establishes a prior.
- `low_confidence_fraction_proxy`: Local AFDB metadata fractionPlddtVeryLow + fractionPlddtLow when unambiguous.
- `long_range_PAE_proxy`: Maximum existing A0-summary long-range q90; never recomputed here.
- `existing_PDB_AFDB_disagreement_proxy`: Existing pair_geometry_qc median_aligned_ca_distance; never recomputed here.
- `ligand_annotation_available`: Presence of explicit local mmCIF _pdbx_nonpoly_scheme.
- `interface_annotation_available`: Null because no dedicated local interface-annotation artifact was identified.

### Missing local inputs

- `afdb_metadata`: 148
- `afdb_model`: 148
- `afdb_pae`: 148
- `afdb_plddt`: 148
- `pair_qc`: 187
- `pdb_structure`: 146
- `residue_mapping`: 187
- `sifts_mapping`: 146

### Sampling priors

- `easy_control_prior`: 15
- `low_confidence_local_prior`: 1
- `high_pae_long_range_prior`: 2
- `state_disagreement_prior`: 4
- `uncertain_or_unclassified`: 191

### Prior observability audit

A non-unknown sampling prior requires explicit positive evidence and complete required local provenance. Global pLDDT is ranking metadata only.

- Evidence status `observed_no_positive_prior`: 2
- Evidence status `positive_complete`: 22
- Evidence status `unobserved`: 189
- Prior-observability complete: 23
- Multi-prior overlaps: 2
  - Index 148 `3zoj_A__F2QVG4`: high_pae_long_range_prior, low_confidence_local_prior; selected `high_pae_long_range_prior` by documented A0 precedence.
  - Index 171 `8pb5_A__P0DPA9`: state_disagreement_prior, high_pae_long_range_prior, low_confidence_local_prior; selected `state_disagreement_prior` by documented A0 precedence.

## Round-1 versus remaining pool

Descriptive comparison only; it is not a statistical population inference.
Round-1 is strongly enriched for locally complete inputs; its global-pLDDT distribution is modestly higher. Mechanism-rich priors in Round-1 reflect existing completed diagnostics and must not be interpreted as remaining-pool prevalence.

- Round-1 count: 26
- Remaining count: 187
- Round-1 global pLDDT: n=26, min=88.1, q25=93.1, median=95.1, q75=96.6, max=98.6
- Remaining global pLDDT: n=187, min=40.3, q25=92.1, median=94.5, q75=96.6, max=98.6
- Round-1 local-data-complete: 26
- Remaining local-data-complete: 0

## Batch-1 acquisition panel

The plan contains 48 candidates (target 48), with unique UniProt accessions and no repeated known sequence cluster. Unknown clusters are left unknown rather than inferred. Round-1 overlap is 0. Every row in `batch1_plan_v1.tsv` records its selection reason. This acquisition panel is not a final inferential or formal mechanism-balanced panel. Post-acquisition re-stratification is mandatory. This is a plan only; P0/P1 were not run.

## Scale feasibility scenarios

These are descriptive scenarios, not confidence intervals.

- A1 (~64): plausible_only_at_about_30_percent_or_higher. A 30% yield is 63.9, so A1 has essentially no margin at that boundary.
- A2 (128–256): 128_requires_at_least_60.1_percent;_256_is_impossible_from_213. The >=40% scenario alone does not establish support for 128, and membership caps the current universe below 256 even at 100% admission.
- Likely limiting sampling priors: low_confidence_local_prior, high_pae_long_range_prior, state_disagreement_prior
- Expansion trigger: Start external-pool planning after Batch-1 if observed conservative yield projected over 213 is below the target, or if any required uncertainty-rich stratum remains too sparse for its planned representation; do not wait for full-pool exhaustion.

## Guardrails

No candidate was admitted or rejected. No download, P0, P1, P2, ProteinMPNN, or evaluator execution occurred.
