# PDB/AFDB Pair Validity Analysis

Descriptive pair comparability only; no new uncertainty metric, P/M/SDFI, ranking, regret, mechanism, or H1 analysis.

## Pair validity

- Proteins: 127
- Explicit clean identity/mapping subset: 68
- Groups: {"highly_comparable_identity_mapping": 68, "potentially_confounded_coverage_or_mapping": 36, "unresolved": 23}
- No weighted pair-validity score was constructed.
- The explicit clean subset uses formal exact admission plus mapped_residue_count/canonical_sequence_length >= 0.90, reusing the existing quality value as a descriptive sensitivity criterion rather than changing admission.
- Missing metadata remains unresolved; AFDB confidence values are global descriptors when present, not analyzed-region ground truth.

## Relation to remodeling

- Existing protein-level median |D|, full cohort: 0.0017133271632093772
- Existing protein-level median |D|, explicit clean subset: 0.0012478368553101304
- Clean-minus-full median difference: -0.0004654903078992467

## Interpretation

The comparison uses the frozen cohort, existing common masks, admission evidence, and the existing structural-response protein summary. It does not establish biological structural uncertainty; it describes paired structural representation variation and experimental–predicted structural discrepancy.

## Next scientific gate

If pair validity is judged adequate, the next separately authorized analysis is LOCAL_INVERSE_FOLDING_REMODELING_ANALYSIS.
