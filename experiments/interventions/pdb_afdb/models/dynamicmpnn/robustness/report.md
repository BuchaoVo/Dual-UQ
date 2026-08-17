# DynamicMPNN baseline evaluation

VERDICT: **LIMITED**

The DynamicMPNN result is scoped to 365 proteins with complete DynamicMPNN, ProteinMPNN, and ESM-IF1 scoring paths. The frozen baseline contains 1357 proteins.
Generation exclusions were {'both_conformations_missing_coordinate': 991, 'dynamicmpnn_featurizer_unavailable': 1}.

No absolute compatibility score is interpreted across unrelated proteins. Protein is the inference unit; evaluator scales remain separate.

## Q1. Cross-evaluator robustness

Against best single-state, DynamicMPNN improved WorstCompat under both evaluators for 0/365 proteins; it improved ProteinMPNN only for 1 and ESM-IF1 only for 0. WorstCompat worsened under both for 364 proteins.
Against simple MULTI, the corresponding counts were 0 improved under both and 363 worsened under both.

## Q2. Evaluator-specific compatibility

ProteinMPNN DynamicMPNN WorstCompat median/q10/q90 = -299.917/-615.345/-32.299; ESM-IF1 = -784.367/-1438.035/-125.536.
The DynamicMPNN result is therefore not an evaluator-general improvement over the frozen single-state baseline in this evaluable subset.

## Q3. DynamicMPNN versus simple MULTI

Median DynamicMPNN-minus-simple-MULTI WorstCompat = -442.231 for ProteinMPNN and -644.938 for ESM-IF1.

## Q4. Heterogeneity

DynamicMPNN diversity median = 0.353 (ProteinMPNN scoring view) and 0.353 (ESM-IF1 scoring view); the protein-level q10/q90 ranges above show broad heterogeneity rather than a uniform shift.

## Q5. Relation to frozen sensitivity descriptors

- esm_if1 vs best_single: esm_if1_js_bits_mean_64, Spearman rho=0.508 (n=365)
- esm_if1 vs best_single: esm_if1_d_excess_64, Spearman rho=0.446 (n=365)
- esm_if1 vs multi: esm_if1_js_bits_mean_64, Spearman rho=0.442 (n=365)
- esm_if1 vs best_single: proteinmpnn_js_bits_mean_64, Spearman rho=0.441 (n=365)

## Boundaries

The analysis tests model-level multi-state compatibility only. It does not claim biological fitness, stability, function, or causal mechanism.
