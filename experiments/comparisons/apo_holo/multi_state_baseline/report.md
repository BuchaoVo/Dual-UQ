# Multi-state APO/HOLO baseline summary

Protein cohort: **1357** shared proteins.

The protein is the inference unit. Sequence diversity is a marginal Hamming summary and is not a joint-distribution or biological replicate claim.

## A–G result inventory

- **A/E.** All model-specific rows are WT-normalized APO/HOLO compatibility endpoints with a MULTI equal-weight joint-decoder condition.
- **B/D.** MULTI is compared with the separate APO and HOLO ensembles; the protein table retains median, q10 and q90 endpoint summaries.
- **F.** `worst_compat` is the primary condition-comparison endpoint; mean compatibility and state gap are secondary.
- **G.** Associations are Spearman summaries against frozen upstream generative/local-response descriptors; no sequence rows are treated as independent proteins.

## Numeric answers

- **Q1.** APO-vs-HOLO marginal between-condition divergence minus the within-condition mean has median **0.1260** and is positive for **0.997** of proteins. This is a marginal Hamming result, not a full joint-sequence distribution claim.
- **Q2/Q3.** The MULTI condition is retained as the equal-weight joint decoder; upstream local-response/generative descriptors are joined without redefining them. Descriptor associations are reported below.
- **ProteinMPNN.** MULTI median worst compatibility = 166.6806; MULTI−best-single median = 37.5778 (positive in 0.976 of proteins); MULTI state-gap median = 9.3272.
- **ESM-IF1.** MULTI median worst compatibility = -129.9729; MULTI−best-single median = -219.7419 (positive in 0.019 of proteins); MULTI state-gap median = 35.8122.
- **ESM-IF1.** strongest recorded descriptor association: esm_if1_js_bits_mean_64 (Spearman rho = 0.4983, n = 1357).
- **ProteinMPNN.** strongest recorded descriptor association: proteinmpnn_local_burden_mean (Spearman rho = 0.7353, n = 1357).
- **Q4.** The state-gap and worst-compatibility columns retain both cross-state compatibility behavior and its heterogeneity; interpretation is restricted to model-level compatibility.
- **Q5.** Heterogeneity is represented by protein-level q10/q90 columns for every endpoint and by the sign fractions above; no sequence-level independence assumption is used.

## Materialized columns

- Protein summary rows: **1357**
- Protein summary columns: **105**
- Full interpretation remains limited to the frozen shared APO/HOLO cohort and the ProteinMPNN/ESM-IF1 model semantics.

Absolute scores are model-level compatibility quantities; no claim about fitness, stability, function, or experimental outcome is made.
