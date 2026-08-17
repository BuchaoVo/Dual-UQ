# APO/HOLO Generative Propagation

- Model: `ESM-IF1`
- Protein-level cohort: `1359`
- Generated records: `173952`
- Conditions: independent APO and HOLO ensembles; 64 sequences per condition at temperature 0.1.
- Nested checkpoints reuse the same 64-sequence ensembles: 16, 32, and 64.

## Q1 — Between-structure divergence

- D_excess (D_AH − mean(D_APO, D_HOLO)) median `0.246451`, q10 `0.144682`, q90 `0.373192`.
- Positive D_excess: `1357/1359` proteins.
- This is a within-protein Hamming-based divergence comparison, not a full joint-sequence distribution distance.

## Q2 — Position-level generative shift

- Mean position-level Jensen–Shannon burden at 64 samples: median `0.292704` bits, q10 `0.170150`, q90 `0.466698`.
- Position-level empirical distributions are reported in the canonical position table.

## Q3 — Upstream descriptor propagation

- Protein-level association with frozen local-response and structural descriptors is a downstream join; no new composite uncertainty metric is introduced.

## Q4 — Cross-structure compatibility

- Cross-structure compatibility scoring is a separate analysis layer and is not recomputed here.

## Q5 — Heterogeneity

- Protein-level D_excess spans q10–q90 `0.144682`–`0.373192`; positive-fraction reporting preserves protein-level heterogeneity.

Interpretation is limited to ProteinMPNN/ESM-IF1 generated-distribution separation under the frozen apo/holo protocol; no claims about fitness, stability, function, or causality are made.
