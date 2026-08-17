# APO/HOLO Generative Propagation — Cross-Model Analysis

- Verdict: `LIMITED_PRIMARY_INTERSECTION`
- Primary requested cohort: `1359` proteins
- Actual shared generation intersection: `1357` proteins
- Statistical unit: protein; generated sequences are not treated as independent biological replicates.

## Q1 — Does between-state divergence exceed within-state diversity?

- ProteinMPNN: positive D_excess in 1353/1357 proteins; median 0.12598214446346334, q10 0.04782584641902436, q90 0.27567748057979125.
- ESM-IF1: positive D_excess in 1355/1357 proteins; median 0.2464505224500012, q10 0.14462565342958356, q90 0.37191475581412753.
- D_excess is independent-sample normalized Hamming divergence minus the mean within-state divergence; it describes marginal sequence separation, not a full joint distribution distance.

## Q2 — Does local state response propagate to generated position distributions?

- ProteinMPNN: protein-level generative JS burden median 0.18909516540901075 bits (q10 0.0814472061688277, q90 0.40853452296057363).
- ESM-IF1: protein-level generative JS burden median 0.29270440283733634 bits (q10 0.1700928101266747, q90 0.4667197232986421).
- Position-level JS values are empirical 20-AA distributions from the same 64-sequence ensembles and are summarized at protein level.

## Q3 — Does local response track generative propagation?

- `proteinmpnn_local_to_d_excess`: Spearman rho 0.8854768505731722 (n=1357).
- `proteinmpnn_local_to_generative_js`: Spearman rho 0.8730727963825923 (n=1357).
- `esm_if1_local_to_d_excess`: Spearman rho 0.8473573711978422 (n=1357).
- `esm_if1_local_to_generative_js`: Spearman rho 0.8347065597991151 (n=1357).
- These are descriptive protein-level associations; no p-values or new composite uncertainty metric are introduced.

## Q4 — Is generative propagation shared across architectures?

- D_excess cross-model Spearman rho: `0.7235314836678607`.
- Generative JS cross-model Spearman rho: `0.78042013728218`.
- Frozen local-response reference rho: `0.783822`; observed local rho on this shared generation intersection: `0.7834792418668111`.

## Q5 — How heterogeneous are the effects?

- ProteinMPNN: D_excess q10/median/q90 = 0.04782584641902436 / 0.12598214446346334 / 0.27567748057979125; JS q10/median/q90 = 0.0814472061688277 / 0.18909516540901075 / 0.40853452296057363 bits.
- ESM-IF1: D_excess q10/median/q90 = 0.14462565342958356 / 0.2464505224500012 / 0.37191475581412753; JS q10/median/q90 = 0.1700928101266747 / 0.29270440283733634 / 0.4667197232986421 bits.

## Convergence

- ESM-IF1 n=16: D_excess median 0.24665428321678323, positive 1358/1359, rank rho to final 0.9932924429415378.
- ESM-IF1 n=32: D_excess median 0.24554396712158807, positive 1357/1359, rank rho to final 0.9980043475026387.
- ESM-IF1 n=64: D_excess median 0.2464505224500012, positive 1357/1359, rank rho to final 1.0.
- ProteinMPNN n=16: D_excess median 0.12651367187499996, positive 1355/1357, rank rho to final 0.9972589799785799.
- ProteinMPNN n=32: D_excess median 0.12626790081778355, positive 1355/1357, rank rho to final 0.9990769895455409.
- ProteinMPNN n=64: D_excess median 0.12598214446346334, positive 1353/1357, rank rho to final 1.0.

## Structural association

- proteinmpnn proteinmpnn_d_excess_64 vs aligned_ca_rmsd: Spearman rho 0.8001211964758745 (n=1357).
- proteinmpnn proteinmpnn_js_bits_mean_64 vs aligned_ca_rmsd: Spearman rho 0.7995324492595887 (n=1357).
- proteinmpnn proteinmpnn_d_excess_64 vs median_residue_displacement: Spearman rho 0.8070326361379313 (n=1357).
- proteinmpnn proteinmpnn_js_bits_mean_64 vs median_residue_displacement: Spearman rho 0.8086801189171509 (n=1357).
- proteinmpnn proteinmpnn_d_excess_64 vs p90_residue_displacement: Spearman rho 0.8142485700841423 (n=1357).
- proteinmpnn proteinmpnn_js_bits_mean_64 vs p90_residue_displacement: Spearman rho 0.8146382125089174 (n=1357).
- proteinmpnn proteinmpnn_d_excess_64 vs median_local_pairwise_distance_change: Spearman rho 0.8959367697235936 (n=1357).
- proteinmpnn proteinmpnn_js_bits_mean_64 vs median_local_pairwise_distance_change: Spearman rho 0.9000269731022639 (n=1357).
- proteinmpnn proteinmpnn_d_excess_64 vs contact_turnover_fraction_mean: Spearman rho 0.8767372825150804 (n=1357).
- proteinmpnn proteinmpnn_js_bits_mean_64 vs contact_turnover_fraction_mean: Spearman rho 0.8722159494428053 (n=1357).
- esm_if1 esm_if1_d_excess_64 vs aligned_ca_rmsd: Spearman rho 0.569043607114003 (n=1357).
- esm_if1 esm_if1_js_bits_mean_64 vs aligned_ca_rmsd: Spearman rho 0.5943788310360368 (n=1357).
- esm_if1 esm_if1_d_excess_64 vs median_residue_displacement: Spearman rho 0.579734916372578 (n=1357).
- esm_if1 esm_if1_js_bits_mean_64 vs median_residue_displacement: Spearman rho 0.60786935360282 (n=1357).
- esm_if1 esm_if1_d_excess_64 vs p90_residue_displacement: Spearman rho 0.5856788914412254 (n=1357).
- esm_if1 esm_if1_js_bits_mean_64 vs p90_residue_displacement: Spearman rho 0.6107849006601985 (n=1357).
- esm_if1 esm_if1_d_excess_64 vs median_local_pairwise_distance_change: Spearman rho 0.6871682669308298 (n=1357).
- esm_if1 esm_if1_js_bits_mean_64 vs median_local_pairwise_distance_change: Spearman rho 0.7288923829405637 (n=1357).
- esm_if1 esm_if1_d_excess_64 vs contact_turnover_fraction_mean: Spearman rho 0.6382693983450088 (n=1357).
- esm_if1 esm_if1_js_bits_mean_64 vs contact_turnover_fraction_mean: Spearman rho 0.657072844964362 (n=1357).

## Ligand-related generative response

- proteinmpnn explicit_zero_proximal: n=128, D_excess median 0.16977002635348762, JS median 0.2685094359529119 bits.
- proteinmpnn ligand_proximal: n=1229, D_excess median 0.1231238053037676, JS median 0.18535569196253052 bits.
- esm_if1 explicit_zero_proximal: n=128, D_excess median 0.29307249631704535, JS median 0.38932048176745304 bits.
- esm_if1 ligand_proximal: n=1229, D_excess median 0.24362133289802856, JS median 0.2838465960805812 bits.
- These geometry and ligand summaries are descriptive within the available released intersection; they do not establish causality, allostery, fitness, stability, or function.

## Interpretation boundary

Protein-level descriptive generative propagation only; no biological fitness, function, stability, causality, or ligand-mechanism claim.

## Next

`MULTI_STATE_BASELINE_EVALUATION`
