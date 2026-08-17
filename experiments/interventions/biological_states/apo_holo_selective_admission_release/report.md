# Apo/Holo Selective-Admission Release Verification

## VERDICT

`LIMITED`

The release is internally consistent after a lossless geometry implementation repair. The model-evaluable cohort is large (1,394/1,410 primary pairs), but independent 30% identity cluster counts remain unresolved and all geometry/model exclusions are retained as structured unavailability rather than admission changes.

## RELEASE

- candidate pairs: 4,246
- admitted pairs: 1,636
- primary pairs: 1,410
- alternatives: 226
- unresolved asset: 75
- excluded: 2,535
- independent clusters: unresolved (`UNRESOLVED_NO_FROZEN_IDENTITY_SOURCE`) for every cohort; no identity clustering was inferred
- admission decisions and exact mappings were not changed by this verification/repair

## GEOMETRY DIAGNOSIS

- final geometry descriptors available: 1,399/1,410
- final geometry failures: 11
- dominant final failure taxonomy: `mapping_correspondence_limitation` (11 pairs)
- the prior 1,250 failures were dominated by a technical array/row-alignment defect in the descriptor implementation; canonical-position pairing repaired that defect without changing admission or mapping semantics
- the remaining 11 failures cannot realize three paired finite C-alpha coordinates under the exact released correspondence; they are coordinate/mapping evaluability limits, not evidence of invalid apo/holo identity
- no final geometry failure is marked as a genuine admission-validity concern

## ANALYSIS COHORTS

- `PRIMARY_ADMITTED`: 1,410 pairs / 1,410 proteins; cluster count unresolved
- `MODEL_EVALUABLE`: 1,394 pairs / 1,394 proteins; strict N/CA/C/O and ESM-IF1 N/CA/C preflight pass
- `GEOMETRY_EVALUABLE`: 1,399 pairs / 1,399 proteins; exact canonical-coordinate geometry available
- `LIGAND_LOCALIZATION_EVALUABLE`: 1,267 pairs / 1,267 proteins with at least one proximal residue row; 132 geometry pairs have an explicit zero-proximal result
- independent 30%-identity clusters: unresolved for all four cohorts because no frozen source was available

## SELECTION CHARACTERIZATION

- model-unavailable pairs are descriptively longer (median canonical length 440.0 vs 335.0) and fail complete-backbone representation; sequence identity, common mapping, and assembly admission fields remain comparable
- geometry-unavailable pairs are only 11 cases; their median common mapped count is 159.0 versus 329.0 among available pairs, reflecting correspondence limitations rather than a changed admission rule
- these comparisons are descriptive only; no model outcome was used for selection and no biological enrichment claim is made

## STATE CHARACTERIZATION

- aligned C-alpha RMSD: median 0.8301 Å, q90 3.5557 Å
- median residue displacement: median 0.4437 Å, q90 2.2873 Å
- upper-tail (per-pair p90) displacement: median 1.0332 Å, q90 5.0697 Å
- local pairwise-distance change: median 0.1110 Å, q90 0.2926 Å
- contact-turnover fraction: median 0.0000, q90 0.2000; nonzero in 0.3521 of residue rows
- ligand-site coverage: 1,267 pairs with proximal residues, 132 with explicit zero proximal residues, 61,135 proximal residue rows
- the descriptor range spans small, localized, and broader state changes; no composite severity score was created

## SCIENTIFIC INTERPRETATION

- inverse-folding local-response analysis may proceed only on `MODEL_EVALUABLE` (1,394 pairs / proteins), with the exact common residue set and model-specific backbone requirements enforced
- geometry-response and ligand-localization analyses must be restricted to `GEOMETRY_EVALUABLE` and, for proximal-residue summaries, the nested ligand-localization subset; the 11 correspondence-limited pairs remain unavailable, not rejected
- the release supports no claim about model response, biological fitness, or causal mechanism before the next analysis stage

## NEXT

`APO_HOLO_CROSS_MODEL_LOCAL_RESPONSE`

No inverse-folding scoring or other model execution was started in this verification task.
