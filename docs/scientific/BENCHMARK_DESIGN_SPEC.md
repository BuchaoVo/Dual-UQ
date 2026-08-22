# StructCal Benchmark Design Specification

## Identity and scope

The public benchmark is **StructCal**. The frozen first release is **StructCal v1**
with release identifier `structcal_v1`. The repository and package retain the names
`Dual-UQ` and `dual_uq`.

StructCal evaluates whether an inverse-folding model remains invariant to nuisance or
representation-level structural variation while retaining selective sensitivity to
biologically meaningful structural-state variation. It is not a generic uncertainty
benchmark, and structural differences are not globally labelled as uncertainty.

## Arm and Track

Arm is the source of structural variation:

- `representation_variation`: frozen PDB/AFDB representation pairs;
- `ligand_state`: frozen Apo/Holo PRIMARY pairs;
- `functional_state`: frozen functional-state PRIMARY pairs;
- `controlled_perturbation`: frozen controlled relational-geometry confirmatory pairs.

Track is the evaluation objective:

1. `TRACK_I_STRUCTURAL_INVARIANCE` asks whether model response remains low under
   nuisance or representation-level variation. StructCal v1 uses the existing 68-pair
   clean PDB/AFDB characterization as `INVARIANCE_NATURALISTIC`. This does not assert
   that every PDB/AFDB difference is ground-truth irrelevant.
2. `TRACK_II_FUNCTIONAL_SENSITIVITY` characterizes response to Apo/Holo and frozen
   functional-state variation. A larger response is not defined as universally better.
3. `TRACK_III_INVARIANCE_SENSITIVITY_CALIBRATION` jointly reports Track-I nuisance
   response and Track-II sensitivity retention. It is a Pareto/multidimensional
   contract, not a third cohort and not an arbitrary scalar weighted score.

The controlled confirmatory cohort remains in Core and structural annotations as a
calibration resource. It is not placed in Track-I because its frozen LOW/MEDIUM/HIGH
tiers intentionally manipulate relational geometry rather than establish nuisance
invariance.

## Global protein identity, clustering, and split

One canonical biological UniProt identity maps to one `protein_id`, even when it
appears in multiple Arms. StructCal v1 clusters exactly one frozen canonical sequence
per formal protein with MMseqs2 18.8cc5c using minimum identity 0.30, coverage 0.80,
coverage mode 0, and connected-component interpretation.

The global cluster assignment is distinct from every historical method-development or
Arm-specific split. Whole 30%-identity clusters are assigned once to `TRAIN`,
`VALIDATION`, or `LOCKED_TEST` with root seed 20260822. Clusters are indivisible.
Identity-cluster count is not an optimization target. The 70/15/15 target fractions
are applied jointly to normalized protein count, formal-pair count, per-Arm pair count,
and per-state-family pair count. The deterministic greedy objective minimizes the
summed squared target-relative deviation across those pre-outcome metrics. This can
produce asymmetric cluster counts when larger clusters are assigned to validation or
test partitions. No model result, model availability, score, response, divergence, or
evaluator outcome may affect clustering, splitting, membership, or Track assignment.

## Release cardinality hierarchy

Counts at different scientific levels are not interchangeable. The PDB/AFDB Arm has
127 structural pairs covering 113 canonical proteins; 14 proteins contribute two
formal PDB/AFDB comparisons each. Its frozen Track-I clean subset has 68 pairs from 60
proteins in 57 global identity clusters.

The controlled confirmatory Arm has 567 pair-level benchmark observations, formed by
three relational tiers (`LOW_RELATIONAL`, `MEDIUM_RELATIONAL`, and `HIGH_RELATIONAL`)
within each of 189 complete parent-dose strata from 98 parent proteins. Thus
`567 = 189 × 3`; these observations and repeated dose strata are not independent
biological replicates. Twenty-nine incomplete parent-dose strata remain in the frozen
construction audit and contribute no selected Core pair.

## Statistical contract

The pair is the benchmark measurement unit, the protein is the primary biological
aggregation unit, and the global 30%-identity cluster is the inferential independence
and bootstrap unit. Aggregation proceeds residue → pair → protein → identity cluster →
cohort. Repeated state pairs or perturbation doses from one protein are not independent
biological replicates.

Primary uncertainty intervals use 10,000 percentile-bootstrap replicates at 95%
confidence, resampling identity clusters with equal cluster contribution. Comparisons
between systems use paired cluster bootstrap with the same resampled clusters.

## Generation and seed contract

The canonical protocol requests 64 sequences per structural condition. Models with a
meaningful native standard temperature use 0.1; otherwise temperature is
`NOT_APPLICABLE`. Native autoregressive generation cannot be replaced by independent
per-position sampling. Paired conditions request explicit seeds 0 through 63, without
altering native decoding semantics. Hash-derived seed generation is not used.

## Evaluator contract

The initial reference evaluator panel is ProteinMPNN and ESM-IF1. Generator and
evaluator roles remain distinct. SELF, cross-ProteinMPNN, and cross-ESM-IF1 results are
reported separately. Raw scores from different evaluators are never averaged.
Evaluator outputs are model-based sequence–structure compatibility, not direct
measurements of physical stability, experimental fitness, or biological function.

## Frozen upstream boundary

StructCal release conversion is deterministic projection only. It does not rerun or
change candidate discovery, scientific admission, PRIMARY/ALTERNATIVE decisions,
orientation, residue mapping, structural annotation, controlled intervention design,
model inference, generation, scoring, or mechanism analysis.
