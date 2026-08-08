# Stage0-2A Fixed-Probe Sensitivity Analysis Design

**Status:** APPROVED FOR IMPLEMENTATION
**Date:** 2026-08-08
**Scientific scope:** Stage-0 fixed-probe structural-condition sensitivity only

## 1. Goal and interpretation boundary

Analyze the immutable Stage0-2A ProteinMPNN score artifacts using paired
PDB/AFDB decoding realizations. The primary estimand is the change in a
mutation's WT-relative compatibility effect under the AFDB versus PDB
structural condition. Protein is the pilot-level interpretation unit.

The analysis is descriptive. It does not perform hypothesis testing, binary
uncertainty classification, candidate ranking, regret analysis, sequence
generation, evaluator execution, Arm-B work, or Stage0-2B work.

## 2. Architecture

Create one analysis module:

`src/dual_uq/dataset/fixed_probe_sensitivity.py`

and one thin CLI:

`scripts/dataset/analyze_fixed_probe_sensitivity.py`

The module uses only the repository's existing Pandas, NumPy, JSON, and
Parquet stack. It does not extend the frozen scoring implementation. The CLI
resolves project-relative paths and calls the reusable API; it contains no
statistical logic.

The canonical flow is:

```text
frozen scoring manifest + WT + raw scores + same-state null
    -> input integrity gate
    -> paired candidate-repeat result
    -> candidate summaries
    -> position summaries
    -> protein summaries
    -> one structured analysis result
    -> four Parquet artifacts + one JSON manifest
```

The paired candidate-repeat result is the only analysis layer allowed to
consume raw PDB/AFDB candidate scores. Every downstream aggregate consumes
that paired result or its canonical summaries.

## 3. Frozen inputs and gates

Require the exact paths, SHA-256 values, row counts, schemas, and scientific
keys declared by the Stage0-2A analysis task:

- scoring manifest: `ec9882604cdfad0d61fbd30c1314466fbc7cbbea1fe7173d84c9913309783b00`;
- WT scores: `373948042b6868611ba6bc9ea80253d47cf0b441e002dcc6c10eed57f23b8d31`, 480 rows;
- raw scores: `ac0cf8e1f251770fd4ade08580b8df19000a47124531350e9af63dd71f773abf`, 2,040,600 rows;
- same-state null: `fa248c180444498e89541582edab12ef6d42a822ddbb65bc92e358b383621403`, 68,020 rows.

Also validate the Stage0-1 protein-manifest and fixed-probe hashes bound by the
scoring manifest. Any drift, duplicate key, incomplete grid, schema mismatch,
or non-finite score produces a structured `BLOCKED` result before pairing or
canonical output creation.

Recompute all input hashes immediately before final materialization to prove
the upstream artifacts remained byte-unchanged.

## 4. Paired structural effects

Pair candidate rows exactly on:

```text
protein_id
sequence_hash
repeat_index
decoding_realization_sha256
```

Require exactly one PDB and one AFDB row in every pair and one shared
realization fingerprint. Pair WT scores on protein, repeat, and realization.

The comparison orientation is defined once as `AFDB_minus_PDB`:

```text
raw_structural_shift = raw_AFDB - raw_PDB
wt_structural_shift = WT_AFDB - WT_PDB
structural_mutation_interaction = delta_AFDB - delta_PDB
```

Numerically require:

```text
structural_mutation_interaction
== raw_structural_shift - wt_structural_shift
```

within a strict floating tolerance of `atol=1e-12`, `rtol=1e-12`. Store native
floating values without display rounding.

Canonical paired ordering is frozen protein order, frozen candidate order,
then repeat `0..29`.

## 5. Candidate summaries

For every one of the 34,010 candidates, require exactly 30 repeats and compute:

- mean, population standard deviation (`ddof=0`), minimum, maximum, median;
- `q05`, `q25`, `q50`, `q75`, and `q95` using the linear quantile method;
- positive, negative, and zero-within-tolerance fractions;
- PDB and AFDB same-state technical standard deviations;
- pooled technical scale;
- absolute interaction mean divided by technical scale where defined.

The zero rule is:

```text
abs(structural_mutation_interaction) <= 1e-6
```

Positive and negative fractions use values above `+1e-6` and below `-1e-6`,
respectively, so the three fractions partition the 30 realizations.

The pooled scale is:

```text
sqrt((technical_sd_pdb^2 + technical_sd_afdb^2) / 2)
```

When the scale is zero, the ratio is null and status is
`zero_technical_scale`; otherwise status is `defined`. The ratio remains a
descriptive quantity, not a z-score or decision threshold.

Quantiles are explicitly labelled `decoding_realization_quantile_interval`.

## 6. Position summaries

Aggregate only from the canonical candidate summaries. Each protein-position
must contain exactly 19 unique substitutions. Compute:

- mean interaction;
- mean, RMS, median, and maximum absolute interaction;
- substitution giving the maximum absolute interaction;
- mean technical scale;
- mean defined interaction-to-technical-scale ratio;
- positive/negative mutation counts and fractions.

Maximum ties are resolved by frozen candidate order. Position output follows
frozen protein and common-mask position order. No binary label is emitted.

## 7. Protein summaries

Aggregate only from candidate and position summaries and preserve frozen
protein order. For each of eight proteins record candidate and position counts,
candidate-level mean/RMS/median absolute interaction, linear absolute-interaction
quantiles `q05/q25/q50/q75/q95`, maximum absolute interaction, mean interaction
SD, mean technical scale, defined-ratio count and distribution, and positive/
negative candidate fractions.

Also record position-level mean, median, `q75`, `q95`, and maximum of position
mean-absolute interaction. These are descriptive pilot summaries, not rankings.

## 8. Structured result and outputs

One immutable `FixedProbeSensitivityResult` contains:

- validated input provenance;
- paired effects;
- candidate summaries;
- position summaries;
- protein summaries;
- analysis protocol metadata.

Render exactly:

1. `fixed_probe_structural_effects.parquet` — 1,020,300 rows;
2. `fixed_probe_candidate_sensitivity.parquet` — 34,010 rows;
3. `fixed_probe_position_sensitivity.parquet` — 1,790 rows;
4. `fixed_probe_protein_sensitivity.parquet` — 8 rows;
5. `fixed_probe_sensitivity_manifest.json` — one manifest.

All five use immutable semantics: identical content returns
`reused_identical`; conflicting existing content is rejected. The manifest is
written last and records input paths/hashes, protocol
`stage0_fixed_probe_structural_sensitivity_v1`, orientation, estimand,
technical-scale definition, repeat count, `ddof=0`, quantile convention,
output paths/hashes/counts, and explicit negative declarations for hypothesis
testing and binary classification.

## 9. Failure handling

The public API raises structured `FixedProbeSensitivityError` codes. Input
failures are `BLOCKED`; pairing or algebra failures are `FAIL`. No partial
canonical artifact is eligible for manifest creation. Temporary Parquet files
are removed after success or failure.

## 10. Test contract

Use strict RED -> GREEN layers:

1. input hash/schema/count/key gates;
2. paired PDB/AFDB joins and orientation;
3. interaction algebra;
4. candidate repeat aggregation and technical scale;
5. 19-mutation position aggregation;
6. frozen-membership protein aggregation;
7. deterministic immutable materialization and unchanged upstream hashes.

After focused tests, run dataset release tests, dataset tests, full pytest,
Ruff on touched Python, compileall, JSON validation, and a second identical
analysis invocation proving all outputs are `reused_identical`.
