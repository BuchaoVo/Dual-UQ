# Stage0-2A Local Mechanism Closure Design

Status: approved for implementation; scientific execution remains limited to frozen Stage0 artifacts.

## Scope

This analysis joins the already-frozen continuous structural-sensitivity and local decision-sensitivity releases for all 1,790 canonical positions. It reconstructs the existing WT-plus-19-mutant score landscapes from frozen score tables, derives Top-1/Top-2 margins, computes protein-wise descriptive associations, and records one project-fork recommendation. It does not execute ProteinMPNN or create candidates.

## Trust root and inputs

The Stage0 scientific trust root is checkpoint commit:

`9cf144656b96ef26a14c42d7a7ff6d400e2c8f15`

The completed decision-sensitivity manifest is the immediate upstream contract. Its own bytes are bound to the checkpoint; its input/output records provide the canonical paths, SHA256 values, and row counts for:

- scoring manifest;
- WT score table;
- fixed-probe score table;
- continuous-sensitivity manifest and position table;
- decision-sensitivity position table.

All SHA, schema, count, protein-membership, position-key, repeat, realization, and 20-amino-acid checks complete before mechanism outputs are constructed. A mismatch produces a structured `BLOCKED` result.

## Canonical dataflow

```text
checkpoint-bound decision manifest
        +
frozen continuous and decision position tables
        +
frozen WT and 19-mutant scores
        ↓
input integrity gate
        ↓
exact (protein_id, position) 1,790-row join
        ↓
existing 20-AA local-score builder and frozen rank semantics
        ↓
repeat-level Top-1/Top-2 margins (in memory only)
        ↓
position margin aggregation and perturbation/margin ratio
        ↓
protein-wise descriptive Pearson/Spearman matrix
        ↓
single FixedProbeMechanismResult
        ↓
two immutable Parquet outputs + manifest-last JSON
```

Only the repeat-level reconstruction touches raw WT/mutant scores. Position and protein outputs consume the canonical in-memory results.

## Interfaces

`MechanismInputs` is an immutable, lightweight binding for paths, hashes, protein order, the two position tables, and the existing validated decision inputs.

`FixedProbeMechanismResult` contains:

- `input_provenance`;
- `position_mechanism`;
- `protein_associations`;
- `project_fork`;
- frozen protein order and repeat count.

The public entrypoint is:

```python
run_fixed_probe_mechanism(project_root: Path) -> FixedProbeMechanismResult
```

The CLI parses the repository root, calls the API, and materializes outputs. It contains no scientific calculations.

## Exact position cohort

The join key is exactly `(protein_id, position)`. Both sides must contain 1,790 unique keys and the same frozen per-protein counts. The joined table preserves the frozen protein order and ascending canonical position. No position selection or tail filtering precedes summaries.

## Margin semantics

For each `(protein_id, position, backbone_condition, repeat_index)` landscape, ranks use the existing higher-is-better score order and canonical amino-acid exact-tie break. The margin is:

```text
top1_top2_margin = highest score - second-highest score
```

Values below the frozen floating tolerance fail; tiny roundoff may be normalized to zero. Practical ties use the existing absolute and relative tolerance and do not alter strict ranks.

Per-position fields are:

- `margin_mean_pdb`, `margin_median_pdb`;
- `margin_mean_afdb`, `margin_median_afdb`;
- `margin_mean_both`, `margin_median_both`;
- `margin_min`;
- `practical_tie_fraction`.

Repeat-level margins remain internal.

## Perturbation relative to margin

For positive `margin_mean_both`:

```text
perturbation_to_margin = position_mean_abs_interaction / margin_mean_both
margin_ratio_status = defined
```

For exactly zero margin, the ratio is null and `margin_ratio_status = zero_margin`. No epsilon is introduced.

## Descriptive association matrix

Pearson and Spearman coefficients are calculated separately for each protein for exactly eight primary pairs:

1. perturbation vs Top-1 disagreement;
2. perturbation vs symmetric regret;
3. perturbation vs normalized rank displacement;
4. margin vs Top-1 disagreement;
5. margin vs symmetric regret;
6. perturbation/margin vs Top-1 disagreement;
7. perturbation/margin vs symmetric regret;
8. perturbation/margin vs normalized rank displacement.

Each pair records valid-row count, Pearson, Spearman, and structured statuses/reasons. Constant or insufficient vectors yield null coefficients, never zero. No pooled inference, hypothesis testing, confidence interval, or p-value is computed.

## Project-fork evidence

The canonical result records full-cohort evidence for:

- direction and recurrence of perturbation associations;
- protective direction and recurrence of margin associations;
- relative magnitude and recurrence of ratio associations;
- concentration across proteins;
- Top-1 flip regret and margin distributions, including a substantive tail.

The final decision is exactly `GO_STAGE0_2B` or `HOLD_STAGE0_2B`. It is a Stage0 evidence recommendation, not a claim that full Dual-UQ is established and not authorization to execute Stage0-2B.

## Release behavior

The position table, protein table, and manifest are rendered from one validated structured result. Parquet files are written first; the manifest is written last. Existing identical bytes return `reused_identical`; conflicting bytes are rejected. Every input is rehashed immediately before and after materialization.

## Explicit exclusions

The implementation performs no model execution, sequence generation, data acquisition, new scoring, p-values, binary UQ labels, biological mutation recommendation, figure generation, Arm-B work, evaluator-UQ, or Stage0-2B execution.
