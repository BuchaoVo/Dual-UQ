# Stage0-2A Fixed-Probe Sensitivity Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Materialize the five immutable Stage0-2A structural-sensitivity artifacts from the frozen ProteinMPNN scoring release without executing a model.

**Architecture:** A new reusable `dual_uq.dataset.fixed_probe_sensitivity` module owns all input gates, the sole raw-score pairing layer, hierarchical summaries, validation, and immutable rendering. A thin CLI resolves the repository root and invokes one public execution function.

**Tech Stack:** Python 3.11, Pandas, NumPy, PyArrow-backed Parquet, pytest, Ruff.

## Global Constraints

- Consume only the four exact Stage0-2A scoring artifacts and their manifest-bound Stage0-1 inputs.
- Structural orientation is always `AFDB_minus_PDB`.
- Quantiles are `q05/q25/q50/q75/q95`, method `linear`.
- Zero-within-tolerance is `abs(interaction) <= 1e-6`.
- Repeat standard deviation is population `ddof=0`.
- No ProteinMPNN execution, generation, ranking, regret, hypothesis testing, binary labels, external evaluators, Arm-B, or Stage0-2B.
- Preserve all pre-existing dirty/untracked work and do not commit.

---

### Task 1: Frozen Input Contract

**Files:**
- Create: `src/dual_uq/dataset/fixed_probe_sensitivity.py`
- Create: `tests/dataset/releases/test_fixed_probe_sensitivity.py`

**Interfaces:**
- Produces: `SensitivityInputs`, `FixedProbeSensitivityError`, `load_frozen_sensitivity_inputs(project_root: Path) -> SensitivityInputs`.
- Consumes: canonical Stage0 paths and SHA-256 constants from the approved design.

- [ ] **Step 1: Write failing input-gate tests**

Cover exact real hashes/counts plus synthetic wrong hash, missing schema column,
duplicate raw key, and incomplete repeat grid. Assert stable error codes such as
`scoring_manifest_hash_mismatch`, `raw_score_schema_mismatch`,
`duplicate_raw_score_key`, and `raw_repeat_grid_mismatch`.

- [ ] **Step 2: Run RED**

```bash
conda run -n dual-uq pytest tests/dataset/releases/test_fixed_probe_sensitivity.py -q
```

Expected: collection failure because `dual_uq.dataset.fixed_probe_sensitivity`
does not exist.

- [ ] **Step 3: Implement minimal frozen loader**

Define immutable input metadata and load the manifest, WT, raw, and null tables.
Validate all required columns, exact scientific keys, finite numerical fields,
row counts, repeat grids, input hashes, and manifest-bound Stage0-1 hashes
before returning any DataFrame.

- [ ] **Step 4: Run GREEN**

Run the focused test file and require all Layer-1 tests to pass.

---

### Task 2: Sole Paired Candidate-Repeat Layer

**Files:**
- Modify: `src/dual_uq/dataset/fixed_probe_sensitivity.py`
- Modify: `tests/dataset/releases/test_fixed_probe_sensitivity.py`

**Interfaces:**
- Produces: `build_paired_structural_effects(inputs: SensitivityInputs) -> pd.DataFrame`.
- Output key: `(protein_id, sequence_hash, repeat_index)` with one shared `decoding_realization_sha256`.

- [ ] **Step 1: Add RED fixtures**

Create small paired PDB/AFDB/WT fixtures. Assert failures for missing PDB,
missing AFDB, duplicate condition, mismatched realization fingerprint, and
reversed orientation.

- [ ] **Step 2: Implement paired pivot/merge**

Pair on protein, sequence hash, repeat, and realization fingerprint; join WT by
protein/repeat/fingerprint. Compute raw shift, WT shift, and interaction only as
AFDB minus PDB. Retain position and substitution provenance.

- [ ] **Step 3: Enforce algebra and ordering**

Require `delta_AFDB - delta_PDB == raw_shift - wt_shift` using
`atol=rtol=1e-12`. Sort by frozen protein order, frozen candidate order, and
repeat index.

- [ ] **Step 4: Run Layer-2/3 GREEN**

Run focused tests and require paired coverage, orientation, and algebra tests
to pass.

---

### Task 3: Candidate and Technical-Null Summaries

**Files:**
- Modify: `src/dual_uq/dataset/fixed_probe_sensitivity.py`
- Modify: `tests/dataset/releases/test_fixed_probe_sensitivity.py`

**Interfaces:**
- Produces: `summarize_candidate_sensitivity(paired: pd.DataFrame, scoring_null: pd.DataFrame, candidate_order: pd.DataFrame) -> pd.DataFrame`.

- [ ] **Step 1: Add RED aggregation tests**

Test exact 30-repeat enforcement, `ddof=0`, linear quantiles, three sign
fractions partitioning at `1e-6`, pooled technical scale, defined ratio, and
null ratio with `zero_technical_scale`.

- [ ] **Step 2: Implement candidate aggregation**

Group the canonical paired table once per candidate, compute the approved
statistics, pivot the existing null table to PDB/AFDB technical SD fields, and
merge one-to-one by candidate identity. Do not read raw scores in this layer.

- [ ] **Step 3: Run Layer-4 GREEN**

Require exact candidate identity/order and no missing technical fields.

---

### Task 4: Position and Protein Summaries

**Files:**
- Modify: `src/dual_uq/dataset/fixed_probe_sensitivity.py`
- Modify: `tests/dataset/releases/test_fixed_probe_sensitivity.py`

**Interfaces:**
- Produces: `summarize_position_sensitivity(candidate_summary: pd.DataFrame) -> pd.DataFrame`.
- Produces: `summarize_protein_sensitivity(candidate_summary: pd.DataFrame, position_summary: pd.DataFrame, protein_order: tuple[str, ...]) -> pd.DataFrame`.

- [ ] **Step 1: Add RED position tests**

Require 19 unique mutations per protein-position, correct maximum-substitution
tie-breaking by frozen candidate order, and exact 1,790-row reconciliation.

- [ ] **Step 2: Implement position summary**

Compute approved absolute, RMS, technical, ratio, and directional metrics only
from candidate summaries.

- [ ] **Step 3: Add RED protein tests**

Require exact frozen membership/order, eight rows, and exact candidate/position
count reconciliation.

- [ ] **Step 4: Implement protein summary**

Compute candidate-level and position-level descriptive distributions without
ranking or labels.

- [ ] **Step 5: Run Layer-5/6 GREEN**

Run focused tests and require all aggregation invariants to pass.

---

### Task 5: Structured Result, Immutable Rendering, and CLI

**Files:**
- Modify: `src/dual_uq/dataset/fixed_probe_sensitivity.py`
- Create: `scripts/dataset/analyze_fixed_probe_sensitivity.py`
- Modify: `tests/dataset/releases/test_fixed_probe_sensitivity.py`

**Interfaces:**
- Produces: `FixedProbeSensitivityResult`.
- Produces: `run_fixed_probe_sensitivity(project_root: Path) -> FixedProbeSensitivityResult`.
- Produces: `materialize_fixed_probe_sensitivity(result, output_root: Path) -> dict[str, str]`.
- CLI command: `python scripts/dataset/analyze_fixed_probe_sensitivity.py --project-root <root>`.

- [ ] **Step 1: Add RED release tests**

Test exact row counts, finite output, forbidden-label/metric absence,
deterministic ordering, input rehash before output, identical rerun reuse, and
conflicting output rejection.

- [ ] **Step 2: Implement result validation and immutable writers**

Use atomic temporary Parquet writes and byte-identical reuse. Render the JSON
manifest last with relative paths, all hashes, protocol/estimand metadata, and
negative scope declarations.

- [ ] **Step 3: Implement thin CLI**

Resolve `ProjectPaths`, invoke the reusable API, render a structured PASS or
BLOCKED/FAIL response, and contain no Pandas/NumPy operations.

- [ ] **Step 4: Run focused GREEN and Ruff**

```bash
conda run -n dual-uq pytest tests/dataset/releases/test_fixed_probe_sensitivity.py -q
conda run -n dual-uq ruff check src/dual_uq/dataset/fixed_probe_sensitivity.py scripts/dataset/analyze_fixed_probe_sensitivity.py tests/dataset/releases/test_fixed_probe_sensitivity.py
```

---

### Task 6: Real Offline Analysis and Release Verification

**Files:**
- Create: `experiments/p2_design_baseline/stage0/fixed_probe_structural_effects.parquet`
- Create: `experiments/p2_design_baseline/stage0/fixed_probe_candidate_sensitivity.parquet`
- Create: `experiments/p2_design_baseline/stage0/fixed_probe_position_sensitivity.parquet`
- Create: `experiments/p2_design_baseline/stage0/fixed_probe_protein_sensitivity.parquet`
- Create: `experiments/p2_design_baseline/stage0/fixed_probe_sensitivity_manifest.json`

- [ ] **Step 1: Execute the analysis once**

```bash
conda run -n dual-uq python scripts/dataset/analyze_fixed_probe_sensitivity.py --project-root /home/zbc/data/AI4S/ProteinDesign/Dual-UQ
```

Expected: PASS with 1,020,300 paired rows, 34,010 candidates, 1,790 positions,
8 proteins, and five `created` outputs.

- [ ] **Step 2: Independently validate artifacts**

Verify hashes, schemas, row counts, unique keys, repeat/mutation grids, finite
values, exact orientation/algebra, technical-scale semantics, frozen order,
absence of forbidden fields, and unchanged upstream SHA-256 values.

- [ ] **Step 3: Execute an identical rerun**

Expected: all five outputs `reused_identical`; no upstream mutation.

- [ ] **Step 4: Run regressions**

```bash
conda run -n dual-uq pytest tests/dataset/releases -q
conda run -n dual-uq pytest tests/dataset -q
conda run -n dual-uq pytest -q
conda run -n dual-uq python -m compileall -q src scripts/dataset/analyze_fixed_probe_sensitivity.py
```

If the known legacy namespace/history-path check is the only failure, rerun
dataset and full suites with exactly that test deselected and report both raw
and deselected results. Do not rewrite frozen historical paths.

- [ ] **Step 5: Final scope audit**

Confirm no active model process, no network/download, no generated sequences,
no Top-k/rank/regret or hypothesis-test columns, no binary label, no Arm-B, no
Stage0-2B, no commit, and all unrelated worktree changes preserved.
