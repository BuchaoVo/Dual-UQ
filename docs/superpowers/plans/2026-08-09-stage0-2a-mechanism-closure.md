# Stage0-2A Local Mechanism Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the immutable, offline Stage0-2A mechanism closure from frozen continuous sensitivity, local decision sensitivity, and score artifacts.

**Architecture:** A single `dual_uq.dataset.fixed_probe_mechanism` module validates checkpoint-bound inputs, constructs one canonical position table and one protein-wise association table, and renders both plus a manifest. The CLI is a thin path/exception adapter. Existing score reconstruction, tie ranking, and immutable writers are reused rather than copied.

**Tech Stack:** Python 3.11, pathlib, dataclasses, Pandas, NumPy, PyArrow-backed Parquet, pytest, Ruff.

## Global Constraints

- Work only from frozen Stage0 artifacts at checkpoint `9cf144656b96ef26a14c42d7a7ff6d400e2c8f15`.
- Never execute ProteinMPNN, generate sequences, acquire data, or start Stage0-2B.
- Use all 1,790 canonical positions and the frozen eight-protein order.
- Use exactly 30 repeats and exactly 20 standard amino acids per local landscape.
- Compute descriptive Pearson/Spearman coefficients only; no hypothesis testing or p-values.
- Write immutable outputs and rehash all upstream inputs before and after materialization.

---

### Task 1: Input integrity and exact position join (Layers 1–2)

**Files:**
- Create: `tests/dataset/releases/test_fixed_probe_mechanism.py`
- Create: `src/dual_uq/dataset/fixed_probe_mechanism.py`

**Interfaces:**
- Produces: `MechanismError`, `MechanismInputs`, `load_frozen_mechanism_inputs(Path)`, `join_position_mechanism_inputs(...)`.
- Consumes: `load_frozen_decision_inputs(Path)` and checkpoint-bound decision manifest.

- [ ] Write failing tests for missing/hash-mismatched inputs, duplicate/missing position keys, protein-order mismatch, and a successful exact join.
- [ ] Run `conda run -n dual-uq pytest tests/dataset/releases/test_fixed_probe_mechanism.py -q` and confirm failure because the module/API is absent.
- [ ] Implement structured `BLOCKED` input failures and exact `(protein_id, position)` join with frozen ordering.
- [ ] Re-run the focused tests and confirm the new Layer 1–2 tests pass.

### Task 2: Landscape reconstruction and margins (Layers 3–7)

**Files:**
- Modify: `tests/dataset/releases/test_fixed_probe_mechanism.py`
- Modify: `src/dual_uq/dataset/fixed_probe_mechanism.py`

**Interfaces:**
- Consumes: `build_local_amino_acid_scores`, `rank_local_amino_acid_scores`.
- Produces: `build_repeat_margins`, `summarize_position_margins`, `attach_margin_diagnostics`.

- [ ] Add failing tests for exact 20-AA landscapes, Top-1/Top-2 calculation, canonical exact-tie behavior, negative-margin rejection, practical ties, zero-margin null ratio, and positive-margin ratio.
- [ ] Run the focused file and confirm the expected RED assertions.
- [ ] Implement repeat margins from rank 1/2 rows, aggregate the eight frozen margin diagnostics, and attach ratios without epsilon.
- [ ] Re-run focused tests and confirm Layers 3–7 are GREEN.

### Task 3: Protein-wise descriptive associations (Layers 8–9)

**Files:**
- Modify: `tests/dataset/releases/test_fixed_probe_mechanism.py`
- Modify: `src/dual_uq/dataset/fixed_probe_mechanism.py`

**Interfaces:**
- Produces: `descriptive_correlation`, `summarize_protein_mechanism_associations`.

- [ ] Add failing tests for the exact eight-pair matrix, per-protein isolation, finite Pearson/Spearman coefficients, constant-variable nulls, insufficient-valid-row nulls, and absence of p-value fields.
- [ ] Run focused tests and confirm RED.
- [ ] Implement pairwise finite filtering and structured status/reason fields for every coefficient.
- [ ] Re-run focused tests and confirm Layers 8–9 are GREEN.

### Task 4: Project-fork assessment and result validation (Layer 10)

**Files:**
- Modify: `tests/dataset/releases/test_fixed_probe_mechanism.py`
- Modify: `src/dual_uq/dataset/fixed_probe_mechanism.py`

**Interfaces:**
- Produces: `assess_project_fork`, `FixedProbeMechanismResult`, `build_fixed_probe_mechanism_result`, `validate_mechanism_result`.

- [ ] Add failing tests requiring exactly one allowed recommendation, a concise evidence summary answering Q1–Q4, no categorical position labels, and no forbidden statistical fields.
- [ ] Run focused tests and confirm RED.
- [ ] Implement a deterministic multi-criterion evidence renderer over recurrence, direction, magnitude, protein concentration, and full-cohort flip tails.
- [ ] Re-run focused tests and confirm Layer 10 is GREEN.

### Task 5: Deterministic immutable materialization (Layers 11–13)

**Files:**
- Modify: `tests/dataset/releases/test_fixed_probe_mechanism.py`
- Modify: `src/dual_uq/dataset/fixed_probe_mechanism.py`
- Create: `scripts/dataset/analyze_fixed_probe_mechanism.py`

**Interfaces:**
- Produces: `materialize_fixed_probe_mechanism`, `run_fixed_probe_mechanism`, CLI `main()`.

- [ ] Add failing tests for deterministic row/column order, manifest-last output, `reused_identical`, conflicting-output rejection, pre/post input rehashing, and structured CLI failure.
- [ ] Run focused tests and confirm RED.
- [ ] Implement immutable two-Parquet-plus-manifest rendering and the thin CLI.
- [ ] Re-run focused tests and confirm Layers 11–13 are GREEN.

### Task 6: Formal execution and regression validation

**Files:**
- Create by execution: `experiments/p2_design_baseline/stage0/fixed_probe_position_mechanism.parquet`
- Create by execution: `experiments/p2_design_baseline/stage0/fixed_probe_protein_mechanism_association.parquet`
- Create by execution: `experiments/p2_design_baseline/stage0/fixed_probe_mechanism_manifest.json`

**Interfaces:**
- Command: `conda run -n dual-uq python scripts/dataset/analyze_fixed_probe_mechanism.py --project-root .`

- [ ] Hash all frozen inputs immediately before execution.
- [ ] Run the CLI once and require 1,790/8 rows plus a complete manifest.
- [ ] Run it again and require all three write statuses to be `reused_identical`.
- [ ] Compare pre/post input hashes byte-for-byte.
- [ ] Run `conda run -n dual-uq pytest tests/dataset/releases/test_fixed_probe_mechanism.py -q`.
- [ ] Run `conda run -n dual-uq pytest tests/dataset/releases -q`.
- [ ] Run `conda run -n dual-uq pytest -q` and document any known unrelated baseline failure separately.
- [ ] Run Ruff on the new module, CLI, and test; run `python -m json.tool` on the manifest and scoped `git diff --check`.
- [ ] Inspect the final manifest and tables for all invariants and produce the required PASS/BLOCKED/FAIL report without starting the recommended next task.
