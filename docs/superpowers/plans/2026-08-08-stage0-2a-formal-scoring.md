# Stage-0-2A Formal Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Materialize the frozen 30-repeat ProteinMPNN WT, fixed-probe, and same-state-null score artifacts without performing cross-condition scientific analysis.

**Architecture:** Extend the existing Stage0 scoring orchestrator with immutable G2/upstream/model gates, one resumable shard per protein/backbone/repeat, deterministic consolidation, and a single model-environment worker. Keep Torch isolated in the model worker and keep the CLI thin.

**Tech Stack:** Python, pandas/PyArrow, NumPy, ProteinMPNN/PyTorch, Parquet, pytest.

## Global Constraints

- Reuse the exact 8-protein manifest, 34,010 probes, G2 SHA, model commit, checkpoint SHA, protocol, 30 repeats, and seeds 0..29.
- Never generate sequences, rebuild masks, infer cross-condition effects, download assets, or overwrite conflicting immutable artifacts.
- Preserve unrelated dirty/untracked files and do not commit.
- Logical shards are `protein_id × backbone_condition × repeat_index`; candidate batches are execution-only.

---

### Task 1: Freeze formal-input gates

**Files:**
- Modify: `src/dual_uq/dataset/fixed_probe_scoring.py`
- Test: `tests/dataset/releases/test_fixed_probe_scoring.py`

- [ ] Add failing tests for exact G2 SHA/contract and 30-seed schedule.
- [ ] Implement a pure frozen-G2 loader and validate all upstream/model bindings before model execution.
- [ ] Run focused tests to GREEN.

### Task 2: Define and validate resumable shard contracts

**Files:**
- Modify: `src/dual_uq/dataset/fixed_probe_scoring.py`
- Test: `tests/dataset/releases/test_fixed_probe_scoring.py`

- [ ] Add failing tests for shard bindings, incomplete rows, candidate identity drift, realization mismatch, non-finite scores, and immutable/reusable status.
- [ ] Implement canonical shard metadata, deterministic row schema, exact validator, and atomic shard writer.
- [ ] Run focused tests to GREEN.

### Task 3: Execute formal shards through the model boundary

**Files:**
- Modify: `src/dual_uq/dataset/services/proteinmpnn_scoring.py`
- Modify: `src/dual_uq/dataset/fixed_probe_scoring.py`
- Modify: `scripts/dataset/proteinmpnn_g2_worker.py`
- Modify: `scripts/dataset/score_fixed_probes.py`
- Test: `tests/dataset/releases/test_fixed_probe_scoring.py`

- [ ] Add failing tests for formal request identity, shared realization, WT inclusion, exact candidate count, batching invariance, and CLI formal mode.
- [ ] Extend the process-boundary worker with one shard request/response mode.
- [ ] Implement serial shard scheduling/resume and structured progress without any G2 rerun.
- [ ] Run focused tests to GREEN.

### Task 4: Consolidate and validate canonical outputs

**Files:**
- Modify: `src/dual_uq/dataset/fixed_probe_scoring.py`
- Test: `tests/dataset/releases/test_fixed_probe_scoring.py`

- [ ] Add failing tests for exact WT pairing/delta, row counts, per-protein arithmetic, complete grids, key uniqueness, pairing fingerprints, deterministic ordering, and upstream immutability.
- [ ] Implement deterministic consolidation to 480 WT and 2,040,600 probe rows.
- [ ] Implement the 68,020-row empirical same-state null using population standard deviation (`ddof=0`).
- [ ] Implement immutable final writes and a manifest created only after all final validations pass.
- [ ] Run focused tests to GREEN.

### Task 5: Execute and verify the frozen formal run

**Runtime outputs:**
- `experiments/p2_design_baseline/stage0/fixed_probe_scoring_manifest.json`
- `experiments/p2_design_baseline/stage0/fixed_probe_wt_scores.parquet`
- `experiments/p2_design_baseline/stage0/fixed_probe_scores.parquet`
- `experiments/p2_design_baseline/stage0/fixed_probe_scoring_null.parquet`
- resumable shards under `runs/design_baseline/stage0-2a/`

- [ ] Reverify upstream, G2, submodule, checkpoint, and clean tracked worktree.
- [ ] Execute all 480 logical shards with resume enabled and no force.
- [ ] Consolidate only after every shard validator passes.
- [ ] Validate exact counts, hashes, finite values, joins, pairing, ordering, and immutable reuse.
- [ ] Run focused, dataset, full pytest, Ruff, compileall, JSON, Parquet, and scoped-diff checks.
- [ ] Confirm no cross-condition analysis, generation, evaluator, network, Arm-B, or Stage0-2B work occurred.
