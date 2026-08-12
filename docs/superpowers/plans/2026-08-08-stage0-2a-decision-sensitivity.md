# Stage0-2A Local Decision Sensitivity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Derive immutable local 20-amino-acid PDB-versus-AFDB decision-sensitivity and same-state decoding-order null artifacts from the frozen Stage0-2A scores without executing ProteinMPNN.

**Architecture:** A new `dual_uq.dataset.fixed_probe_decision_sensitivity` module owns one raw-score entry layer, reconstructs and ranks the canonical 20-AA local landscapes once, and derives repeat, technical-null, position, and protein results from that canonical representation. A thin CLI resolves portable paths and invokes the API. Four Parquet outputs and one JSON manifest are rendered from a single structured result with immutable reuse semantics.

**Tech Stack:** Python 3.10, Pandas, NumPy, PyArrow/Parquet, pytest, Ruff.

## Global Constraints

- Consume only the frozen Stage0 scoring/sensitivity artifacts and verify every bound SHA before analysis.
- Do not modify or invoke `fixed_probe_scoring.py` or ProteinMPNN.
- Use `score_mean_logp_mask`, with higher scores preferred.
- Use the canonical amino-acid order `ACDEFGHIKLMNPQRSTVWY` for deterministic exact-tie resolution.
- Record practical top-score ties with the frozen scoring tolerance `atol=1e-6`, `rtol=1e-6`; practical-tie status is descriptive and does not alter strict score ordering.
- Use structural orientation AFDB versus PDB and nonnegative local regrets.
- Treat 30 realizations and 435 within-condition repeat pairs as descriptive ensembles, not independent biological replicates.
- Do not compute p-values, FDR, binary uncertainty labels, sequence recommendations, or biological-risk claims.
- Do not start Generated Candidate Union, Arm-B, Stage0-2B, external evaluators, or training.
- Do not commit in this task; preserve all pre-existing dirty/untracked files.

---

### Task 1: Frozen input gate

**Files:**
- Create: `src/dual_uq/dataset/fixed_probe_decision_sensitivity.py`
- Create: `tests/dataset/releases/test_fixed_probe_decision_sensitivity.py`

**Interfaces:**
- Produces: `DecisionSensitivityError`, `DecisionInputs`, `load_frozen_decision_inputs(project_root: Path) -> DecisionInputs`.
- Consumes: exact scoring and sensitivity manifests plus the nine required frozen artifacts.

- [ ] Write tests that load the real frozen release and assert exact hashes/counts/protein order.
- [ ] Write tests that monkeypatch one hash and expect structured `BLOCKED` before any result construction.
- [ ] Write tests for missing schema columns and duplicate scientific keys.
- [ ] Run focused tests and confirm RED because the module does not exist.
- [ ] Implement the minimal loader, manifest-bound path resolution, SHA/schema/count/key gates, and relative-path provenance.
- [ ] Run focused tests and confirm GREEN.

### Task 2: Canonical 20-AA local score table

**Files:**
- Modify: `src/dual_uq/dataset/fixed_probe_decision_sensitivity.py`
- Modify: `tests/dataset/releases/test_fixed_probe_decision_sensitivity.py`

**Interfaces:**
- Produces: `build_local_amino_acid_scores(inputs, repeat_count=30) -> pd.DataFrame`.
- Output key: `protein_id, position, backbone_condition, repeat_index, aa`.

- [ ] Write a synthetic fixture containing one protein, two positions, two backbones, and three repeats.
- [ ] Test exact one-WT-plus-19-mutant reconstruction without inserting WT into `fixed_probe_candidates`.
- [ ] Test rejection of missing/duplicate/nonstandard amino acids and wrong WT identity.
- [ ] Test exact 20-AA alphabet and realization provenance for every local set.
- [ ] Run focused tests and confirm RED on missing API.
- [ ] Implement WT analytical expansion, mutant integration, canonical ordering, and full grid validation.
- [ ] Run focused tests and confirm GREEN.

### Task 3: Ranking and paired structural decisions

**Files:**
- Modify: `src/dual_uq/dataset/fixed_probe_decision_sensitivity.py`
- Modify: `tests/dataset/releases/test_fixed_probe_decision_sensitivity.py`

**Interfaces:**
- Produces: `rank_local_amino_acid_scores(local_scores, ...) -> pd.DataFrame` and `build_repeat_decision_effects(ranked_scores, ...) -> pd.DataFrame`.
- Repeat result key: `protein_id, position, repeat_index, decoding_realization_sha256`.

- [ ] Test higher-is-better ranking and canonical exact-tie ordering independent of DataFrame row order.
- [ ] Test practical tie counts/flags at the frozen tolerance without changing strict rankings.
- [ ] Test PDB/AFDB mismatch/missing realization rejection.
- [ ] Test Top-1 disagreement, Top-3/Top-5 instability, Spearman rank, mean/max normalized displacement, WT status, and both directional/symmetric regrets using hand-computed landscapes.
- [ ] Test all regret values are nonnegative within strict numerical tolerance.
- [ ] Run focused tests and confirm RED.
- [ ] Implement a single ranking helper and paired structural decision derivation.
- [ ] Run focused tests and confirm GREEN.

### Task 4: Same-state decision technical null

**Files:**
- Modify: `src/dual_uq/dataset/fixed_probe_decision_sensitivity.py`
- Modify: `tests/dataset/releases/test_fixed_probe_decision_sensitivity.py`

**Interfaces:**
- Produces: `summarize_same_state_decision_null(ranked_scores, repeat_count=30) -> pd.DataFrame`.
- Output key: `protein_id, position, backbone_condition`.

- [ ] Test exactly `C(n,2)` repeat pairs are summarized for a small synthetic repeat grid.
- [ ] Test modal Top-1 with canonical tie break, concentration, distinct count, pairwise Top-1 disagreement, Top-3/Top-5 instability, rank displacement, and symmetric regret summaries.
- [ ] Test missing repeats and negative-regret numerical violations fail structurally.
- [ ] Run focused tests and confirm RED.
- [ ] Implement within-condition repeat-pair metrics without materializing pairwise rows.
- [ ] Run focused tests and confirm GREEN.

### Task 5: Position and protein aggregation

**Files:**
- Modify: `src/dual_uq/dataset/fixed_probe_decision_sensitivity.py`
- Modify: `tests/dataset/releases/test_fixed_probe_decision_sensitivity.py`

**Interfaces:**
- Produces: `summarize_position_decisions(...) -> pd.DataFrame` and `summarize_protein_decisions(...) -> pd.DataFrame`.
- Position output joins the existing continuous position descriptors and PDB/AFDB technical-null summaries.

- [ ] Test exactly 30 repeat rows per position and frozen protein/position membership.
- [ ] Test required means, medians, linear regret quantiles, modal Top-1 identities, unique Top-1 counts, WT-change fraction, and tie frequencies.
- [ ] Test side-by-side PDB/AFDB technical-null fields and descriptive structural-minus-technical contrasts.
- [ ] Test protein counts reconcile and frozen protein order is preserved.
- [ ] Test no ranking, significance, recommendation, or binary-label fields enter summaries.
- [ ] Run focused tests and confirm RED.
- [ ] Implement deterministic position and protein aggregation from canonical upstream summaries only.
- [ ] Run focused tests and confirm GREEN.

### Task 6: Structured result, immutable renderer, and CLI

**Files:**
- Modify: `src/dual_uq/dataset/fixed_probe_decision_sensitivity.py`
- Create: `scripts/dataset/analyze_fixed_probe_decisions.py`
- Modify: `tests/dataset/releases/test_fixed_probe_decision_sensitivity.py`

**Interfaces:**
- Produces: `DecisionSensitivityResult`, `run_fixed_probe_decision_sensitivity(project_root)`, and `materialize_fixed_probe_decision_sensitivity(result, output_root)`.
- Outputs: repeat effects, position summary, technical null, protein summary, manifest.

- [ ] Test final row counts `53,700 / 1,790 / 3,580 / 8`, bounds, finite metrics, nonnegative regrets, and complete membership.
- [ ] Test immutable writers return `reused_identical` and reject conflicts.
- [ ] Test manifest records all input hashes, 20-AA definition, score/tie/rank/Top-k/regret/null protocols, repeat count, and explicit scope exclusions.
- [ ] Test CLI is thin and exposes `--project-root`.
- [ ] Run focused tests and confirm RED.
- [ ] Implement structured orchestration, final validator, immutable renderers, manifest-last writing, and thin CLI.
- [ ] Run focused tests and confirm GREEN.

### Task 7: Formal offline execution and verification

**Files:**
- Create: `experiments/p2_design_baseline/stage0/fixed_probe_local_decision_effects.parquet`
- Create: `experiments/p2_design_baseline/stage0/fixed_probe_position_decision_sensitivity.parquet`
- Create: `experiments/p2_design_baseline/stage0/fixed_probe_local_decision_null.parquet`
- Create: `experiments/p2_design_baseline/stage0/fixed_probe_protein_decision_sensitivity.parquet`
- Create: `experiments/p2_design_baseline/stage0/fixed_probe_decision_sensitivity_manifest.json`

- [ ] Record all upstream SHA values immediately before execution.
- [ ] Run the offline CLI once and require exact output counts.
- [ ] Independently recompute representative ranking/regret/null/aggregation invariants from the outputs.
- [ ] Run the CLI again and require all five outputs to report `reused_identical`.
- [ ] Verify every upstream SHA remains byte-identical.
- [ ] Run focused tests, `tests/dataset/releases`, `tests/dataset`, and full pytest; report the known unrelated canonical-namespace failure separately if unchanged.
- [ ] Run Ruff on touched Python, compileall, JSON validation, and scoped whitespace checks.
- [ ] Confirm no ProteinMPNN/model process ran and no forbidden stage/output was created.
