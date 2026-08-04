# Dual-UQ A0 Candidate Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the audited, fragment-aware, resumable A0 candidate pipeline and select a 12-protein panel only after at least 24 full-quality diagnostics exist.

**Architecture:** Keep network/data acquisition, residue-number normalization, candidate lifecycle auditing, batch orchestration, confidence scoring, and A0 classification in separate testable modules. Every stage writes an explicit machine-readable artifact; later stages join by `screening_index` plus `(pdb_id, chain_id, uniprot_id)` and never infer completion from one downstream file.

**Tech Stack:** Python 3.11, pandas, NumPy, PyArrow, Biopython, SciPy, Pydantic, pytest, YAML, RCSB/SIFTS/AlphaFold DB APIs.

## Global Constraints

- Execute T1 through T10 strictly in order.
- Add or update tests for every task.
- Run `conda run -n dual-uq pytest -q` before every task commit.
- Use one single-purpose git commit per task and inspect the staged diff first.
- Keep full-length mapping coverage at or above 0.90.
- Keep sequence identity at or above 0.95.
- Keep observed-CA fraction at or above 0.90.
- Never call PDB–AFDB structural disagreement “AlphaFold error”.
- Never hard-code an AFDB residue offset.
- Do not start GearNet or ProteinMPNN training.

---

### Task 1: Candidate lifecycle audit

**Files:**
- Create: `src/dual_uq/lifecycle.py`
- Create: `scripts/21_build_candidate_lifecycle.py`
- Create: `tests/test_lifecycle.py`
- Create: `reports/candidate_lifecycle.csv`
- Create: `reports/candidate_lifecycle_audit.json`
- Modify: `docs/superpowers/plans/2026-07-30-dual-uq-a0-candidate-pipeline.md`

**Interfaces:**
- Consumes screening pool, preflight manifest, pair directories, structured/legacy status logs, geometry QC, robust diagnostics, segment context, and A0 summary.
- Produces `build_candidate_lifecycle(...) -> tuple[pd.DataFrame, dict[str, Any]]`.

- [ ] Write tests proving stage statuses, exclusion reasons, quality flags, and audit counts are derived without treating missing files as success.
- [ ] Run `pytest tests/test_lifecycle.py -q` and confirm failure because `dual_uq.lifecycle` is absent.
- [ ] Implement deterministic joins and explicit statuses: `not_started`, `complete`, `failed`, `successful_no_segments`, `unsupported_afdb_fragment`, and preflight skip states.
- [ ] Generate the CSV and audit JSON from current repository artifacts.
- [ ] Run `conda run -n dual-uq pytest -q`.
- [ ] Stage only Task 1 files, inspect `git diff --cached`, and commit `feat: add candidate lifecycle audit`.

### Task 2: Index 9 numbering and fragment diagnosis

**Files:**
- Create: `src/dual_uq/indexing_diagnostics.py`
- Modify: `scripts/19_diagnose_pair_indexing.py`
- Create: `tests/test_indexing_diagnostics.py`
- Create: `reports/index9_numbering_diagnosis.json`

**Interfaces:**
- Consumes SIFTS mapping, mmCIF atom-site identifiers, and all AFDB prediction records.
- Produces `diagnose_numbering_intersections(...) -> dict[str, Any]` with auth, label, UniProt, and fragment interval intersections plus a categorical root cause.

- [ ] Write synthetic tests distinguishing `auth_label_numbering_mismatch`, `afdb_fragment_selection_mismatch`, `afdb_residue_offset_mismatch`, and unsupported coverage.
- [ ] Run the focused test and confirm RED.
- [ ] Parse `_atom_site.auth_asym_id`, `_atom_site.label_asym_id`, `_atom_site.auth_seq_id`, `_atom_site.label_seq_id`, insertion code, and AFDB fragment intervals without offsets.
- [ ] Run the Index 9 diagnostic and save the evidence JSON.
- [ ] Run full pytest and commit `diagnose: classify index 9 mapping failure`.

### Task 3: Mapped-interval-aware AFDB model selection

**Files:**
- Modify: `src/dual_uq/afdb.py`
- Modify: `src/dual_uq/pairing.py`
- Modify: `src/dual_uq/preflight.py`
- Modify: `scripts/16_preflight_screening_pool.py`
- Modify: `tests/test_step1_utils.py`
- Create or modify: `tests/test_afdb_fragments.py`

**Interfaces:**
- Produces `select_prediction_for_interval(records, mapped_start, mapped_end)` and `UnsupportedAFDBFragment`.
- `fetch_afdb_prediction(..., mapped_interval=(start, end))` downloads the covering model; no covering model returns the structured unsupported outcome.

- [ ] Write failing tests for selecting F2/F3 when they cover the mapped interval, preferring full coverage, deterministic tie-breaking, and raising `UnsupportedAFDBFragment`.
- [ ] Implement interval extraction from AFDB metadata and coverage-based selection.
- [ ] Reorder pairing/preflight so SIFTS mapped interval is known before the model is selected.
- [ ] Run focused and full tests; commit `fix: select AFDB fragment by mapped interval`.

### Task 4: Preserve auth and label residue identifiers

**Files:**
- Modify: `src/dual_uq/sifts.py`
- Modify: `src/dual_uq/structure_io.py`
- Modify: `src/dual_uq/preflight.py`
- Modify: `src/dual_uq/pair_geometry.py`
- Modify: `src/dual_uq/schema.py`
- Create: `tests/test_residue_numbering.py`

**Interfaces:**
- Residue tables retain `auth_asym_id`, `label_asym_id`, `auth_seq_id`, `label_seq_id`, and `insertion_code`.
- Geometry joins use explicit auth identifiers with a validated label fallback.

- [ ] Write failing mmCIF/SIFTS fixtures covering different auth/label chain and sequence identifiers plus insertion codes.
- [ ] Implement explicit identifier extraction and mapping-schema migration.
- [ ] Preserve backward-compatible aliases only at I/O boundaries.
- [ ] Regenerate affected Index 9 mapping and re-run its diagnostic.
- [ ] Run full pytest; commit `fix: preserve auth and label residue numbering`.

### Task 5: Preflight-aware resumable screening runner

**Files:**
- Create: `src/dual_uq/screening_runner.py`
- Modify: `scripts/14_run_screening_pool.py`
- Modify: `scripts/15_summarize_screening_status.py`
- Create: `tests/test_screening_runner.py`

**Interfaces:**
- Produces one structured status record per candidate and stage with timestamps, runtime, seed, error type, and output validation.
- Stage terminal states include `complete`, `skipped_preflight`, `successful_no_segments`, `unsupported_afdb_fragment`, and `failed_<stage>`.

- [ ] Write failing tests for default pass-only filtering, explicit warning inclusion, fail skipping, stage-level resume, output validation, deterministic seeds, and no-segment success.
- [ ] Implement a pure stage-planning/state-reduction layer, then the subprocess adapter.
- [ ] Make permutation seeds explicit and configurable; preserve download cache behavior.
- [ ] Run full pytest; commit `feat: make screening runner resumable and preflight aware`.

### Task 6: Complete geometry pilot

**Files:**
- Update: `data/manifests/geometry_pilot.tsv`
- Create: `reports/geometry_pilot_mechanisms.csv`
- Create: `reports/geometry_pilot_audit.json`
- Create: `tests/test_geometry_pilot_audit.py`

**Interfaces:**
- Consumes complete diagnostics for indices 6, 8, 24, 36, 35, plus 1AKE.
- Produces one evidence-backed mechanism label per sample.

- [ ] Write failing validation tests requiring every requested sample and all diagnostic stages.
- [ ] Run missing/resumable stages using the Task 5 runner with fixed seeds.
- [ ] Build the mechanism table from segment context and geometry evidence, keeping construct difference separate from low confidence.
- [ ] Verify at least four positive samples, at least two mechanisms, and correct confidence-file/model pairing.
- [ ] Run full pytest; commit `data: register geometry pilot mechanisms`.

### Task 7: Lower-confidence replacement discovery

**Files:**
- Create: `src/dual_uq/replacements.py`
- Modify: `scripts/20_build_lower_conf_replacement_pool.py`
- Modify: `configs/legacy/a0_screening/lower_conf_replacement.yaml`
- Create: `tests/test_replacements.py`
- Create: `data/manifests/lower_conf_replacement_pool.tsv`
- Create: `reports/lower_conf_replacement_audit.json`

**Interfaces:**
- First-stage filter requires PDB/UniProt length ratio 0.90–1.10, both mapping coverages ≥0.90, identity ≥0.95, and observed CA ≥0.90.

- [ ] Write failing boundary and missing-value tests for the full-length proxy.
- [ ] Implement filtering/ranking from the original eligible discovery table.
- [ ] Discover 15–20 replacements and preflight them without lowering thresholds.
- [ ] Record pass/warn/fail counts and exclusion reasons.
- [ ] Run full pytest; commit `feat: build full-length lower-confidence replacements`.

### Task 8: Mapped-region confidence segments

**Files:**
- Create: `src/dual_uq/mapped_confidence.py`
- Create: `scripts/22_score_mapped_confidence.py`
- Create: `tests/test_mapped_confidence.py`
- Create: `reports/replacement_mapped_confidence.csv`

**Interfaces:**
- Produces mapped pLDDT min/q10/median and longest below-70/below-80 contiguous segment with observed-CA and nearest-terminus evidence.

- [ ] Write failing tests for internal, terminal, unmapped, missing-CA, and interrupted low-confidence runs.
- [ ] Implement segment detection strictly over mapped positions with observed CA.
- [ ] Score replacement pass samples and mark `is_low_conf_local` only for internal runs of length ≥5.
- [ ] Run full pytest; commit `feat: score mapped-region confidence segments`.

### Task 9: Multi-label A0 classification

**Files:**
- Create: `src/dual_uq/a0_classification.py`
- Modify: `scripts/12_build_a0_candidate_summary.py`
- Modify: `configs/legacy/a0_screening/a0_selection.yaml`
- Create: `tests/test_a0_classification.py`
- Update: `reports/a0_candidate_summary.csv`

**Interfaces:**
- Produces six independent booleans and a deterministic `primary_category`; length is never a final high-PAE label.

- [ ] Write failing tests for overlapping labels, priority, construct controls, missing-coordinate stress, mapped low-confidence segments, and high-PAE requirements.
- [ ] Implement pure classification functions and add long-range ≥15 Å statistics and separation strata.
- [ ] Rebuild the candidate summary from complete diagnostics.
- [ ] Run full pytest; commit `feat: classify A0 candidates with multiple labels`.

### Task 10: Diagnose 24+ samples and select final A0 panel

**Files:**
- Create: `src/dual_uq/a0_selection.py`
- Create: `scripts/23_select_final_a0_panel.py`
- Create: `tests/test_a0_selection.py`
- Create: `data/manifests/a0_final_panel.tsv`
- Create: `reports/a0_final_selection_audit.json`

**Interfaces:**
- Selection refuses to run below 24 quality-pass complete diagnostics.
- Selects three unique full-quality samples from each primary category and retains secondary controls separately.

- [ ] Write failing tests for the 24-sample gate, per-category quota, uniqueness, threshold enforcement, and deterministic tie-breaking.
- [ ] Run additional preflight-pass candidates through the fixed-seed resumable pipeline until 24 complete quality-pass diagnostics exist.
- [ ] Rebuild lifecycle and multi-label summaries.
- [ ] Select the final 12 only if all gates and quotas pass; otherwise emit a structured blocked audit without fabricating a panel.
- [ ] Run full pytest; commit `data: select final A0 panel`.

## Self-review

- T1–T10 are represented in the required order and each has a RED/GREEN test cycle, full pytest gate, and single-purpose commit.
- Mapping, identity, and observed-CA thresholds are copied exactly and never relaxed.
- AFDB selection is interval-based and unsupported coverage is explicit.
- Numbering retains both auth and label identifiers and never relies on a hard-coded offset.
- T10 explicitly refuses premature selection below 24 complete quality-pass samples.
- No GearNet or ProteinMPNN training appears in the plan.
