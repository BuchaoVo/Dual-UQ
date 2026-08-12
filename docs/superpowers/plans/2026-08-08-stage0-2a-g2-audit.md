# Stage-0-2A G2 Audit Implementation Plan

> Scope: implement and execute only the frozen upstream/model integrity,
> common-mask projection, explicit decoding-realization, and G2 stochasticity
> audit milestone. Do not score the formal 34,010-probe library.

**Design authority:** `docs/design/stage0_scoring.md`

**Goal:** Produce one immutable, identity-bound G2 audit that determines whether
formal Stage-0-2A scoring requires one repeat or 30 decoding-order repeats.

**Architecture:** Add a lazy-loaded ProteinMPNN scoring adapter under
`dual_uq.dataset.services`, with Stage-0-specific input gates and G2 state-machine
orchestration in `dual_uq.dataset.fixed_probe_scoring`. The CLI remains a thin
audit-only entrypoint. Frozen Stage-0-1 mapping/backbone implementations are
reused through adapters; no scientific algorithm is copied or changed.

**Runtime:** Unit tests run in `dual-uq` without importing Torch at module import.
The real checkpoint integration audit runs in `dual-uq-model` on one selected GPU.

---

## Task 1: Materialize the frozen scoring configuration

**Files:**

- Modify: `configs/experiments/design_baseline/stage0.yaml`
- Test: `tests/dataset/releases/test_fixed_probe_scoring.py`

1. Add RED tests asserting that loading fails for a missing scoring section,
   altered input path/SHA, checkpoint path/SHA, implementation commit, scoring
   mode, coordinate system, atom list, noise, dtype, tolerance, or seed schedule.
2. Run the focused test and confirm RED because no scoring binding exists yet.
3. Add the exact authorized model identity and frozen scoring/audit settings to
   `stage0.yaml`, including explicit audit seeds `0,1,2,3` and formal schedule
   `0..29` without ellipsis.
4. Implement the smallest pure config parser needed by Task 2; it must resolve
   repository-relative paths and must not import Torch.
5. Re-run focused tests and confirm GREEN.

## Task 2: Gate upstream artifacts and authorized model identity

**Files:**

- Create: `src/dual_uq/dataset/fixed_probe_scoring.py`
- Create: `src/dual_uq/dataset/services/proteinmpnn_scoring.py`
- Test: `tests/dataset/releases/test_fixed_probe_scoring.py`

1. Add RED tests for all three frozen upstream SHA mismatches, wrong counts
   (8 proteins, 1,790 mask positions, 34,010 probes), duplicate identities, an
   absent checkpoint, checkpoint hash drift, submodule commit drift, and any
   attempted fallback checkpoint.
2. Run focused tests and confirm RED.
3. Implement immutable dataclasses for scoring configuration, upstream bindings,
   model identity, projection, decoding realization, score rows, and G2 result.
4. Implement a pure upstream loader that validates the exact admitted subset,
   protein manifest, fixed-probe file, identities, row counts, and SHA256 values
   before any model load or output write.
5. Implement an authorized model verifier that accepts exactly the configured
   ProteinMPNN implementation commit and checkpoint SHA. Lazy-load Torch and the
   submodule only inside the runtime loader.
6. Re-run focused tests and confirm GREEN.

## Task 3: Build exact frozen-mask scoring projections

**Files:**

- Modify: `src/dual_uq/dataset/services/proteinmpnn_scoring.py`
- Modify: `src/dual_uq/dataset/fixed_probe_scoring.py`
- Test: `tests/dataset/releases/test_fixed_probe_scoring.py`

1. Add RED fixtures for compressed UniProt numbering, reordered masks,
   PDB/AFDB domain mismatch, missing N/CA/C/O atoms, coordinate-bearing identity
   conflict, candidate sequence-hash drift, and candidate mutation mismatch.
2. Run focused tests and confirm RED.
3. Reuse the frozen Stage-0-1 SIFTS, mapping, exact-fragment, atom-selection, and
   residue-pairing helpers to project PDB and AFDB N/CA/C/O coordinates onto the
   manifest-authoritative mask.
4. Validate that PDB and AFDB position vectors exactly equal the manifest mask.
   Set ProteinMPNN `residue_idx` to the actual one-based UniProt positions; never
   compress positions or include mask-external structural/sequence context.
5. Validate candidate `full_sequence`, `sequence_hash`, WT/mutant metadata, and
   mask projection before scoring.
6. Re-run focused tests and confirm GREEN.

## Task 4: Freeze explicit decoding realizations and direct logp scoring

**Files:**

- Modify: `src/dual_uq/dataset/services/proteinmpnn_scoring.py`
- Test: `tests/dataset/releases/test_fixed_probe_scoring.py`

1. Add RED tests demonstrating that per-batch RNG draws break scientific pairing.
2. Add RED tests for same `(protein_id, repeat_index, protocol)` reproducibility,
   distinct audit seeds, fingerprint validation, reuse across WT/candidates and
   PDB/AFDB, and batch partition invariance.
3. Run focused tests and confirm RED.
4. Implement one explicit realization per `protein_id × repeat_index`, derived
   from a SHA-bound seed domain and materialized as the exact ProteinMPNN decoding
   order/noise input. Record algorithm version, length, seed, and SHA256.
5. Load the authorized `v_48_020` model with frozen `backbone_noise=0.0`; use the
   official forward implementation without invoking generation.
6. Score target residues by directly gathering the model's `log_probs`; return
   `score_sum_logp_mask` and `score_mean_logp_mask`, where higher is better.
7. Re-run focused tests with fake model/log-prob tensors and confirm sign, mean,
   sum, finite-value, and scored-residue-count behavior.

## Task 5: Implement the G2 state machine and audit-only CLI

**Files:**

- Modify: `src/dual_uq/dataset/fixed_probe_scoring.py`
- Create: `scripts/dataset/score_fixed_probes.py`
- Test: `tests/dataset/releases/test_fixed_probe_scoring.py`

1. Add RED tests for all G2 branches:
   - same-realization instability -> `BLOCKED`;
   - batch-size disagreement -> `FAIL`;
   - stable and seed-insensitive -> one repeat, seed 0;
   - stable and seed-sensitive -> 30 repeats, seeds 0..29.
2. Add RED tests asserting the exact two proteins, six positions, deterministic
   first-four-non-WT mutation rule, 24 mutants, two WT sequences, seeds 0..3,
   and no formal scoring-output creation.
3. Run focused tests and confirm RED.
4. Implement the G2 audit using the exact frozen fixtures and tolerances. Reuse
   one realization object/content across repeated calls, both backbones, all
   candidates, and batch sizes 1 and 8.
5. Implement atomic writing of only
   `runs/design_baseline/stage0-2a/g2_audit.json`. Bind it to upstream hashes,
   protocol identity, implementation commit, checkpoint SHA, environment
   provenance, audit candidates, scores, deviations, fingerprints, classification,
   and selected formal repeat policy. Refuse silent replacement of a different
   existing audit.
6. Implement a thin `--audit-only` CLI. In this milestone it must refuse any
   request to start formal 34,010-probe scoring.
7. Re-run focused tests and confirm GREEN.

## Task 6: Execute and verify the real G2 audit

**Files:**

- Runtime artifact only: `runs/design_baseline/stage0-2a/g2_audit.json`

1. Run focused unit tests in `dual-uq`.
2. Run existing Stage-0-1 release regressions in `dual-uq`.
3. Recompute the real checkpoint SHA and verify the ProteinMPNN submodule commit.
4. Run the audit-only CLI in `dual-uq-model` on one explicitly recorded GPU.
5. Validate the JSON syntax and all upstream/model/protocol bindings.
6. Independently verify the G2 classification from recorded numeric deviations
   and realization fingerprints.
7. Run the dataset test suite, full pytest, Ruff on touched Python files,
   `compileall`, and scoped whitespace/diff checks.
8. Confirm no canonical formal score files were created and no generated-union,
   ESM-IF1, evaluator-UQ, cross-condition analysis, or sequence generation ran.
9. Stop and report the G2 result for human review. Do not start formal scoring.

## Completion boundary

This plan is complete only when G2 has a validated result and the repository is
stopped before formal scoring. The G2 classification may select a future repeat
policy, but it does not authorize or execute the 34,010-probe formal run.
