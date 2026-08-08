# Stage0-2A Fixed-Probe Sensitivity Handoff

## 1. Handoff status

This document records the repository and scientific state after completion of
Stage0-2A fixed-probe ProteinMPNN scoring and paired PDB-versus-AFDB
structural-condition sensitivity analysis.

```text
snapshot date: 2026-08-08
branch: protocol/dataset-a-h2
HEAD: a9a51bb198766be21ef80b944d2a162090bc6193
worktree: dirty; current Stage-0 implementation and artifacts are uncommitted
```

Scientific execution state:

```text
Stage0 intervention declaration: 11 proteins, frozen historical declaration
Stage0 executable admission subset: 8 proteins
Stage0-1 common masks and fixed probes: complete for the admitted subset
Stage0-2A fixed-probe scoring: complete
Stage0-2A structural-condition sensitivity analysis: complete
Generated Candidate Union: not started
Arm-B: not started
Stage0-2B: not started
Model training: not started
```

These artifacts form a Stage-0 pilot. They are not a powered population study,
a biological fitness experiment, or a final held-out benchmark.

## 2. Read-first map

Read these files in order:

1. `docs/handoff/STAGE0_2A_SENSITIVITY_HANDOFF.md` — this document.
2. `configs/experiments/design_baseline/stage0.yaml` — Stage-0 bindings.
3. `docs/design/stage0_scoring.md` — frozen scoring contract.
4. `docs/superpowers/specs/2026-08-08-stage0-2a-sensitivity-design.md` — analysis contract.
5. `experiments/p2_design_baseline/stage0/fixed_probe_scoring_manifest.json` — scoring release.
6. `experiments/p2_design_baseline/stage0/fixed_probe_sensitivity_manifest.json` — analysis release.
7. `docs/protocols/PDR-01_D1-D2_条款草案_v0.2.md` — D1/D2 protocol state.
8. `docs/protocols/STAGE_NAMING_REGISTRY.md` — protocol/workstream naming boundary.

The scoring design still says cross-condition analysis has not started. That
status line is stale relative to the immutable sensitivity manifest; it does
not invalidate the completed analysis.

## 3. Stage-0 cohort and admission state

### 3.1 Frozen 11-protein declaration

```text
path: experiments/p2_design_baseline/stage0/stage0_intervention_panel_v1.jsonl
records: 11
SHA256: 62d6e8576ed21663947c9bef893bcb9e74f7e6fba7b347910a4c0479b4e07e0d
```

This is a frozen experimental declaration transcribed from prior human review.
It is not a reconstruction of the full candidate census and is not equivalent
to the currently executable subset.

### 3.2 Materialized admission state

```text
path: experiments/p2_design_baseline/stage0/stage0_intervention_admission_v1.jsonl
records: 11
SHA256: 1fa86cde49cdac68433572d929a99c964574c8edc7f9f62832e0c2b3d4a529c0
ADMITTED: 8
PENDING_HUMAN_VARIANT_REVIEW: 2
IDENTITY_CONTRACT_FAIL: 1
```

The ledger materializes an existing human panel-version state. It does not
recompute scientific admission or assign new variant authorization.

| Protein | State | Frozen evidence | Consequence |
| --- | --- | --- | --- |
| `6jgj_A__P42212` | `IDENTITY_CONTRACT_FAIL` | Five observed variants; paired identity `0.9779735683`; mismatch budget is 3 and identity threshold is 0.99 | Requires a new protocol/panel decision. |
| `2ykz_A__P00138` | `PENDING_HUMAN_VARIANT_REVIEW` | `A39V`; mismatch count 1; paired identity `0.9920634921` | Exact variant authorization remains a human decision. |
| `1ix9_A__P00448` | `PENDING_HUMAN_VARIANT_REVIEW` | `Y175F`; mismatch count 1; paired identity `0.9951219512` | Exact variant authorization remains a human decision. |

Do not silently drop mismatch-bearing coordinates, infer authorization, or
replace these proteins with an unversioned panel.

### 3.3 Executable 8-protein subset

```text
path: experiments/p2_design_baseline/stage0/stage0_intervention_admitted_v1.jsonl
records: 8
SHA256: 0f29fe5309a63ae0e24eda2ebdacfad861c418165bb2552b6f68e40855d6d3f5
```

Frozen order:

1. `5gv8_A__P83686`
2. `5mn1_A__P00760`
3. `1fn8_A__P35049`
4. `1pjx_A__Q7SIG4`
5. `3pyp_A__P16113`
6. `6s2s_A__P02689`
7. `4ce8_A__Q9HYN5`
8. `5avh_A__P24300`

## 4. Stage0-1 frozen candidate space

| Artifact | Rows/content | SHA256 |
| --- | ---: | --- |
| `protein_manifest.json` | 8 proteins; 1,790 common-mask positions | `fd34ae871c3d5feebba1dbe38bce24141634764882a9d51db1ce79cd7581d30b` |
| `fixed_probe_candidates.parquet` | 34,010 WT single-mutant probes | `26dc56745c005c01e78007ad3c3b6dbc59708da23087ac7bd2337acc2ea27ef0` |

Every common-mask position contributes exactly 19 non-WT substitutions.
Candidate sequences are backbone-independent and are scored on both PDB and
AFDB. The common-mask domain uses canonical UniProt positions and paired
N/CA/C/O coordinates. Continuous output numbering must not be interpreted as
peptide adjacency across a true UniProt gap.

## 5. Frozen ProteinMPNN scorer

Stage0-2A has exactly one authorized internal inverse-folding scorer:

```text
family: ProteinMPNN
role: stage0_internal_inverse_folding_scorer
implementation: third_party/ProteinMPNN
implementation commit: 8907e6671bfbfc92303b5f79c4b5e6ce47cdef57
checkpoint: third_party/ProteinMPNN/vanilla_model_weights/v_48_020.pt
checkpoint SHA256: c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd
```

Frozen scoring semantics:

```text
protocol: stage0_fixed_sequence_autoregressive_mask_logp_v1
mode: fixed_sequence_autoregressive_mask_logp
domain: frozen_common_mask_projection
backbone atoms: N, CA, C, O
backbone noise: 0.0
residue index: one-based canonical UniProt position
dtype: float32
score direction: higher is better
sequence generation: false
repeats: 30, seeds 0..29
```

No fallback checkpoint, model download, commit update, or evaluator-registry
substitution is authorized by this handoff.

Recorded execution environment provenance is Python 3.10.20, Torch
2.12.1+cu130, CUDA runtime 13.0, NVIDIA A800-SXM4-80GB on `cuda:1`, and batch
size 128. Environment provenance does not replace model commit/checkpoint
identity.

## 6. Stage0-2A scoring release

```text
manifest: experiments/p2_design_baseline/stage0/fixed_probe_scoring_manifest.json
status: complete
SHA256: ec9882604cdfad0d61fbd30c1314466fbc7cbbea1fe7173d84c9913309783b00
logical shards: 480
```

| Artifact | Rows | SHA256 |
| --- | ---: | --- |
| `fixed_probe_wt_scores.parquet` | 480 | `373948042b6868611ba6bc9ea80253d47cf0b441e002dcc6c10eed57f23b8d31` |
| `fixed_probe_scores.parquet` | 2,040,600 | `ac0cf8e1f251770fd4ade08580b8df19000a47124531350e9af63dd71f773abf` |
| `fixed_probe_scoring_null.parquet` | 68,020 | `fa248c180444498e89541582edab12ef6d42a822ddbb65bc92e358b383621403` |

The same-state null uses the empirical 30-repeat distribution and population
standard deviation (`ddof=0`). It measures ProteinMPNN decoding-order
variability only; it is not a biological structural null. Do not rerun
ProteinMPNN merely to reproduce the sensitivity analysis.

## 7. Stage0-2A sensitivity analysis

### 7.1 Estimand and pairing

```text
structural_mutation_interaction(k,r)
    = delta_score_vs_wt(k,AFDB,r) - delta_score_vs_wt(k,PDB,r)

orientation = AFDB - PDB
```

The validated equivalent is candidate raw AFDB-PDB shift minus WT raw
AFDB-PDB shift. Positive means relatively greater compatibility under AFDB
according to the frozen scorer; negative means relatively greater
compatibility under PDB. Neither direction is a biological fitness claim.

Cross-condition pairing uses exactly:

```text
protein_id
sequence_hash
repeat_index
decoding_realization_sha256
```

### 7.2 Implementation and dataflow

```text
API: src/dual_uq/dataset/fixed_probe_sensitivity.py
CLI: scripts/dataset/analyze_fixed_probe_sensitivity.py
tests: tests/dataset/releases/test_fixed_probe_sensitivity.py
```

Only the canonical paired candidate-repeat layer reads raw PDB/AFDB scores.
Candidate, position, and protein summaries derive from that result and its
canonical summaries. The implementation uses Pandas/NumPy/Parquet and does not
add model code, DuckDB, Polars, or scoring logic.

### 7.3 Analysis release

```text
manifest: experiments/p2_design_baseline/stage0/fixed_probe_sensitivity_manifest.json
status: complete
protocol: stage0_fixed_probe_structural_sensitivity_v1
orientation: AFDB_minus_PDB
standard deviation: population_ddof0
quantiles: q05/q25/q50/q75/q95, linear interpolation
zero tolerance: abs(interaction) <= 1e-6
SHA256: 148354c1ac1d4389ac8cb73240fb457dd193bee7c105e183b65f8d4db048a3ff
```

| Artifact | Rows | SHA256 |
| --- | ---: | --- |
| `fixed_probe_structural_effects.parquet` | 1,020,300 | `171504e5dc75828b931ac91ee2c0deedf3583cf1f32de8f6902f18a897ff39e4` |
| `fixed_probe_candidate_sensitivity.parquet` | 34,010 | `a7276db340e23334f9ee5c8d6287e0a0427c628b0bf15294170884d8162f50ca` |
| `fixed_probe_position_sensitivity.parquet` | 1,790 | `6faa79d1d1ff3a0999a878ad35aeb3e9ffdb22df058c308d727e95fc06ed0eb9` |
| `fixed_probe_protein_sensitivity.parquet` | 8 | `91464052845e5c02250ca7dff5b27ef93c77279296a47736ffe8d6ba52f65e18` |

An identical rerun returns `reused_identical` for all Parquet files and the
manifest. Conflicting content is rejected rather than overwritten.

### 7.4 Integrity results

```text
paired candidate-repeat rows: 1,020,300
missing PDB pairs: 0
missing AFDB pairs: 0
duplicate pairs: 0
realization-fingerprint mismatches: 0
repeats per candidate: 30
mutations per position: 19
interaction algebra maximum absolute error: 0.0
candidate-to-position reaggregation maximum error: 0.0
candidate-to-protein reaggregation maximum error: 0.0
forbidden rank/Top-k/regret/testing/classification fields: 0
```

### 7.5 Descriptive pilot result

```text
candidate interaction mean range: -0.0517236 to 0.0499750
median candidate interaction mean: -0.0000222510
mean abs(candidate interaction mean): 0.00203149
median abs(candidate interaction mean): 0.00110115
q95 abs(candidate interaction mean): 0.00706706
median interaction population SD: 0.000891444
median same-state technical scale: 0.00184871
median abs(interaction mean) / technical scale: 0.591076
q95 abs(interaction mean) / technical scale: 3.29151
```

| Protein | Candidates | Positions | Mean abs. interaction | RMS | Median abs. | Q95 abs. | Max abs. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `5gv8_A__P83686` | 5,168 | 272 | 0.001948 | 0.002806 | 0.001323 | 0.005967 | 0.025259 |
| `5mn1_A__P00760` | 4,237 | 223 | 0.001626 | 0.002438 | 0.001073 | 0.004837 | 0.028775 |
| `1fn8_A__P35049` | 4,256 | 224 | 0.001767 | 0.002819 | 0.001093 | 0.005222 | 0.029503 |
| `1pjx_A__Q7SIG4` | 5,966 | 314 | 0.001103 | 0.001582 | 0.000792 | 0.003184 | 0.019509 |
| `3pyp_A__P16113` | 2,375 | 125 | 0.005924 | 0.008469 | 0.003927 | 0.017921 | 0.051724 |
| `6s2s_A__P02689` | 2,508 | 132 | 0.003411 | 0.005152 | 0.002259 | 0.010213 | 0.038078 |
| `4ce8_A__Q9HYN5` | 2,166 | 114 | 0.003775 | 0.005411 | 0.002568 | 0.011401 | 0.049975 |
| `5avh_A__P24300` | 7,334 | 386 | 0.000986 | 0.001435 | 0.000673 | 0.003044 | 0.009483 |

These are descriptive values in frozen cohort order, not a ranking. No protein
has been assigned a binary structural-UQ label.

## 8. Reproduction and validation

### 8.1 Offline analysis rerun

The analysis can be reproduced without invoking ProteinMPNN:

```bash
conda run -n dual-uq \
  python scripts/dataset/analyze_fixed_probe_sensitivity.py \
  --project-root /path/to/Dual-UQ
```

The project-root argument is an execution-time override only. It does not
participate in scientific identity; all manifest paths are repository
relative.

Expected rerun result:

```text
status: PASS
paired_effect_rows: 1,020,300
candidate_summary_rows: 34,010
position_summary_rows: 1,790
protein_summary_rows: 8
all write_status values: reused_identical
```

### 8.2 Verification baseline

Focused analysis tests:

```bash
conda run -n dual-uq \
  pytest tests/dataset/releases/test_fixed_probe_sensitivity.py -q
```

Current result: `19 passed`.

Release regression:

```bash
conda run -n dual-uq pytest tests/dataset/releases -q
```

Current result: `127 passed`.

Raw full-suite result in the current dirty worktree:

```text
pytest -q
1 failed, 1036 passed
```

The sole failure is:

`tests/dataset/test_canonical_namespace.py::test_active_code_and_configuration_do_not_reference_legacy_namespace`

It reports two frozen historical report bindings in
`src/dual_uq/dataset/fixed_probes.py`:

```text
reports/dataset_a_scale/batch1_acquisition_run_v1.json
reports/dataset_a_scale/batch1_acquisition_plan_v1.json
```

These paths predate the sensitivity analysis and were not introduced by it.
With this exact known check deselected, the result is `1036 passed, 1
deselected`. Ruff on the sensitivity module, CLI, and tests passes. Compileall,
JSON validation, and scoped whitespace checks also pass.

## 9. Scientific interpretation boundary

The completed pilot supports this statement:

> ProteinMPNN's fixed-probe mutation compatibility landscape exhibits a
> measured degree of sensitivity to the PDB-versus-AFDB structural condition
> under paired autoregressive decoding realizations.

It does not establish:

- biological fitness or an experimental mutation effect;
- that structural uncertainty causes design failure;
- that any protein is structurally uncertainty-positive;
- that PDB is better or worse than AFDB;
- a powered population-level effect or calibrated significance threshold;
- Top-k, ranking, regret, Pareto, or selection consequences;
- biological conformational uncertainty, which requires later Arm-B evidence.

The 34,010 candidates and 1,790 positions are nested descriptive measurements,
not independent biological replicates. Protein is the Stage-0 pilot
interpretation unit, and there are eight admitted proteins.

## 10. Known engineering debt

Address these items before reusing the scorer for a new formal scoring release.
They do not retroactively change the current immutable artifacts:

1. The G2 numerical tolerance helper uses `np.allclose(reference, observed)`;
   the documented relative-error contract is explicitly relative to the
   reference value. Boundary behavior should be made explicit and tested.
2. The G2 stochasticity audit compares nonzero seeds to seed 0 rather than
   checking every realization pair before declaring determinism. The current
   audit remains stochastic because observed differences were well beyond the
   tolerance.
3. The model identity gate verifies the ProteinMPNN Git commit but should also
   enforce a clean tracked worktree/tree-content condition before future model
   execution. The worktree was recorded clean for the current release.
4. `docs/design/stage0_scoring.md` has a stale implementation-status line that
   says cross-condition analysis is not started.
5. The two legacy `reports/dataset_a_scale/...` bindings above keep the raw full
   test suite from being completely green. Relocation must preserve historical
   evidence and manifest compatibility rather than rewriting paths casually.

Do not combine these maintenance fixes with a new scientific experiment or
silently regenerate current outputs.

## 11. Worktree and version-control warning

At this handoff the current Stage-0 configuration, implementation, tests, and
experiment artifacts are uncommitted. The worktree also contains pre-existing
deletions/modifications such as `.env.example`, `project_charter.md`, and
dataset package initialization changes.

Before any checkpoint:

1. run `git status --short`;
2. classify every dirty/untracked path by provenance;
3. stage only explicitly reviewed Stage-0 paths;
4. inspect `git diff --cached` and `git diff --cached --check`;
5. never use `git add .`, `git add -A`, `git reset`, `git stash`, or
   `git clean` to simplify this worktree.

Large immutable Parquet artifacts need an explicit repository/release-storage
decision before commit. Do not assume that untracked means disposable.

## 12. Recommended next gates

Proceed in this order unless a human owner explicitly changes the plan:

1. Review this handoff and checkpoint the already-reviewed Stage0-1,
   Stage0-2A scoring, and Stage0-2A analysis work in logically separated
   commits or release-storage bindings.
2. Resolve the three non-admitted members only through explicit human
   panel-version/variant decisions; do not modify the existing 8-protein
   release retrospectively.
3. Repair and test scorer-audit engineering debt before any new formal scoring
   execution.
4. Decide and freeze the next scientific endpoint separately. Top-k, ranking,
   regret, generated candidate union, Arm-B, and Stage0-2B remain outside the
   current contract.
5. Preserve the scoring and sensitivity manifests as immutable provenance
   anchors for subsequent decision-level analysis.

No next gate is authorized merely by the existence of this handoff.

## 13. Explicit stop conditions

Without a new reviewed task and applicable human authorization, do not:

- rerun ProteinMPNN or switch its checkpoint;
- generate protein sequences;
- start Generated Candidate Union;
- calculate Top-k/rank/regret or Pareto outputs from these files;
- assign binary uncertainty labels;
- run hypothesis tests treating candidates or positions as independent;
- execute an external evaluator;
- start Arm-B or Stage0-2B;
- start GearNet, ProteinMPNN, Joint-UQ, or other model training;
- alter admission, PDR-01 decisions, or variant authorization;
- overwrite immutable Stage-0 artifacts.

## 14. Final handoff statement

Stage0-2A fixed-probe scoring and paired structural-condition sensitivity
analysis are complete for the frozen 8-protein executable subset. The release
is internally consistent, realization-matched, immutable on identical rerun,
and bounded to descriptive Stage-0 provenance sensitivity.

The declared 11-protein panel is not fully executable under current admission
state. Two exact variants remain pending human review and one protein fails the
frozen identity contract. No generated-union, decision-level, Arm-B,
Stage0-2B, evaluator, or training stage has started.
