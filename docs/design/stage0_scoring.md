# Stage-0-2A ProteinMPNN Fixed-Probe Scoring Design

**Repository path:** `docs/design/stage0_scoring.md`

**Status:** DESIGN FROZEN
**Human approval:** CONFIRMED 2026-08-08
**Implementation status:** FORMAL SCORING COMPLETE / CROSS-CONDITION ANALYSIS NOT STARTED
**Scientific scope:** Stage-0-2A fixed-probe internal scoring only

---

## 1. Purpose

Stage-0-2A measures whether the frozen ProteinMPNN compatibility landscape changes when the structural condition changes from the experimentally observed PDB backbone to the corresponding AFDB backbone, while holding fixed:

* protein cohort;
* canonical residue domain;
* candidate sequence space;
* model identity;
* model checkpoint;
* scoring semantics;
* autoregressive decoding realization.

The Stage-0-1 candidate library is deliberately backbone-independent.

Therefore the primary experimental object is:

\[
S(k,B,r)
\]

where:

* \(k\) = fixed-probe candidate;
* \(B\in\{\mathrm{PDB},\mathrm{AFDB}\}\) = structural condition;
* \(r\) = explicit autoregressive decoding realization.

No cross-condition effect is analyzed in this task. Stage-0-2A only materializes validated scores and same-backbone scoring variability.

---

# 2. Frozen upstream inputs

Stage-0-2A MUST consume existing Stage-0 configuration.

Configuration:

`configs/experiments/design_baseline/stage0.yaml`

Required frozen inputs:

### 2.1 Executable cohort

`stage0_intervention_admitted_subset`

Expected artifact:

`experiments/p2_design_baseline/stage0/stage0_intervention_admitted_v1.jsonl`

Expected SHA256:

`0f29fe5309a63ae0e24eda2ebdacfad861c418165bb2552b6f68e40855d6d3f5`

Expected proteins:

1. `5gv8_A__P83686`
2. `5mn1_A__P00760`
3. `1fn8_A__P35049`
4. `1pjx_A__Q7SIG4`
5. `3pyp_A__P16113`
6. `6s2s_A__P02689`
7. `4ce8_A__Q9HYN5`
8. `5avh_A__P24300`

### 2.2 Stage-0-1 manifest

`experiments/p2_design_baseline/stage0/protein_manifest.json`

Expected SHA256:

`fd34ae871c3d5feebba1dbe38bce24141634764882a9d51db1ce79cd7581d30b`

Expected:

* 8 proteins;
* 1,790 frozen common-mask positions.

### 2.3 Fixed-probe library

`experiments/p2_design_baseline/stage0/fixed_probe_candidates.parquet`

Expected SHA256:

`26dc56745c005c01e78007ad3c3b6dbc59708da23087ac7bd2337acc2ea27ef0`

Expected:

* 34,010 probes;
* exactly 19 substitutions per common-mask position;
* one mutation per probe.

Any upstream SHA mismatch is a hard:

`BLOCKED`

before model execution or scientific output creation.

---

# 3. Frozen ProteinMPNN identity

Stage-0-2A has exactly one authorized internal model.

### Implementation

`third_party/ProteinMPNN/`

ProteinMPNN commit:

`8907e6671bfbfc92303b5f79c4b5e6ce47cdef57`

### Checkpoint

`third_party/ProteinMPNN/vanilla_model_weights/v_48_020.pt`

SHA256:

`c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd`

Role:

`stage0_internal_inverse_folding_scorer`

No checkpoint fallback is allowed.

Before model loading:

1. resolve configured checkpoint path;
2. recompute SHA256;
3. verify exact commit/version binding;
4. reject mismatch.

A mismatch produces:

`BLOCKED`.

The current evaluator registry is not an authorization source for Stage-0-2A.

---

# 4. Stage-0 scoring configuration

The existing `stage0.yaml` should contain a frozen scoring section equivalent in semantics to:

```yaml
scoring:
  model:
    family: ProteinMPNN
    role: stage0_internal_inverse_folding_scorer
    implementation_path: third_party/ProteinMPNN
    implementation_commit: 8907e6671bfbfc92303b5f79c4b5e6ce47cdef57
    checkpoint_path: third_party/ProteinMPNN/vanilla_model_weights/v_48_020.pt
    checkpoint_sha256: c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd

  mode: fixed_sequence_autoregressive_mask_logp
  scoring_domain: frozen_common_mask_projection
  residue_index_coordinate_system: uniprot_position_1based
  backbone_atoms: [N, CA, C, O]
  backbone_noise: 0.0
  numerical_dtype: float32

  audit:
    atol: 1.0e-6
    rtol: 1.0e-6
    stochastic_seed_schedule:
      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
       10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
       20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
```

Exact YAML hierarchy may follow existing repository conventions.

The semantics above are frozen.

---

# 5. Scoring domain

## 5.1 Structural input

For each protein, ProteinMPNN receives only coordinates corresponding to the Stage-0-1 frozen common mask.

For both:

* PDB;
* AFDB;

the residue-position set MUST be exactly identical.

If:

\[
M_p
\]

is the manifest mask for protein \(p\), then:

\[
M_p^{PDB}=M_p^{AFDB}=M_p.
\]

Required atoms:

* N
* CA
* C
* O

No mask-external backbone coordinates may enter the Stage-0-2A scorer.

This prevents the intervention from changing both backbone geometry and available structural context.

---

## 5.2 Sequence input

Candidate scientific identity remains the Stage-0-1:

`full_sequence + sequence_hash`

However, ProteinMPNN receives the candidate sequence projected to the same frozen mask:

\[
x^{(k)}_{M_p}.
\]

Thus PDB and AFDB receive:

* identical sequence positions;
* identical sequence projection;
* identical residue indexing;
* different backbone coordinates only.

No mask-external sequence context enters the scorer.

---

# 6. Residue indexing

ProteinMPNN `residue_idx` MUST encode the actual frozen canonical UniProt positions.

Example:

if the common mask is:

`24, 25, ..., 246`

then `residue_idx` represents:

`24, 25, ..., 246`

rather than:

`0, 1, ..., 222`

or:

`1, 2, ..., 223`.

No coordinate compression is allowed.

This remains required even though the current eight Stage-0 masks each consist of one contiguous segment.

The purpose is to preserve the coordinate contract for future masks containing genuine gaps.

---

# 7. ProteinMPNN scoring semantics

The adapter reuses the authorized ProteinMPNN:

* model implementation;
* featurization behavior where applicable;
* forward implementation.

It does not invoke sequence generation.

For candidate \(k\), backbone \(B\), and decoding realization \(r\):

\[
S_{\mathrm{sum}}(k,B,r) =
\sum_{i\in M_p}
\log p_\theta
\left(
x_i^{(k)}
\mid
B_{M_p},
x_{M_p}^{(k)},
\pi_r
\right)
\]

and:

\[
S_{\mathrm{mean}}(k,B,r) =
\frac{1}{|M_p|}
S_{\mathrm{sum}}(k,B,r).
\]

Canonical output names:

* `score_sum_logp_mask`
* `score_mean_logp_mask`

Higher score means greater compatibility under this frozen scoring convention.

The implementation MUST extract target-residue log probabilities directly from ProteinMPNN `log_probs`.

Do not use the official helper's positive-NLL `_scores` sign convention as the Stage-0 score.

---

# 8. Interpretation boundary

`score_mean_logp_mask` is:

> a fixed-sequence, fixed-decoding-realization, autoregressive ProteinMPNN compatibility score over the frozen common mask.

It MUST NOT be described as:

* an order-marginalized sequence likelihood;
* experimental fitness;
* folding free energy;
* calibrated design success probability;
* biological truth.

This distinction must remain in code comments, manifest documentation, and subsequent reports.

---

# 9. Explicit decoding realization

## 9.1 Scientific invariant

For a fixed:

`protein × repeat`

the exact same autoregressive decoding realization MUST be shared across:

* WT;
* every fixed probe;
* PDB;
* AFDB;
* every candidate batch.

Formally:

\[
\pi_{p,r}^{WT,PDB}
= \pi_{p,r}^{WT,AFDB}
= \pi_{p,r}^{k,PDB}
= \pi_{p,r}^{k,AFDB}
\]

for all candidates \(k\).

Only two quantities may vary:

1. candidate sequence;
2. structural condition.

---

## 9.2 Seed alone is insufficient

The implementation MUST NOT rely on:

> “reset RNG to the same seed before every batch”

as the scientific pairing mechanism.

That design is forbidden because RNG consumption may depend on:

* batch size;
* call order;
* candidate count;
* future implementation changes.

Instead, the adapter must explicitly construct one decoding realization for each:

`protein_id × repeat_index`

and reuse that exact realization across scoring calls.

The realization may be represented using the exact random/noise tensor expected by the authorized ProteinMPNN forward implementation.

The tensor is then repeated/tiled across candidate batches without drawing new scientific randomness.

---

## 9.3 Realization provenance

For every:

`protein × repeat`

record at least:

* `repeat_index`
* `seed`
* realization algorithm/version
* realization length
* decoding-realization SHA256 or equivalent deterministic fingerprint.

The scoring manifest should make:

\[
(protein_id,repeat_index)\rightarrow decoding\ realization
\]

explicitly auditable.

---

# 10. WT pairing

WT is scored using exactly the same:

* projected residue domain;
* backbone;
* realization;
* model;
* numerical configuration

as its corresponding fixed probes.

For candidate \(k\):

\[
\Delta S(k,B,r)
= S_{\mathrm{mean}}(k,B,r)
- S_{\mathrm{mean}}(WT,B,r).
\]

Store as:

`delta_score_vs_wt`

The pairing key is:

`protein_id × backbone_condition × repeat_index`.

No WT score may be reused across a different repeat realization.

---

# 11. G2 stochasticity audit

No full 34,010-probe scoring may start before this audit passes.

## 11.1 Audit proteins

Use exactly two admitted proteins:

### Protein A

`5gv8_A__P83686`

Reason:

* full canonical mask;
* mask starts at UniProt 1;
* mask length 272.

Audit positions:

* first: `1`
* interior: `136`
* last: `272`

### Protein B

`5mn1_A__P00760`

Reason:

* truncated but valid frozen mask;
* tests non-1 UniProt coordinate origin;
* mask is `24–246`;
* mask length 223.

Audit positions:

* first: `24`
* interior: `135`
* last: `246`

The positions above are audit fixtures only and must be verified against the frozen manifest before scoring.

---

# 12. Audit mutation set

For each of the six audit positions:

1. retrieve the actual WT residue;
2. traverse the canonical alphabet:

`ACDEFGHIKLMNPQRSTVWY`

3. take the first four residues that differ from WT.

This yields:

* 6 positions;
* 4 mutations per position;
* 24 fixed-probe mutants total.

Additionally score WT for each protein.

Per protein:

* 12 mutant sequences;
* 1 WT.

Total sequence identities:

* 24 mutants;
* 2 WT.

Both PDB and AFDB are scored.

This deterministic selection rule avoids manually choosing mutations based on expected score behavior.

---

# 13. G2 audit stages

## G2-A — Same-realization numerical reproducibility

Use:

`seed = 0`

Construct the realization once per protein.

Repeat the exact audit scoring at least three times using the SAME realization object/content.

All corresponding:

* `score_sum_logp_mask`
* `score_mean_logp_mask`

values must satisfy:

```text
abs(a-b) <= atol + rtol * abs(reference)
```

with:

* `atol = 1e-6`
* `rtol = 1e-6`

for every WT/candidate/backbone score.

If this fails:

`BLOCKED`

for scientific execution.

Do not compensate by increasing repeat count.

Do not classify GPU nondeterminism as scientific scoring stochasticity.

---

## G2-B — Batch-size invariance

Using the same realization, score the same audit candidates using at least:

* batch size `1`;
* a second batch size such as `8` or the entire per-protein audit set.

Candidate scores must agree within the same frozen tolerance.

Failure indicates an adapter/execution defect:

`FAIL`

before full scoring.

---

## G2-C — Decoding-order sensitivity

Construct realizations for:

* seed `0`
* seed `1`
* seed `2`
* seed `3`

Verify their realization/order fingerprints are not accidentally identical.

Score the identical audit set under all four seeds.

Compare scores across different realizations.

### Outcome 1

All different-realization scores remain within frozen tolerance.

Classification:

`deterministic_under_frozen_scoring_protocol`

Formal repeat policy:

`1`

Formal seed:

`0`

Do not create artificial duplicate repeats.

### Outcome 2

Same-realization repetitions are stable, but at least one different decoding realization changes at least one scientific score beyond tolerance.

Classification:

`stochastic_due_to_decoding_order`

Formal repeat policy:

`30`

Formal seeds:

`0..29`

### Outcome 3

Same-realization repetitions themselves exceed tolerance.

Classification:

`numerically_unstable_same_realization`

Result:

`BLOCKED`

No formal Stage-0-2A scoring may start.

---

# 14. G2 audit expected state machine

```text
UPSTREAM + MODEL VERIFIED
          |
          v
SAME REALIZATION REPEAT
          |
      stable?
       /   \
     no     yes
     |       |
 BLOCKED     v
        BATCH INVARIANCE
             |
          passes?
           /   \
         no     yes
         |       |
        FAIL     v
          DIFFERENT REALIZATIONS
                 |
       scientific scores change?
             /          \
           no            yes
           |              |
    DETERMINISTIC     STOCHASTIC
      repeats=1       repeats=30
       seed=0         seeds=0..29
```

The audit result itself becomes frozen scoring provenance.

---

# 15. Formal scoring outputs

Only after G2 passes may formal scoring start.

Required artifacts:

### 15.1 Scoring manifest

`experiments/p2_design_baseline/stage0/fixed_probe_scoring_manifest.json`

### 15.2 WT scores

`experiments/p2_design_baseline/stage0/fixed_probe_wt_scores.parquet`

### 15.3 Fixed-probe raw scores

`experiments/p2_design_baseline/stage0/fixed_probe_scores.parquet`

### 15.4 Same-state scoring null

`experiments/p2_design_baseline/stage0/fixed_probe_scoring_null.parquet`

No PDB-vs-AFDB effect table is generated in Stage-0-2A.

---

# 16. Output row-count contract

If G2 classifies the scorer as deterministic:

WT:

\[
8\times2=16
\]

rows.

Fixed probes:

\[
34010\times2=68020
\]

rows.

If G2 classifies it as decoding-order stochastic:

WT:

\[
8\times2\times30=480
\]

rows.

Fixed probes:

\[
34010\times2\times30=2,040,600
\]

rows.

Formal execution must derive row counts from the frozen G2 classification rather than selecting the cheaper option.

---

# 17. Same-state null semantics

`fixed_probe_scoring_null.parquet` describes only variability from repeated scoring on the SAME backbone.

It MUST NOT contain a PDB-vs-AFDB effect estimate.

### Deterministic classification

Store:

* `n_repeats = 1`
* `scoring_null_type = "deterministic_zero"`

Variability fields may explicitly record zero where semantically appropriate.

Do not construct confidence intervals from fabricated repeated copies.

### Stochastic classification

Store:

* `n_repeats = 30`
* `scoring_null_type = "empirical_repeat_distribution"`

Summarize the 30 realized scores for the same:

`protein × candidate × backbone_condition`.

Descriptive statistics only.

---

# 18. Batch and shard architecture

Formal scoring may be chunked.

A shard is execution infrastructure, not a scientific result.

Each shard must be bound to:

* upstream fixed-probe SHA;
* manifest SHA;
* model checkpoint SHA;
* scoring-protocol version;
* protein ID;
* backbone condition;
* repeat index;
* decoding-realization fingerprint;
* candidate identity range/set.

A shard may be resumed only when all identity fields match.

Never resume a shard solely because a filename exists.

Final canonical outputs are written only after all required cells pass:

* completeness;
* uniqueness;
* finite-value validation;
* candidate joins;
* backbone joins;
* repeat joins.

---

# 19. Canonical scientific score keys

Fixed probes:

```text
(
  protein_id,
  sequence_hash,
  backbone_condition,
  repeat_index
)
```

WT:

```text
(
  protein_id,
  backbone_condition,
  repeat_index
)
```

Shard number, batch number, GPU ID, and dataframe row number are not scientific identities.

---

# 20. Proposed implementation architecture

## Reusable ProteinMPNN adapter

`src/dual_uq/dataset/services/proteinmpnn_scoring.py`

Responsibilities:

* verify/load authorized model;
* construct frozen common-mask backbone projection;
* construct sequence projection;
* preserve UniProt `residue_idx`;
* materialize explicit decoding realization;
* run ProteinMPNN forward;
* extract target-AA `log_probs`;
* compute sum/mean mask logp;
* batch execution;
* numerical validation.

It MUST NOT know Stage-0 admission logic.

---

## Stage-0 fixed-probe scoring service

`src/dual_uq/dataset/fixed_probe_scoring.py`

Responsibilities:

* upstream integrity gates;
* config resolution;
* G2 audit;
* repeat-policy freeze;
* WT pairing;
* fixed-probe iteration;
* shards/resume;
* score schema validation;
* same-state null rendering;
* final immutable artifact materialization.

---

## Thin CLI

`scripts/dataset/score_fixed_probes.py`

Responsibilities:

* project/config resolution;
* invoke scoring service;
* structured status reporting.

No scientific scoring logic should live in the CLI.

---

# 21. TDD plan

Implementation proceeds in the following RED → GREEN order.

No later layer should be implemented to bypass a failing earlier scientific contract.

---

## Layer 0 — Upstream integrity

### RED tests

Fail when:

* admitted subset SHA differs;
* manifest SHA differs;
* fixed-probe SHA differs;
* cohort count is not 8;
* probe count is not 34,010;
* manifest mask total is not 1,790;
* config does not resolve required bindings.

### GREEN target

A pure upstream loader validates all frozen inputs without loading ProteinMPNN.

No GPU required.

---

# 22. Layer 1 — Model identity and loading

### RED tests

Fail when:

* checkpoint path missing;
* SHA mismatch;
* commit/version mismatch;
* unconfigured checkpoint supplied;
* automatic checkpoint fallback attempted.

### GREEN target

One model loader accepts exactly the authorized:

`v_48_020`

identity.

Separate unit tests should use dependency injection/mocks where practical.

A GPU integration test verifies the actual checkpoint can load in `dual-uq-model`.

---

# 23. Layer 2 — Frozen scoring projection

### RED tests

Construct deliberately incorrect projections:

* compressed residue indices;
* PDB/AFDB unequal masks;
* reordered positions;
* missing N/CA/C/O;
* candidate projection inconsistent with full-sequence identity.

All must fail.

### GREEN target

Adapter produces for each protein:

* identical PDB/AFDB UniProt position vector;
* projected coordinates;
* projected sequence;
* exact manifest mask length;
* real UniProt residue indices.

---

# 24. Layer 3 — Decoding realization

### RED tests

Demonstrate that naïvely drawing randomness per batch changes realization assignment or breaks batch invariance.

### GREEN target

A deterministic API such as conceptually:

```text
make_decoding_realization(
    protein_id,
    mask_length,
    seed,
    protocol_version
)
```

returns an explicit realization that can be tiled across arbitrary candidate batches.

Tests verify:

* same protein/seed → same realization bytes/fingerprint;
* different frozen seeds → distinct realization fingerprints;
* batch partitioning does not change realization.

---

# 25. Layer 4 — G2 numerical audit

### RED tests

Use controlled fake scorers to exercise all three state-machine branches:

1. same-realization instability → `BLOCKED`;
2. stable and seed-insensitive → deterministic;
3. stable and seed-sensitive → stochastic 30-repeat policy.

### GREEN integration audit

Run the actual authorized ProteinMPNN only on the frozen 24-mutant + 2-WT audit set.

No formal dataset scoring yet.

Produce an auditable structured G2 result.

---

# 26. Layer 5 — WT scoring

### RED tests

Fail on:

* wrong backbone;
* wrong mask;
* wrong realization;
* wrong residue count;
* non-finite score;
* repeat mismatch.

### GREEN target

Correct WT scoring for:

`protein × PDB/AFDB × required repeat`

with canonical score schema.

---

# 27. Layer 6 — Fixed-probe candidate scoring

### RED tests

Fail on:

* sequence-hash mismatch;
* incorrect mutation projection;
* missing candidate;
* duplicated candidate;
* wrong realization;
* inconsistent batch result;
* wrong scored residue count.

### GREEN target

Small fixed fixture first, then scalable batch path.

Verify direct `log_probs` extraction and score sign using a manually inspected small case.

---

# 28. Layer 7 — WT-relative normalization

### RED tests

Construct mismatched:

* protein;
* backbone;
* repeat

WT joins and require failure.

### GREEN target

Compute:

```text
delta_score_vs_wt =
candidate score_mean_logp_mask
-
matching WT score_mean_logp_mask
```

using an exact key join.

---

# 29. Layer 8 — Shards and resume

### RED tests

Reject shard reuse when any binding differs:

* upstream SHA;
* checkpoint SHA;
* protocol;
* realization fingerprint;
* candidate set;
* backbone;
* repeat.

### GREEN target

Idempotent valid shard reuse and deterministic consolidation.

Partial shards never become canonical scientific outputs.

---

# 30. Layer 9 — Same-state null summary

### RED tests

Fail if:

* deterministic mode fabricates repeated observations;
* stochastic mode has fewer/more than 30 repeats;
* repeat cells are incomplete;
* PDB and AFDB are mixed into a cross-condition effect statistic.

### GREEN target

Render only same-backbone descriptive scoring variability.

---

# 31. Layer 10 — Final artifact integrity

### RED tests

Reject final materialization if there is:

* missing score cell;
* duplicate score key;
* NaN/Inf;
* wrong mask length;
* wrong candidate join;
* wrong model SHA;
* wrong backbone SHA;
* wrong row count;
* mutation of Stage-0-1 upstream artifacts.

### GREEN target

Only after complete validation write:

* scoring manifest;
* WT score table;
* fixed-probe score table;
* scoring-null table.

Use the existing immutable release-writing semantics.

---

# 32. Layer 11 — Regression and reproducibility

Run:

* focused Stage-0-2A tests;
* dataset release tests;
* relevant dataset suite;
* full suite if practical;
* Ruff;
* compileall.

Known unrelated deleted-`Makefile` failure remains excluded from task scope.

Re-running a completed valid scoring materialization must not silently replace immutable scientific outputs.

---

# 33. G2 review artifact

Before formal scoring begins, implementation should expose a concise G2 audit record containing:

`runs/design_baseline/stage0-2a/g2_audit.json`

This is a pre-formal runtime review artifact, not one of the four canonical
Stage-0-2A scientific outputs. It must be written atomically and bound to the
upstream, scoring-protocol, implementation-commit, and checkpoint identities;
an existing valid artifact must not be silently replaced.

* model/checkpoint identity;
* exact two audit proteins;
* exact six positions;
* actual WT residues;
* exact 24 selected mutants;
* seed 0 same-realization repeat comparison;
* batch-size comparison;
* seeds 0–3 realization fingerprints;
* maximum same-realization absolute/relative score deviation;
* whether different realizations changed scientific scores;
* resulting classification;
* resulting formal repeat count;
* formal seed schedule.

This audit artifact should be reviewable before the 34,010-candidate formal run.

---

# 34. Explicit G2 expected outcomes

No outcome is scientifically preferred.

### Expected outcome A

```text
same realization: stable
different realization: score-sensitive
```

Then:

`stochastic_due_to_decoding_order`

and:

`30 repeats, seeds 0..29`

### Expected outcome B

```text
same realization: stable
different realization: score-insensitive
```

Then:

`deterministic_under_frozen_scoring_protocol`

and:

`1 repeat, seed 0`

### Invalid outcome

```text
same realization: unstable beyond tolerance
```

Then:

`BLOCKED`

The implementation must not continue to formal scoring.

---

# 35. Explicit non-goals

Stage-0-2A does NOT:

* generate sequences;
* construct Generated Candidate Union;
* calculate PDB-vs-AFDB score effects;
* rank candidates across conditions;
* compute Top-k instability;
* compute Spearman/rank displacement;
* calculate regret;
* perform hypothesis testing;
* label proteins uncertainty-positive;
* run ESM-IF1;
* run Rosetta/DFIRE;
* perform Arm-B;
* alter admission;
* alter Stage-0-1 mask or probes.

---

# 36. Scientific interpretation after Stage-0-2A

A successful Stage-0-2A produces a validated measurement table of:

\[
S(k,B,r)
\]

on a common sequence intervention domain.

It does not by itself establish structural uncertainty or design risk.

Only a later analysis task may compare the PDB and AFDB conditions.

The appropriate scientific claim at completion is therefore:

> The frozen fixed-probe candidate library has been evaluated under paired PDB and AFDB structural conditions using a single frozen ProteinMPNN model and explicitly controlled autoregressive decoding realizations.

No stronger causal or biological conclusion is permitted at this stage.

---

# 37. Implementation gate

Implementation may begin only after this design and TDD plan are accepted.

The first executable milestone is:

**upstream/model integrity + projection tests + G2 audit only.**

The full 34,010-candidate formal scoring run is forbidden until G2 has produced and frozen one of the two valid repeat policies:

* deterministic → 1 repeat;
* decoding-order stochastic → 30 repeats.
