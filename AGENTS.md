# AGENTS.md

## 1. Mission

This repository develops scientific software for protein design, sequence/structure modeling, dataset construction, uncertainty estimation, model inference, evaluation, and related computational biology research.

Optimize decisions in this order:

1. scientific correctness;
2. quality of evidence;
3. reproducibility;
4. data integrity;
5. clarity of scientific meaning;
6. maintainability and reuse;
7. testability;
8. computational efficiency;
9. implementation convenience;
10. code brevity.

The repository should make scientific questions easier to formulate, test, reproduce, and revise.

Prefer implementations that encode reusable scientific concepts rather than the chronology of individual experiments.

Use `ARCHITECTURE.md` for repository structure and dependency boundaries, but do not let architectural neatness take priority over answering the scientific question correctly.

---

## 2. Scientific Reasoning Is the Default

For non-trivial research work, do not treat the task as only a coding request. Identify the scientific question that the code, analysis, or experiment is meant to resolve.

Use this loop:

```text
QUESTION
  ↓
HYPOTHESIS / SCIENTIFIC EXPECTATION
  ↓
ASSUMPTIONS + POSSIBLE CONFOUNDERS
  ↓
PREDICTION / OBSERVABLE CONSEQUENCE
  ↓
MINIMAL INFORMATIVE TEST OR EXPERIMENT
  ↓
EXECUTE
  ↓
INSPECT ACTUAL OUTPUTS
  ↓
INTERPRET
  ↓
UPDATE THE HYPOTHESIS OR IMPLEMENTATION
  ↺
```

Repeat only while each iteration can materially reduce uncertainty or resolve a concrete discrepancy. Avoid open-ended experimentation without a decision criterion.

Before substantial scientific work, determine:

- what question is actually being asked;
- what quantity or phenomenon is being inferred;
- what evidence would support the proposed explanation;
- what evidence would contradict it;
- which assumptions are necessary;
- which alternative explanations are plausible;
- which confounders or leakage paths could produce the same result;
- what the smallest decisive experiment is.

### Agent scientific judgment

Do not execute a scientific workflow mechanically. Before committing to an implementation or experiment, judge whether the requested operation can actually answer the underlying question.

For each non-trivial scientific task, make an internal decision at these checkpoints:

```text
Is the scientific question well-defined enough to act on?
  ↓
Does the proposed measurement or experiment identify that question?
  ↓
Could leakage, confounding, aggregation, or preprocessing create the same observation?
  ↓
What result would discriminate the main explanations?
  ↓
Will another run reduce uncertainty enough to change a decision?
```

If a requested analysis cannot support the claimed inference, do not blindly treat it as sufficient. Use the smallest scientifically valid interpretation or test that remains within task scope, and state the limitation.

Distinguish exploratory work from confirmatory work. Exploration may generate hypotheses; it should not be presented as independent confirmation of hypotheses selected from the same observations.

Do not confuse successful execution with scientific support. Code can run correctly while the scientific conclusion is wrong.

Do not confuse a failed hypothesis with a software failure. A scientifically negative result is valid evidence when the experiment itself is sound.

When results disagree with expectation, investigate the discrepancy before modifying thresholds, filters, metrics, or code to make the result look expected.

Prefer direct evidence from actual outputs over additional speculative pre-checks when direct execution is safe and informative.

---

## 3. Separate Observation, Interpretation, and Decision

Keep these distinct:

```text
OBSERVATION
what the data/output actually shows

INTERPRETATION
what mechanism or explanation may account for it

DECISION
what experiment, model, or code change should follow
```

Never present an interpretation as if it were directly observed.

When multiple explanations remain consistent with the evidence, state that ambiguity and design a discriminating test instead of selecting one explanation by convenience.

Prefer falsifiable explanations over post-hoc narratives.

Do not optimize only for a headline metric. Check whether an apparent improvement comes from:

- data leakage;
- split contamination;
- changed sample composition;
- altered missing-value handling;
- favorable aggregation;
- threshold tuning on evaluation data;
- duplicated or homologous examples;
- different preprocessing;
- accidental filtering of difficult cases;
- changes in stochastic sampling;
- changed model/checkpoint identity.

Negative, null, and unexpected results should remain visible when scientifically valid.

---

## 4. Preserve Scientific Meaning

Never silently change scientific semantics.

Scientifically meaningful behavior includes, where relevant:

- dataset membership;
- inclusion/exclusion and admission rules;
- clustering and redundancy definitions;
- train/validation/test splitting;
- mutation and probe semantics;
- residue and chain indexing;
- sequence/structure mapping;
- amino-acid alphabet or ordering;
- masks and missingness;
- normalization scope;
- model scoring semantics;
- uncertainty definitions;
- metric definitions;
- aggregation and resampling units;
- stochastic sampling;
- seed derivation.

Refactoring should preserve these meanings unless the task explicitly intends to change the science.

If scientific meaning changes, validate the changed claim directly rather than relying only on generic software tests.

For protein and structure workflows, explicitly check indexing, chain identity, residue mapping, masks, insertion/deletion handling, missing residues, alternate coordinates when relevant, and whether aggregation is per-residue, per-sequence, per-structure, or per-protein.

Do not silently turn `missing`, `invalid`, `masked`, `unavailable`, `failed`, and `unobserved` into one generic state or numeric zero.

Do not silently drop scientific records.

---

## 5. Canonical Scientific Ownership

Maintain one clear implementation for each reusable scientific concept.

Before creating new logic:

```text
search
→ identify the existing scientific owner
→ reuse
→ extend
→ adapt
→ create only when genuinely new
```

Do not create a second implementation merely because it is convenient for one experiment.

Do not duplicate scientific definitions across scripts, workflows, notebooks, reports, tests, and experiment directories.

An independent implementation is appropriate when it serves as a deliberate reference/oracle for validation.

Experiment chronology belongs primarily in configuration, run metadata, or output directories, not in reusable implementation names.

Repository-wide scientific names are:

- `StructCal` for the benchmark (`structcal` in paths and Python identifiers);
- `ReSC` for Reliable Structural Conditioning (`resc` in paths and Python identifiers).

Avoid reusable names such as `a0`, `stage1`, `scale1b`, `step2`, `00_...` when a scientific name is available.

Prefer names such as `admission`, `selection`, `redundancy`, `mapping`, `probes`, `scoring`, `evaluation`, `calibration`, `uncertainty`, and `planning`.

---

## 6. Keep Engineering Proportional to the Research Need

Choose the simplest implementation that answers the current scientific or computational need correctly.

Avoid premature:

- abstraction;
- framework construction;
- registries or plugin systems;
- generic managers;
- workflow engines;
- configuration layers;
- compatibility layers;
- helper dumping grounds.

Prefer:

```text
function
→ cohesive module
→ small policy/specification object
→ interface only when real variation requires it
```

Prefer composition over inheritance.

Use mature existing libraries when they already solve the problem adequately.

Do not preserve obsolete internal APIs by default. When an internal implementation has been superseded, migrate current callers and remove the obsolete path unless a real external dependency still requires it.

Do not reorganize large parts of the repository merely to make one local change look architecturally elegant.

---

## 7. Inspect Before Changing

For non-trivial work, inspect only the relevant repository state.

Typical sources include:

- `AGENTS.md` and more-specific scoped instructions;
- `ARCHITECTURE.md`;
- affected modules;
- callers;
- focused tests;
- experiment/configuration files;
- relevant datasets or schemas;
- CLI/workflow entry points;
- repository tooling.

Prefer targeted exploration with `rg`, known module paths, imports, and relevant tests. Avoid repeated whole-repository scans.

Before editing, identify:

- the scientific question or requested behavior;
- the current implementation responsible for it;
- scientific invariants that must remain true;
- likely confounders and failure modes;
- expected files to change;
- the smallest useful validation.

Do not invent a new module, schema, workflow, or helper before checking whether an adequate implementation already exists.

---

## 8. Experiment and Analysis Discipline

Design experiments to distinguish hypotheses, not merely to produce more outputs.

A useful experiment should have a clear reason for being run and a clear interpretation for major possible outcomes.

Prefer controlled comparisons. Change one scientifically important factor at a time when feasible.

When comparing methods, keep non-target factors aligned unless the difference itself is part of the hypothesis, including:

- dataset membership;
- preprocessing;
- split construction;
- checkpoint/model identity;
- decoding or sampling settings;
- random seeds or seed policy;
- evaluation subset;
- metric implementation;
- aggregation unit.

Check for data leakage and dependence before interpreting performance differences.

For train/validation/test or benchmark construction, consider biological redundancy and homology, not only row-level duplication.

For uncertainty or calibration work, distinguish predictive quality from calibration quality. Do not infer one solely from the other.

For aggregate metrics, inspect the unit of aggregation and, when relevant, the distribution of per-protein or per-example values rather than relying only on a mean.

When a thresholded success metric is used, inspect both threshold satisfaction and the underlying continuous measurements.

Do not tune scientific conclusions to a single random seed. Use repeated seeds or an explicitly justified deterministic setup when stochastic variation could change the conclusion.

---

## 9. Failure Diagnosis

Classify failures before reacting to them.

### Software / invariant failure

Examples: impossible internal state, malformed input representation, inconsistent residue mapping, violated schema assumption.

Action: fail clearly and fix the implementation or invalid input handling.

### Scientific / data outcome

Examples: a candidate fails a scientifically defined criterion, no valid mapping exists, a hypothesis is unsupported, a dataset is insufficient for the intended inference.

Action: record the outcome and reason. Do not disguise it as a software error.

### Infrastructure / operational failure

Examples: GPU failure, network timeout, missing remote asset, subprocess crash, storage failure, transient service failure.

Action: report it as operational. Retry only when appropriate.

Never reinterpret infrastructure failure as scientific rejection.

When debugging a surprising scientific result, check in roughly this order:

1. whether the result is measured correctly;
2. whether the compared populations are actually comparable;
3. whether indexing/mapping/masking is correct;
4. whether preprocessing or filtering changed;
5. whether metric aggregation changed;
6. whether stochasticity can explain the discrepancy;
7. whether the scientific hypothesis itself is wrong.

Do not begin by changing the result-producing code simply because the output is surprising.

---

## 10. Reproducibility and Provenance

Keep enough provenance to reconstruct scientifically meaningful results without turning ordinary research iteration into release engineering.

Record the relevant subset of:

- source dataset identity;
- important data-selection policy;
- model/checkpoint;
- preprocessing;
- scientific configuration;
- seed or seed policy;
- code revision;
- important uncommitted changes when they affect results.

Use stable logical identifiers rather than machine-specific absolute paths as scientific identity.

Do not add SHA256 fields, per-file checksum ledgers, duplicate manifests, immutable-write mechanisms, or audit artifacts by default.

Use cryptographic hashes only when they solve a concrete integrity problem, are required by an external/release/publication interface, or are already part of an authoritative artifact format.

For ordinary research iteration, prefer simpler evidence such as:

- explicit run configuration;
- stable record IDs;
- dataset version/release identifiers;
- row/sample counts;
- schema checks;
- targeted deterministic comparisons;
- Git history.

Do not repeatedly hash or fully replay large artifacts when a focused scientific check establishes the claim being tested.

---

## 11. Randomness and Determinism

Scientific randomness must be intentional and reproducible enough for the claim being made.

Prefer explicit root seeds and stable semantic identifiers.

Do not derive scientific randomness from:

- wall-clock time;
- process ID;
- filesystem iteration order;
- worker scheduling order;
- unstable language hashes.

Parallel execution should not silently change scientific meaning.

When exact bitwise determinism is expensive or unnecessary, prioritize reproducibility of the scientific conclusion and characterize stochastic variability instead of adding heavy engineering solely to force identical bytes.

---

## 12. Caching

Treat incorrect cache reuse as a scientific correctness bug.

Cache keys must represent inputs that can change the scientific result, such as:

- protein/input identity;
- structure identity;
- model/checkpoint;
- preprocessing;
- scientific configuration;
- scientifically meaningful version identifiers.

Do not include unrelated runtime metadata merely because it exists globally. Log level, report path, timestamp, worker count, and plotting settings are normally non-semantic unless they actually affect the result.

Prefer transparent caches that are easy to invalidate and inspect over elaborate cache governance.

### Persistent storage and materialization

Before creating a source file, script, experiment directory, or run directory, search for the existing module, function, CLI, workflow, and configuration that own the same operation. A new task, pilot, ablation, scale-up, repeat, or report normally changes arguments or configuration; it does not create another implementation or version-suffixed directory.

A structure with the same scientific identity should have one canonical persistent file under `data/`. Experiments and runs store `structure_id`, the repository-relative canonical path, parent identity, and perturbation parameters instead of copying the structure. Do not use a content hash as a substitute for scientific structure identity.

When an external program needs an isolated writable directory, materialize inputs in a project-local temporary workspace. Extract the scientific result, then remove that workspace. Do not retain copied inputs, build/repair directories, subprocess logs, shards, chunk manifests, or caches after a consolidated result replaces them.

Keep `runs/` small by default. Retain result tables, summaries, figures, metadata needed to interpret conditions, and only the checkpoint needed for subsequent use. Intermediate checkpoints are restart state while training is active; after a predefined selection is complete, retain the selected checkpoint and its training/validation curves unless the scientific protocol explicitly requires the full trajectory.

Do not add SHA256, freeze, release, preflight, smoke, or audit machinery merely to justify pruning regenerable runtime files. When cleanup is requested, verify replacement by semantic identity, expected keys/counts, and the final scientific output that consumes it.

---

## 13. Validation Strategy

Validation should answer: **what evidence is sufficient to trust this specific change or conclusion?**

Use:

```text
smallest decisive scientific check
→ focused software test
→ targeted regression/comparison
→ broader validation only if uncertainty remains or blast radius is large
```

Do not run the full test suite by default.

Use broader validation when changes affect shared scientific behavior, dataset construction, model scoring, persistent serialization, major cross-package APIs, or multiple top-level subsystems.

For scientific changes, prefer checks that interrogate the actual semantic consequence. Examples include:

- compare dataset membership before/after;
- verify sequence-to-structure mapping on representative edge cases;
- compare per-protein score distributions;
- test invariance to batching when batching should be irrelevant;
- confirm split independence or redundancy criteria;
- compare stochastic variation across seeds;
- inspect calibration curves or error stratification rather than only one summary metric.

Never claim a test, experiment, or validation passed unless it actually ran.

When a test fails, determine whether it reveals a bug, an outdated expectation, or a genuine scientific change before modifying the test.

---

## 14. Performance

Scientific correctness precedes optimization.

Optimize measured or clearly dominant bottlenecks.

Prefer established techniques such as batching, vectorization, streaming, memory-aware tensor operations, caching, and validated mixed precision.

Performance changes must preserve scientifically relevant behavior, including:

- masks;
- residue mappings;
- normalization scope;
- sampling distributions;
- random streams where required;
- aggregation units.

Do not accidentally convert per-protein semantics into per-batch semantics.

Do not sacrifice scientific interpretability or reproducibility for minor speed improvements.

---

## 15. Task Scope and Repository Safety

For analysis/review/planning tasks, inspect and report without modifying files unless modification is also requested.

For implementation/fix/refactor/update tasks, make the smallest coherent local change and run relevant non-destructive validation without asking for confirmation for ordinary edits.

Explicit authorization is still required for committing, pushing, destructive repository operations, persistent external writes, or materially expanding the requested scope.

Preserve unrelated dirty and untracked state.

Do not use destructive operations such as `git reset`, `git clean`, force checkout, or force push unless explicitly requested.

Do not commit or push unless explicitly requested.

Do not use `git add .` or `git add -A` when unrelated changes may exist. Stage only relevant files when staging is requested.

Do not fix unrelated issues, rename unrelated APIs, broadly reformat files, upgrade unrelated dependencies, or perform opportunistic repository cleanup.

---

## 16. Documentation

Do not create task-specific engineering reports such as `design_report.md`, `implementation_report.md`, `review_report.md`, `task_summary.md`, `execution_notes.md`, or `codex_report.md` unless explicitly requested.

Prefer durable scientific documentation:

```text
ARCHITECTURE.md
    repository structure and dependency boundaries

docs/scientific/
    scientific definitions, assumptions, metrics, datasets, and interpretation

docs/protocols/
    reusable experimental and evaluation procedures

docs/decisions/
    durable scientific or architectural decisions when they genuinely need recording
```

Do not promote one-task operational details into `AGENTS.md`.

---

## 17. Definition of Done

A non-trivial research or modification task is complete when:

1. the requested behavior or scientific question has been addressed;
2. the relevant scientific meaning is preserved or intentionally changed;
3. the evidence actually supports the stated conclusion;
4. plausible confounders and alternative explanations have been considered when material;
5. canonical scientific ownership remains clear;
6. focused validation has actually run;
7. surprising or negative results have not been hidden or normalized away;
8. unrelated repository state is preserved;
9. remaining scientific uncertainty is stated clearly.

Before finishing, perform one final research check:

```text
What do I now believe?
What evidence changed that belief?
What important alternative explanation remains?
What result would make the current conclusion wrong?
Is another experiment necessary, or would it only produce more data without reducing uncertainty?
```

Stop when the task is answered with sufficient evidence. Do not continue refactoring, experimenting, or adding governance for its own sake.

### Final response

Keep the final response proportional to the task.

For substantial modification or research tasks, summarize:

```text
RESULT
- what was changed or learned

EVIDENCE
- checks/experiments actually executed

INTERPRETATION
- what the evidence supports

UNCERTAINTY
- remaining material risk or alternative explanation, if any

FILES
- relevant modified paths
```

Do not reproduce the task prompt and do not create a separate execution-report Markdown file unless explicitly requested.
