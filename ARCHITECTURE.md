# ARCHITECTURE.md

## 1. Purpose

Dual-UQ is a scientific software repository for computational protein design, sequence/structure modeling, uncertainty analysis, and reproducible evaluation.

Its architecture exists to make scientific questions easier to formulate, test, compare, reproduce, and revise.

The repository should support work such as:

- protein sequence and structure processing;
- dataset and cohort construction;
- PDB / AlphaFold DB integration;
- structure-conditioned protein design;
- sequence scoring and model inference;
- structure, evaluator, and joint uncertainty;
- calibration and reliability analysis;
- mechanism and counterfactual analysis;
- StructCal benchmark construction and evaluation;
- ReSC method development and evaluation.

The architectural objective is not maximal abstraction, release engineering, or platform-like governance.

The objective is the smallest stable structure that protects scientific meaning while enabling informative experiments and reuse.

`AGENTS.md` defines how an agent should reason and act. This file defines where scientific responsibilities live, how data and computation flow, and which dependency boundaries should remain stable.

---

## 2. Research-First Architecture

Architecture should follow the scientific reasoning loop used by the repository:

```text
scientific question
    ↓
hypothesis / competing explanations
    ↓
scientific objects + controlled comparison
    ↓
model or analysis operation
    ↓
measurement
    ↓
aggregation / uncertainty analysis
    ↓
interpretation
    ↓
next experiment or revised hypothesis
```

Each architectural layer should make one or more of these transitions explicit.

For non-trivial experiments, the architecture should make it possible to recover the key assumptions, plausible confounders, and the evidence that discriminates competing explanations. These do not require a heavyweight metadata system; a typed experiment specification, concise protocol note, and preserved primitive outputs are often sufficient.

A useful boundary is one that prevents scientifically different concepts from being silently mixed, for example:

- sequence index vs structure-array index;
- PDB condition vs AFDB condition;
- model score vs derived metric;
- per-residue observation vs per-protein estimand;
- missing data vs scientific exclusion;
- exploratory analysis vs confirmatory evaluation;
- structure uncertainty vs evaluator uncertainty;
- data construction vs model eligibility;
- observation vs interpretation.

Do not create architectural machinery merely because a system could theoretically be more formal.

Prefer direct scientific clarity over additional wrappers, registries, schemas, hashes, manifests, or lifecycle states unless they solve a demonstrated problem.

---

## 3. Core Architectural Principles

### 3.1 Organize by scientific capability

Reusable code should be named and organized around stable scientific or computational responsibilities such as:

```text
mapping
comparability
admission
redundancy
selection
structure conditioning
design
scoring
uncertainty
metrics
evaluation
calibration
reporting
```

Do not organize reusable implementation around experiment chronology such as:

```text
stage1
stage2
scale1a
scale1b
step3
p2
```

Historical identifiers may remain in experiment metadata, output metadata, or reproduction-only paths.

### 3.2 One active owner per scientific concept

A reusable scientific concept should have one clear active implementation.

Before adding new logic:

```text
find current owner
→ reuse
→ extend
→ adapt at an existing boundary
→ create only if scientifically distinct
```

Do not maintain multiple active implementations of mapping, admission, scoring, metrics, aggregation, or uncertainty definitions merely for different experiments.

A second implementation is justified when it is deliberately used as an independent reference or oracle.

### 3.3 Dependencies point toward more stable meaning

High-level experiment composition may depend on lower-level scientific capabilities.

Low-level scientific modules must not depend on:

- CLI code;
- report rendering;
- run directories;
- historical experiment names;
- task-specific orchestration.

External source and model details should remain behind adapters.

### 3.4 Preserve information before summarizing it

When computationally reasonable, preserve information-rich intermediate results so later scientific questions do not require unnecessary recomputation.

For example, prefer preserving per-candidate or per-residue model outputs when downstream metrics may change, rather than persisting only one aggregate score.

Summaries belong after the primitive scientific measurements they summarize.

### 3.5 Engineering effort should be proportional to scientific risk

Ordinary research iteration should use ordinary Python modules, focused tests, explicit configurations, clear experiment directories, and semantic provenance.

Do not default to:

- preflight or readiness phases;
- repository-wide audits before execution;
- dry-run or validation gates that duplicate evidence available from real execution;
- release services;
- artifact registries;
- immutable-write frameworks;
- workflow engines;
- state databases;
- checksum chains;
- broad compatibility layers;
- schema frameworks for transient internal objects.

Use stronger machinery only when the scientific or external interface genuinely requires it.

### 3.6 Continuity over task-local code

The repository should evolve by extending existing scientific capabilities, not by accumulating one-off files and directories for each requested task. A task name is not an architectural boundary.

Before writing new processing logic, inspect the relevant existing code path with targeted searches: the current scientific owner, nearby functions, callers, tasks/runners, metrics/evaluators, configuration, and tests when they are directly relevant. The purpose is to find existing logic that can be reused or extended, not to create a separate preflight phase.

For a new experiment or analysis, prefer:

```text
find existing scientific owner
→ reuse existing function/module
→ extend its coherent interface if needed
→ reuse the existing task/runner/CLI
→ express experiment-specific variation in arguments or configuration
```

over:

```text
new task
→ new folder
→ new Python script
→ copied or slightly modified logic
→ another folder/script for the next task
```

Do **not** create a new directory merely because a new task, experiment, figure, cohort, ablation, threshold, model checkpoint, or comparison is requested. Create a directory only when it represents a durable scientific namespace, a genuinely distinct reusable subsystem, or a real storage boundary that cannot be expressed cleanly in an existing location.

Do **not** create a new `.py` file merely because the requested operation is new to the current task. First determine whether the behavior belongs in an existing module. Prefer adding a focused function, method, policy field, task option, runner branch, or evaluator operation to the existing owner when the scientific responsibility is unchanged.

Experiment directories should primarily contain scientific intent, configuration, compact notes when needed, and outputs. They must not become parallel source trees containing their own copies of processing, scoring, filtering, plotting, or evaluation logic.

A one-off exploratory script is acceptable only when the work is genuinely disposable and has no reusable scientific logic. Once logic is scientifically meaningful or likely to be reused, place it in the existing canonical owner; keep invocation thin.

Do not create a new metric, task, evaluator, model wrapper, parser, or helper merely to give one experiment a new name. Different datasets, thresholds, subsets, arms, checkpoints, output locations, or presentation names normally reuse the same implementation.

When an existing implementation is close but incomplete, **extend it rather than clone it**. Duplication is not an acceptable way to isolate tasks.

### 3.7 Parameterize real experimental variation

Do not hard-code values that are expected to vary across scientifically meaningful runs. Prefer explicit function arguments or small typed experiment specifications for values such as:

- dataset or cohort identity;
- model and checkpoint choice;
- structural condition or comparison arm;
- thresholds and perturbation magnitudes;
- sampling budget and decoding settings;
- seeds or seed policy;
- evaluation subset;
- aggregation or resampling choices when they are part of the analysis;
- input/output locations;
- device, batch size, and worker count.

Avoid branches such as `if experiment == "scale1b"` or hidden constants embedded in task-specific code when the difference is actually a parameter or scientific choice.

Parameterization does **not** mean moving every constant into YAML. Stable scientific definitions should remain explicit in their canonical owner. Examples include amino-acid ordering, residue-index semantics, score sign conventions, and a metric equation whose definition is intended to be fixed.

Use the following distinction:

```text
scientific invariant        → canonical code owner
scientifically varying choice → typed experiment specification / explicit argument
runtime-only choice         → runtime configuration
```

Do not replace hard-coded values with an unstructured global configuration dictionary. Expose only variation that is real and useful.

---

## 4. Scientific Composition vs Python Dependencies

Scientific workflow composition and Python import direction are different concepts.

A typical scientific execution path is:

```text
experiment specification
        ↓
runner / workflow composition
        ↓
scientific capabilities + source/model adapters
        ↓
primitive scientific outputs
        ↓
metrics / aggregation / uncertainty analysis
        ↓
figures, tables, interpretation-ready results
```

This does not imply that every downstream stage owns the objects it consumes.

For example:

- a model adapter may produce model-native outputs;
- a normalized scientific representation may live outside the model adapter;
- metrics consume outputs but do not own model execution;
- evaluation consumes metrics but does not redefine them;
- reporting presents evaluation results but does not alter scientific meaning.

---

## 5. Target Repository Layout

The following is a target organization, not a requirement to create empty directories or perform a large rewrite.

```text
repository/
│
├── AGENTS.md
├── ARCHITECTURE.md
├── README.md
├── pyproject.toml
│
├── configs/
│   ├── datasets/
│   ├── models/
│   ├── experiments/
│   ├── runtime/
│   └── local/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── experiments/
├── runs/
├── artifacts/
│
├── src/
│   └── dual_uq/
│       ├── core/
│       ├── structure/
│       ├── construction/
│       ├── benchmark/
│       ├── design/
│       ├── models/
│       ├── tasks/
│       ├── uncertainty/
│       ├── metrics/
│       ├── evaluation/
│       ├── reporting/
│       ├── runners/
│       └── cli/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── docs/
│   ├── scientific/
│   ├── protocols/
│   ├── decisions/
│   └── archive/
│
└── third_party/
```

Historical packages may remain while active work migrates toward these boundaries.

Do not mechanically move files simply to match this tree. Migrate when active scientific work exposes a concrete ownership or duplication problem.

### 5.1 `dataset/` vs `construction/`

New reusable dataset-construction work should prefer `construction/` when that is the active canonical owner.

Historical `dataset/` implementations may remain for reproduction or until their behavior is understood well enough to migrate safely.

Do not maintain both as competing active implementations of the same scientific operation.

### 5.2 `workflows/` / `inference/` vs `runners/` / `tasks/`

If historical `workflows/` or `inference/` modules exist, preserve them when needed for current callers, but new architecture should separate:

- semantic scientific operations in `tasks/`;
- model-specific behavior in `models/`;
- thin experiment composition in `runners/`;
- statistical interpretation in `evaluation/`.

Do not create a second private evaluation or scoring path merely to support a new experiment.

---

## 6. Dependency Direction

A practical default dependency model is:

| Package | May depend on |
| --- | --- |
| `core` | foundational libraries |
| `structure` | `core` |
| `construction` | `core`, `structure` |
| `benchmark` | `core`, `structure`, selected construction-level scientific representations |
| `design` | `core`, `structure` |
| `models` | `core`, `structure`, `design` where required |
| `tasks` | scientific capabilities and model adapters required for the operation |
| `uncertainty` | `core`, normalized scientific/model outputs |
| `metrics` | `core`, stable scientific representations |
| `evaluation` | `metrics`, `uncertainty`, normalized scientific results |
| `reporting` | `metrics`, `uncertainty`, `evaluation` |
| `runners` | the lower-level capabilities required by a concrete experiment |
| `cli` | `runners` and boundary-level configuration |

This table is guidance, not blanket permission for arbitrary imports.

Before introducing a new top-level dependency, ask:

1. Which package scientifically owns the concept being imported?
2. Does this dependency move toward more stable meaning?
3. Would it create peer-package coupling or a cycle?
4. Is the shared object truly common, or is one package leaking orchestration into another?

Avoid cycles such as:

```text
metrics ↔ evaluation
construction ↔ reporting
design ↔ models
uncertainty ↔ evaluation
```

When a cycle appears, inspect ownership before using dynamic imports, monkey-patching, service locators, or moving arbitrary code into `core/`.

---

## 7. `core/`

`core/` contains only genuinely cross-cutting primitives with no natural scientific owner.

A concept belongs in `core/` when:

1. no scientific package naturally owns it;
2. multiple independent packages need it;
3. its semantics are stable independently of those callers;
4. moving it there improves dependency direction.

Possible examples include:

```text
shared identifiers
small error/result primitives
portable provenance primitives
common typed metadata
```

Do not create dumping grounds such as:

```text
core/utils.py
core/helpers.py
core/common.py
```

Do not create a hashing subsystem for ordinary research code. Stable scientific identifiers and semantic metadata are the default; hash-based validation is reserved only for an explicit external requirement.

---

## 8. Scientific Objects and Data Boundaries

Scientific computations should operate on meaningful scientific objects, typed specifications, arrays/tensors with explicit semantics, or iterables of such objects.

They should not depend unnecessarily on repository path layouts.

Prefer:

```python
value = sequence_recovery(
    prediction,
    native_sequence,
    mask,
)
```

rather than:

```python
value = sequence_recovery(
    experiment_directory,
    prediction_csv,
    output_directory,
)
```

Filesystem discovery, serialization, remote retrieval, and output placement belong near runners, adapters, or persistence boundaries.

### 8.1 Identity

Keep identity semantic and inspectable.

Useful fields may include:

```text
protein_id
structure_id
candidate_id
condition
backbone_source
generation_source
scoring_source
model_id
checkpoint_id
seed
replicate
```

Do not encode a large Cartesian product of scientific attributes into opaque path names.

Opaque IDs are acceptable when metadata clearly defines their meaning.

### 8.2 Missingness and exclusions

Keep distinct:

```text
missing
invalid
masked
unavailable
scientifically excluded
operationally failed
unobserved
```

A downstream metric must not infer these states from numeric sentinels when an explicit representation is feasible.

### 8.3 Observation unit

Scientific objects should make the observation unit explicit when aggregation depends on it.

Possible units include:

```text
residue
mutation/probe
candidate sequence
structure
condition pair
protein
family/cluster
cohort
```

Do not let dataframe row structure silently define the scientific unit.

---

## 9. Structure and Residue Mapping

`structure/` owns protein structural representation and residue-coordinate semantics.

Responsibilities may include:

- PDB/mmCIF parsing;
- chain representation;
- residue identity;
- missing-residue handling;
- sequence/structure mapping;
- coordinate validation;
- geometry;
- structure confidence.

Potential modules:

```text
structure/
├── records.py
├── parsing.py
├── mapping.py
├── geometry.py
├── confidence.py
└── validation.py
```

Residue identity is an explicit mapping problem.

Relevant coordinate spaces may include:

```text
author/PDB residue identifier
canonical sequence index
structure-array index
model/tensor index
alignment index
```

Conversions between these spaces should be centralized.

Dataset construction, model adapters, metrics, uncertainty, and evaluation code must not independently reconstruct residue mapping semantics.

Mapping representations should preserve enough information to diagnose insertions, deletions, missing coordinates, chain mismatches, and masking differences.

Round-trip and boundary cases should be directly tested.

---

## 10. Construction

`construction/` owns reusable scientific operations that determine what objects enter an analysis and how those objects are normalized and compared.

Typical responsibilities include:

```text
source acquisition
normalization
scientific validity checks
sequence/structure comparability
admission
redundancy / clustering
candidate or cohort selection
structural annotations
functional-state pairing
model-independent eligibility
```

Potential organization:

```text
construction/
├── records.py
├── acquisition.py
├── normalization.py
├── comparability.py
├── admission.py
├── redundancy.py
├── selection.py
└── annotations.py
```

### 10.1 Acquisition

Acquisition coordinates source retrieval and source-specific status.

External API or filesystem details belong in source adapters, for example:

```text
construction/sources/
├── rcsb.py
├── afdb.py
└── uniprot.py
```

A network or source failure is operational failure, not scientific rejection.

### 10.2 Scientific validity and admission

Parsing answers whether an artifact can be interpreted.

Scientific validity asks whether the interpreted object is usable for the intended scientific operation.

Admission asks whether it belongs in a particular scientific population or analysis under an explicit policy.

Keep these distinctions visible.

Admission results should retain the candidate/protein identity, decision, reason, and relevant evidence.

### 10.3 Redundancy and leakage control

Redundancy logic owns clustering and related biological dependence calculations.

For train/validation/test or benchmark construction, biological homology and cluster structure matter more than row-level duplication alone.

Do not duplicate clustering logic for each experiment.

### 10.4 Selection

Selection should expose the scientific policy explicitly.

Examples include:

```text
cluster-first selection
stratified selection
candidate prioritization
reserve selection
cohort selection
```

Avoid hiding selection policy in stage-specific scripts or ad hoc dataframe filters.

---

## 11. Benchmark Architecture

`benchmark/` owns model-independent benchmark semantics that are genuinely shared across benchmark construction and evaluation.

It may own:

- benchmark identities;
- semantic condition names and orientation;
- benchmark instance definitions;
- split and cluster assignments when they are part of benchmark meaning;
- normalized benchmark tables required by downstream evaluation;
- relational checks needed to interpret those tables.

It must not own:

- concrete model invocation;
- checkpoint loading;
- method-specific private evaluation paths;
- report-specific metric redefinitions;
- historical orchestration merely because it produced the first dataset.

Do not turn `benchmark/` into a general governance package.

Use schemas only where stable persisted benchmark tables benefit materially from machine-readable validation.

Do not create a schema for every transient Python object.

---

## 12. Design

`design/` owns model-independent protein-design concepts.

Responsibilities may include:

- candidate sequence representation;
- mutation and probe definitions;
- allowed and fixed positions;
- sequence-distance constraints;
- design constraints;
- model-independent sampling specifications.

Potential modules:

```text
design/
├── candidates.py
├── probes.py
├── constraints.py
└── sampling.py
```

Model-native tensor manipulation, vendor-specific file formats, checkpoint layout, and model execution do not belong here.

---

## 13. Models and Third-Party Boundaries

`models/` owns model-specific behavior and adapters.

Examples may include:

- ProteinMPNN;
- LigandMPNN;
- MoMPNN;
- DynamicMPNN;
- ESM-family models;
- RFdiffusion-family models.

Do not create a universal model framework before multiple concrete models demonstrate the same substitutable capability.

Treat capabilities separately. A model may implement both sequence design and sequence scoring without making those two operations the same abstraction.

Model adapters should localize:

- checkpoint loading;
- native vocabulary/alphabet;
- model-native input/output formats;
- tensor conversion;
- device invocation;
- model-specific batching;
- third-party directory assumptions.

Vendored code under `third_party/` is an implementation detail and should not leak throughout the repository.

### 13.1 ProteinMPNN scoring

`dual_uq.models.proteinmpnn` is the active owner of concrete ProteinMPNN scoring behavior.

It owns ProteinMPNN-specific:

- amino-acid ordering used by the adapter;
- structure and candidate tensor construction;
- sequence projection into model coordinates;
- decoding-order / realization interpretation;
- model loading and checkpoint binding;
- model-native batching;
- target-amino-acid log-probability extraction;
- score aggregation over the declared valid residue mask.

When the scientific analysis uses both summed and mean masked log-probability, keep their meanings explicit, e.g. `score_sum_logp_mask`, `score_mean_logp_mask`, and `scored_residue_count`. Do not collapse them into an ambiguous generic `score`.

WT and mutated/probe candidates should remain semantically distinguishable. WT records should not require fake mutation sentinels merely to share a table representation.

The model adapter does **not** own PDB/AFDB pair policy, functional-state pairing, common-mask policy, cohort membership, or population-level comparison. Those are construction/task/evaluation concerns.

A logical scoring request should identify the structure condition, comparable residue domain, candidate collection, stochastic realization when relevant, and score meaning. Worker IDs, shard paths, output directories, and device assignments are execution details rather than scientific identity.

The adapter should return project-level scientific outputs rich enough for downstream comparison without forcing evaluation code to understand ProteinMPNN internals.

Historical ProteinMPNN scoring paths may remain when needed to reproduce earlier results, but new work should use the active scoring owner rather than adding another versioned wrapper.

---

## 14. Tasks

`tasks/` owns semantic scientific operations that compose one or more lower-level capabilities around a clear estimand or intervention.

Examples may include:

```text
local sensitivity
generative propagation
sequence scoring
multistate generation
controlled intervention evaluation
```

A task should express what scientific operation is being performed, not which model implements it and not which experiment first requested it.

Tasks may depend on model capability interfaces when real substitution exists.

Tasks should not own:

- model-specific invocation details;
- dataset construction;
- report formatting;
- alternative private metric equations.

A new experiment should normally configure an existing task rather than create a near-identical new task module. Changing a threshold, cohort, checkpoint, sampling setting, condition, or aggregation request is normally a parameterization change, not a reason for a new task or Python file.

When an existing task is almost sufficient, extend its coherent scientific interface instead of cloning it. Split into a new task only when the operation has a different estimand, intervention, input/output semantics, or scientific responsibility.

---

## 15. Scientific Model Outputs

Normalized model outputs should preserve the information needed for scientifically valid downstream comparisons.

Possible fields include:

```text
protein_id
candidate_id
structure / condition identity
residue mapping
amino-acid ordering
raw or normalized model outputs
valid mask
model_id
checkpoint_id
seed / realization identity
```

Do not decide ownership of a normalized output type merely from who produces it. Ownership should follow the stable scientific meaning consumed by multiple downstream packages.

Preserve primitive outputs when storage cost is reasonable so that new metrics, calibration analyses, and error stratifications can reuse existing inference.

Do not rerun expensive model inference merely because a downstream summary definition changed, when stored primitive outputs already contain the required information.

---

## 16. Uncertainty

`uncertainty/` owns uncertainty definitions, estimators, and reusable uncertainty computations.

Distinct concepts may include:

```text
structure uncertainty
evaluator uncertainty
joint uncertainty
```

Do not collapse these into a generic uncertainty framework simply because they share mathematical utilities.

A fitted transformation that changes the meaning of an uncertainty estimator belongs with that estimator or a clearly adjacent scientific owner.

Keep uncertainty quantity, predictive quality, and calibration quality conceptually distinct.

---

## 17. Metrics

`metrics/` owns deterministic scientific metric definitions.

Examples include:

- sequence recovery;
- ranking quantities;
- stability and diversity metrics;
- calibration-error metrics;
- reliability-related deterministic statistics;
- score transformations that are themselves estimands.

A metric should have a clear scientific meaning, sign convention, normalization, and observation domain.

Metric APIs should consume scientific objects or arrays, not experiment directory structures.

Do not create a new metric because a new report, arm, experiment, dataset, threshold, or display label changes.

A new metric requires a genuinely different estimand or equation. Different thresholds, subsets, grouping choices, or presentation names should normally reuse the same metric implementation and be expressed in evaluation/configuration/reporting as appropriate.

Do not copy a metric into an experiment-specific module to change one constant. Parameterize the real variation at the correct boundary while keeping the equation owned here.

Thresholded metrics should not erase their underlying continuous measurements when those measurements remain scientifically informative.

---

## 18. Evaluation and Statistical Inference

`evaluation/` owns aggregation, comparison, resampling, and population/cohort-level scientific inference.

Responsibilities may include:

```text
per-protein aggregation
cluster/family aggregation
confidence intervals
bootstrapping / resampling
model comparison
cohort comparison
calibration assessment
reliability analysis
stratified error analysis
```

Evaluation should make explicit:

- observational unit;
- resampling unit;
- pairing structure;
- aggregation rule;
- treatment of missing or invalid observations;
- comparison population.

Do not treat residues or probes as independent proteins when the scientific estimand is protein-, family-, or cluster-level.

Do not compare methods on silently different populations unless population difference is itself the scientific question.

For aggregate results, preserve access to per-protein or per-example distributions when feasible.

Evaluation consumes metric values and primitive model outputs where necessary; it does not own model execution.

---

## 19. Reporting

`reporting/` owns presentation-oriented transformations.

Responsibilities may include:

- summary tables;
- figure-ready data;
- report-card views;
- human-readable scientific summaries;
- plotting data preparation.

Reporting may organize or visualize scientific results, but it must not redefine:

- metric equations;
- cohort membership;
- admission criteria;
- thresholds used in the underlying scientific analysis;
- uncertainty semantics;
- aggregation units.

Keep observation and interpretation distinguishable in generated reports.

---

## 20. Runners and CLI

`runners/` composes reusable capabilities into explicit experiment procedures.

A runner may coordinate:

- configuration loading;
- input/output paths;
- adapters;
- task execution;
- evaluation;
- artifact writing;
- concise execution status.

Runners should remain thin relative to the scientific owners they compose.

Prefer several clear runners over a universal pipeline with many mode flags.

`cli/` should remain thinner still:

```text
parse arguments
→ construct typed inputs and dependencies
→ invoke runner
→ report concise status
```

Reject malformed required inputs where they are consumed. Do not insert a separate preflight/readiness stage merely to prove that execution is allowed to begin.

Scientific algorithms do not belong in CLI modules.

---

## 21. Configuration Boundary

Configuration files are external representations, not domain APIs.

Preferred flow:

```text
YAML / JSON / CLI
        ↓
parse into typed scientific or runtime specification
        ↓
scientific implementation
```

Structural parsing errors should fail at this boundary. Do not turn configuration handling into a separate readiness workflow.

Distinguish:

### 21.1 Scientific definitions

Definitions that determine scientific meaning, for example:

- amino-acid ordering;
- mask semantics;
- metric equation;
- normalization semantics;
- mutation semantics.

These should normally remain with their scientific owner rather than becoming freely adjustable global configuration.

### 21.2 Experiment configuration

Values defining the scientific comparison, for example:

- dataset/cohort;
- structural conditions;
- model/checkpoint;
- thresholds;
- selected metrics;
- seed policy;
- perturbation magnitude;
- sampling budget.

### 21.3 Runtime configuration

Execution-only values, for example:

- device;
- batch size;
- worker count;
- cache location;
- output location;
- log level.

Runtime configuration should not intentionally change scientific meaning.

### 21.4 Local configuration

Machine-specific settings such as local paths or credentials should normally be ignored by version control.

Do not pass repository-wide untyped dictionaries deep into scientific modules.

Experiment-varying values should not be hidden as module constants, duplicated literals, filename conventions, or branches on experiment names. If changing a value is a normal scientific operation, make that variation explicit at the smallest appropriate interface.

Prefer extending an existing typed specification over creating a new configuration format for each experiment. Closely related experiments should normally differ by configuration values and selected reusable capabilities, not by separate Python implementations.

Do not parameterize for hypothetical future variation. Parameterize values that already vary, are expected to vary in the active research program, or materially determine the scientific interpretation.

---

## 22. Experiment Architecture

`experiments/` contains version-controlled scientific intent.

An experiment should be understandable in terms of:

```text
question
hypothesis or competing explanations
assumptions and plausible confounders
population / cohort
intervention or varying factor
controlled factors
measurement
aggregation / analysis plan
discriminating outcomes
```

Not every exploratory run needs a long protocol document. A small configuration plus a concise README or protocol note is often sufficient.

Experiment names should describe scientific purpose rather than chronology.

Prefer:

```text
structure_uncertainty
scoring_sensitivity
functional_state_response
calibration_analysis
mechanism_analysis
counterfactual_analysis
```

rather than:

```text
stage2
scale1b_v3
final_final
run_new2
```

Historical IDs may remain in metadata where useful for reproduction.

### 22.1 Controlled comparisons

When experiments compare methods or conditions, keep non-target factors aligned whenever possible:

- cohort membership;
- preprocessing;
- model/checkpoint;
- decoding/sampling settings;
- seed policy;
- evaluation subset;
- metric implementation;
- aggregation unit.

When they cannot be aligned, make the mismatch explicit rather than treating the result as a clean causal comparison.

### 22.2 Exploratory vs confirmatory work

Exploratory analyses may generate hypotheses and useful visualizations.

Confirmatory analyses should avoid silently tuning thresholds, filters, cohorts, or metrics on the same evidence later presented as independent confirmation.

Architecture should not force every exploratory analysis into formal release machinery.

---

## 23. Data, Runs, and Artifacts

Keep these concepts distinct:

```text
data        reusable scientific inputs
experiments version-controlled scientific intent
runs        concrete execution instances
artifacts   durable outputs worth preserving or sharing
```

### 23.1 `data/`

`data/raw/` contains source-native inputs such as PDB/mmCIF, AlphaFold DB structures, UniProt metadata, and authoritative mappings.

`data/interim/` contains reusable normalized or intermediate scientific representations.

`data/processed/` contains processed datasets or tables that are stable enough to serve as inputs to later analyses.

Raw source data should not be casually rewritten.

Persistent structures belong to reusable data, not to experiment chronology. A `structure_id` should normally resolve to one canonical repository-relative file, for example:

```text
data/processed/structures/<scientific-namespace>/<structure-id>.cif
```

Experiment tables record structure identity, canonical path, parent structure, source/condition, and perturbation parameters. They do not copy the same structure into `pilot_v2/structures`, `pilot_v3/structures`, and later variants. Historical experiment-local structures may remain when moving them would damage reproduction, but active constructors must use the canonical store. A read-only external consumer may receive a symlink; a consumer that mutates input receives a temporary copy.

### 23.2 `runs/`

`runs/` contains concrete execution state. During execution it may include restart state and temporary pieces, but the retained form should be compact:

```text
runs/<experiment>/<run-id>/
├── config.*
├── logs/
├── results.parquet
├── summary.json
├── figures/
└── selected_checkpoint.pt
```

Runs are normally mutable and ignored by Git.

Run IDs may use timestamps or opaque identifiers because they identify executions, not scientific concepts.

Do not treat `runs/` as a permanent external-tool filesystem or an input archive. Once a consolidated result exists, remove regenerable shards, chunks, temporary manifests, copied source structures, screening/build/repair directories, subprocess work products, and superseded checkpoints. Preserve information-rich scientific measurements when they support later diagnosis; directory trees whose only purpose was execution do not qualify.

External tools run in a project-local temporary workspace:

```text
canonical structure or generated sequence
    → project-local temporary input/work directory
    → external program
    → compact result records
    → temporary workspace removed
```

Temporary copies must not become persistent run artifacts. If an execution can resume, intermediate checkpoints may exist while it is active. After the predefined selection rule runs, retain the selected/best checkpoint, training curve, and validation metrics; default retention is `best_only` or `keep_last: 1` where no best-model selection exists.

Shard and chunk boundaries are execution details. Consolidation must validate expected semantic keys or row counts, after which the redundant partitions are deleted. Do not add a new cleanup script for each workflow; cleanup belongs to the existing runner/CLI that creates the temporary state or to the existing run-archive capability.

### 23.3 `artifacts/`

`artifacts/` contains outputs that are scientifically worth preserving beyond one run, such as:

- selected benchmark data;
- reusable checkpoints;
- compact published tables;
- durable model outputs;
- release-ready benchmark material when an actual release exists.

Do not promote every generated file into `artifacts/`.

Do not require every artifact to have a manifest or content hash.

For ordinary research outputs, explicit configuration, semantic identity, counts, and code revision are usually more useful than cryptographic provenance.

---

## 24. Reproducibility and Provenance

Record what can materially affect scientific interpretation.

Relevant provenance may include:

- source dataset identity;
- cohort or selection policy;
- structural condition;
- model/checkpoint;
- preprocessing;
- scientifically meaningful parameters;
- seed or seed policy;
- software/code revision;
- important uncommitted changes;
- external data source version when relevant.

Do not record every incidental runtime detail simply because it is available.

Machine-specific absolute paths must not serve as scientific identity.

Prefer repository-relative logical paths or stable external identifiers.

### 24.1 Hashing

SHA256, checksum, and content-hash validation are **not part of the default research workflow**. Do not add, calculate, compare, or repeatedly verify hashes merely to strengthen provenance, detect ordinary file changes, validate regenerable outputs, or make a task appear more rigorous.

Prefer semantic scientific identity instead: dataset/cohort identity, model and checkpoint name/version, experiment parameters, seed, code revision, row counts or key scientific summaries when useful, and direct inspection of the outputs that matter to the scientific question.

Do not introduce hash fields, checksum sidecars, hash ledgers, hash-based cache identity, manifest hashes, or hash-validation steps for ordinary Parquet/CSV/JSON files, figures, summaries, source files, experiment intermediates, predictions, or derived tables.

A cryptographic hash is an exception, not a default. Use one only when an explicit external interface already requires content-addressed identity or when the task specifically requires verifying an otherwise unidentifiable external binary/checkpoint. Do not generalize that exception into repository-wide hash validation.

Never build hash chains such as `input hash → output hash → manifest hash → release hash`, and do not rerun SHA256 validation as a routine completion check.

---

## 25. Randomness

Scientific randomness should be intentional and interpretable.

Use explicit root seeds or documented seed policies when stochastic variation can affect the conclusion.

Do not derive scientific randomness from:

- wall-clock time;
- process ID;
- filesystem iteration order;
- worker scheduling order;
- unstable language hashes.

Parallelization should not silently change the statistical experiment.

When exact bitwise determinism is costly and scientifically unnecessary, characterize stochastic variability rather than introducing heavy machinery solely to reproduce identical bytes.

---

## 26. Caching and Reuse

Incorrect cache reuse is a scientific correctness error.

Cache identity should include inputs capable of changing the scientific result, such as:

- protein/input identity;
- structure or condition identity;
- model/checkpoint;
- preprocessing;
- scientific configuration;
- relevant semantic version identifiers.

Normally exclude unrelated runtime metadata such as:

- log level;
- report path;
- timestamp;
- worker count;
- plotting settings.

Prefer transparent, inspectable caches that can be invalidated easily.

A stored primitive model output should be reused for new downstream analyses when its scientific inputs and semantics match the new question.

---

## 27. Direct Scientific Evidence

Research work should move toward informative execution, not toward proving readiness to execute.

The default loop is:

```text
scientific question
→ smallest meaningful computation or experiment
→ inspect real output
→ investigate an observed ambiguity, discrepancy, or failure
→ update the scientific interpretation
```

Do not create a separate preflight, readiness check, pre-run audit, dry-run validation, or validation-gate phase for ordinary research work. Do not scan the repository, inputs, environment, or generated artifacts merely to establish that work is "ready" when the real computation can provide stronger evidence at acceptable cost.

Required inputs and invariants should be checked naturally by the component that consumes them. If an input is malformed or an invariant is violated, fail clearly at that boundary and fix the concrete problem. Do not build a second preliminary workflow that predicts whether the same boundary might fail.

Use a targeted preliminary check only when direct execution is materially irreversible, exceptionally expensive, security-sensitive, or capable of corrupting important external data. Even then, check only the specific risk that justifies the check.

Tests are tools for resolving concrete uncertainty, not mandatory stages. Prefer direct scientific evidence such as residue/mask inspection, cohort comparison, per-protein distributions, leakage analysis, or seed sensitivity when those quantities determine the conclusion. Do not escalate mechanically into broader suites after the relevant uncertainty is resolved.

---

## 28. Failure Semantics

Architecture should preserve three broad failure classes.

### Software / invariant failure

Examples:

- impossible internal state;
- malformed scientific representation;
- inconsistent mapping;
- violated required shape or index assumption.

Fail clearly.

### Scientific / data outcome

Examples:

- candidate does not satisfy a scientific criterion;
- no valid residue mapping exists;
- requested comparison population is insufficient;
- hypothesis is unsupported.

Represent the outcome and reason without disguising it as a software error.

### Infrastructure / operational failure

Examples:

- GPU failure;
- network timeout;
- missing remote asset;
- subprocess crash;
- storage failure.

Report operationally and retry only when appropriate.

Never reinterpret operational failure as scientific exclusion.

---

## 29. Performance

Correctness and scientific equivalence precede speed.

Optimize measured or clearly dominant bottlenecks.

Prefer standard techniques such as:

- batching;
- vectorization;
- streaming;
- memory-aware tensor operations;
- transparent caching;
- validated mixed precision.

Performance changes must preserve:

- masks;
- residue mappings;
- normalization scope;
- sampling distribution;
- random streams where required;
- aggregation unit.

Do not silently change per-protein semantics into per-batch semantics.

---

## 30. Extension Decision Process

When adding behavior, first inspect the relevant existing implementation and ask whether the requested difference is new science or only a new setting of existing science. Keep this inspection targeted to likely owners and callers; it is normal code understanding, not a preflight workflow.

```text
Does relevant handling logic already exist?
    |
    +-- YES → reuse it directly or extend that owner
    |
    +-- NO / INCOMPLETE
         |
         Does an existing function/module/task/runner own the same scientific operation?
         |
         +-- YES → extend its coherent interface
         |          and express the new variation through arguments/configuration
         |
         +-- NO
              |
              Is this only a different experiment composition?
              |
              +-- YES → reuse existing runner/task/CLI + configuration
              |
              +-- NO
                   |
                   Is the behavior model/source specific?
                   |
                   +-- YES → extend or add the adapter near the representation it produces
                   |
                   +-- NO
                        |
                        Is there a genuinely new reusable scientific responsibility?
                        |
                        +-- YES → create one cohesive owner
                        |
                        +-- NO → reconsider the requested abstraction
```

A new `.py` file or directory is the last option in this process, not the default unit of progress. A new task name, experiment name, ablation, figure, checkpoint, cohort, or analysis request does **not** justify a new source file or folder.

Before creating either one, be able to state why the existing owner cannot express the behavior cleanly. If the answer is only “to keep this task separate,” do not create it.

A new top-level package should normally require:

1. a coherent scientific responsibility;
2. multiple meaningful operations or dependencies;
3. no suitable existing owner;
4. a clearer dependency structure.

The first use of a concept does not automatically justify a framework, package, task folder, or standalone script. Prefer a small extension of existing code over either premature framework construction or repeated task-specific implementations.

---

## 31. Historical Code and Migration

Historical stage-oriented code should be treated as evidence about earlier scientific behavior, not as the preferred template for new implementation.

When active work requires migration:

```text
understand current scientific behavior
→ identify its real responsibility
→ identify invariants and edge cases
→ move or consolidate into the active owner
→ migrate current callers
→ run focused scientific comparisons
→ remove obsolete duplication when safe
```

Do not perform a repository-wide rewrite solely for aesthetic consistency.

Do not add new wrappers, hashes, or checking machinery merely to keep two duplicate implementations synchronized.

Prefer one active implementation and a clearly labeled reproduction-only historical path when historical reproduction remains necessary.

---

## 32. StructCal Boundary

`StructCal` is the repository-wide public benchmark identity (`structcal` in paths and Python identifiers).

Dual-UQ remains the repository/package identity unless explicitly changed elsewhere.

StructCal studies invariance–sensitivity calibration under structural conditions.

The current scientific distinction is:

```text
Track I — structural invariance
    clean PDB/AFDB representation-variation pairs
    expected behavior: unnecessary response should remain small

Track II — functional sensitivity
    Apo/Holo PRIMARY pairs and functional-state PRIMARY pairs
    expected behavior: biologically meaningful structural change may require response

Track III — invariance–sensitivity calibration
    joint evaluation of Track-I and Track-II evidence
    this is not a duplicated third cohort
```

Arm identifies the source/type of structural variation. Track identifies the scientific evaluation objective. They are different concepts and must not be inferred from one another.

### 32.1 Global comparison population

StructCal v1 uses one global 30%-sequence-identity clustering and one shared split across the complete formal protein universe. This is part of the scientific comparison design, not an execution convenience.

`splits.parquet` is the scientific source for cluster and split assignment. Arm-specific or historical method-development splits may be retained as provenance but must not silently replace the shared comparison split.

Do not create arm-specific splits that change the comparison population unless the scientific question explicitly requires a different population.

### 32.2 Model-independent benchmark core

The public model-independent benchmark core contains the scientific objects needed to define the comparison:

```text
proteins
structures
condition pairs
residue mappings
benchmark instances
cluster/split assignments
```

Keep model responses, model-specific eligibility, model scores, and execution state outside this core. Generic scientific eligibility and model capability are different questions; combine them only when evaluating a concrete model.

Structural annotations that are useful for stratification or mechanism analysis may remain a separate scientific layer rather than being forced into the minimal benchmark core.

### 32.3 Historical cohorts

Historical PDB/AFDB, Apo/Holo, and other construction paths may remain as reproduction sources.

Projection into the active StructCal representation should preserve scientifically meaningful decisions already established by those sources—such as admission, PRIMARY selection, condition orientation, residue mapping, descriptor calculation, and controlled-intervention construction—unless the research task explicitly intends to revisit one of them.

Do not rerun upstream discovery or model work merely to make the directory structure cleaner.

Do not require release machinery during ordinary StructCal development. Stronger artifact-identity checks are appropriate only for a real external/publication boundary or another concrete integrity need.

---

## 33. ReSC Boundary

`ReSC` is the repository-wide name for Reliable Structural Conditioning (`resc` in paths and Python identifiers).

ReSC method development should reuse the same model adapters, task semantics, metrics, and evaluation paths used for comparable baseline methods wherever the scientific operation is the same.

Do not create a private ReSC-only implementation of a metric, data construction rule, or evaluation path merely to simplify one experiment.

Method-specific code belongs in a ReSC-specific module only when the behavior is genuinely part of the ReSC method itself, for example a learned transformation, conditioning rule, or method-specific parameterization.

The architectural goal is:

```text
shared scientific population
+ shared task semantics
+ shared metrics/evaluation
+ ordinary model adapter
+ ReSC-specific method component
→ comparable evidence about ReSC
```

This separation makes it easier to determine whether an observed improvement comes from the method rather than from a different evaluation path, cohort, preprocessing rule, or metric implementation.

---

## 34. Architecture for Scientific Diagnosis

The repository should make surprising results diagnosable without rewriting the pipeline.

For any major comparison, it should be possible to inspect or recover enough intermediate information to ask:

1. Was the quantity measured correctly?
2. Were the compared proteins/conditions actually comparable?
3. Were residue mapping and masks correct?
4. Did preprocessing or filtering differ?
5. Did aggregation or missingness semantics differ?
6. Could stochasticity explain the effect?
7. Does the result remain after reasonable stratification?
8. Is the scientific hypothesis itself unsupported?

Architecture should preserve the data needed for these questions when the storage cost is reasonable.

Do not optimize the pipeline so aggressively around a single headline output that scientific diagnosis becomes impossible.

---

## 35. Architecture Success Criteria

The architecture is working when:

- a new scientific question can often be answered by composing existing scientific capabilities;
- a new model does not require rewriting dataset construction or metrics;
- a new metric usually does not require rerunning model inference when primitive outputs already exist;
- residue/index mappings are explicit and centralized;
- cohort and comparison populations are inspectable;
- biological redundancy and split leakage can be checked directly;
- scientific observation units and aggregation units are explicit;
- uncertainty quantities remain distinguishable;
- external model/source details stay behind adapters;
- experiment chronology does not leak into reusable implementation names;
- one scientific concept normally has one active implementation;
- a new experiment usually reuses existing Python modules and differs through explicit inputs/configuration rather than adding another task-specific `.py` file;
- scientific choices that legitimately vary are explicit rather than buried in literals, paths, or experiment-name branches;
- stable scientific definitions remain centralized instead of being duplicated or made arbitrarily configurable;
- reporting cannot silently redefine scientific results;
- surprising outputs can be traced through mapping, filtering, scoring, aggregation, and stochasticity;
- ordinary research work begins with informative computation rather than preflight, audit, or readiness gates;
- ordinary research work does not require unnecessary hashes, manifests, release gates, or framework code;
- stronger reproducibility machinery is introduced only where it changes the trustworthiness of an actual scientific result or external artifact;
- adding an experiment mostly means selecting a cohort, configuring a controlled comparison, composing existing operations, and interpreting evidence.

---

## 36. Final Architectural Principle

Dual-UQ should encode:

```text
scientific questions
+
explicit scientific objects
+
one owner per reusable concept
+
continuous extension of existing code
+
explicit parameterization of real variation
+
clear source/model boundaries
+
controlled experiment composition
+
information-rich measurements
+
explicit metrics and aggregation
+
proportional reproducibility
```

not:

```text
roadmap chronology
+
copied stage scripts
+
new Python files for every experiment
+
near-duplicate tasks or metrics
+
hard-coded experiment choices and paths
+
parallel scientific implementations
+
model assumptions spread across the repository
+
global configuration dictionaries
+
boolean-mode pseudo-frameworks
+
unnecessary schema/hash/manifest machinery
+
report-driven scientific definitions
```

The architecture is successful when it helps the repository answer protein-design research questions with fewer hidden assumptions, cleaner comparisons, and evidence that is easier to challenge and reproduce.
