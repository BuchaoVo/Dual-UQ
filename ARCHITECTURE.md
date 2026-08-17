# ARCHITECTURE.md

## 1. Purpose

Dual-UQ is a long-lived scientific software repository for computational protein design and uncertainty-aware evaluation.

The repository supports capabilities including:

* protein sequence and structure processing;
* dataset construction and scientific release;
* PDB / AlphaFold DB integration;
* structure-conditioned protein design;
* sequence scoring;
* model inference and comparison;
* structure uncertainty;
* evaluator uncertainty;
* joint uncertainty;
* calibration and reliability analysis;
* mechanism analysis;
* counterfactual analysis.

The architecture follows one principle:

> Stable scientific concepts live in reusable modules; experiment specifications configure and compose those capabilities; execution state and frozen scientific artifacts have explicit lifecycles.

This document defines:

```text
module ownership
dependency direction
cross-package contracts
configuration boundaries
I/O boundaries
artifact lifecycle
extension rules
migration strategy
```

It does not define Agent execution behavior.

Agent behavior belongs in `AGENTS.md`.

This is a target architecture, not a requirement for a Big-Bang rewrite.

---

## 2. Architectural Principles

The repository is organized primarily by stable scientific capability rather than experiment chronology.

Reusable code should encode concepts such as:

```text
admission
mapping
acquisition
redundancy
candidate selection
sequence design
sequence scoring
uncertainty
metrics
evaluation
provenance
release validation
```

rather than historical identifiers such as:

```text
P2
Stage0
Scale1A2
Scale1A3
Scale1B
```

Historical identifiers remain valid as:

* experiment identities;
* release identities;
* provenance metadata;
* compatibility aliases;
* protocol history.

They should not determine reusable package architecture.

The goal is not maximum abstraction.

The goal is the smallest stable architecture that preserves scientific meaning while allowing future experiments to reuse existing capabilities.

---

## 3. Conceptual Composition vs Code Dependencies

Do not confuse scientific workflow composition with code dependency direction.

### 3.1 Conceptual Composition

Scientific execution generally looks like:

```text
experiment specification
        ↓
workflow
        ↓
scientific capabilities + adapters + policies
        ↓
validated scientific artifacts
```

This diagram describes how scientific work is composed.

It is not a Python import graph.

### 3.2 Code Dependency Principle

Dependencies should point toward more stable concepts.

High-level orchestration may depend on lower-level scientific capabilities.

Reusable scientific capabilities must not depend on:

* workflows;
* CLI;
* reporting;
* experiment directories;
* historical experiment identities.

Project-level scientific contracts must not depend on concrete source/model adapters.

Adapters translate external systems into project-level contracts.

---

## 4. Target Repository Layout

```text
repository/
│
├── AGENTS.md
├── ARCHITECTURE.md
├── README.md
├── pyproject.toml
├── LICENSE
├── CITATION.cff
│
├── configs/
│   ├── datasets/
│   ├── models/
│   ├── experiments/
│   ├── runtime/
│   └── local/
│
├── schemas/
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── manifests/
│
├── experiments/
├── runs/
│
├── artifacts/
│   ├── releases/
│   └── checkpoints/
│
├── src/
│   └── dual_uq/
│       ├── core/
│       ├── structure/
│       ├── dataset/
│       ├── design/
│       ├── models/
│       ├── inference/
│       ├── uncertainty/
│       ├── metrics/
│       ├── evaluation/
│       ├── workflows/
│       ├── reporting/
│       └── cli/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── regression/
│   ├── contracts/
│   └── fixtures/
│
├── docs/
│   ├── architecture/
│   ├── scientific/
│   ├── protocols/
│   ├── decisions/
│   └── archive/
│
└── third_party/
```

This is a target state.

Do not create empty directories merely to conform to the diagram.

Existing code should migrate incrementally as active work justifies it.

---

## 5. Dependency Rules

A practical default dependency model is:

| Package       | May depend on                                                        |
| ------------- | -------------------------------------------------------------------- |
| `core`        | foundational libraries                                               |
| `structure`   | `core`                                                               |
| `dataset`     | `core`, `structure`                                                  |
| `design`      | `core`, `structure`                                                  |
| `models`      | `core`, `structure`, stable design/contracts where required          |
| `inference`   | `core`, `structure`, `design`, `models`, stable prediction contracts |
| `uncertainty` | `core`, stable prediction/scientific contracts                       |
| `metrics`     | `core`, stable scientific contracts                                  |
| `evaluation`  | `core`, `metrics`, `uncertainty`, stable data contracts              |
| `reporting`   | `metrics`, `uncertainty`, `evaluation`                               |
| `workflows`   | required lower-level capabilities                                    |
| `cli`         | `workflows`, configuration/application boundaries                    |

This table is guidance, not blanket permission for arbitrary imports.

Before introducing a dependency between top-level packages, verify:

1. the dependency follows the intended direction;
2. the depended-on package owns the required concept;
3. the dependency does not create a cycle;
4. importing a smaller stable contract would not create a cleaner boundary.

A new top-level import is an architectural decision.

Avoid peer-package dependency drift such as:

```text
dataset ↔ evaluation
design ↔ models
uncertainty ↔ evaluation
```

When two peer packages appear to require one another, first determine whether a stable lower-level contract has unclear ownership.

---

## 6. `core/`

`core/` contains genuinely cross-cutting primitives only when no scientific package naturally owns them.

A concept belongs in `core/` only when:

1. no scientific/domain package is its natural owner;
2. at least two independent top-level capabilities require it;
3. its semantics are stable independently of those callers;
4. placing it in `core/` improves dependency direction.

Typical candidates include:

```text
identity
provenance primitives
stable hashing
shared invariant/error types
artifact identity
```

Possible modules:

```text
core/
├── identity.py
├── provenance.py
├── hashing.py
└── errors.py
```

Do not create dumping-ground modules such as:

```text
core/utils.py
core/helpers.py
core/common.py
```

Scientific logic belongs to its scientific owner.

---

## 7. I/O Boundary

Reusable scientific computation should normally operate on:

* domain/scientific objects;
* typed policies/specifications;
* arrays/tensors with explicit contracts;
* iterables;
* stable data contracts;

rather than repository directory layouts.

Prefer:

```python
value = sequence_recovery(
    prediction,
    native_sequence,
    mask,
)
```

over:

```python
value = sequence_recovery(
    experiment_directory,
    prediction_csv,
    output_directory,
)
```

Filesystem layout, serialization, remote retrieval, and artifact placement belong at boundary/orchestration layers unless persistence is itself the responsibility of the capability.

Scientific computation should remain independently testable from repository paths where practical.

Source adapters may naturally perform remote/file I/O.

Artifact/release modules may naturally own serialization or persistence.

The I/O boundary rule prevents arbitrary I/O from leaking into unrelated scientific logic.

---

## 8. Configuration Boundary

Configuration files are external representations, not domain APIs.

Preferred flow:

```text
YAML / JSON / CLI
        ↓
parse + validate
        ↓
typed policy/specification
        ↓
scientific implementation
```

Prefer:

```python
policy = AdmissionPolicy(
    max_mismatch=3,
    min_identity=0.99,
)

decision = evaluate_admission(
    record,
    policy,
)
```

over:

```python
decision = evaluate_admission(
    record,
    global_config,
)
```

Do not pass repository-wide unstructured dictionaries deep into reusable scientific modules.

Configuration has four conceptual categories.

### Scientific Definitions

Define scientific meaning.

Examples:

* amino-acid alphabet;
* mask semantics;
* metric equation;
* normalization semantics;
* mutation semantics.

### Experiment Configuration

Defines a scientific experiment.

Examples:

* dataset release;
* cohort;
* checkpoint;
* thresholds;
* metrics;
* seed.

### Runtime Configuration

Controls execution.

Examples:

* device;
* batch size;
* workers;
* cache location;
* output location;
* log level.

Runtime configuration should not intentionally change scientific meaning.

### Local Configuration

Machine-specific settings.

Usually gitignored.

Do not move canonical scientific definitions into configuration files merely to make them adjustable.

---

## 9. Behavioral Variation

Do not model independent scientific behaviors through accumulating boolean flags or generic mode strings.

Avoid:

```python
evaluate(
    strict=True,
    legacy=False,
    scale2=True,
    use_afdb=False,
)
```

when those flags encode coherent scientific policies or fundamentally distinct capabilities.

Prefer typed policy/specification objects for coherent scientific variation.

For example:

```python
decision = evaluate_admission(
    record,
    policy=admission_policy,
)
```

Prefer separate cohesive capabilities when behaviors are fundamentally different.

A function whose behavior is dominated by combinations of boolean flags is a signal that its abstraction boundary should be reconsidered.

---

## 10. `structure/`

`structure/` owns protein structural representation and residue-coordinate semantics.

Potential responsibilities include:

* PDB/mmCIF parsing;
* chain representation;
* residue identity;
* missing-residue representation;
* sequence/structure mapping;
* coordinate validation;
* geometry;
* structure confidence.

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

### Residue Mapping Contract

Residue identity is an explicit mapping problem.

Relevant spaces may include:

```text
author/PDB residue identifier
canonical sequence index
structure-array index
model/tensor index
alignment index
```

Conversions between these spaces must be centralized.

Dataset, model, metric, uncertainty, and evaluation code must not independently reconstruct residue-mapping semantics.

A stable `ResidueMapping` contract is appropriate when multiple capabilities share these transformations.

Round-trip and boundary invariants should be tested.

---

## 11. `dataset/`

`dataset/` owns reusable scientific dataset-construction capabilities.

Expected conceptual ownership:

```text
dataset/
├── records.py
├── acquisition.py
├── validation.py
├── admission.py
├── redundancy.py
├── selection.py
├── manifests.py
├── audits.py
└── releases.py
```

Source adapters may initially live under:

```text
dataset/sources/
├── rcsb.py
├── afdb.py
└── uniprot.py
```

when their current primary responsibility is dataset construction and source-specific complexity warrants separation.

Historical experiment stages should normally be represented through:

```text
policy/configuration
release specification
provenance
workflow orchestration
```

rather than new stage-named reusable modules.

Avoid reusable modules such as:

```text
scale1a1.py
scale1a2.py
scale1a3.py
stage2.py
```

when their actual responsibilities are admission, redundancy, selection, acquisition, etc.

---

## 11.1 Source Adapter Ownership

Source-adapter ownership follows the stable project-level contract produced by the adapter, not merely the first workflow that uses it.

For example, an adapter whose stable output is a project-level structure contract may ultimately belong with structure-facing infrastructure even if its first consumer is dataset construction.

Do not move an adapter solely because another workflow starts using it.

Reconsider ownership only when the stable output contract and dependency direction justify the change.

Do not introduce a generic top-level `sources/` package until multiple scientific domains require a genuinely shared source abstraction.

---

## 11.2 Dataset Records

Dataset-level records may encode stable scientific boundaries.

Examples may include:

```text
ProteinRecord
DatasetRecord
AdmissionDecision
RejectionReason
ProvenanceRecord
```

Do not introduce a dedicated type for every scalar.

A type should normally:

* enforce an invariant;
* prevent category errors;
* distinguish incompatible index spaces;
* define a stable boundary;
* group coherent scientific state.

---

## 11.3 Acquisition

Acquisition owns:

```text
source binding
retrieval coordination
acquisition provenance
cache semantics
source-level operational status
```

Source retrieval failure is not scientific rejection.

Source-specific API and filesystem conventions belong in adapters rather than downstream admission/evaluation logic.

---

## 11.4 Validation

Validation owns scientific/data validity after parsing or normalization.

Parsing answers:

```text
Can this artifact be interpreted?
```

Validation answers:

```text
Does the interpreted scientific object satisfy the required contract?
```

These are different responsibilities.

---

## 11.5 Admission

Admission owns reusable formal admission semantics.

Preferred conceptual API:

```python
decision = evaluate_admission(
    record,
    policy,
)
```

Results should preserve:

```text
candidate/protein identity
terminal decision
structured reason
supporting evidence
policy/version
```

Policies may vary while canonical evaluation machinery remains reusable when scientific semantics permit.

Do not create experiment-specific classifiers that duplicate an existing canonical evaluator.

---

## 11.6 Redundancy

Redundancy owns:

```text
cluster binding
redundancy assessment
cluster census
representative precedence/ranking
non-redundant capacity
```

Do not duplicate clustering logic because a later experiment uses the same clustering convention.

Thresholds or conventions that vary belong in explicit policy/release specifications.

---

## 11.7 Selection

Selection owns reusable prospective selection mechanisms such as:

```text
cluster-first selection
stratified selection
candidate prioritization
reserve selection
cohort selection
```

Selection policy must remain explicit.

Do not encode scientific selection rules through stage names or hidden dataframe filters.

---

## 11.8 Audits

Audits own reusable census and integrity calculations.

Examples:

```text
sampling-frame completeness
redundancy diagnostics
attrition
provenance completeness
data-quality census
```

Frozen historical expectations belong in release/contract specifications, not generic audit implementations.

---

## 11.9 Releases

Release utilities own generic:

```text
manifest validation
membership verification
hash verification
release identity
compatibility checks
promotion validation
```

Generic release code must not hard-code one historical Scale/Stage release.

Release-specific scientific expectations belong in experiment/release specifications or contract fixtures.

---

## 12. `design/`

`design/` owns model-independent protein-design concepts.

Potential responsibilities:

* candidate sequence representation;
* mutation/probe definitions;
* allowed/fixed positions;
* Hamming-distance constraints;
* sampling policy;
* design constraints.

Potential modules:

```text
design/
├── candidates.py
├── probes.py
├── constraints.py
└── sampling.py
```

Model-native tensor manipulation, checkpoint semantics, or vendor-specific file formats do not belong here.

---

## 13. `models/`

`models/` owns model-facing adapters and model-specific behavior.

Examples may include:

* ProteinMPNN;
* LigandMPNN;
* MoMPNN;
* DynamicMPNN;
* ESM;
* RFdiffusion.

Do not create a universal model framework prematurely.

Treat model capabilities independently.

For example:

```text
SequenceScorer
SequenceDesigner
```

are different capabilities even when implemented by the same external model.

Only introduce Protocols/interfaces when concrete substitution value exists.

External-model assumptions should be localized:

* checkpoint loading/layout;
* native vocabulary;
* native input/output formats;
* tensor conversion;
* device-specific invocation;
* vendor directory structure.

### 13.1 ProteinMPNN Scoring Boundary

`dual_uq.models.proteinmpnn` is the canonical concrete ProteinMPNN scoring
adapter. It owns the frozen ProteinMPNN alphabet, checkpoint and implementation
identity, explicit decoding-realization interpretation, native structure and
candidate tensors, model-native batching, model loading, target-amino-acid
log-probability extraction, and the precise ProteinMPNN score aggregation.

Dataset services may retain historical PDB/AFDB projection compatibility,
request adaptation, shard handling, and persistence, but they delegate model
behavior to this adapter. The adapter treats the condition label as opaque and
does not own paired-intervention policy or common-mask construction.

`dual_uq.models.scoring.ScoreRecord` is the normalized successful-score
envelope. It preserves condition and structure identity, explicit WT or PROBE
variant semantics, repeat/seed/realization identity, scorer implementation and
checkpoint identity, score-contract identity, and the contract-specific fields
`score_sum_logp_mask`, `score_mean_logp_mask`, and
`scored_residue_count`. WT measurements carry no mutation sentinels.

`dual_uq.models.scoring.ScoreRequest` is one logical, collection-level,
model-independent scientific scoring request. It binds a concrete
`StructureCondition`, the external comparable residue domain, one WT plus its
fixed candidate collection, repeat/seed/realization identity, and the required
score contract. Scorer implementation/checkpoint identity remains an external
`ScorerBinding`; dataset grouping, filesystem, worker, device, and persistence
metadata are excluded.

`dual_uq.models.scoring.SequenceScorer` consumes one `ScoreRequest` and returns
a normalized `tuple[ScoreRecord, ...]`: one WT record followed by one record
per fixed candidate. This logical cardinality is independent of model-native
batching. `ProteinMPNNScorer` implements the capability while retaining
ProteinMPNN-specific sequence projection, decoding-order interpretation,
tensors, batching, and invocation behind the adapter.
Resolved coordinate projections must carry and match the request's immutable
structure SHA. A ProteinMPNN scorer reports implementation/checkpoint identity
from its concrete verified adapter; it must not attach authorized constants to
an arbitrary injected runtime.

Historical Scale-1B-v2 plan interpretation remains workflow/compatibility
ownership in `dual_uq.workflows.final_confirmatory_protocol`. Its adapter maps
frozen PDB/AFDB schema and provenance to the generic request/scorer contracts;
the reusable model contracts contain no Scale, Stage, PDB/AFDB pairing, or
frozen-Parquet assumptions.

Formal scoring retains its historical planning, shard, resume, atomic-write,
consolidation, and persisted-schema contracts. At the worker seam, a
compatibility adapter turns the historical runtime request into
`ScoreRequest` objects, `execute_score_request` invokes a `SequenceScorer` and
validates normalized `ScoreRecord` results, and a persistence adapter maps
those records back to the authoritative shard representation. Operational
shard names, output paths, device data, and resume state remain outside the
scientific request/result identity. Model-native batching remains scorer-owned;
this integration does not introduce a scheduler or execution framework.
Legacy formal-request v1 artifacts remain immutable but cannot be promoted to
`ScoreRequest` because they lack canonical full-sequence mutation identity;
missing work is rebuilt as the explicitly enriched v2 request from the frozen
plan rather than by inferring scientific identity from projected sequences.

Do not scatter vendor imports throughout dataset, uncertainty, metrics, evaluation, or reporting code.

---

## 14. Third-Party Boundaries

Vendored implementation directories such as:

```text
third_party/ProteinMPNN/
```

are external implementation details.

Dedicated adapters should be the primary code that knows:

* third-party internal paths;
* native JSONL formats;
* native output layout;
* checkpoint organization;
* model-specific alphabet/tensor conventions.

Other packages should consume project-level contracts.

---

## 15. `inference/`

`inference/` owns model execution mechanics.

Potential responsibilities:

* batching;
* device placement;
* execution coordination;
* inference mode;
* prediction materialization.

Potential modules:

```text
inference/
├── formal.py
└── materialization.py
```

`dual_uq.inference.formal` owns model-independent formal request inventory,
deterministic orchestration identities, artifact-binding decisions, derived
resume/reuse state, fresh-work selection, and scorer-dispatch composition.
The orchestration identity is deterministically derived from the scientific
fingerprint, but remains a distinct operational role; artifact paths never
enter the scientific request identity. Inventory tables are immutable derived
snapshots, not mutable status databases.

`dual_uq.inference.materialization` exclusively owns validation, discovery,
atomic persistence, and immutable-conflict handling for normalized formal
score shards. It delegates record-level correctness to the shared
`ScoreRequest`/`ScoreRecord` dispatch validator rather than introducing a
second resume contract.

Workflow adapters may prove exact historical compatibility under an
authoritative reuse policy. Authorization and per-request compatibility remain
separate predicates. Accepted historical results are copied through the same
canonical materializer with source execution/artifact, reuse-contract, and
compatibility-validation provenance; source artifacts remain immutable.

Model architecture and checkpoint-specific behavior remain owned by model adapters.

Inference may produce, materialize, or serialize predictions.

Inference does not automatically own the shared scientific prediction contract.

Inference should not own scientific metrics or evaluation policy.

---

## 16. Prediction Contracts

Prediction contracts are model-independent scientific data contracts, not concrete model-adapter or execution-engine implementation details.

Do not define a general `PredictionRecord` inside a concrete adapter such as:

```text
models/proteinmpnn.py
```

Do not assume that `inference/` is automatically the canonical owner merely because inference produces predictions.

Before introducing a final normalized prediction abstraction:

1. audit existing model outputs;
2. audit persisted prediction formats;
3. audit downstream consumers;
4. identify stable common semantics;
5. determine the most appropriate model-independent owner.

A future normalized prediction contract may include:

```text
protein_id
candidate_id
input/backbone identity
residue_mapping
aa_alphabet
scores/log_probs
valid_mask
model_id
checkpoint_id
provenance
```

The canonical owner must remain model-independent.

Do not create a new top-level `prediction/` subsystem until concrete usage justifies it.

Downstream uncertainty, metrics, and evaluation should prefer normalized project-level prediction contracts over model-native files where practical.

Preserve information-rich predictions when storage cost is reasonable so changes in metrics do not require unnecessary inference reruns.

Successful scalar score measurements use `dual_uq.models.scoring.ScoreRecord`.
The record envelope is model-independent, but score meaning remains bound to
its explicit scorer and `score_contract_id`; model-specific scientific fields
are not collapsed into a generic `score` or `primary_score` value.

---

## 17. `uncertainty/`

`uncertainty/` owns uncertainty definitions, uncertainty estimators, and reusable uncertainty computations.

Potential capabilities include:

```text
structure uncertainty
evaluator uncertainty
joint uncertainty
```

Potential modules:

```text
uncertainty/
├── structure.py
├── evaluator.py
└── joint.py
```

Do not organize reusable uncertainty code by roadmap identifiers such as:

```text
p3.py
p4.py
p5.py
```

Distinct uncertainty concepts may share lower-level functions without being collapsed into one generic framework.

A fitted transformation that materially changes an uncertainty estimator belongs with the capability that owns that estimator.

Do not introduce a generic `uncertainty/calibration.py` merely because calibration-related analysis exists elsewhere.

---

## 18. `metrics/`

`metrics/` owns deterministic scientific metric definitions.

Examples include:

```text
sequence recovery
ranking
calibration metrics
stability
diversity
reliability statistics
```

Calibration-related responsibilities in `metrics/` are deterministic scientific quantities, such as appropriate calibration-error or scoring metrics.

Metrics do not own cohort-level inference, resampling, or reliability-study orchestration.

Metric APIs should consume scientific objects/arrays rather than experiment directory layouts.

Prefer:

```python
value = sequence_recovery(
    predicted_sequence,
    native_sequence,
    mask,
)
```

not:

```python
value = calculate_scale1_recovery(
    csv_path,
    output_dir,
)
```

Metric definitions must not live in reporting or experiment scripts.

---

## 19. `evaluation/`

`evaluation/` owns statistical aggregation, comparison, and population/cohort-level scientific interpretation.

Potential responsibilities include:

```text
protein-level aggregation
cluster/family aggregation
confidence intervals
bootstrapping/resampling
model comparison
cohort comparison
calibration assessment
reliability analysis
```

Calibration ownership in `evaluation/` refers to population/cohort-level analysis such as:

* reliability curves;
* aggregation;
* confidence intervals;
* resampling;
* model/cohort comparison.

Deterministic calibration metrics remain owned by `metrics/`.

Estimator-specific calibration transformations remain with the estimator capability that they modify.

Observational and resampling units must be explicit.

Evaluation consumes scientific results.

It should not own model execution.

Avoid pseudo-replication caused by treating residues or probes as independent proteins when the scientific unit is protein, family, or cluster.

---

## 20. `workflows/`

`workflows/` composes reusable capabilities into explicit scientific procedures.

Examples:

```text
build_dataset
acquire_and_validate_records
select_cohort
generate_candidates
score_sequences
estimate_uncertainty
evaluate_predictions
release_dataset
```

A workflow may coordinate:

* I/O;
* domain capabilities;
* adapters;
* policies;
* artifact writing.

Canonical formulas and scientific definitions remain in lower-level owners.

Avoid a universal:

```python
DualUQPipeline(mode=...)
```

with many mode flags.

Prefer several explicit, auditable workflows.

---

## 21. `reporting/`

`reporting/` owns presentation-oriented transformations.

Responsibilities may include:

* summary tables;
* figure-ready data;
* human-readable scientific summaries;
* scientific report rendering.

Reporting consumes scientific results.

It must not redefine:

* metrics;
* admission policy;
* selection policy;
* scientific thresholds;
* uncertainty semantics.

Task-specific Codex design/review/implementation reports are not architectural reporting artifacts.

---

## 22. `cli/`

CLI code should remain thin.

Its responsibilities are:

```text
parse arguments
validate application configuration
construct dependencies
invoke workflow
report concise status
```

Scientific algorithms do not belong in CLI modules.

CLI/file configuration should be converted into typed domain/application specifications before entering reusable scientific modules.

---

## 23. Scientific Artifact Lifecycle

Keep four concepts distinct:

```text
data
experiments
runs
artifacts
```

They are not interchangeable.

### 23.1 `data/`

Contains information that can serve as input to subsequent scientific computation.

Recommended structure:

```text
data/
├── raw/
├── interim/
├── processed/
└── manifests/
```

#### `raw/`

Source-native external inputs such as:

* PDB/mmCIF;
* AlphaFold DB;
* UniProt metadata;
* authoritative external mappings.

Raw inputs should be treated as immutable where practical.

#### `interim/`

Reusable normalized/intermediate scientific data.

Examples:

* normalized structures;
* resolved mappings;
* normalized source metadata.

#### `processed/`

Canonical processed datasets suitable as inputs to later scientific computation.

#### `manifests/`

Dataset identity, source binding, membership, and provenance records.

Do not store execution logs, figures, task reports, or worker responses under `data/`.

---

### 23.2 `experiments/`

Contains version-controlled scientific intent.

Experiment directories may contain:

```text
README/protocol description
configs/
protocols/
release_specs/
small reference manifests
```

Experiment directories should not become the primary location for:

* runtime logs;
* large predictions;
* worker responses;
* temporary execution output.

Experiment names should describe scientific purpose.

Examples:

```text
model_behavior_validation
design_baseline
structure_uncertainty
evaluator_uncertainty
joint_uncertainty
pareto_reliability
mechanism_analysis
counterfactual_analysis
```

Historical roadmap IDs may remain in metadata.

Dataset construction and release experiments use the semantic namespace
`experiments/dataset/`.  Its subdirectories separate construction
(`construction/sampling_frame`, `construction/admission`,
`construction/full_frame`, `construction/redundancy`, and
`construction/expansion`) from release material (`releases/cohort`,
`releases/scoring_protocol`, and `releases/confirmatory`) and derived
evaluation (`analysis/structural_response` and `analysis/pair_validity`).
Historical paths such as `scale1a1` and `scale1b_v2` remain read-only inputs
when frozen manifests bind them; they are not active output namespaces.

---

### 23.3 `runs/`

Contains execution instances.

Conceptually:

```text
runs/
└── <experiment>/
    └── <run-id>/
        ├── run_manifest.json
        ├── logs/
        ├── intermediate/
        ├── predictions/
        └── metrics/
```

Run IDs are execution identities and may contain timestamps, hashes, or opaque identifiers.

Runs are usually mutable and gitignored.

---

### 23.4 `artifacts/`

Contains promoted, validated, durable scientific outputs.

Examples:

```text
artifacts/
├── checkpoints/
└── releases/
```

Canonical release flow:

```text
compute
→ staging
→ validate
→ manifest/hash
→ promote
```

Do not partially overwrite frozen releases.

---

## 24. Naming Architecture

Use four naming rules:

```text
Code names       → scientific/engineering capabilities
Experiment names → scientific purpose
Artifact names   → artifact meaning + optional version
Metadata         → historical/legacy identity
```

Prefer:

```text
dataset/admission.py
design_baseline
paired_structure_dataset_v1
```

over:

```text
scale1a1.py
p2/
final_dataset
```

Version suffixes are valid when the semantic base name is meaningful:

```text
dataset_manifest_v2
prediction_schema_v3
design_baseline_cohort_v1
```

Historical identifiers may remain in provenance:

```yaml
legacy_ids:
  - scale1a3
```

Code names encode concepts.

Metadata preserves history.

---

## 25. Candidate and Cohort Identity

Distinguish stable object identity from scientific semantics.

Opaque IDs are acceptable:

```text
candidate_000042
```

when metadata defines their meaning.

Avoid packing many scientific dimensions into semi-semantic IDs such as:

```text
index36_matched_afdb_03_c07
```

when every component must be decoded.

Prefer explicit structured metadata:

```text
protein_id
candidate_id
backbone_source
generation_source
scoring_source
condition
hamming_distance
replicate
seed
model_id
checkpoint_id
```

Do not encode the full Cartesian product of experimental conditions into nested directory paths.

Partition storage only where it materially improves I/O.

Cohort names should describe scientific purpose rather than size.

Prefer:

```text
paired_structure_cohort
design_baseline_cohort
cross_mechanism_cohort
```

with counts stored as metadata:

```yaml
cohort_id: paired_structure_cohort
n_proteins: 36
```

---

## 26. Provenance and Versioning

Scientific artifacts should preserve relevant:

* source identities;
* source hashes;
* dataset/release identity;
* scientific policy;
* model/checkpoint;
* configuration;
* seed;
* schema version;
* scientific-definition version;
* code revision.

Historical stage identifiers may also be retained.

Example:

```yaml
release_id: design_baseline_cohort_v1

legacy_ids:
  - scale1b
```

Do not use one ambiguous `version` field for unrelated concepts.

Distinguish where relevant:

```text
schema_version
artifact_format_version
scientific_definition_version
protocol_version
model_version
checkpoint_id
```

A serialization-format change is not necessarily a scientific-definition change.

A metric-equation change is not merely a schema change.

Machine-specific absolute paths must not serve as scientific identity.

Repository-owned artifacts should normally use repository-relative logical paths.

External/shared artifacts should use stable identifiers or portable URIs with appropriate hashes/provenance.

---

## 27. Schemas

Persistent machine-readable repository-level contracts belong under:

```text
schemas/
```

Examples:

```text
dataset_manifest.schema.json
prediction_record.schema.json
release_manifest.schema.json
```

Python-internal contracts may remain with their scientific owner.

Schemas are not configuration.

Do not place persistent schemas under `configs/`.

---

## 28. Test Architecture

Use one primary organizational axis:

```text
tests/
├── unit/
├── integration/
├── regression/
├── contracts/
└── fixtures/
```

### Unit Tests

Cover stable reusable scientific behavior.

Example:

```text
tests/unit/
├── structure/
│   └── test_mapping.py
├── dataset/
│   ├── test_admission.py
│   ├── test_redundancy.py
│   └── test_selection.py
├── models/
├── uncertainty/
└── evaluation/
```

General unit tests should use semantic rather than stage-based names.

### Integration Tests

Cover component/workflow boundaries.

Examples:

```text
source acquisition → normalization
model adapter → normalized prediction representation
dataset components → dataset workflow
```

### Regression Tests

Protect previously observed failures.

Regression identity should describe the protected behavior where practical.

### Contract Tests

Protect frozen:

* releases;
* manifests;
* schemas;
* membership;
* scientifically important historical expectations.

Example:

```text
tests/contracts/releases/
└── design_baseline_cohort_v1/
    ├── test_membership.py
    ├── test_manifest.py
    └── expected.yaml
```

Reusable contract assertions should be shared rather than copied between releases.

Prefer:

* fixtures;
* parameterization;
* small helper assertions;

over inheritance-heavy test frameworks.

Tests should normally import production scientific definitions rather than duplicate them.

Independent simple reference implementations may be used as oracles where appropriate.

Do not create expected values by copying the production implementation line-for-line.

---

## 29. Extension Decision Tree

When adding behavior:

```text
Does an existing package own the concept?
        |
        +-- YES
        |     ↓
        |   extend a cohesive module there
        |
        +-- NO
              |
              Is this orchestration?
              |
              +-- YES → workflows/
              |
              +-- NO
                    |
                    Is it source/model-specific?
                    |
                    +-- YES → adapter near the stable contract it produces
                    |
                    +-- NO
                          |
                          Is it a stable scientific capability
                          with multiple meaningful responsibilities?
                          |
                          +-- YES → consider new package
                          |
                          +-- NO → reconsider the abstraction
```

A new package should normally satisfy all of:

1. coherent scientific responsibility;
2. multiple meaningful operations or dependencies;
3. no appropriate existing owner;
4. improved dependency direction.

Do not create a package merely because one task requires several files.

New abstractions must pay for themselves by removing or preventing concrete coupling.

---

## 30. Stage-to-Capability Migration

Historical stage-oriented code should migrate incrementally.

Preferred transformation:

```text
historical stage implementation
        ↓
characterize existing behavior
        ↓
identify stable scientific responsibility
        ↓
extract canonical reusable capability
        ↓
policy/config + release contract
        ↓
migrate callers
        ↓
validate frozen behavior
        ↓
remove obsolete duplication
```

Example:

```text
Scale1A3-specific redundancy implementation
```

may become:

```text
src/dual_uq/dataset/redundancy.py
+
experiment-specific redundancy policy
+
release provenance legacy_id=scale1a3
```

Do not mechanically rename historical files before understanding their scientific responsibility.

---

## 31. Compatibility Wrappers

Temporary compatibility is allowed:

```text
legacy entry point
        ↓
canonical reusable capability
```

Example:

```python
def legacy_scale1_admission(...):
    return evaluate_admission(
        ...,
        policy=legacy_policy,
    )
```

Compatibility wrappers preserve documented contracts during migration.

They are not canonical scientific implementations.

Do not retain obsolete wrappers indefinitely after all relevant callers have migrated.

---

## 32. Migration Safety

Do not perform a Big-Bang rewrite.

Before moving scientifically meaningful behavior, establish characterization/contract coverage for relevant:

* admission outcomes;
* rejection reasons;
* cohort membership;
* cluster assignments;
* redundancy counts;
* residue mappings;
* manifest identity;
* release hashes.

Then migrate using:

```text
1. characterize existing behavior
2. identify stable responsibility
3. extract reusable implementation
4. retain policy/release-specific data separately
5. migrate callers
6. validate frozen contracts
7. remove obsolete duplication
```

Architecture improvement should normally be driven by active scientific work, not broad mechanical cleanup.

---

## 33. Dependency and Cycle Discipline

Top-level scientific packages should remain acyclic where practical.

If a cycle appears, do not resolve it using:

```text
dynamic imports
runtime monkey-patching
global service locators
moving arbitrary code into core
```

Instead inspect ownership.

Typical causes of cycles include:

* a scientific contract owned by the wrong package;
* orchestration leaking into domain code;
* adapter details leaking upward;
* shared types placed too high or too low;
* peer packages owning overlapping concepts.

Resolve the ownership problem before introducing technical workarounds.

---

## 34. Architectural Change Discipline

Architecture should evolve from demonstrated project needs.

Do not modify repository-wide architecture merely because a cleaner theoretical design is imaginable.

A durable architecture change should normally be justified by at least one of:

* repeated duplication;
* repeated dependency-direction problems;
* repeated scientific-contract ambiguity;
* multiple real callers requiring the same abstraction;
* an active migration from stage-specific code;
* a demonstrated testing/provenance limitation.

Task-specific architectural decisions that are not yet repository-wide should normally be documented under:

```text
docs/decisions/
```

rather than immediately promoted into `ARCHITECTURE.md`.

---

## 35. Architecture Success Criteria

The architecture is working when:

* a new model does not require rewriting dataset logic;
* a new metric does not require unnecessary model inference;
* historical experiment IDs do not leak into reusable implementation;
* scientific definitions have canonical owners;
* residue/index mappings are explicit and centralized;
* project-level contracts are model/source independent;
* external model/source details remain behind adapters;
* configuration does not leak as unstructured dictionaries through reusable scientific code;
* top-level dependency cycles are absent;
* experiment directories describe scientific intent rather than runtime state;
* frozen artifacts remain reconstructable and auditable;
* tests describe scientific behavior rather than roadmap chronology;
* adding a new experiment mostly means configuring and composing existing capabilities;
* reusable components operate on compatible new datasets or frames through inputs and policies rather than internal rewrites.

---

## 36. Final Architectural Principle

Dual-UQ should encode:

```text
scientific concepts
+
explicit contracts
+
reusable capabilities
+
clear dependency direction
+
source/model boundaries
+
experiment specifications
+
provenance-controlled artifacts
```

not:

```text
roadmap chronology
+
copied stage scripts
+
boolean-mode pseudo-abstractions
+
global configuration dictionaries
+
model-specific assumptions everywhere
+
deep condition-specific directory trees
+
duplicated scientific definitions
```

The architectural objective is not maximum abstraction.

It is the minimum stable architecture required to make protein-design research software scientifically correct, reproducible, reusable, extensible, and auditable.
