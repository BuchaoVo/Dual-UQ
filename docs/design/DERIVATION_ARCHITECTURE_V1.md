# DERIVE-ARCH-1: Reusable Derivation Architecture V1

| Item | Value |
| --- | --- |
| Status | IMPLEMENTED / VALIDATED |
| Date | 2026-08-04 |
| Branch baseline | `protocol/dataset-a-h2` at `bf81c19e085982aa90f28a4de98586586d0b2b4c` |
| Scientific baseline | DERIVE-PILOT = `DERIVATION_PIPELINE_PASS_WITH_DATA_FAILURES` |
| Scope | Refactor reusable raw-to-observable derivation only |

This design establishes `dual_uq.dataset` as the long-lived capability namespace.
“Dataset A” remains a scientific benchmark and protocol name; it does not become a
new package name. The design does not migrate the frozen A1–A8 modules, change
scientific semantics, or authorize DERIVE-48.

## 1. Current monolith responsibility map

The current `scripts/dataset_a/derivation/derive_pilot.py` contains 1,599 lines and
combines pilot selection, portable-path concerns, identity validation, mapping,
fragment resolution, observability, orchestration and three report renderers.
The following map assigns every current top-level block to exactly one target
responsibility. Line numbers refer to the pre-refactor file at the baseline above.

| Current lines / symbol | Current responsibility | Target responsibility | Target location |
| --- | --- | --- | --- |
| 51–56 `DerivePilotError` | Structured domain error | data model | `dual_uq.dataset.models` |
| 59–63 `EXPECTED_SOURCE_BINDINGS` | Pilot baseline binding | genuinely pilot-specific | thin pilot entrypoint |
| 66–108 `validate_preflight_contract` | Validate the registered 48-candidate ACQ-5 baseline | genuinely pilot-specific | thin pilot entrypoint |
| 111–134 `_as_records` | Normalize AFDB metadata collection | identity | `dual_uq.dataset.identity` |
| 137–144 `_exact_records` | Exact-accession partition | identity | `dual_uq.dataset.identity` |
| 147–177 `extract_canonical_sequence` | Unique exact record, sequence and model identity | identity | `dual_uq.dataset.identity` |
| 180–187 `verify_bound_hash` | Verify a logical asset against its source binding | derivation orchestration | `dual_uq.dataset.derivation` using frozen hashing |
| 190–197 `_known_number` | Pilot selection metadata normalization | CLI / pilot selection | thin pilot entrypoint |
| 200–325 `select_pilot_panel` | Eight-protein stress-panel policy | genuinely pilot-specific | thin pilot entrypoint |
| 328–345 `attach_batch1_ranks` | Bind the pilot selector to the frozen Batch-1 order | genuinely pilot-specific | thin pilot entrypoint |
| 348–353 `prepare_confidence_mapping` | Adapt explicit UniProt coordinate names | mapping | `dual_uq.dataset.mapping` |
| 356–439 `annotate_gap_semantics` | D3 segment and true-gap semantics | mapping | `dual_uq.dataset.mapping` |
| 442–447 `are_peptide_adjacent` | True peptide adjacency predicate | mapping | `dual_uq.dataset.mapping` |
| 450–472 `audit_sequence_discrepancies` | Mapping/PDB/AFDB mismatch sets | mapping | `dual_uq.dataset.mapping` |
| 475–484 `_fragment_from_record` | Convert an exact prediction record to frozen fragment type | fragment resolution | `dual_uq.dataset.fragments` |
| 487–524 `resolve_exact_fragment` | Require exactly one exact-accession full-cover record | fragment resolution | `dual_uq.dataset.fragments` |
| 527–545 `validate_bound_arrays` | PAE/confidence/model-length binding | fragment resolution | `dual_uq.dataset.fragments` |
| 551–569 `account_stage_failure` | Primary/dependent stage propagation | data model | `dual_uq.dataset.models` |
| 572–609 `recompute_sampling_prior` | G0-C-R positive-evidence sampling prior | observability | `dual_uq.dataset.observability` |
| 612–615 `_json_cell` | Deterministic table-cell serialization | reporting | `dual_uq.dataset.reporting` |
| 618–677 `write_outputs` | JSON/TSV/Markdown rendering | reporting | `dual_uq.dataset.reporting` |
| 680–687 `_load_json` | Read registered pilot inputs | CLI / pilot selection | thin pilot entrypoint |
| 690–695 `_sha256_file` | File hashing | frozen-module adapter | reuse `dual_uq.dataset_a_scale.hashing.sha256_file` |
| 698–730 `_validated_preflight` | Pilot binding files and ACQ-5 gate | genuinely pilot-specific | thin pilot entrypoint, using `ProjectPaths` |
| 733–738 `_load_resolution` | Load pilot's historical exact-record evidence | CLI / pilot selection | thin pilot entrypoint context adapter |
| 741–757 `_successful_asset_records` | Assemble pilot ACQ-1/ACQ-5 logical asset bindings | CLI / pilot selection | thin pilot entrypoint context adapter |
| 760–788 `_metadata_path` | Convert legacy pilot evidence into one logical metadata ref | genuinely pilot-specific | thin pilot entrypoint context adapter; generic identity receives the ref |
| 791–823 `_asset_path` | Convert legacy pilot evidence into logical model/PAE/confidence refs | genuinely pilot-specific | thin pilot entrypoint context adapter; generic fragment code receives the refs |
| 826–851 `validate_asset_model_binding` | Require one selected model across dependent assets | fragment resolution | `dual_uq.dataset.fragments` |
| 854–857 `_verify_optional_hash` | Hash or verify a resolved asset | derivation orchestration | `dual_uq.dataset.derivation` |
| 860–887 `_pair_quality` | Frozen preflight metrics and classification adapter | mapping | `dual_uq.dataset.mapping` |
| 890–908 `pair_qc_attrition` | Separate QC predicate from causal availability | mapping | `dual_uq.dataset.mapping` |
| 911–931 `_mapping_with_provenance` | Auth/label join and D3 annotation | mapping | `dual_uq.dataset.mapping` |
| 934–935 `_nullable_int` | Mapping report serialization helper | mapping | `dual_uq.dataset.mapping` |
| 938–965 `_residue_mapping_audit_records` | Explicit residue provenance records | mapping | `dual_uq.dataset.mapping` |
| 968–992 `_afdb_ca_by_uniprot` | Strict model-local to UniProt coordinate mapping | mapping | `dual_uq.dataset.mapping` |
| 995–1014 `_state_segments` | State-disagreement segment inputs | observability | `dual_uq.dataset.observability` |
| 1017–1131 `_mechanism_evidence` | Four mechanism-observability feature families | observability | `dual_uq.dataset.observability` |
| 1134–1165 `_candidate_base` | Initialize one structured candidate result | data model | `dual_uq.dataset.models` |
| 1168–1403 `_derive_one` | Candidate-level stage execution | derivation orchestration | `dual_uq.dataset.derivation` |
| 1406–1416 `_current_failure_stage` | Determine current stage | data model | `dual_uq.dataset.models` |
| 1419–1440 `_failed_result` | Convert domain failure to structured result | derivation orchestration | `dual_uq.dataset.derivation` |
| 1443–1476 `_descriptive_attrition` | Run-level descriptive aggregation | pipeline orchestration | `dual_uq.dataset.pipeline` when constructing the run result |
| 1479–1576 `run_pilot` | Mixed pilot loading, batch execution and summary | pipeline orchestration | replace with generic `dual_uq.dataset.pipeline.run_derivation`; the script separately composes pilot inputs |
| 1579–1585 `parse_args` | CLI | CLI / pilot selection | thin pilot entrypoint |
| 1588–1597 `main` | CLI composition root | CLI / pilot selection | thin pilot entrypoint |

### 1.1 Logic already implemented elsewhere

The refactor must not reproduce the following algorithms:

- File hashing: `dual_uq.dataset_a_scale.hashing.sha256_file`.
- SIFTS parsing and insertion-code preservation:
  `dual_uq.sifts.parse_sifts_residue_mapping`.
- Pair-QC metrics and thresholds: `dual_uq.preflight.compute_preflight_metrics`
  and `dual_uq.preflight.classify_preflight`.
- Auth/label CA joins: `dual_uq.structure_io.join_residue_mapping_to_ca`.
- PDB/AFDB CA loading: `dual_uq.structure_io.load_chain_ca_table`.
- AFDB fragment/PAE coordinate objects and strict PAE loading:
  `dual_uq.dataset_a_scale.pae`.
- AFDB artifact URL/model validation: the existing P0 validators in
  `dual_uq.dataset_a_scale.stages.p0`.
- Mapped confidence: `dual_uq.mapped_confidence`.
- Disagreement segments: `dual_uq.robust_stats.contiguous_segments`.
- PAE strata: `dual_uq.a0_classification.compute_pae_strata`.
- Geometry alignment: `dual_uq.geometry.kabsch_align`.

`dual_uq.afdb.select_prediction_for_interval` is not the canonical D2 adapter for
this runner: it deterministically chooses among multiple covering records, while
the approved D2 contract requires `ambiguous_full_covering_fragments`. The existing
pilot resolver therefore moves to `dual_uq.dataset.fragments`; it must not be
replaced by the permissive selector.

The ACQ scripts contain equivalent exact-accession collection partitioning, but
script entrypoints are not a reusable package API. `dual_uq.dataset.identity` will
become the long-lived derivation-side implementation of that already-reviewed
contract. Historical ACQ scripts and reports remain untouched.

## 2. Target module map

Only modules justified by current responsibilities are created.
`dataset/acquisition.py`, `core/config.py`, `core/io.py`, `core/provenance.py` and
`core/failures.py` are deliberately not created in this migration.

```text
src/dual_uq/
├── core/
│   ├── __init__.py
│   └── paths.py
└── dataset/
    ├── __init__.py
    ├── models.py
    ├── identity.py
    ├── mapping.py
    ├── fragments.py
    ├── observability.py
    ├── derivation.py
    ├── pipeline.py
    └── reporting.py
```

| Module | Public responsibility | Explicitly excluded |
| --- | --- | --- |
| `core.paths` | Resolve repository and runtime roots; produce portable logical references | Scientific validation, downloading, biological identity |
| `dataset.models` | Immutable contexts, logical asset refs, stage and run result types | File I/O and scientific computation |
| `dataset.identity` | Exact-accession collection partition and canonical sequence/model identity | Accession fallback and acquisition |
| `dataset.mapping` | Pair-QC adapter, residue provenance, true gaps, discrepancy audit and strict coordinates | Threshold redefinition and offset rescue |
| `dataset.fragments` | Exact full-cover D2 resolution and model/PAE/confidence identity binding | Downloading, first-record/F1 fallback and stitching |
| `dataset.observability` | Mechanism evidence inputs and sampling-only prior | Final P6 label or admission |
| `dataset.derivation` | Execute one candidate and translate expected failures into stage results | Panel selection and report formatting |
| `dataset.pipeline` | Validate an already-built panel, deterministic batch/subset execution, dependency propagation and run-level aggregation | Candidate-count branches, acquisition-ledger loading and scientific algorithms |
| `dataset.reporting` | One canonical report object and deterministic JSON/TSV/Markdown renderers | Recomputing scientific metrics independently per format |

`dual_uq.dataset.__init__` exposes only the stable entrypoints and models:
`CandidateContext`, `StageResult`, `CandidateDerivationResult`,
`DerivationRunResult`, `DerivationConfig`, `ProjectPaths` and `run_derivation`.
Internal adapters remain imported from their capability modules.

## 3. Frozen dependency map

| Frozen dependency | Reuse path | Adapter | Assumptions | API sufficient? |
| --- | --- | --- | --- | --- |
| SHA-256 | `dual_uq.dataset_a_scale.hashing.sha256_file` | none | raw bytes, no normalization | yes |
| AFDB fragment and PAE types | `dual_uq.dataset_a_scale.pae.AFDBFragment`, `PAEMappingError`, `load_pae_json`, `validate_pae_matrix` | thin conversion from exact metadata record | inclusive UniProt interval and exact model length | yes |
| P0 AFDB URL/model validators | `dual_uq.dataset_a_scale.stages.p0._validate_artifact_identities`, `_validate_model_mmcif` | named wrapper in `dataset.fragments` | same AFDB version and exact model identity | yes, but private-API coupling is a documented risk |
| SIFTS | `dual_uq.sifts.parse_sifts_residue_mapping` | normalize column names only | auth/label IDs and insertion code remain separate | yes |
| Pair-QC | `dual_uq.preflight.compute_preflight_metrics`, `classify_preflight` | `dataset.mapping.compute_pair_quality` | frozen thresholds supplied as one bound config | yes |
| Structure I/O | `dual_uq.structure_io` | D3 provenance annotation around returned tables | no guessed auth/label conversion | yes |
| Confidence | `dual_uq.confidence.load_plddt` and `dual_uq.mapped_confidence` | fragment interval passed explicitly | JSON remains authoritative, no padding | yes |
| State segments | `dual_uq.robust_stats.contiguous_segments` | observability thresholds applied after segment formation | UniProt positions determine continuity | yes |
| PAE strata | `dual_uq.a0_classification.compute_pae_strata` | none | symmetric mapped PAE matrix | yes |
| Geometry | `dual_uq.geometry.kabsch_align` | none | minimum three mapped CA pairs | yes |
| A1–A8 P0/P1 contracts | `dual_uq.dataset_a_scale.stages.p0/p1` and associated models | read/validate only; no migration | frozen identity, immutability and coordinate semantics | yes |

`FROZEN_MODULE_ARCHITECTURE_CONFLICT: none`.

The private P0 validators are sufficient for exact behavioral reuse. Promoting them
to a public API would modify a frozen module and is outside this task. The adapter
must call them rather than copy their implementation.

## 4. ProjectPaths design

`core.paths.ProjectPaths` is an immutable path-resolution object. It contains no
scientific decisions.

```python
@dataclass(frozen=True)
class ProjectPaths:
    repository_root: Path
    data_root: Path
    raw_root: Path
    processed_root: Path
    reports_root: Path
    runs_root: Path
    artifacts_root: Path

    @classmethod
    def discover(
        cls,
        *,
        project_root: Path | None = None,
        anchor: Path | None = None,
        data_root: Path | None = None,
        reports_root: Path | None = None,
        runs_root: Path | None = None,
        artifacts_root: Path | None = None,
    ) -> "ProjectPaths": ...

    def logical_ref(self, path: Path) -> str: ...
```

### 4.1 Root discovery precedence

1. Explicit `project_root` argument.
2. `DUAL_UQ_PROJECT_ROOT`, when explicitly set by the runtime.
3. Upward marker search from the explicit `anchor`; the pilot passes its own
   `__file__`, never `Path.cwd()`.
4. Upward marker search from `core.paths.__file__` for an editable source checkout.
5. Structured `ProjectPathError("repository_root_unresolved")`.

A repository marker requires both `pyproject.toml` and `src/dual_uq`; an arbitrary
directory named `Dual-UQ` is insufficient. There is no CWD fallback.

Each runtime root uses explicit constructor/CLI override first, then a corresponding
environment override (`DUAL_UQ_DATA_ROOT`, `DUAL_UQ_REPORTS_ROOT`,
`DUAL_UQ_RUNS_ROOT`, `DUAL_UQ_ARTIFACTS_ROOT`), then the repository-relative
default. `raw_root` and `processed_root` derive from `data_root` and are not
independently overridden in V1.

`logical_ref()` serializes portable namespace-relative references such as
`data/raw/pdb/2vb1.cif` and `artifacts/audits/...`. It never serializes a developer
absolute path. Default roots reproduce existing report path strings exactly;
overridden roots change physical resolution without changing logical references.
Paths that lie outside every declared root cause a structured path error instead of
leaking an absolute path.

## 5. Data models

### 5.1 Stable biological identity

```python
@dataclass(frozen=True)
class BiologicalIdentity:
    pair_id: str
    pdb_id: str
    chain_id: str
    uniprot_accession: str
    polymer_entity_id: str
```

The tuple above, not a filesystem location, is the biological identity. PDB IDs and
UniProt accessions are normalized once when a context is constructed. No implicit
accession substitution is permitted.

### 5.2 Logical assets and CandidateContext

```python
@dataclass(frozen=True)
class LogicalAssetRef:
    asset_type: str
    logical_path: str
    sha256: str | None
    provenance: str
    expected_model_identity: str | None = None

@dataclass(frozen=True)
class CandidateContext:
    candidate_index: int
    identity: BiologicalIdentity
    exact_afdb_accession: str
    expected_afdb_model_identity: str
    assets: tuple[LogicalAssetRef, ...]
    source_bindings: tuple[tuple[str, str], ...]
    protocol_binding: str
    selection_roles: tuple[str, ...] = ()
    selection_reason: str | None = None
    prederivation_evidence: tuple[tuple[str, object], ...] = ()
```

`candidate_index` is an inventory/run convenience ID. It is not permanent protein
identity or admission. Asset references are logical and hash-bound; `ProjectPaths`
resolves them to physical paths at runtime. `protocol_binding` is the digest/version
of the complete validated scientific configuration, not a mutable CLI label.

### 5.3 StageResult

```python
StageStatus = Literal["complete", "failed", "unobservable", "skipped_dependency"]

@dataclass(frozen=True)
class StageResult:
    stage: str
    status: StageStatus
    primary_failure_code: str | None = None
    warnings: tuple[str, ...] = ()
    metrics: Mapping[str, object] = field(default_factory=dict)
    artifact_refs: tuple[LogicalAssetRef, ...] = ()
    dependent_unavailable_stages: tuple[str, ...] = ()
```

Construction validates these invariants:

- `complete` has no primary failure code.
- `failed` and `unobservable` require a primary failure code.
- `skipped_dependency` identifies its upstream dependency and is never counted as
  an independent failure.
- A pair-QC quality predicate failure remains a `complete` stage with warning
  `pair_qc_threshold_not_met`; it does not masquerade as an execution failure.
- Metrics and artifact refs are carried explicitly; scientific/data failures are
  not represented solely by exceptions.

Expected low-level exceptions are converted at the candidate boundary into
`StageResult`. Unexpected exceptions produce
`implementation_or_schema_issue` and prevent scale-up.

### 5.4 Candidate and run results

```python
@dataclass(frozen=True)
class CandidateDerivationResult:
    context: CandidateContext
    stages: tuple[StageResult, ...]
    evidence: Mapping[str, object]
    report_record: Mapping[str, object]

@dataclass(frozen=True)
class DerivationRunResult:
    schema_version: str
    preflight: Mapping[str, object]
    candidates: tuple[CandidateDerivationResult, ...]
    summary: Mapping[str, object]
    attrition_bias_probe: Mapping[str, object]
    scope: Mapping[str, object]
```

`report_record` is the compatibility projection for the current report schema. It
is constructed once from stages/evidence. Renderers never independently infer
stage state or recompute scientific metrics.

### 5.5 Configuration model

```python
@dataclass(frozen=True)
class DerivationConfig:
    protocol_version: str
    protocol_binding: str
    preflight_thresholds: Mapping[str, float]
    observability_thresholds: Mapping[str, object]
```

The constructor defensively copies and validates mappings. Numeric scientific
thresholds enter only through a bound, reviewed configuration; CLI flags cannot
override individual predicates.

## 6. Generic runner and stage dependency graph

The only scalable public execution entrypoint is:

```python
def run_derivation(
    panel: Sequence[CandidateContext],
    config: DerivationConfig,
    paths: ProjectPaths,
) -> DerivationRunResult: ...
```

The runner validates unique candidate indices and biological identities, then uses
stable ascending `candidate_index` order. Empty panels fail structurally. It has no
branches based on `len(panel)` and no identifiers named pilot, Round-2, 48, 213,
A1 or A2.

```text
validated CandidateContext / raw bindings
                  │
                  ▼
       exact-record identity + canonical sequence
                  │
                  ▼
         pair-QC calculation and predicate
                  │
        ┌─────────┴─────────────────────┐
        │ complete, including QC fail  │ unobservable/failed
        ▼                              ▼
 explicit residue mapping       mapping = skipped_dependency
        │                        fragment/PAE/confidence/
        ▼                        observability = skipped_dependency
 exact full-cover fragment
        │
        ├──────────────┬────────────────┐
        ▼              ▼                ▼
 model identity      PAE binding    confidence binding
        └──────────────┴────────────────┘
                       │
                       ▼
            mechanism observability evidence
```

`derive_candidate(context, config, paths)` in `dataset.derivation` executes one
candidate. `dataset.pipeline.run_derivation` provides deterministic iteration,
candidate isolation, summary construction and pipeline verdict. One candidate's
data failure does not stop another candidate. An implementation/schema failure is
retained distinctly and sets the run verdict to `IMPLEMENTATION_BUG`.

The existing rule is preserved: pair-QC quality failure does not block mapping or
observability, but positive mechanism evidence cannot become a supported sampling
prior without `pass_full_length`.

## 7. Pilot wrapper

The final `scripts/dataset_a/derivation/derive_pilot.py` remains a compatibility
entrypoint and is expected to be approximately 250–350 lines because the approved
eight-role selector is genuinely pilot-specific. It may only:

1. Parse CLI path overrides and output location.
2. Construct `ProjectPaths` without relying on CWD.
3. Load the registered inventory, Batch-1 plan, ACQ-1/ACQ-5 ledgers,
   exact-record-resolution evidence and pilot configs.
4. Revalidate pilot-specific 48/48 source bindings.
5. Deterministically select the eight stress candidates.
6. Resolve the historical pilot evidence into explicit logical asset refs and
   construct `CandidateContext` values. The generic pipeline never searches legacy
   ledgers or guesses physical assets.
7. Call `run_derivation(panel, config, paths)`.
8. Call the common reporter and return status.

The following must disappear from the script: exact-record parsing, sequence
extraction, pair-QC algorithms, gap annotation, discrepancy audit, fragment
resolution, PAE/confidence validation, geometry, mechanism evidence, dependency
propagation, summary calculation and format-specific report construction.

The script may retain `validate_preflight_contract`, `select_pilot_panel`,
`attach_batch1_ranks`, CLI parsing and the three approved source-binding constants.

## 8. Reporting and behavioral-equivalence gate

### 8.1 Path audit

The baseline JSON, TSV and Markdown were scanned for `/home/`, `/mnt/` and Windows
drive paths. No machine-specific absolute path occurs. All recorded paths are
repository-relative, for example `data/raw/...` and `artifacts/audits/...`.
Therefore no report schema or path exception is approved.

### 8.2 Locked baseline

| Report | Required SHA-256 after refactor |
| --- | --- |
| `reports/dataset_a_scale/derive_pilot_v1.json` | `90ba56dba15237b348f7e3e67d947c43db80a37afcefacd37f1731b52a8248ef` |
| `reports/dataset_a_scale/derive_pilot_v1.tsv` | `e645f46270694c777d03d1c92e6e824e0b22a0b58ed4c04c1dc34ce4bb30f2be` |
| `reports/dataset_a_scale/derive_pilot_v1.md` | `1211ff7fbac6256bc2760409316a2823a362cf7f09993156605241af0ea677b6` |

All three hashes are hard gates. JSON key order/indentation, LF endings, TSV column
order and empty-cell handling, Markdown ordering, float serialization, path values,
stage status, evidence and verdict must remain byte-identical.

`dataset.reporting.build_report(run_result)` creates one canonical compatibility
mapping. `render_json`, `render_tsv` and `render_markdown` consume only that mapping.
No renderer calls mapping, fragment, confidence or observability code.

## 9. Configuration boundaries

### 9.1 Protocol invariants

These are code/config-version invariants and are not runtime-overridable:

- Literal exact-accession metadata partition; zero or multiple exact records fail.
- `uniprotSequence`, then `sequence`, is the only sequence-field priority.
- Exactly one exact-accession fragment must fully cover the mapped interval.
- Auth, label and insertion-code residue provenance remain distinct.
- UniProt position defines biological continuity; output position does not.
- Pair-QC and mechanism thresholds are loaded as complete hash-bound sets.
- Upstream failure creates dependent `skipped_dependency` results.
- Quality failure does not itself block evidence derivation.
- No sibling substitution, first-record/F1 fallback, nearest fragment, stitching,
  offset inference, intersection trimming, sequence trimming, numerical padding or
  truncation rescue.

### 9.2 Experiment configuration

- Panel membership and deterministic panel order.
- Pilot selection roles/reasons.
- Expected source/manifest hashes.
- Protocol/config binding and report schema version.

Changing an experiment configuration chooses a different registered subset; it
does not enable different scientific implementation.

### 9.3 Runtime configuration

- Physical repository/data/reports/runs/artifacts roots.
- Output report directory and filename prefix.
- Logging verbosity and continue-to-next-candidate behavior after structured data
  failure.

Runtime configuration cannot alter identity, mapping, fragment or numerical rescue
semantics. A CLI option that could enable any forbidden fallback is prohibited.

## 10. RED-first TDD migration plan

No GREEN implementation belongs to the design task. The implementation task must
execute these steps in order and record each expected RED before production edits.

### Step 1 — Architecture and current-behavior characterization

- **New failing test:** import `dual_uq.dataset.run_derivation` and the structured
  models; verify the package does not yet exist. Record the three report SHA values
  above as immutable integration expectations.
- **Minimal GREEN later:** package initializers exposing only the approved public
  interface stubs, followed by real types in subsequent steps.
- **Protected invariant:** package name is `dual_uq.dataset`; no `dataset_a` package
  and no output-baseline ambiguity.

### Step 2 — Portable path resolution

- **New failing tests:** explicit root at `tmp_path`, clone marker discovery from a
  non-CWD anchor, independent data/report/run/artifact overrides, logical reference
  generation and rejection of out-of-root absolute paths.
- **Minimal GREEN later:** `core.paths.ProjectPaths` and `ProjectPathError` only.
- **Protected invariant:** arbitrary clone location and no CWD/machine path leakage.

### Step 3 — Immutable contexts and logical assets

- **New failing tests:** construct valid `BiologicalIdentity`, `LogicalAssetRef`,
  `CandidateContext` and `DerivationConfig`; reject empty identities, duplicate
  assets, malformed SHA values and use of paths as biological IDs.
- **Minimal GREEN later:** frozen dataclasses and constructor validation in
  `dataset.models`.
- **Protected invariant:** filesystem location is not biological identity and every
  run is manifest/config-bound.

### Step 4 — StageResult and dependency propagation

- **New failing tests:** enforce the four statuses; require failure codes for
  failed/unobservable; propagate one mapping failure to skipped fragment/PAE/
  confidence/observability; keep a pair-QC predicate failure complete with warning.
- **Minimal GREEN later:** `StageResult`, result containers and dependency helper in
  `dataset.models`.
- **Protected invariant:** primary failure accounting and scientific/data failure
  semantics remain structured.

### Step 5 — Capability extraction and frozen adapters

- **New failing tests:** port the existing exact sibling, gap/adjacency, mismatch,
  zero/multiple-cover, PAE/confidence length, no-rescue, multi-label and quality-gate
  tests to imports from `dual_uq.dataset.identity/mapping/fragments/observability`.
  Each test is RED because the target capability module is absent.
- **Minimal GREEN later:** move the current reviewed functions without algorithmic
  edits and replace duplicated hashing/QC/PAE/structure calculations with the frozen
  imports documented in §3.
- **Protected invariant:** all scientific edge cases and prohibited fallbacks.

### Step 6 — Candidate derivation and generic runner

- **New failing tests:** execute one synthetic candidate; execute input-order-
  permuted subsets; isolate one candidate failure; parameterize panel sizes
  1/8/48/213 with a controlled candidate executor and assert identical orchestration
  logic and stable candidate-index order.
- **Minimal GREEN later:** `derive_candidate` and
  `run_derivation(panel, config, paths)` with no count branches.
- **Protected invariant:** one scientific implementation for all scales.

### Step 7 — Common reporter equivalence

- **New failing tests:** render one structured synthetic run through all three
  formats; prove all formats consume the same compatibility mapping; assert
  deterministic bytes, LF endings, candidate-index ordering and no trailing TSV
  whitespace.
- **Minimal GREEN later:** `dataset.reporting` using one canonical report mapping.
- **Protected invariant:** formats cannot drift by recomputing their own summaries.

### Step 8 — Thin pilot entrypoint and full behavioral equivalence

- **New failing tests:** AST/import boundary test rejects scientific helper
  definitions in the script; an invocation test verifies delegation to
  `run_derivation` and common reporting; CWD-independent invocation runs from a
  temporary directory.
- **Minimal GREEN later:** reduce `derive_pilot.py` to the responsibilities in §7.
- **Protected invariant:** historical CLI compatibility with no second pipeline.

Finally run the real eight-candidate pilot offline and require all three SHA values
in §8.2. Any mismatch is a failed migration. No output drift is auto-approved.

## 11. Architecture acceptance tests

| Acceptance claim | Designed proof |
| --- | --- |
| `dual_uq.dataset`, not `dataset_a` | import test succeeds for the former; repository scan rejects creation of `src/dual_uq/dataset_a` |
| No developer absolute paths | source/output scan plus `logical_ref` rejection test |
| Arbitrary repository root | marker tree under `tmp_path` |
| Arbitrary data/report roots | overrides resolve physical files while logical refs remain portable |
| CWD-independent | `chdir(tmp_path)` before CLI/library call |
| Reusable candidate/context model | immutable identity/asset/context tests |
| Structured stage result | status/failure/dependency invariant tests |
| One-candidate execution | real generic runner test with one fixture candidate |
| Arbitrary subset | permuted non-contiguous context subset test |
| Deterministic ordering | repeated/permuted inputs produce candidate-index order |
| No 8/48/213 branching | parameterized 1/8/48/213 orchestration test and source review |
| Frozen implementation reused | import/delegation tests around preflight, SIFTS, PAE, mapped confidence and P0 validators |
| Shared reporting source | all renderers accept the same canonical compatibility mapping only |
| Thin pilot entrypoint | AST boundary and delegation test |
| Pilot output unchanged | exact three-file SHA gate |

## 12. 8 → 48 scale proof

After this refactor, a future authorized DERIVE-48 invocation consists only of:

```python
paths = ProjectPaths.discover(project_root=approved_root, ...)
config = load_bound_derivation_config(approved_protocol_config)
panel = load_candidate_contexts(approved_batch1_manifest)  # 48 contexts
result = run_derivation(panel=panel, config=config, paths=paths)
write_reports(result, output_dir=approved_run_output)
```

The eight-candidate wrapper differs only in how `panel` is selected and where the
report is written. Identity, pair-QC, mapping, fragment, PAE/confidence,
observability, failure propagation and rendering are the same functions. If
DERIVE-48 would require a new scientific module, a new stage implementation or a
candidate-count flag, DERIVE-ARCH-1 has failed acceptance.

This section is a scale proof, not authorization to execute 48 candidates.

## 13. Exact implementation file scope

### 13.1 Files to create in the later implementation task

```text
src/dual_uq/core/__init__.py
src/dual_uq/core/paths.py
src/dual_uq/dataset/__init__.py
src/dual_uq/dataset/models.py
src/dual_uq/dataset/identity.py
src/dual_uq/dataset/mapping.py
src/dual_uq/dataset/fragments.py
src/dual_uq/dataset/observability.py
src/dual_uq/dataset/derivation.py
src/dual_uq/dataset/pipeline.py
src/dual_uq/dataset/reporting.py
tests/dataset/test_paths.py
tests/dataset/test_models.py
tests/dataset/test_capabilities.py
tests/dataset/test_pipeline.py
tests/dataset/test_reporting.py
```

### 13.2 Files to modify in the later implementation task

```text
scripts/dataset_a/derivation/derive_pilot.py
tests/dataset_a_scale/test_derive_pilot.py
```

The three existing reports are regenerated only for hash verification. Their bytes
must not change and therefore they must not acquire a content diff.

### 13.3 Files explicitly left untouched

```text
src/dual_uq/dataset_a_scale/**
src/dual_uq/sifts.py
src/dual_uq/preflight.py
src/dual_uq/structure_io.py
src/dual_uq/mapped_confidence.py
src/dual_uq/robust_stats.py
src/dual_uq/a0_classification.py
src/dual_uq/geometry.py
src/dual_uq/afdb.py
scripts/dataset_a/acquisition/**
scripts/dataset_a/census/**
data/raw/**
data/processed/**
reports/dataset_a_scale/batch1_acquisition_*.json
reports/dataset_a_scale/batch1_acquisition_*.tsv
reports/dataset_a_scale/batch1_acquisition_*.md
```

No `dual_uq.dataset_a`, `dataset/acquisition.py`, `derive_48.py` or new scientific
pipeline is allowed.

## 14. Known risks and resolved architecture questions

1. **Private frozen validator API.** The P0 model/URL validators are private. V1
   accepts a thin adapter rather than copying or modifying frozen code. A later
   public-API promotion requires its own reviewed task.
2. **Exact-record logic exists in historical ACQ scripts.** Importing script code
   would make the package depend on entrypoints. The reviewed exact-accession
   contract is moved once into `dataset.identity`; ACQ history is not rewritten.
3. **Pair-QC status has two meanings.** A calculated QC failure is scientific
   eligibility evidence, not necessarily a failed stage. `StageResult` makes this
   distinction explicit and the compatibility projection preserves current fields.
4. **Output equivalence is serialization-sensitive.** Dict order, float rendering,
   TSV terminal columns and Markdown aggregation are locked by byte hashes and
   renderer tests.
5. **Logical versus physical paths.** Default logical references already match the
   current reports. Runtime overrides alter physical resolution only; unknown paths
   fail rather than leak absolutes.
6. **Existing `dataset_a_scale` name.** It remains as a frozen dependency during
   this migration. New long-lived code uses `dual_uq.dataset`; renaming frozen
   modules is explicitly out of scope.
7. **Pilot complexity role is pre-derivation.** The wrapper retains the approved
   metadata-based proxy and does not use future P0/P1 outcomes to select candidates.
8. **Dirty/untracked baseline.** Implementation must stage or commit nothing
   implicitly and must preserve unrelated worktree state. The exact five pilot
   artifacts remain the behavioral reference regardless of their current Git state.

No unresolved issue requires a scientific contract change. There is no
`FROZEN_MODULE_ARCHITECTURE_CONFLICT`, no approved report-schema change and no
authorization to start DERIVE-48.

## 15. DERIVE-ARCH-1 implementation report

Implementation date: 2026-08-04. This report accompanies the engineering-only
architecture checkpoint and does not authorize a 48-candidate derivation run.

### 15.1 Reusable modules implemented

The approved `dual_uq.core` and `dual_uq.dataset` modules in §2 are implemented.
No `dual_uq.dataset_a`, `dataset.acquisition`, `derive_48.py` or second scientific
pipeline was created. `dual_uq.dataset.run_derivation` is the stable public runner.

### 15.2 ProjectPaths

`ProjectPaths` implements explicit root, environment override, nearest repository
marker discovery and structured unresolved-root failure in that order. It provides
logical `data`, `raw`, `processed`, `reports`, `runs` and `artifacts` roots, rejects
parent traversal and paths outside registered roots, and never consults the CWD as
an implicit root. Unit tests cover arbitrary repository/runtime roots and an
unrelated current working directory. A real pilot invocation from `/tmp` produced
the locked output bytes.

### 15.3 Models and CandidateContext

`BiologicalIdentity`, `LogicalAssetRef`, `CandidateContext`, `DerivationConfig`,
`StageResult`, `CandidateDerivationResult` and `DerivationRunResult` are immutable
dataclasses. Nested scientific configuration/result mappings are recursively
frozen. `CandidateContext` stores only identity, portable logical asset references,
hash/provenance bindings, selection metadata and protocol binding; parsed PDB,
SIFTS, PAE and confidence payloads remain stage-local.

### 15.4 StageResult

The runner uses `complete`, `failed`, `unobservable` and `skipped_dependency`.
Expected failures retain one `primary_failure_code`; later stages are dependency
skips rather than independent failures. Candidate results reject duplicate or
out-of-order stage records. A pair-QC threshold failure remains a completed stage
with a warning and does not stop mapping. Raw input failure sets both the structured
stage and compatibility `raw_complete` field consistently.

### 15.5 Extracted capabilities

- `identity`: metadata collections, literal exact-accession partition and canonical
  sequence/model extraction.
- `mapping`: frozen preflight adapter, auth/label provenance, true UniProt gaps,
  AFDB model-coordinate mapping and three independent discrepancy sets.
- `fragments`: exact full-cover resolution, strict bound arrays and exact model
  binding, with no sibling/nearest/F1/stitch/offset/padding fallback.
- `observability`: mapped confidence, geometry, PAE strata, state segments and the
  positive-evidence sampling prior without assigning a P6 mechanism label.
- `derivation`: one candidate's strict stage execution and causal failure
  propagation.

### 15.6 Generic runner

`run_derivation(panel, config, paths)` validates unique candidate indices, pair IDs,
complete biological identity tuples and protocol bindings, orders deterministically
by candidate index, isolates candidate-level scientific/data failures and constructs
one run result. Tests exercise 1, 8, 48 and 213 contexts plus a permuted,
non-contiguous subset. The pipeline contains no candidate-count switch.

### 15.7 Reporting architecture

`dataset.reporting.build_report` is the sole compatibility projection. JSON, TSV
and Markdown consume that mapping and do not recompute scientific evidence.
Serialization thaws immutable values deterministically. Run metadata can configure
the output basename and Markdown presentation, so scale-up does not require another
renderer; pilot-compatible defaults retain the historical bytes.

### 15.8 Thin pilot wrapper

The historical script now performs only ACQ-5/source binding validation,
pre-derivation eight-candidate selection, legacy ledger-to-`CandidateContext`
adaptation, bound YAML loading, portable path resolution, generic runner invocation
and common reporting. It contains no candidate derivation, mapping, fragment,
geometry, confidence or observability implementation. Scientific configuration is
bound by a canonical SHA-256 over protocol version plus both threshold mappings.

### 15.9 Frozen dependency reuse

Raw hashing uses `dataset_a_scale.hashing.sha256_file`; fragment/PAE and P0 model
validation use the frozen Dataset-A APIs; SIFTS, preflight, structure I/O, mapped
confidence, Kabsch, PAE strata and contiguous segments are imported from their
existing implementations. `P0ValidationError` is translated without losing its
structured failure code. No A1–A8 frozen module was modified.

### 15.10 Portability and architecture tests

The approved five `tests/dataset` files cover path resolution, immutable models,
scientific contract adapters, dependency propagation, arbitrary panel sizes,
identity duplication, common reporting, absence of machine paths, absence of a
new `dataset_a` package, frozen adapter imports and the pilot AST boundary. The
existing pilot tests now characterize reusable modules through package imports.

### 15.11 Behavioral SHA equivalence

The post-refactor offline pilot is byte-identical to the locked reports:

```text
JSON  90ba56dba15237b348f7e3e67d947c43db80a37afcefacd37f1731b52a8248ef
TSV   e645f46270694c777d03d1c92e6e824e0b22a0b58ed4c04c1dc34ce4bb30f2be
MD    1211ff7fbac6256bc2760409316a2823a362cf7f09993156605241af0ea677b6
```

No report schema or portable path exception was introduced.

### 15.12 Exact 8-to-48 operation

An authorized future run loads 48 `CandidateContext` values, supplies the bound
`DerivationConfig` and a different output profile/root, then calls the same
`run_derivation` and `write_reports` functions. The parameterized architecture test
executes this 48-context orchestration without running DERIVE-48 scientific work.
No stage implementation changes with panel size.

### 15.13 Remaining technical debt

The fragment adapter still calls private frozen P0 validators because changing A1–A8
is outside scope. The compatibility report projection retains historical pilot field
names. The pilot wrapper remains longer than a new-clean-project CLI because it must
adapt several immutable historical acquisition ledgers; moving that adapter would
require a separately reviewed module/scope change. None changes current scientific
semantics or requires a second pipeline.

### 15.14 Pilot script line count and validation

The pilot entrypoint decreased from 1,599 to 681 lines (the exact formatted count may
vary by mechanical formatting); all removed scientific work now resides in reusable
capability modules. Final validation completed with 62 focused architecture/pilot
tests, 484 Dataset-A-scale tests, 845 full tests, Ruff on all touched Python, source
compilation, JSON/TSV validation, CWD-independent execution and scoped whitespace
checks. Expected raw-read errors and frozen model-artifact errors also have explicit
causal-stage regressions. DERIVE-48, P0/P1/P2, evaluators, ProteinMPNN and network
access were not run.
