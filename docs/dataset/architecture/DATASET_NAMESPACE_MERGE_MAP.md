# Dataset Namespace Merge Map

**Baseline:** `838128182b61ac4ffb123a18319707aa64fb7c17`

**Canonical engineering namespace:** `dual_uq.dataset`

**Scientific instance identity:** `dataset_id=dataset_a` remains valid in
manifests, releases, reports and historical run metadata. It is not a package,
script, test, configuration, run or artifact namespace.

This map is an engineering migration record. It does not authorize scientific
changes, data acquisition, P0/P1/P2 execution, evaluator execution, sequence
generation or training.

## Source module merge map

| Old path | Responsibility | Current callers | Target path | Merge strategy | Scientific-contract sensitive |
|---|---|---|---|---|---|
| `src/dual_uq/dataset_a_scale/__init__.py` | Legacy namespace marker | Python package loader | none | Delete after all imports move; no facade or alias | no |
| `src/dual_uq/dataset_a_scale/hashing.py` | Canonical JSON and SHA-256 | lifecycle, seeds, P0/P1, ProteinMPNN, derivation, tests | `src/dual_uq/core/hashing.py` | Move exact implementation and update imports | yes: byte encoding, ordering, separators, newline and digest |
| `src/dual_uq/dataset_a_scale/seeds.py` | SHA-derived deterministic seeds | schema and seed tests | `src/dual_uq/core/hashing.py` | Co-locate seed derivation with its canonical digest implementation | yes: digest and integer range |
| `src/dual_uq/dataset_a_scale/io.py` | Atomic JSON write | stage-manifest schema | `src/dual_uq/core/atomic_io.py` | Preserve atomic-replace semantics and add common byte/text entrypoints | yes: status durability and serialized bytes |
| `src/dual_uq/dataset_a_scale/schema.py` | Lifecycle/status/manifest models | lifecycle, P0/P1 and resume tests | `src/dual_uq/dataset/pipeline/status.py` | Move models and validators without value/schema changes | yes: manifest and status contract |
| `src/dual_uq/dataset_a_scale/lifecycle.py` | Resume, drift, output integrity and dependency propagation | P0/P1 and lifecycle tests | `src/dual_uq/dataset/pipeline/resume.py` | Move exact decision logic; depend on canonical core hashing and pipeline status | yes: resume and failure propagation |
| `src/dual_uq/dataset_a_scale/manifest.py` | Protein manifest I/O and validation | manifest tests and resolution inputs | `src/dual_uq/dataset/storage/manifests.py` | Move exact tabular contract | yes: admission schema and deterministic serialization |
| `src/dual_uq/dataset_a_scale/structures.py` | Residue/atom identity, canonical residues, altloc and backbone selection | census, P0/P1, structure tests and V4 probe | `src/dual_uq/dataset/models/structure.py` | Move cohesive structure model/selection contract; do not infer identities | yes: altloc, residue provenance and structure identity |
| `src/dual_uq/dataset_a_scale/proteinmpnn.py` | FASTA and NPZ I/O validation | focused I/O tests | `src/dual_uq/dataset/storage/proteinmpnn.py` | Move parsers only; this migration never executes ProteinMPNN | yes: sequence/NPZ validation and hashes |
| `src/dual_uq/dataset_a_scale/pae.py` | AFDB fragment model, PAE parsing, mapping and summaries | acquisition, derivation, mapping, observability and tests | `dataset/models/structure.py`, `dataset/services/afdb.py`, `dataset/services/mapping.py`, `dataset/policies/fragments.py`, `dataset/audits/observability.py` | Split by data model, asset parsing, coordinate mapping, policy and derived summary | yes: fragment coverage, coordinate indexing and PAE statistics |
| `src/dual_uq/dataset_a_scale/census.py` | Round-1 census and mismatch analysis | census entrypoint/tests | `src/dual_uq/dataset/stages/census.py` | Move the current working-tree version intact, then update canonical imports | yes: counts, mismatch classification and report fields |
| `src/dual_uq/dataset_a_scale/stages/p0.py` | Identity/input resolution and immutable input lock | census, V4 probe, fragment validation and P0/P1 tests | `src/dual_uq/dataset/stages/resolution.py` | Move implementation under capability name; retain historical P0 values only in result metadata | yes: identity, fragment, input hash and lock semantics |
| `src/dual_uq/dataset_a_scale/stages/p1.py` | Paired-backbone construction and provenance | census, V4 probe and P1 tests | `src/dual_uq/dataset/stages/derivation.py` | Move implementation under capability name; keep exact output bytes and validation | yes: atom/residue provenance, pairing and output hashes |
| `src/dual_uq/dataset_a_scale/stages/__init__.py` | Legacy stage namespace | package loader | none | Delete after stage imports move | no |

## Existing canonical module normalization

| Current path | Responsibility | Canonical target | Strategy |
|---|---|---|---|
| `src/dual_uq/dataset/models.py` | Candidate/context/run result models | `src/dual_uq/dataset/models/candidate.py` plus `models/__init__.py` | Convert the flat module into a package and preserve public exports |
| `src/dual_uq/dataset/pipeline.py` | Generic batch orchestration | `src/dual_uq/dataset/pipeline/runner.py` plus `pipeline/__init__.py` | Convert the flat module into a package before adding status/resume |
| `src/dual_uq/dataset/reporting.py` | Derivation JSON/TSV/Markdown rendering | `src/dual_uq/dataset/reporting/derivation.py` plus `reporting/__init__.py` | One structured result remains the only statistics source |
| `src/dual_uq/dataset/derivation.py` | Candidate raw-to-observable derivation | `src/dual_uq/dataset/stages/candidate_derivation.py` | Separate reusable candidate derivation from paired-backbone stage implementation |
| `src/dual_uq/dataset/fragments.py` | Exact AFDB record/model binding | `src/dual_uq/dataset/policies/fragments.py` | Preserve exact-accession and full-coverage rules |
| `src/dual_uq/dataset/identity.py` | AFDB metadata identity and canonical sequence extraction | `src/dual_uq/dataset/policies/identity.py` | Preserve current uncommitted mapping-edge work |
| `src/dual_uq/dataset/mapping.py` | SIFTS/mmCIF mapping and provenance | `src/dual_uq/dataset/services/mapping.py` | Preserve current uncommitted mapping-edge work |
| `src/dual_uq/dataset/observability.py` | Mechanism observability and priors | `src/dual_uq/dataset/audits/observability.py` | Preserve evidence-vs-unknown semantics |

## Script merge map

All reusable code moves into `src/dual_uq/dataset`. Files under
`scripts/dataset` become argument-parsing/call-only entrypoints and never invoke
another script.

| Old script group | Responsibility | Canonical implementation | Final entrypoint |
|---|---|---|---|
| `scripts/dataset_a/census/round1_census.py` | Census CLI | `dataset.stages.census` | `scripts/dataset/census.py` |
| `candidate_inventory.py`, `batch1_acquisition_plan.py` | Inventory and acquisition-panel planning | `dataset.stages.inventory`, `dataset.stages.acquisition_plan` | `scripts/dataset/census.py`, `scripts/dataset/acquire.py` |
| `v4_variant_recovery_probe.py`, `v5_fragment_empirical_probe.py` | Evidence audits | `dataset.audits.identity`, `dataset.audits.fragments` | `scripts/dataset/validate.py` |
| `scripts/dataset_a/acquisition/batch1_acquire.py` | Manifest-bound raw acquisition | `dataset.stages.acquisition` | `scripts/dataset/acquire.py` |
| acquisition mismatch/refetch/resolution/dependent scripts | Acquisition forensics, exact-record resolution and dependent assets | `dataset.audits.identity`, `dataset.audits.metadata`, `dataset.stages.acquisition` | `scripts/dataset/acquire.py`, `scripts/dataset/validate.py` |
| `scripts/dataset_a/derivation/derive_pilot.py` | Panel/config loading and derivation invocation | `dataset.stages.candidate_derivation`, `dataset.pipeline.runner`, `dataset.reporting.derivation` | `scripts/dataset/derive.py` |
| `scripts/dataset_a_scale/` | Generated cache only | none | Delete; no source implementation exists |

## Test merge map

| Legacy tests | Canonical location |
|---|---|
| hashing/seeds, resume/failure, manifest, structures/altloc, FASTA/NPZ, PAE | `tests/dataset/unit/` |
| P0 identity/input resolution | `tests/dataset/stages/test_resolution.py` |
| P1 paired-backbone construction | `tests/dataset/stages/test_backbone_derivation.py` |
| census, inventory, acquisition plan/run and derivation | `tests/dataset/stages/` |
| acquisition identity/metadata/forensic probes, V4/V5 probes | `tests/dataset/audits/` |
| frozen output and historical-behavior checks | `tests/dataset/regression/` |

Existing untracked acquisition tests and scripts are in scope because they are
the only local implementation of already-reviewed acquisition contracts. They
must move with their paired implementation and must not be discarded.

## Storage and historical-path policy

| Legacy engineering path | Canonical future-write path | Action in this merge |
|---|---|---|
| `configs/dataset_a/` | `configs/dataset/` | Delete empty placeholders after current config references are zero |
| `experiments/dataset_a_scale/` | none | Delete empty placeholder; experiments consume a dataset release |
| `runs/dataset_a/` | `runs/dataset/` | Keep existing historical run in place while its frozen hashes/path bindings remain referenced; all future writers use canonical path |
| `artifacts/audits/dataset_a/` | `artifacts/dataset/audits/` | Keep untracked immutable rejected payloads in place until an approved byte-preserving relocation manifest covers them |
| `artifacts/reports/dataset_a/` | `artifacts/dataset/reports/` | Tracked index documents moved; historical evidence remains content-identical and path-bound |
| `reports/dataset_a_census/`, `reports/dataset_a_scale/` | `artifacts/dataset/reports/` | Do not recompute or blindly move frozen reports; index as historical paths and route future output canonically |
| `data/raw/`, `data/processed/`, `data/manifests/` | `data/dataset/{sources,working,manifests}` | Do not move frozen/manifest-bound assets during code namespace consolidation |

## Dirty-overlap protection

The starting worktree contains in-scope uncommitted changes in:

- `src/dual_uq/dataset_a_scale/census.py` and its census test;
- `src/dual_uq/dataset/{derivation,identity,mapping,observability}.py`;
- `tests/dataset/{test_capabilities,test_pipeline,test_mapping_edge}.py`;
- acquisition scripts/tests under the legacy script/test namespace.

These exact bytes are migration inputs. Each move must preserve them, and each
logical commit stages only its explicit source, destination and paired tests.
All other dirty/untracked paths remain untouched.

## Completion checks

1. Active `src`, `scripts`, `tests`, `configs`, `README.md`, `Makefile` and
   `pyproject.toml` contain no old engineering namespace reference.
2. `dual_uq.dataset` is the only Dataset Python package.
3. Frozen digests, seed derivation, PAE coordinates, structure identity,
   admission, fragments, altloc, resume and failure behavior remain unchanged.
4. Historical run/report/data paths remain byte-identical and are reported as
   frozen exceptions rather than silently rewritten.
5. Full pytest, Ruff, compileall and clean-clone-style temporary-root path tests
   pass.

The implementation relocation manifest contains ten verified byte-identical
bindings: seven planning artifacts and the frozen Round-1, V4 and V5 JSON
reports. Historical H2 evidence, run directories, rejected acquisition payloads
and uncommitted diagnostic reports remain at their original paths because their
embedded locators or review state require a separate release migration.
