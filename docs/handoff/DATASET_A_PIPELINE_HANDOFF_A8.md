# Dual-UQ Dataset A Pipeline Handoff after Task A8

## 1. Handoff identity

This document is the normative engineering handoff for the new Dataset A scale
pipeline through Task A8.

```text
Project: /mnt/data/users/zbc/AI4S/ProteinDesign/Dual-UQ
Branch: codex/a0-candidate-pipeline
Handoff date: 2026-08-01
Frozen implementation HEAD: 825110065fabf1637ccc6efe647c3e6325b9a6d2
Last completed task: A8 — P1 Build & Validate Paired Backbones
Next pipeline stage: not started
```

The repository also contains an older, untracked
`CODEX_HANDOFF_DUAL_UQ.md`. It describes an earlier candidate-pipeline state
and is not authoritative for Dataset A A1–A8. Do not overwrite or delete it;
it belongs to the existing dirty workspace.

## 2. Two project states that must not be conflated

### 2.1 A0-dev v0.9

The earlier A0 candidate workflow is frozen as a development dataset. Its
authoritative documents are:

- `docs/DATASET_A0_DEV_CARD.md`;
- `docs/handoff/MODEL_STAGE_HANDOFF.md`;
- `reports/a0_dev_v0_9_freeze_manifest.json`;
- `reports/a0_final_selection_audit.json`.

A0-dev v0.9 is not a balanced final benchmark. The final 12-protein panel is
empty because its diagnostic and category gates did not pass. This state may
support method development and exploratory mechanism analysis, but it must not
be represented as a final held-out evaluation set.

### 2.2 Dataset A scale pipeline

Tasks A1–A8 establish a new, strict, deterministic pipeline foundation. It is
an engineering implementation, not a completed Dataset A release. No real
Dataset A scale run has been launched, no P2 or later stage exists, and no
training job has been started.

The current implemented flow is:

```text
explicit manifest row
        │
        ▼
P0 resolve, validate, and freeze inputs
        │ immutable input lock + frozen residue mapping
        ▼
P1 build and validate paired canonical backbones
        │ PDB backbone + AFDB backbone + atom provenance
        ▼
STOP — no later stage implemented
```

## 3. Completed task and commit ledger

| Task | Contract | Commit |
| --- | --- | --- |
| A1 | Manifest schema foundation | `f715364abe5bf9e5a9854d499c5bbf77d6424171` |
| A2 | Canonical hashing and SHA-derived seeds | `bff94845b8deb27b7631179b3d93accb621d0a47` |
| A3 | Lifecycle, resume, drift, and failure propagation | `b69e88d645f3007cf3efb79fd4a8821176c6c99a` |
| A4 | Structure identity, altloc, and residue provenance | `28f0abfe43fa7b0bd4408d52f65f2ee149dd8cba` |
| A5 | ProteinMPNN FASTA/NPZ validation foundation | `d5dbcc239a0bb9ac4936f23b0acf163b6913112c` |
| A6 | AFDB fragment and PAE coordinate mapping | `789e889359a9cf9f0b96eade0f283598483c3001` |
| A7 | Immutable P0 input resolution and freeze | `580bb339005a689c84e0cd576963165381588274` |
| A8 | Validated P1 paired backbone construction | `825110065fabf1637ccc6efe647c3e6325b9a6d2` |

Each commit is single-purpose. Preserve this property for later stages.

## 4. Source and test map

### 4.1 Shared infrastructure

| File | Responsibility |
| --- | --- |
| `src/dual_uq/dataset_a_scale/manifest.py` | Load, validate, and deterministically write protein manifests |
| `src/dual_uq/dataset_a_scale/hashing.py` | Canonical JSON bytes and SHA-256 for bytes, files, and canonical objects |
| `src/dual_uq/dataset_a_scale/seeds.py` | Stable SHA-derived tool seeds |
| `src/dual_uq/dataset_a_scale/schema.py` | Lifecycle enums, validation records, output declarations, and stage manifests |
| `src/dual_uq/dataset_a_scale/lifecycle.py` | Resume decisions, drift detection, output-integrity checks, and upstream blocking |
| `src/dual_uq/dataset_a_scale/io.py` | Atomic JSON write helper |
| `src/dual_uq/dataset_a_scale/structures.py` | Residue/atom identity, canonical amino acids, altloc selection, and backbone extraction |
| `src/dual_uq/dataset_a_scale/proteinmpnn.py` | Strict single-candidate FASTA and ProteinMPNN NPZ readers/validators |
| `src/dual_uq/dataset_a_scale/pae.py` | AFDB fragment coverage, PAE parsing, coordinate mapping, and long-range summaries |

### 4.2 Implemented stages

| File | Responsibility |
| --- | --- |
| `src/dual_uq/dataset_a_scale/stages/p0.py` | Resolve seven explicit inputs, validate scientific identity, and write an immutable P0 lock |
| `src/dual_uq/dataset_a_scale/stages/p1.py` | Consume P0, build exact PDB/AFDB paired backbones, and retain atom-level provenance |

There is no Dataset A scale CLI or multi-protein runner yet. P0 and P1 are
library APIs. Do not silently reuse an older A0 screening runner as the new
orchestrator; its state and artifact contracts differ.

### 4.3 Tests

```text
tests/dataset_a_scale/test_manifest_schema.py
tests/dataset_a_scale/test_hashing_and_seeds.py
tests/dataset_a_scale/test_resume.py
tests/dataset_a_scale/test_failure_propagation.py
tests/dataset_a_scale/test_structure_identity.py
tests/dataset_a_scale/test_altloc.py
tests/dataset_a_scale/test_fasta_and_npz.py
tests/dataset_a_scale/test_pae_mapping.py
tests/dataset_a_scale/test_p0.py
tests/dataset_a_scale/test_p1.py
```

At A8 HEAD, the verified baseline is:

```text
P1 focused tests: 21 passed
Dataset A tests: 347 passed
Full repository tests: 667 passed
Ruff on A8 files: passed
```

## 5. Non-negotiable engineering contracts

### 5.1 Identity is explicit

Never use a DataFrame row index, path basename, structure row order, or a
guessed residue offset as scientific identity. Relevant identities must be
stored explicitly and included in validation.

### 5.2 File hashes use raw bytes

Physical inputs and declared outputs use SHA-256 over raw bytes. Do not
normalize line endings, use modification time, rely on file size, or use a
path alone as file identity.

Canonical scientific/config identities use `sha256_canonical`, not manually
concatenated strings.

### 5.3 Outputs are immutable

P0 and P1 write new files atomically and refuse to overwrite existing files.
There is no force-overwrite mode. A stage manifest is written only after all
declared stage outputs have been created.

If a process is interrupted after one output is written but before the stage
manifest is written, the directory can contain orphaned partial outputs. A
retry will correctly refuse to overwrite them. A human must audit that stage
directory and move the entire orphaned directory aside before retrying; do not
delete individual files blindly and do not add a force flag.

### 5.4 Resume is validation-based

`evaluate_resume` may return `skipped_validated` only when all of the following
are true:

1. the historical stage manifest is valid and successful;
2. current input and config digests match history;
3. every declared output exists, remains inside the stage directory, is not a
   symlink escape, and matches its raw SHA;
4. current scientific validation passes.

Input or config changes produce `blocked_input_drift`. Corrupt outputs produce
`failed_validation`; they must not be reused.

### 5.5 Failures are structured

Stage failures return:

```text
validation_pass = false
failure_code
failure_message
details
```

Do not replace structured failures with warnings, empty tables, silent residue
drops, or runtime stack traces.

## 6. A1–A6 shared interfaces

### 6.1 Manifest

The canonical protein identity fields are:

```text
protein_id
screening_index
pair_id
mechanism_label
tier
```

`protein_id` is explicit; DataFrame row position is never an identifier.

### 6.2 Hashing and seeds

Use:

```python
from dual_uq.dataset_a_scale.hashing import sha256_canonical, sha256_file
from dual_uq.dataset_a_scale.seeds import derive_seed
```

Seeds must derive from stable scientific identity. Do not use Python's process-
randomized `hash()`.

### 6.3 Lifecycle

Primary APIs:

```python
evaluate_resume(...)
evaluate_upstream_dependencies(...)
verify_declared_outputs(...)
read_stage_manifest(...)
write_stage_manifest(...)
```

Successful upstream statuses are `complete` and `skipped_validated`.

### 6.4 Structure identity

The author identity key is:

```text
(source_id, auth_chain_id, auth_seq_id, insertion_code)
```

Label identity is independent provenance:

```text
(label_chain_id, label_seq_id)
```

`42`, `42A`, and `42B` are three distinct residues. Missing insertion tokens
normalize to an empty string, but a real insertion code is retained.

A4 handles:

- standard amino-acid canonicalization;
- `MSE -> M` while retaining `MSE/HETATM` provenance;
- finite coordinates and occupancy bounds;
- deterministic occupancy/altloc selection;
- rejection of scientifically ambiguous duplicate atoms;
- explicit missing N/CA/C/O reporting.

Do not reimplement these rules in later stages.

### 6.5 ProteinMPNN I/O

A5 validates FASTA and NPZ inputs/outputs, but no ProteinMPNN generation or
training is part of A1–A8. Later code must call these validators rather than
trusting a filename or reshaping an array to fit.

### 6.6 AFDB PAE mapping

A6 requires explicit AFDB fragment coverage and validates matrix dimensions.
UniProt-to-model mapping is derived from the frozen fragment interval. It does
not select a nearest fragment, infer an offset from matrix length, concatenate
fragments, pad, truncate, or reshape inconsistent PAE data.

## 7. P0 contract — Resolve & Freeze Inputs

### 7.1 Public API

```python
from dual_uq.dataset_a_scale.stages.p0 import (
    P0_INPUT_PATH_FIELDS,
    P0Resolution,
    P0RunResult,
    P0ValidationError,
    resolve_p0_inputs,
    run_p0,
)
```

`resolve_p0_inputs` is read-only. `run_p0` validates and writes the immutable
stage.

### 7.2 Required manifest path fields

```text
pair_qc_path
residue_mapping_path
pdb_structure_path
afdb_metadata_path
afdb_model_path
afdb_plddt_path
afdb_pae_path
```

All must be explicit non-empty paths to regular files of the expected type.

### 7.3 Scientific validation

P0 verifies:

- pair QC is `quality_flag == pass`;
- pair identity and all internally recorded paths agree;
- PDB mmCIF identity and atom-site content;
- selected AFDB model identity, version, URL identities, and fragment interval;
- full coverage of the mapped UniProt interval;
- AFDB model residue count;
- pLDDT local positions and model length;
- PAE identity and matrix length through A6;
- frozen residue mapping uniqueness and explicit auth/label provenance.

P0 deliberately validates mapping but does not build a paired structure.

### 7.4 Digest contract

`input_digest` canonicalizes:

- manifest identity;
- each logical input name;
- original and project-relative paths;
- each raw file SHA.

`config_digest` includes pipeline version and canonical config. A path-only
change to identical bytes is still input drift.

### 7.5 Outputs

```text
P0_resolve_and_freeze_inputs/
├── stage_manifest.json
├── validation.json
└── outputs/
    ├── frozen_inputs.json
    ├── residue_mapping.tsv
    └── input_lock.json
```

The lock includes model identity/version, fragment interval, input records,
hashes, input/config digests, mapping summary, and validation result.

## 8. P1 contract — Build & Validate Paired Backbones

### 8.1 Public API

```python
from dual_uq.dataset_a_scale.stages.p1 import (
    P1Resolution,
    P1RunResult,
    P1ValidationError,
    PairedResidue,
    resolve_p1_inputs,
    run_p1,
)
```

`resolve_p1_inputs` validates and builds results in memory without writing.
`run_p1` writes the immutable stage.

### 8.2 Upstream boundary

P1 accepts a validated P0 stage directory, not arbitrary current mapping or
structure paths. It verifies the P0 stage manifest, all declared P0 outputs,
the P0 lock, and the current raw PDB/AFDB model hashes before pairing.

### 8.3 Output numbering

If frozen mapping already has `output_position`, it must be exactly contiguous
`1..L`. If absent, P1 sorts the frozen mapping by strictly increasing UniProt
position and creates deterministic `1..L` output numbering. Structure row
order never determines output order.

### 8.4 PDB pairing

PDB residues are joined only through:

```text
(auth_chain_id, auth_seq_id, insertion_code)
```

If mapping label identity is present, it must agree with the mmCIF residue. If
mapping label identity is absent, it is not used as a fallback join key; the
actual mmCIF label identity is still retained in provenance.

### 8.5 AFDB pairing

AFDB model positions must be unique, insertion-free, and exactly model-local
`1..N`. A mapped UniProt position connects through:

```text
model_residue_position = uniprot_position - frozen_fragment_start + 1
```

This is an explicit frozen coordinate transformation, not an inferred offset.

### 8.6 Backbone and sequence validation

For every mapped output position, both sources must provide exactly one
selected atom for each of:

```text
N CA C O
```

Mapping canonical AA, PDB canonical AA, and AFDB canonical AA must match at
every output position. P1 never trims to an intersection, drops a mismatched
residue, substitutes `X`, or silently accepts a partial backbone.

### 8.7 PDB writer convention

Both canonical outputs use:

```text
chain: A
residue numbering: output_position 1..L
residue identity: canonical three-letter amino acid
atom order: N, CA, C, O
line endings: LF
```

These are output identities. Source auth/label numbering remains in the
provenance table and is not overwritten by output numbering.

### 8.8 Outputs

```text
P1_build_and_validate_paired_backbones/
├── stage_manifest.json
├── validation.json
└── outputs/
    ├── pdb_backbone.pdb
    ├── afdb_backbone.pdb
    └── backbone_residue_provenance.tsv
```

The provenance table contains one row per selected atom, or `8 × L` rows:
four atoms for each of two sources at each mapped residue. It records source
residue identity, model-local identity, altloc, occupancy, and coordinates.

## 9. Real-data findings that constrain later work

These were read-only checks. No real P0/P1 stage output was written.

### 9.1 Index 9 — `7kr0_A__P0DTD1`

The mapped UniProt interval and selected AFDB fragment do not overlap with full
required coverage. P0 must return:

```text
unsupported_afdb_fragment
```

Do not relabel this as numbering failure, choose F1, search a nearest fragment,
or guess an AFDB offset.

### 9.2 Index 36 — `3zoj_A__F2QVG4`

The historical mapping artifact lacks the explicit T4 auth/label schema, so
strict P0 currently returns:

```text
missing_mapping_column
```

P1 must not add a legacy fallback to bypass P0. Separately, the real mmCIF
contains equal-occupancy A/B altlocs. A read-only A4 check at author residue
`A:33` selected altloc A deterministically for N/CA/C/O, confirming that the
Index 36 altloc policy is connected correctly once a valid frozen mapping is
available.

### 9.3 Index 103 — `5avd_A__P00772`

P0 read-only resolution succeeds:

```text
selected model: AF-P00772-F1
mapped residues: 240
mapped UniProt interval: 27–266
```

Its mapping lacks `output_position` and label IDs, which P1 supports without
using label fallback: output numbering is derived from frozen UniProt order,
and structure label provenance is retained.

However, the strict P1 sequence-equivalence check finds one real mismatch:

```text
output_position: 66
UniProt position: 92
mapping canonical AA: D
PDB canonical AA: N
failure: pdb_amino_acid_mismatch
```

This is consistent with the earlier sequence identity of approximately
0.9958. It is not an implementation crash. Under the current A8 contract,
Index 103 cannot produce a valid P1 backbone unless an explicitly authorized
future data/protocol decision changes the input or acceptance contract. Do not
trim the position or weaken sequence equivalence locally.

## 10. Rejected legacy behavior

Do not migrate or reintroduce any of the following:

- fixed AFDB F1 selection;
- nearest-fragment selection;
- fragment concatenation;
- hard-coded or inferred AFDB residue offsets;
- auth-to-label or label-to-auth fallback joins;
- chainless joins;
- structure-order pairing;
- nearest-residue matching;
- trimming PDB/AFDB to their intersection;
- dropping residues with missing N/CA/C/O;
- replacing unsupported residues with `X`;
- padding, truncating, or reshaping inconsistent pLDDT/PAE/NPZ data;
- reuse based only on file existence;
- force-overwriting a completed or partial immutable stage.

Older modules outside `dataset_a_scale` may intentionally retain compatibility
behavior for the historical A0 workflow. Their presence does not authorize
using that behavior in Dataset A stages.

## 11. Validation commands

Activate no network and run from the project root:

```bash
conda run -n dual-uq pytest tests/dataset_a_scale/test_p0.py -q
conda run -n dual-uq pytest tests/dataset_a_scale/test_p1.py -q
conda run -n dual-uq pytest tests/dataset_a_scale -q
conda run -n dual-uq pytest -q
```

For changed files, run Ruff explicitly:

```bash
conda run -n dual-uq ruff check <changed-python-files>
```

Before committing:

```bash
git diff --check
git status --short
git diff --cached --check
git diff --cached --name-only
```

After committing:

```bash
git diff --check HEAD^ HEAD
git diff-tree --no-commit-id --name-status -r HEAD
```

On this host, `conda run` occasionally returns no captured output for the full
suite. If that occurs, do not infer success. Re-run using the environment's
pytest executable and require an explicit exit code:

```bash
/home/zbc/data/software/miniconda/envs/dual-uq/bin/pytest -q
```

## 12. Dirty-worktree and Git safety

At this handoff the repository has substantial unrelated dirty state,
including modified reports/manifests, deleted historical files, untracked
experiment directories, logs, and a local ProteinMPNN tree. These files belong
to the user or other workstreams.

Rules for every later task:

1. record `git status --short` before editing;
2. preserve unrelated modifications and deletions;
3. never use `git add .` or `git add -A`;
4. stage only the exact task whitelist;
5. inspect the complete cached name list and diff;
6. create one single-purpose commit;
7. do not amend or rewrite older task commits unless explicitly authorized;
8. never use destructive reset/checkout commands to clean the workspace.

At A8 completion, the Dataset A A8 files themselves were clean and the commit
contained only `p1.py` and `test_p1.py`.

## 13. Collaboration protocol for the next task

The next collaborator should begin in this order:

1. verify that `HEAD` contains commit `8251100` or a descendant;
2. read this document and the next task specification completely;
3. inspect current dirty state without cleaning it;
4. run the Dataset A baseline tests;
5. audit the exact upstream interfaces required by the next stage;
6. write the new focused test first and confirm RED;
7. implement only the requested stage and file scope;
8. run focused, Dataset A, full pytest, Ruff, and diff checks;
9. stage exact files and create a single-purpose commit;
10. stop at the task's stated boundary.

If the next task conflicts with a frozen A1–A8 contract, do not quietly change
the earlier module. Report the conflict and request an explicit protocol
decision. In particular, the Index 103 mismatch is a scientific eligibility
decision, not permission to weaken P1 globally.

## 14. Current boundaries and prohibited actions

As of this handoff:

```text
A1–A8: complete
P0: implemented, not run at Dataset A scale
P1: implemented, not run at Dataset A scale
P2 and later: not started
Dataset A CLI/orchestrator: not implemented
real Dataset A outputs: not produced
model training: not started
```

Do not start or create, without a new explicit task:

- P2 or later stage implementation;
- real Dataset A scale execution;
- AFDB/PDB downloads;
- geometry, RMSD, alignment, displacement, or mechanism classification;
- pLDDT/PAE joins beyond the already implemented validation foundations;
- ProteinMPNN generation or scoring jobs;
- GearNet, ProteinMPNN, Joint-UQ, or any other training;
- checkpoints, model configs, or training logs.

## 15. Handoff checklist

Before accepting responsibility for the next stage, confirm:

```text
[ ] Correct project and branch
[ ] A8 commit present
[ ] Existing dirty files recorded and preserved
[ ] A0-dev and Dataset A states understood as separate
[ ] A1–A8 contracts read
[ ] Index 9/36/103 constraints understood
[ ] No fallback or force-overwrite behavior planned
[ ] Focused RED test prepared before production code
[ ] No real-scale execution or training authorized implicitly
```

This is the stop point after A8. It is a coordination handoff, not authority to
start the next pipeline stage.
