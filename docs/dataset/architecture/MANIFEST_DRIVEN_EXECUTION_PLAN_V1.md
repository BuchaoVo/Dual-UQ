# Manifest-Driven Dataset Execution Plan V1

Status: engineering implementation plan; no scientific protocol change.

## Objective

Converge the canonical `dual_uq.dataset` namespace on one scalable execution path:

```text
manifest -> deterministic tasks -> shard/chunk execution
         -> record results -> atomic status -> validation -> release
```

Changing a panel from one record to 8, 48, or 213 records changes only the
manifest and execution options. It does not select a different scientific
implementation.

## Contracts

### DatasetTask

`DatasetTask` is an immutable execution identity. It contains a stable
`record_id`, stage name/version, portable input and dependency references,
input/config/dependency digests, and an optional deterministic seed. It does
not contain parsed structures, PAE matrices, or other large stage-local data.

### Planning and sharding

The planner validates record identity once, rejects duplicate `record_id`
values, and orders tasks by a canonical task key. A task's shard is derived
from a stable SHA-256 digest of that key, never from input row order. Batch,
chunk, and shard layout therefore cannot change task identity or scientific
results.

### Execution

The initial executor is serial and consumes an iterable of tasks in bounded
chunks. Each task is independently resumed, executed, validated, and recorded.
A record-level failure is represented as a structured result and does not stop
other records. Stage logic is injected through a stage handler; the runner is
not coupled to multiprocessing, SLURM, or a particular scientific stage.

### Resume

Reuse requires agreement on record/stage identity, stage version, input digest,
config digest, dependency digest, and declared output hashes. Missing or corrupt
outputs invalidate reuse. Failed records follow an explicit retry policy.
Status writes are atomic and execution status remains distinct from scientific
or dataset disposition.

### Manifest and CLI

The canonical entrypoint is:

```text
dual-uq dataset {census,resolve,acquire,derive,validate,release} --manifest PATH
```

The CLI loads a manifest once and delegates to reusable Python APIs. Existing
scientific implementations remain authoritative and are connected through
adapters. Scripts, where retained for compatibility, only delegate to the same
CLI/API and never invoke one another.

### I/O and paths

Reusable code uses `ProjectPaths`/`DatasetPaths`, portable logical references,
and content hashes. It does not consult the current working directory or encode
developer-specific absolute paths. Batch result/status storage is structured,
deterministic, and atomically replaced.

## TDD sequence

1. Add failing tests for empty manifests, duplicate records, canonical task
   identity, deterministic sharding, batching/chunking, ordering independence,
   partial failures, retry policy, drift, corrupt outputs, and batch-size/shard
   equivalence.
2. Add failing CLI tests for all six manifest-only stage commands.
3. Implement the smallest task, planner, executor, status-store, and CLI layers
   that satisfy those tests while reusing existing status/resume models.
4. Verify the existing Dataset and full suites, path scans, Ruff, compileall,
   and DERIVE-PILOT byte digests.

## Repository cleanup policy

- Commit reusable source, tests, default/example configuration, and durable
  engineering documentation.
- Remove from the index only files proven to be reconstructible runtime output,
  machine-bound manifests, or path-bound products. Preserve their local bytes.
- Keep dataset definitions, schema/release identities, and test baselines
  versioned.
- Put general cache/run/log/local-config rules in `.gitignore`.
- Put only machine-specific unrelated research assets in `.git/info/exclude`.
- Commit deletion of obsolete tracked layout files only when their replacement
  or obsolescence is demonstrable.

## Acceptance gates

- One generic manifest-driven execution API and one canonical Dataset CLI.
- Deterministic record identity, sharding, seeds, and output ordering.
- Record-level failure isolation and validated resume.
- Equivalent results for `batch_size=1`, larger batches, reordered manifests,
  and recombined shard layouts.
- No active `dataset_a_scale` references or runtime machine paths.
- No single-record executable entrypoints.
- Existing Dataset/full tests and Ruff pass.
- Existing DERIVE-PILOT JSON/TSV/Markdown SHA-256 values remain unchanged.
- No scientific thresholds, candidate membership, human decisions, or frozen
  report contents are regenerated or edited.
