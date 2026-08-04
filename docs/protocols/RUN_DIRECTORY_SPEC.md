# Run Directory Specification

This document defines the non-overwriting execution record for Dual-UQ pipelines. Run directories are local runtime state and are ignored by Git.

## Canonical layout

```text
runs/<track>/<stage>/<timestamp>_<dataset>_<gitsha>_<configsha>/
├── resolved_config.yaml
├── command.txt
├── versions.json
├── git_state.json
├── input_manifest.tsv
├── input_digests.json
├── status.json
├── stdout.log
├── stderr.log
├── metrics.json
├── outputs/
└── SUCCESS or FAILED
```

The run identifier combines an immutable UTC timestamp, dataset identifier, Git SHA prefix, and resolved-configuration SHA-256 prefix. A new invocation always creates a new directory; it must never overwrite or recycle a prior run identifier.

## Required provenance

- `resolved_config.yaml`: the fully resolved configuration after defaults, experiment configuration, and local overrides are merged.
- `command.txt`: the exact command and arguments, with secrets omitted.
- `versions.json`: Python, package, operating-system, CUDA, accelerator, and third-party dependency versions that apply to the run.
- `git_state.json`: commit SHA, branch, dirty flag, and a path-only dirty-file inventory.
- `input_manifest.tsv`: the exact ordered scientific input rows.
- `input_digests.json`: SHA-256 values for the manifest and every bound external input.
- `status.json`: current stage, timestamps, structured terminal state, and failure code when applicable.
- `metrics.json`: structured summary metrics; raw predictions remain under `outputs/`.

## Terminal markers

A successful run writes all validated outputs and metadata before atomically creating `SUCCESS`. A failed run writes `FAILED` and `failure.json`; `failure.json` records the stage, exception type, structured failure code, message, and last validated checkpoint. A run must never contain both terminal markers.

Interrupted runs contain neither marker and are not eligible for archival or release.

## Resume binding

Resume is allowed only when all of the following match the prior status record:

1. resolved configuration SHA-256;
2. input manifest SHA-256 and ordered row identity;
3. all external input digests;
4. upstream output digests required by the resumed stage;
5. pipeline implementation version and stage contract version.

Any mismatch is input or configuration drift and must create a new run directory or return a structured drift failure. Existing outputs are never silently accepted by filename alone.

## Output and Git policy

- `runs/` is ignored by Git and may contain logs, model outputs, and restart state.
- Curated, reviewed deliverables may be promoted to `artifacts/audits/`, `artifacts/metrics/`, `artifacts/reports/`, or `artifacts/releases/` with explicit digests.
- Model checkpoints belong under `artifacts/checkpoints/` or a run directory and remain untracked unless an explicit release policy says otherwise.
- ProteinMPNN must write new outputs under `runs/design_baseline/`, `runs/structure_uq/`, or `runs/joint_uq/`, not inside `third_party/ProteinMPNN/outputs/`.

## Archival and verification

`scripts/maintenance/archive_run.py` performs a same-filesystem move only for a run with exactly one terminal marker and refuses to overwrite an archive destination. `scripts/maintenance/verify_release.py` validates a release manifest's relative paths and SHA-256 values. Neither helper downloads data or changes scientific content.
