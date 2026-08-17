# Legacy A0 screening entrypoints

The retained numbered root scripts form the historical A0 candidate pipeline;
removed execution surfaces are represented only in Git history.

## Scope

- The retained historical entrypoints are limited to the frozen-regression paths
  `scripts/analysis/diagnose_pair_robustness.py` and
  `scripts/analysis/characterize_disagreement_segments.py`, plus
  `scripts/12_build_a0_candidate_summary.py` and
  `scripts/13_discover_screening_pool.py`.
- Their configuration lives under `configs/legacy/a0_screening/`.
- Removed numbered execution surfaces are preserved by Git history only; they
  are not duplicated under this directory.

## Maintenance policy

- Fix severe correctness or compatibility bugs only in the retained paths.
- Do not extend this pipeline with new Dataset-A stages.
- Do not recreate removed numbered entrypoints or add compatibility wrappers.

## Execution outputs

Existing reports and logs retain their historical paths until complete run metadata can be reconstructed.
New experimental execution records must follow `docs/protocols/RUN_DIRECTORY_SPEC.md` once that specification is introduced.
