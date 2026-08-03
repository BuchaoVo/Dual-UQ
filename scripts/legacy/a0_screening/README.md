# Legacy A0 screening entrypoints

The numbered root scripts `scripts/00_check_environment.py` through `scripts/23_select_final_a0_panel.py` form the historical A0 candidate pipeline.

## Scope

- Entrypoints remain at their original root paths for Makefile, documentation, lifecycle, and resume compatibility.
- Their configuration lives under `configs/legacy/a0_screening/`.
- They are not duplicated under this directory.

## Maintenance policy

- Fix severe correctness or compatibility bugs only.
- Do not extend this pipeline with new Dataset-A stages.
- Revisit wrappers or a complete move only after the Dataset-A v1 release is frozen.

## Execution outputs

Existing reports and logs retain their historical paths until complete run metadata can be reconstructed.
New experimental execution records must follow `docs/protocols/RUN_DIRECTORY_SPEC.md` once that specification is introduced.
