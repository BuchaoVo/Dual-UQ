# Legacy Dataset-A census stage outputs

These directories are runtime outputs that predate the canonical run-directory
specification. They remain in place because complete run metadata cannot be
reconstructed without changing path-bound evidence.

| Original path | Files | Size | Generating stage | Rebuild status | Review references | Canonical location for new runs |
| --- | ---: | ---: | --- | --- | --- | --- |
| `reports/dataset_a_census/round1_stage_outputs/` | 95 | 796 KB | Dataset-A round1 P0/P1 census, first run | reproducible in principle, metadata incomplete | superseded but retained as historical evidence | `runs/dataset/census/<run-id>/outputs/` |
| `reports/dataset_a_census/round1_stage_outputs_v2/` | 95 | 796 KB | Dataset-A round1 P0/P1 census, new run ID | reproducible in principle, metadata incomplete | directly referenced by V4 and H2 evidence packets | `runs/dataset/census/<run-id>/outputs/` after reference migration |

The v2 directory remains referenced by historical H2 evidence records and by
individual P0 manifests and locks beneath it. Moving either directory without a
coordinated digest and locator migration is deferred. Current dataset entrypoints
write new runtime state below `runs/dataset/`.
