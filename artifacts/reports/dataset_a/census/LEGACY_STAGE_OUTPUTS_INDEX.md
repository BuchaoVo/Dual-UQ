# Legacy Dataset-A census stage outputs

These directories are runtime outputs that predate the canonical run-directory specification. They remain in place because complete run metadata cannot be reconstructed without changing path-bound evidence.

| Original path | Files | Size | Generating stage | Rebuild status | Review references | Recommended future location |
| --- | ---: | ---: | --- | --- | --- | --- |
| `reports/dataset_a_census/round1_stage_outputs/` | 95 | 796 KB | Dataset-A round1 P0/P1 census, first run | reproducible in principle, metadata incomplete | superseded but retained as historical evidence | `runs/dataset_a/census/<run-id>/outputs/` |
| `reports/dataset_a_census/round1_stage_outputs_v2/` | 95 | 796 KB | Dataset-A round1 P0/P1 census, new run ID | reproducible in principle, metadata incomplete | directly referenced by V4 and H2 evidence packets | `runs/dataset_a/census/<run-id>/outputs/` after reference migration |

The v2 directory is hard-coded in `scripts/dataset_a/census/round1_census.py` and `scripts/dataset_a/census/v4_variant_recovery_probe.py`; H2 evidence records also cite individual P0 manifests and locks beneath it. Moving either directory without coordinated digest and locator migration is deferred.
