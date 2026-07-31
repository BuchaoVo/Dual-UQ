# Dual-UQ A0-dev v0.9 Dataset Card

## Status

Dual-UQ is frozen at **A0-dev v0.9** for the current handoff. This is a
development dataset state, not the final balanced A0 benchmark.

The final selection audit is blocked. The required balanced 12-protein main
panel has not been formed, and
`data/manifests/a0_final_panel.tsv` therefore contains a header and zero data
rows.

## Intended use

A0-dev v0.9 may be used for:

- method and interface development;
- diagnostic pipeline validation;
- exploratory mechanism analysis;
- studying overlaps among the four A0 mechanism labels;
- planning additional data acquisition and diagnostic recovery.

It must not be presented as a final frozen held-out benchmark. Results obtained
from this development state do not establish performance on the intended
balanced 12-protein A0 panel.

## Audited composition

All counts below exclude the 1AKE reference unless explicitly stated and match
`reports/a0_final_selection_audit.json`.

| Audit item | Value |
| --- | ---: |
| Summary rows, including 1AKE | 57 |
| Complete classifications | 24 |
| Complete quality-pass diagnostics | 23 |
| Required complete quality-pass diagnostics | 24 |
| Ordinary complete diagnostics | 1 |
| Selected panel rows | 0 |
| Target panel rows | 12 |

Eligible candidates by `primary_category`:

| Primary category | Available | Required |
| --- | ---: | ---: |
| `easy_control` | 15 | 3 |
| `low_confidence_local` | 1 | 3 |
| `high_pae_long_range` | 2 | 3 |
| `high_confidence_state_disagreement` | 4 | 3 |

Multi-label candidates count toward exactly one quota according to their
`primary_category`. The ordinary complete sample contributes to the diagnostic
Gate but cannot fill a mechanism quota.

## Why the panel is blocked

The final audit records:

- `fewer_than_24_complete_quality_pass_diagnostics`;
- `insufficient_low_confidence_local_candidates`;
- `insufficient_high_pae_long_range_candidates`.

Both the diagnostic Gate and the category Gate fail. The current empty panel
is therefore the expected audited output, not a missing result.

## Exclusions and safeguards

The main-panel selector excludes the 1AKE reference and candidates marked as
construct differences, missing-coordinate stress, or unsupported AFDB
fragments. A complete ordinary sample may contribute to the diagnostic count
but never to one of the four main quotas.

Selection is deterministic and independent of input row order. If every Gate
passes in a future dataset revision, candidates are ordered by primary-category
order, then screening index, then pair name.

## Known limitations

- The diagnostic Gate is short by one complete quality-pass candidate.
- The low-confidence-local quota is short by two candidates.
- The high-PAE-long-range quota is short by one candidate.
- Three strict-pass candidates have explicit diagnostic failures in the
  current lifecycle state: two at pair construction and one at geometry.
- The mechanism distribution is intentionally treated as incomplete and
  imbalanced.

## Provenance

The exact A0-dev v0.9 artifact hashes are recorded in
`reports/a0_dev_v0_9_freeze_manifest.json`. The normative selection decision is
`reports/a0_final_selection_audit.json`.

## Model-stage boundary

Future model training is a human-operated activity. Codex does not start
GearNet, ProteinMPNN, Joint-UQ, or any other model training from this handoff.
No checkpoint or training log is part of A0-dev v0.9.
