"""Run the Dataset A round-1 eligibility census against real local pair data.

See docs/Dual-UQ_Dataset-A_实验设计方案_v0.1.md §5.0 and
docs/DATASET_A_PIPELINE_HANDOFF_A8.md. This only touches proteins that
already have real inputs on local disk (no AFDB/PDB download), and it does
not decide the pending D1/D2 protocol questions -- it only measures the
failure_code distribution those decisions need.
"""

from __future__ import annotations

import json
from pathlib import Path

from dual_uq.dataset_a_scale.census import CENSUS_CONFIG, run_round1_census, write_round1_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAIRS_ROOT = PROJECT_ROOT / "data/processed/pairs"
CANDIDATE_LIFECYCLE_PATH = PROJECT_ROOT / "reports/candidate_lifecycle.csv"
REPLACEMENT_LIFECYCLE_PATH = PROJECT_ROOT / "reports/replacement_candidate_lifecycle.csv"
CENSUS_STAGE_ROOT = PROJECT_ROOT / "reports/dataset_a_census/round1_stage_outputs"
REPORT_PATH = PROJECT_ROOT / "reports/dataset_a_census/round1_report.json"


def main() -> None:
    report = run_round1_census(
        pairs_root=PAIRS_ROOT,
        candidate_lifecycle_path=CANDIDATE_LIFECYCLE_PATH,
        replacement_lifecycle_path=REPLACEMENT_LIFECYCLE_PATH,
        project_root=PROJECT_ROOT,
        census_stage_root=CENSUS_STAGE_ROOT,
        config=CENSUS_CONFIG,
    )
    write_round1_report(report, REPORT_PATH)
    print(f"Wrote {REPORT_PATH}")
    print(json.dumps(report.summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
