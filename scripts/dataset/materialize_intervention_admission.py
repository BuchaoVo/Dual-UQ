"""Materialize frozen Stage-0 intervention admission and executable-subset state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dual_uq.core.hashing import sha256_bytes, sha256_file
from dual_uq.core.paths import ProjectPaths
from dual_uq.dataset.releases.intervention_admission import (
    ORIGINAL_PANEL_SHA256,
    PDR01_PROTOCOL_PATH,
    build_admission_records,
    build_admitted_subset,
    build_release_metadata,
    render_admission_jsonl,
    render_variant_review_packets,
    validate_stage0_admission_bindings,
    write_immutable_release,
)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def materialize(repository_root: Path) -> dict[str, object]:
    """Build all deterministic outputs from the frozen panel and audit values."""
    root = repository_root.resolve()
    stage0_dir = root / "experiments/p2_design_baseline/stage0"
    panel_path = stage0_dir / "stage0_intervention_panel_v1.jsonl"
    admission_path = stage0_dir / "stage0_intervention_admission_v1.jsonl"
    admitted_path = stage0_dir / "stage0_intervention_admitted_v1.jsonl"
    metadata_path = stage0_dir / "stage0_intervention_admission_v1.meta.json"
    config_path = root / "configs/experiments/design_baseline/stage0.yaml"
    relative_panel = panel_path.relative_to(root).as_posix()
    relative_admission = admission_path.relative_to(root).as_posix()

    panel_bytes = panel_path.read_bytes()
    panel_sha256 = sha256_bytes(panel_bytes)
    if panel_sha256 != ORIGINAL_PANEL_SHA256:
        raise ValueError(
            f"Frozen Stage-0 panel SHA256 differs: {panel_sha256}"
        )
    panel_records = [
        json.loads(line) for line in panel_bytes.decode("utf-8").splitlines()
    ]
    validate_stage0_admission_bindings(config_path)
    admission_records = build_admission_records(
        panel_records,
        panel_path=relative_panel,
        panel_sha256=panel_sha256,
    )
    admission_bytes = render_admission_jsonl(admission_records)
    admission_sha256 = sha256_bytes(admission_bytes)
    admitted_records = build_admitted_subset(
        admission_records,
        admission_path=relative_admission,
        admission_sha256=admission_sha256,
    )
    admitted_bytes = render_admission_jsonl(admitted_records)
    admitted_sha256 = sha256_bytes(admitted_bytes)
    packets = render_variant_review_packets(admission_records)
    metadata = build_release_metadata(
        panel_sha256=panel_sha256,
        admission_sha256=admission_sha256,
        admitted_subset_sha256=admitted_sha256,
        pdr01_protocol_sha256=sha256_file(root / PDR01_PROTOCOL_PATH),
        admission_records=admission_records,
        admitted_records=admitted_records,
    )

    outputs = {
        admission_path: admission_bytes,
        admitted_path: admitted_bytes,
        metadata_path: _json_bytes(metadata),
        **{stage0_dir / name: payload for name, payload in packets.items()},
    }
    write_status = {
        path.relative_to(root).as_posix(): write_immutable_release(path, payload)
        for path, payload in outputs.items()
    }
    return {
        "admission_ledger_sha256": admission_sha256,
        "admission_status_counts": metadata["admission_status_counts"],
        "admitted_subset_sha256": admitted_sha256,
        "admitted_record_count": len(admitted_records),
        "original_panel_sha256": panel_sha256,
        "write_status": write_status,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = ProjectPaths.discover(
        project_root=args.project_root.resolve() if args.project_root else None,
        anchor=Path(__file__),
    )
    print(json.dumps(materialize(paths.repository_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
