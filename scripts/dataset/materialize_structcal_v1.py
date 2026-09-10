#!/usr/bin/env python3
"""Materialize the frozen StructCal v1 model-independent release."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dual_uq.dataset.structcal_release import (
    build_structcal_v1_bundle,
    materialize_structcal_v1,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path("artifacts/releases/structcal_v1"),
    )
    parser.add_argument(
        "--mmseqs-binary",
        type=Path,
        default=Path(os.environ.get("MMSEQS_BINARY", "mmseqs")),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    root = args.project_root.expanduser().resolve()
    bundle = build_structcal_v1_bundle(root, mmseqs_binary=args.mmseqs_binary)
    output = materialize_structcal_v1(
        bundle,
        project_root=root,
        release_root=args.release_root,
    )
    print(
        json.dumps(
            {
                "release_root": str(output),
                "status": "RELEASED",
                "summary": bundle.metadata["release_manifest"]["summary"],
                "validation": bundle.metadata["release_manifest"]["validation"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
