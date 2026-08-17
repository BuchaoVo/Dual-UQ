"""Validate the locally pinned official ESM-IF1 source and checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dual_uq.models.esm_if1 import validate_esm_if1_source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-root", type=Path, default=Path("third_party/esm"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    checkpoint_value = str(args.checkpoint) if args.checkpoint is not None else os.environ.get("ESM_IF1_CHECKPOINT_PATH")
    if not checkpoint_value:
        parser.error("--checkpoint or ESM_IF1_CHECKPOINT_PATH is required")
    checkpoint = Path(checkpoint_value)
    source_root = args.source_root
    if not source_root.is_absolute():
        source_root = args.project_root / source_root
    observed = validate_esm_if1_source(source_root, checkpoint)
    print(json.dumps(observed, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
