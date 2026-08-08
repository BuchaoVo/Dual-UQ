"""Execute a prepared Stage-0-2A request in the model environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dual_uq.dataset.services.proteinmpnn_scoring import (
    ProteinMPNNScoringError,
    execute_formal_runtime_request,
    execute_g2_runtime_request,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--request", type=Path, help="G2 audit request")
    mode.add_argument("--formal-request", type=Path, help="formal protein request")
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--shard-directory", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.formal_request is not None:
            if args.shard_directory is None:
                raise ProteinMPNNScoringError(
                    "invalid_formal_runtime_request",
                    "--shard-directory is required for formal scoring",
                )

            def progress(record: dict[str, object]) -> None:
                print(json.dumps({"event": "shard_complete", **record}), flush=True)

            response = execute_formal_runtime_request(
                request_path=args.formal_request.resolve(),
                response_path=args.response.resolve(),
                shard_directory=args.shard_directory.resolve(),
                project_root=args.project_root.resolve(),
                device_name=args.device,
                batch_size=args.batch_size,
                progress_callback=progress,
            )
        else:
            response = execute_g2_runtime_request(
                request_path=args.request.resolve(),
                response_path=args.response.resolve(),
                project_root=args.project_root.resolve(),
                device_name=args.device,
            )
    except ProteinMPNNScoringError as exc:
        print(
            json.dumps(
                {"status": "BLOCKED", "failure_code": exc.code, "message": str(exc)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "status": "complete",
                "call_count": len(response.get("calls", [])),
                "completed_shards": response.get("completed_shards", 0),
                "request_sha256": response["request_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
