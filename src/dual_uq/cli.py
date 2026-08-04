"""Top-level Dual-UQ command-line interface."""

from __future__ import annotations

from collections.abc import Sequence

from dual_uq.dataset.cli import build_parser, dispatch


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    return dispatch(parser.parse_args(argv), parser)


if __name__ == "__main__":
    raise SystemExit(main())
