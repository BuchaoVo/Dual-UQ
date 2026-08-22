"""Explicit construction error categories."""


class ExpectedPairDataError(Exception):
    """Expected absence of data for one pair, safe to isolate as unresolved."""

    def __init__(self, reason: str) -> None:
        if not reason or not reason.strip():
            raise ValueError("pair data error requires a non-empty reason")
        self.reason = reason
        super().__init__(reason)
