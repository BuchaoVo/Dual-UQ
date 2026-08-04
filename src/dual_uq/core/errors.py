"""Structured errors shared by portable Dual-UQ capabilities."""


class PAEMappingError(ValueError):
    """Structured failure raised when PAE coordinates cannot be mapped safely."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)
