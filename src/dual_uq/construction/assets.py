"""Asset lookup with operational status separate from scientific admission."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class AssetStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    MALFORMED = "MALFORMED"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True, slots=True)
class AssetResolution:
    status: AssetStatus
    path: Path | None = None
    sha256: str | None = None
    admitted: bool | None = None


def resolve_asset(path: str | Path) -> AssetResolution:
    path = Path(path)
    if not path.is_file():
        return AssetResolution(AssetStatus.UNAVAILABLE)
    if path.stat().st_size == 0:
        return AssetResolution(AssetStatus.MALFORMED, path=path)
    import hashlib
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return AssetResolution(AssetStatus.AVAILABLE, path=path, sha256=digest)
