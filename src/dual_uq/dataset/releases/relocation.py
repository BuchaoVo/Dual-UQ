"""Contracts for byte-preserving dataset resource relocation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from dual_uq.core.hashing import sha256_file
from dual_uq.core.paths import ProjectPaths

SCHEMA_VERSION = "dataset.relocation.v1"
VERIFICATION_SCHEMA_VERSION = "dataset.relocation-verification.v1"
BYTE_IDENTICAL_MODE = "byte_identical_copy_with_legacy_read_compatibility"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class RelocationError(ValueError):
    """A relocation manifest failed a portable or immutable contract."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _portable_relative_path(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise RelocationError("nonportable_path", f"{field} must be a relative path")
    if "\\" in value:
        raise RelocationError(
            "nonportable_path", f"{field} must use portable POSIX separators"
        )
    path = PurePosixPath(value)
    raw_parts = value.split("/")
    if (
        path.is_absolute()
        or re.match(r"^[A-Za-z]:", value)
        or any(part in {"", ".", ".."} for part in raw_parts)
    ):
        raise RelocationError(
            "nonportable_path", f"{field} must not be absolute or contain traversal"
        )
    return path.as_posix()


def _required_text(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RelocationError("missing_required_field", f"{field} must be non-empty")
    return value.strip()


@dataclass(frozen=True)
class RelocationRecord:
    """One immutable legacy-to-canonical resource binding."""

    logical_resource_id: str
    legacy_path: str
    canonical_path: str
    content_sha256: str
    mode: str
    scientific_content_changed: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "logical_resource_id",
            _required_text(self.logical_resource_id, field="logical_resource_id"),
        )
        object.__setattr__(
            self,
            "legacy_path",
            _portable_relative_path(self.legacy_path, field="legacy_path"),
        )
        object.__setattr__(
            self,
            "canonical_path",
            _portable_relative_path(self.canonical_path, field="canonical_path"),
        )
        if self.legacy_path == self.canonical_path:
            raise RelocationError(
                "identical_relocation_paths",
                "legacy_path and canonical_path must identify different locations",
            )
        if not _SHA256_PATTERN.fullmatch(self.content_sha256):
            raise RelocationError(
                "invalid_content_sha256",
                "content_sha256 must be 64 lowercase hexadecimal characters",
            )
        if self.mode != BYTE_IDENTICAL_MODE:
            raise RelocationError("unsupported_relocation_mode", self.mode)
        if self.scientific_content_changed is not False:
            raise RelocationError(
                "scientific_content_change_forbidden",
                "relocation records cannot describe scientific content changes",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_resource_id": self.logical_resource_id,
            "legacy_path": self.legacy_path,
            "canonical_path": self.canonical_path,
            "content_sha256": self.content_sha256,
            "mode": self.mode,
            "scientific_content_changed": self.scientific_content_changed,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RelocationRecord:
        expected = {
            "logical_resource_id",
            "legacy_path",
            "canonical_path",
            "content_sha256",
            "mode",
            "scientific_content_changed",
        }
        if set(value) != expected:
            raise RelocationError(
                "invalid_record_fields", "relocation record fields do not match schema"
            )
        return cls(**value)


@dataclass(frozen=True)
class RelocationManifest:
    """Deterministically ordered collection of relocation records."""

    records: tuple[RelocationRecord, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise RelocationError("unsupported_schema_version", self.schema_version)
        records = tuple(sorted(self.records, key=lambda row: row.logical_resource_id))
        resource_ids = [row.logical_resource_id for row in records]
        if len(resource_ids) != len(set(resource_ids)):
            raise RelocationError(
                "duplicate_logical_resource_id", "logical_resource_id must be unique"
            )
        canonical_paths = [row.canonical_path for row in records]
        if len(canonical_paths) != len(set(canonical_paths)):
            raise RelocationError(
                "duplicate_canonical_path", "canonical_path must be unique"
            )
        object.__setattr__(self, "records", records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "records": [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RelocationManifest:
        if set(value) != {"schema_version", "records"}:
            raise RelocationError(
                "invalid_manifest_fields", "relocation manifest fields do not match schema"
            )
        records = value["records"]
        if not isinstance(records, list):
            raise RelocationError("invalid_records", "records must be a list")
        return cls(
            schema_version=value["schema_version"],
            records=tuple(RelocationRecord.from_dict(row) for row in records),
        )


def render_relocation_manifest(manifest: RelocationManifest) -> bytes:
    """Render a stable UTF-8 JSON representation with a trailing newline."""
    text = json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False)
    return f"{text}\n".encode()


def load_relocation_manifest(path: Path) -> RelocationManifest:
    """Load and validate a relocation manifest from JSON."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RelocationError("manifest_load_failed", path.name) from exc
    if not isinstance(value, dict):
        raise RelocationError("invalid_manifest", "manifest root must be an object")
    return RelocationManifest.from_dict(value)


def verify_relocation_manifest(
    manifest: RelocationManifest,
    project: ProjectPaths,
) -> dict[str, Any]:
    """Verify both locations against the declared digest without mutating files."""
    verified: list[dict[str, Any]] = []
    for record in manifest.records:
        legacy = project.resolve_logical(record.legacy_path)
        canonical = project.resolve_logical(record.canonical_path)
        if not legacy.is_file() or not canonical.is_file():
            raise RelocationError(
                "relocation_file_missing", record.logical_resource_id
            )
        legacy_digest = sha256_file(legacy)
        canonical_digest = sha256_file(canonical)
        if not (
            legacy_digest == canonical_digest == record.content_sha256
        ):
            raise RelocationError(
                "content_hash_mismatch", record.logical_resource_id
            )
        verified.append(
            {
                "logical_resource_id": record.logical_resource_id,
                "legacy_path": record.legacy_path,
                "canonical_path": record.canonical_path,
                "content_sha256": record.content_sha256,
                "status": "verified_byte_identical",
            }
        )
    return {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "record_count": len(verified),
        "verified": verified,
    }
