"""Portable record, failure, and release models shared by Dataset stages."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any


def _text(value: object, field_name: str) -> str:
    if type(value) is not str or not value or value != value.strip() or "\0" in value:
        raise ValueError(f"{field_name} must be a non-empty canonical string")
    return value


def _portable(value: object, field_name: str) -> str:
    checked = _text(value, field_name)
    path = PurePosixPath(checked)
    if (
        "\\" in checked
        or path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{field_name} must be a portable logical path")
    return path.as_posix()


_SHA256 = re.compile(r"[0-9a-f]{64}")


def _value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_value(item) for item in value)
    return value


def _mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("record attributes must be a mapping")
    return _value(value)


def _sha256(value: str | None, field_name: str, *, required: bool = False) -> str | None:
    if value is None and not required:
        return None
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class ProteinRecord:
    """Stable protein identity plus stage-neutral manifest attributes."""

    record_id: str
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_id", _text(self.record_id, "record_id"))
        object.__setattr__(self, "attributes", _mapping(self.attributes))


@dataclass(frozen=True)
class StructureRecord:
    record_id: str
    structure_id: str
    resource_ref: str
    content_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_id", _text(self.record_id, "record_id"))
        object.__setattr__(
            self, "structure_id", _text(self.structure_id, "structure_id")
        )
        object.__setattr__(
            self, "resource_ref", _portable(self.resource_ref, "resource_ref")
        )
        object.__setattr__(
            self,
            "content_sha256",
            _sha256(self.content_sha256, "content_sha256"),
        )


@dataclass(frozen=True)
class MappingRecord:
    record_id: str
    mapping_id: str
    resource_ref: str
    content_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_id", _text(self.record_id, "record_id"))
        object.__setattr__(self, "mapping_id", _text(self.mapping_id, "mapping_id"))
        object.__setattr__(
            self, "resource_ref", _portable(self.resource_ref, "resource_ref")
        )
        object.__setattr__(
            self,
            "content_sha256",
            _sha256(self.content_sha256, "content_sha256"),
        )


@dataclass(frozen=True)
class FailureRecord:
    """Execution failure evidence; never a scientific disposition."""

    code: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _text(self.code, "failure code"))
        object.__setattr__(self, "message", _text(self.message, "failure message"))
        object.__setattr__(self, "details", _mapping(self.details))


@dataclass(frozen=True)
class DatasetRelease:
    """Portable release identity without machine-local paths."""

    dataset_id: str
    release_id: str
    manifest_ref: str
    manifest_sha256: str
    record_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", _text(self.dataset_id, "dataset_id"))
        object.__setattr__(self, "release_id", _text(self.release_id, "release_id"))
        object.__setattr__(
            self, "manifest_ref", _portable(self.manifest_ref, "manifest_ref")
        )
        object.__setattr__(
            self,
            "manifest_sha256",
            _sha256(self.manifest_sha256, "manifest_sha256", required=True),
        )
        if type(self.record_count) is not int or self.record_count < 0:
            raise ValueError("record_count must be a non-negative integer")
