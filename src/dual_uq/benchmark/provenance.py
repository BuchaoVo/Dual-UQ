"""Portable provenance envelopes shared by canonical construction outputs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Provenance:
    source_artifact_ref: str
    source_artifact_sha256: str | None = None
    source_raw_asset_ref: str | None = None
    source_raw_asset_sha256: str | None = None
    construction_code_identity: str | None = None
    schema_version: str | None = None
    annotation_version: str | None = None
    construction_manifest_ref: str | None = None
    created_at: str | None = None
