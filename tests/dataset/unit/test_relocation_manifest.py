from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from dual_uq.core.paths import ProjectPaths
from dual_uq.dataset.releases.relocation import (
    RelocationError,
    RelocationManifest,
    RelocationRecord,
    load_relocation_manifest,
    render_relocation_manifest,
    verify_relocation_manifest,
)

SHA_A = "a" * 64
SHA_B = "b" * 64


def _repository(root: Path) -> Path:
    (root / "src/dual_uq").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    return root


def _record(
    *,
    resource_id: str = "dataset_a:report:round1:v1",
    legacy_path: str = "reports/dataset_a_census/round1_report.json",
    canonical_path: str = "artifacts/dataset/reports/census/round1_report.json",
    content_sha256: str = SHA_A,
    scientific_content_changed: bool = False,
) -> RelocationRecord:
    return RelocationRecord(
        logical_resource_id=resource_id,
        legacy_path=legacy_path,
        canonical_path=canonical_path,
        content_sha256=content_sha256,
        mode="byte_identical_copy_with_legacy_read_compatibility",
        scientific_content_changed=scientific_content_changed,
    )


def test_manifest_serialization_is_deterministic_and_resource_sorted() -> None:
    first = _record()
    second = _record(
        resource_id="dataset_a:report:second:v1",
        legacy_path="reports/legacy_scale/second.json",
        canonical_path="artifacts/dataset/reports/second.json",
        content_sha256=SHA_B,
    )

    forward = RelocationManifest(records=(first, second))
    reverse = RelocationManifest(records=(second, first))

    assert render_relocation_manifest(forward) == render_relocation_manifest(reverse)
    payload = json.loads(render_relocation_manifest(forward))
    assert [row["logical_resource_id"] for row in payload["records"]] == [
        first.logical_resource_id,
        second.logical_resource_id,
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("legacy_path", "/tmp/legacy.json"),
        ("canonical_path", "C:\\data\\canonical.json"),
        ("canonical_path", "C:/data/canonical.json"),
        ("legacy_path", "reports/../outside.json"),
    ],
)
def test_record_rejects_nonportable_paths(field: str, value: str) -> None:
    kwargs = {field: value}

    with pytest.raises(RelocationError) as exc_info:
        _record(**kwargs)

    assert exc_info.value.code == "nonportable_path"


def test_record_rejects_invalid_hash_and_scientific_change() -> None:
    with pytest.raises(RelocationError) as hash_error:
        _record(content_sha256="ABC")
    assert hash_error.value.code == "invalid_content_sha256"

    with pytest.raises(RelocationError) as science_error:
        _record(scientific_content_changed=True)
    assert science_error.value.code == "scientific_content_change_forbidden"


def test_manifest_rejects_duplicate_identity_and_destination() -> None:
    first = _record()
    duplicate_id = _record(canonical_path="artifacts/dataset/reports/other.json")
    duplicate_destination = _record(
        resource_id="dataset_a:report:other:v1",
        legacy_path="reports/legacy_scale/other.json",
    )

    with pytest.raises(RelocationError) as id_error:
        RelocationManifest(records=(first, duplicate_id))
    assert id_error.value.code == "duplicate_logical_resource_id"

    with pytest.raises(RelocationError) as destination_error:
        RelocationManifest(records=(first, duplicate_destination))
    assert destination_error.value.code == "duplicate_canonical_path"


def test_verifier_accepts_only_byte_identical_source_and_destination(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path / "clone")
    project = ProjectPaths.discover(project_root=repository)
    content = b'{"candidate_count": 48}\n'
    import hashlib

    digest = hashlib.sha256(content).hexdigest()
    record = _record(content_sha256=digest)
    legacy = project.resolve_logical(record.legacy_path)
    canonical = project.resolve_logical(record.canonical_path)
    legacy.parent.mkdir(parents=True)
    canonical.parent.mkdir(parents=True)
    legacy.write_bytes(content)
    canonical.write_bytes(content)

    result = verify_relocation_manifest(RelocationManifest(records=(record,)), project)

    assert result == {
        "schema_version": "dataset.relocation-verification.v1",
        "record_count": 1,
        "verified": [
            {
                "logical_resource_id": record.logical_resource_id,
                "legacy_path": record.legacy_path,
                "canonical_path": record.canonical_path,
                "content_sha256": digest,
                "status": "verified_byte_identical",
            }
        ],
    }
    assert str(tmp_path) not in json.dumps(result)

    canonical.write_bytes(b"changed\n")
    with pytest.raises(RelocationError) as mismatch_error:
        verify_relocation_manifest(RelocationManifest(records=(record,)), project)
    assert mismatch_error.value.code == "content_hash_mismatch"


def test_manifest_render_and_load_round_trip_without_absolute_paths(tmp_path: Path) -> None:
    manifest = RelocationManifest(records=(_record(),))
    manifest_path = tmp_path / "relocation.json"
    manifest_path.write_bytes(render_relocation_manifest(manifest))

    loaded = load_relocation_manifest(manifest_path)

    assert loaded == manifest
    assert str(tmp_path) not in render_relocation_manifest(loaded).decode("utf-8")


def test_relocation_schema_declares_portable_immutable_contract() -> None:
    schema_path = (
        Path(__file__).resolve().parents[3]
        / "configs/dataset/schemas/relocation_manifest.yaml"
    )
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))

    assert schema["properties"]["schema_version"]["const"] == "dataset.relocation.v1"
    record = schema["properties"]["records"]["items"]
    assert record["properties"]["scientific_content_changed"]["const"] is False
    assert record["properties"]["content_sha256"]["pattern"] == "^[0-9a-f]{64}$"
    assert record["additionalProperties"] is False
