from __future__ import annotations

from pathlib import Path

import pytest

from dual_uq import cli


@pytest.mark.parametrize(
    ("command", "stage"),
    [
        ("census", "census"),
        ("resolve", "resolution"),
        ("acquire", "acquisition"),
        ("derive", "derivation"),
        ("validate", "validation"),
        ("release", "release"),
    ],
)
def test_canonical_dataset_cli_routes_manifest_only_stage_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    stage: str,
) -> None:
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("record_id\nprotein-001\n")
    calls: list[dict[str, object]] = []

    def controlled_run(**kwargs):
        calls.append(kwargs)
        return ()

    monkeypatch.setattr("dual_uq.dataset.cli.run_manifest_stage", controlled_run)

    result = cli.main(
        [
            "dataset",
            command,
            "--manifest",
            str(manifest),
            "--batch-size",
            "3",
            "--chunk-size",
            "5",
            "--shard-index",
            "1",
            "--num-shards",
            "2",
        ]
    )

    assert result == 0
    assert calls == [
        {
            "manifest_path": manifest,
            "stage": stage,
            "batch_size": 3,
            "chunk_size": 5,
            "shard_index": 1,
            "num_shards": 2,
            "retry_failed": False,
        }
    ]


def test_dataset_cli_requires_manifest() -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["dataset", "derive"])

    assert error.value.code == 2


def test_dataset_cli_rejects_invalid_shard_bounds(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("record_id\nprotein-001\n")

    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "dataset",
                "derive",
                "--manifest",
                str(manifest),
                "--shard-index",
                "2",
                "--num-shards",
                "2",
            ]
        )

    assert error.value.code == 2


def test_pyproject_registers_dual_uq_entrypoint() -> None:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"

    assert 'dual-uq = "dual_uq.cli:main"' in pyproject.read_text()
