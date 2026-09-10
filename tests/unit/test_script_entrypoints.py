from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_network_smoke_uses_semantic_maintenance_path() -> None:
    assert (ROOT / "scripts/maintenance/network_smoke.py").is_file()
    assert not (ROOT / "scripts/13a_network_smoke.py").exists()
