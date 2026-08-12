from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_environment_check_lives_under_maintenance_namespace() -> None:
    path = ROOT / "scripts" / "maintenance" / "check_environment.py"
    assert path.is_file()
    spec = importlib.util.spec_from_file_location("check_environment", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.PACKAGES
    assert callable(module.package_version)
