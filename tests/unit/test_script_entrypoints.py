from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]


def _load(relative_path: str, module_name: str) -> ModuleType:
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate_summary_main_delegates_to_run(monkeypatch) -> None:
    module = _load(
        "scripts/12_build_a0_candidate_summary.py",
        "candidate_summary_entrypoint",
    )
    observed: dict[str, object] = {}

    def fake_run(args):
        observed["project_root"] = args.project_root
        return 7

    monkeypatch.setattr(module, "run", fake_run)

    assert module.main(["--project-root", "/repo"]) == 7
    assert observed["project_root"] == "/repo"


def test_network_smoke_uses_semantic_maintenance_path() -> None:
    assert (ROOT / "scripts/maintenance/network_smoke.py").is_file()
    assert not (ROOT / "scripts/13a_network_smoke.py").exists()
