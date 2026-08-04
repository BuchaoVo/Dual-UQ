from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_SCALE = "dataset_a" + "_scale"
LEGACY_DOMAIN = "dataset" + "_a"


def _active_text_files() -> list[Path]:
    roots = [
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "scripts",
        PROJECT_ROOT / "tests",
        PROJECT_ROOT / "configs",
    ]
    files = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "Makefile",
        PROJECT_ROOT / "pyproject.toml",
    ]
    for root in roots:
        if not root.exists():
            continue
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix in {".py", ".md", ".toml", ".yaml", ".yml"}
        )
    return sorted(set(files))


def test_only_canonical_dataset_engineering_namespace_remains() -> None:
    forbidden_paths = [
        PROJECT_ROOT / "src/dual_uq" / LEGACY_SCALE,
        PROJECT_ROOT / "scripts" / LEGACY_SCALE,
        PROJECT_ROOT / "scripts" / LEGACY_DOMAIN,
        PROJECT_ROOT / "tests" / LEGACY_SCALE,
        PROJECT_ROOT / "configs" / LEGACY_DOMAIN,
        PROJECT_ROOT / "experiments" / LEGACY_SCALE,
    ]

    assert [path.relative_to(PROJECT_ROOT).as_posix() for path in forbidden_paths if path.exists()] == []


def test_active_code_and_configuration_do_not_reference_legacy_namespace() -> None:
    forbidden = (LEGACY_SCALE, f"dual_uq.{LEGACY_SCALE}")
    matches: list[str] = []
    for path in _active_text_files():
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(token in line for token in forbidden):
                relative = path.relative_to(PROJECT_ROOT).as_posix()
                matches.append(f"{relative}:{line_number}:{line.strip()}")

    assert matches == []
