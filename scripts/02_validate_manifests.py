from pathlib import Path

import pandas as pd

from dual_uq.manifests import PROTEIN_COLUMNS, SCORE_COLUMNS, SEQUENCE_COLUMNS, STRUCTURE_COLUMNS


def check(path: Path, expected: list[str]) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    print(f"{path.name}: OK ({len(df)} rows)")

def main() -> None:
    base = Path("data/manifests")
    check(base / "protein_manifest.parquet", PROTEIN_COLUMNS)
    check(base / "structure_manifest.parquet", STRUCTURE_COLUMNS)
    check(base / "sequence_manifest.parquet", SEQUENCE_COLUMNS)
    check(base / "scores.parquet", SCORE_COLUMNS)
if __name__ == "__main__":
    main()
