from __future__ import annotations

from pathlib import Path

import numpy as np

from dual_uq.models.dynamicmpnn import DynamicMPNNInputCase, DynamicMPNNInputError, DynamicMPNNSampleResult
from scripts.analysis.generate_dynamicmpnn_baseline import load_frozen_cases
from scripts.analysis.score_dynamicmpnn_baseline import _dynamic_shards


ROOT = Path(__file__).resolve().parents[2]


def test_frozen_dynamic_inventory_is_outcome_blind_and_shared() -> None:
    evaluable, unavailable = load_frozen_cases(
        ROOT,
        cases_path=ROOT / "runs/analysis/multistate_baseline/cases_esm.pkl",
        cohort_path=ROOT / "experiments/comparisons/apo_holo/multi_state_baseline/protein_summary.parquet",
        primary_pairs_path=ROOT / "experiments/interventions/biological_states/apo_holo_selective_admission_release/primary_pairs.parquet",
    )

    assert len(evaluable) + len(unavailable) == 1357
    assert len(evaluable) == 366
    assert {row["reason"] for row in unavailable} == {
        "both_conformations_missing_coordinate"
    }
    assert all(case.sequences[0] == case.sequences[1] for case in evaluable)


def test_generation_records_featurizer_incompatibility_and_continues(
    monkeypatch, tmp_path: Path
) -> None:
    cases = []
    for protein_id in ("P00001", "P00002", "P00003"):
        coords = np.zeros((6, 2, 3, 3), dtype=np.float32)
        cases.append(
            DynamicMPNNInputCase(
                protein_id=protein_id,
                pair_id=f"pair-{protein_id}",
                canonical_positions=(1, 2, 3, 4, 5, 6),
                sequences=("ACDEFG", "ACDEFG"),
                coordinates=coords,
                coordinate_present=np.ones((6, 2), dtype=bool),
                chain_ids=("A", "A"),
                structure_sha256=("1" * 64, "2" * 64),
            )
        )

    class FakeAdapter:
        checkpoint_sha256 = "3" * 64
        source_commit = "4" * 40
        _model = None

        def __init__(self, **kwargs):
            del kwargs

        def sample(self, case, *, n_samples, seed):
            if case.protein_id == "P00002":
                raise DynamicMPNNInputError(
                    "dynamicmpnn_featurizer_unavailable",
                    "official featurizer rejected the input",
                )
            return DynamicMPNNSampleResult(
                protein_id=case.protein_id,
                pair_id=case.pair_id,
                seed=seed,
                sequences=("ACDEFG",) * n_samples,
                residue_count=case.residue_count,
                checkpoint_sha256=self.checkpoint_sha256,
                source_commit=self.source_commit,
            )

    monkeypatch.setattr(
        "scripts.analysis.generate_dynamicmpnn_baseline.load_frozen_cases",
        lambda *args, **kwargs: (cases, []),
    )
    monkeypatch.setattr(
        "scripts.analysis.generate_dynamicmpnn_baseline.DynamicMPNNAdapter", FakeAdapter
    )

    from scripts.analysis.generate_dynamicmpnn_baseline import run_generation

    summary = run_generation(
        repository_root=ROOT,
        cases_path=Path("unused"),
        cohort_path=Path("unused"),
        primary_pairs_path=Path("unused"),
        output_dir=tmp_path,
        device="cpu",
        n_samples=2,
    )

    assert summary["evaluable_count"] == 2
    assert summary["unavailable_count"] == 1
    inventory = __import__("pandas").read_parquet(tmp_path / "inventory.parquet")
    rejected = inventory[inventory["protein_id"] == "P00002"].iloc[0]
    assert rejected["status"] == "DYNAMICMPNN_UNAVAILABLE"
    assert rejected["reason"] == "dynamicmpnn_featurizer_unavailable"
    assert set(inventory[inventory["status"] == "GENERATED"]["protein_id"]) == {
        "P00001",
        "P00003",
    }


def test_dynamic_shard_loader_reads_flat_merged_shards(tmp_path: Path) -> None:
    shard = tmp_path / "shards" / "P00001.json"
    shard.parent.mkdir()
    shard.write_text(
        '{"schema_version": "apo_holo_multistate_generation_v1", '
        '"protein_id": "P00001", "records": []}\n',
        encoding="utf-8",
    )

    result = _dynamic_shards(tmp_path)

    assert set(result) == {"P00001"}
