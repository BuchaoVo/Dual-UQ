from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

import dual_uq.dataset.admission as formal_admission
from dual_uq.core.hashing import sha256_file
from dual_uq.core.paths import ProjectPaths
from dual_uq.dataset.admission import (
    FORMALLY_ADMITTED,
    IDENTITY_CONTRACT_FAIL,
    PENDING_HUMAN_VARIANT_REVIEW,
    FormalAdmissionConfig,
    FormalAdmissionError,
    audit_formal_admission,
    classify_identity_admission,
    materialize_formal_admission,
    validate_a0_gate,
    validate_local_ready_files,
    validate_stage0_admission_ledger,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def repository_result():
    paths = ProjectPaths.discover(project_root=REPOSITORY_ROOT)
    return audit_formal_admission(paths)


def test_scale1a0_gate_requires_frozen_hashes_and_exact_frame_counts() -> None:
    paths = ProjectPaths.discover(project_root=REPOSITORY_ROOT)
    gate = validate_a0_gate(paths, FormalAdmissionConfig())

    assert len(gate.source_frame) == 213
    assert len(gate.evaluated_frame) == 72
    assert gate.unevaluated_count == 141
    assert gate.evaluated_frame["sampling_frame_index"].is_monotonic_increasing
    assert gate.evaluated_frame["pair_id"].is_unique

    bad = replace(
        FormalAdmissionConfig(),
        expected_scale1a0_table_sha256="0" * 64,
    )
    with pytest.raises(FormalAdmissionError, match="SHA256") as exc:
        validate_a0_gate(paths, bad)
    assert exc.value.code == "upstream_sha256_mismatch"


def test_local_ready_missing_file_is_a_blocking_input_regression() -> None:
    paths = ProjectPaths.discover(project_root=REPOSITORY_ROOT)
    frame = pd.DataFrame(
        [
            {
                "pair_id": "missing_A__P00000",
                "canonical_sequence_source_relative": "data/missing/metadata.json",
                "mapping_metadata_source_relative": "data/missing/mapping.parquet",
                "pdb_file_path_relative": "data/missing/pdb.cif",
                "afdb_file_path_relative": "data/missing/model.cif",
            }
        ]
    )
    with pytest.raises(FormalAdmissionError) as exc:
        validate_local_ready_files(frame, paths)
    assert exc.value.code == "blocked_local_input_regression"


@pytest.mark.parametrize(
    (
        "mismatch_count",
        "paired_identity",
        "observed_variants",
        "authorized_variants",
        "expected_status",
    ),
    [
        (0, 1.0, (), None, FORMALLY_ADMITTED),
        (1, 0.995, ("A39V",), None, PENDING_HUMAN_VARIANT_REVIEW),
        (1, 0.995, ("A39V",), ("A39V",), FORMALLY_ADMITTED),
        (4, 0.995, ("A1V", "A2V", "A3V", "A4V"), None, IDENTITY_CONTRACT_FAIL),
        (1, 0.989, ("A39V",), None, IDENTITY_CONTRACT_FAIL),
    ],
)
def test_identity_contract_separates_numerical_pass_from_human_authorization(
    mismatch_count: int,
    paired_identity: float,
    observed_variants: tuple[str, ...],
    authorized_variants: tuple[str, ...] | None,
    expected_status: str,
) -> None:
    decision = classify_identity_admission(
        mismatch_count=mismatch_count,
        paired_identity=paired_identity,
        observed_variants=observed_variants,
        authorized_variants=authorized_variants,
        mismatch_budget=3,
        identity_threshold=0.99,
    )
    assert decision.status == expected_status


def test_identity_failure_uses_pdr01_priority_and_retains_all_failed_predicates() -> None:
    both = classify_identity_admission(
        mismatch_count=4,
        paired_identity=0.98,
        observed_variants=("A1V", "A2V", "A3V", "A4V"),
        authorized_variants=None,
        mismatch_budget=3,
        identity_threshold=0.99,
    )
    identity_only = classify_identity_admission(
        mismatch_count=1,
        paired_identity=0.98,
        observed_variants=("A1V",),
        authorized_variants=None,
        mismatch_budget=3,
        identity_threshold=0.99,
    )

    assert both.reason_code == "sequence_variant_budget_exceeded"
    assert both.failed_predicates == (
        "sequence_variant_budget_exceeded",
        "sequence_identity_below_threshold",
    )
    assert identity_only.reason_code == "sequence_identity_below_threshold"
    assert identity_only.failed_predicates == (
        "sequence_identity_below_threshold",
    )


def test_stage0_ledger_is_hash_gated_before_authorization_loading() -> None:
    paths = ProjectPaths.discover(project_root=REPOSITORY_ROOT)
    bad = replace(
        FormalAdmissionConfig(),
        expected_stage0_admission_sha256="0" * 64,
    )
    with pytest.raises(FormalAdmissionError) as exc:
        validate_stage0_admission_ledger(paths, bad)
    assert exc.value.code == "upstream_sha256_mismatch"


@pytest.mark.parametrize("mutation", ["injected", "duplicate"])
def test_stage0_ledger_rejects_nonfrozen_identity_sets(
    tmp_path: Path, mutation: str
) -> None:
    source = (
        REPOSITORY_ROOT
        / "experiments/p2_design_baseline/stage0/"
        "stage0_intervention_admission_v1.jsonl"
    )
    lines = source.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    if mutation == "injected":
        record["protein_id"] = "injected_A__P00000"
    lines.append(json.dumps(record, sort_keys=True, separators=(",", ":")))
    ledger = tmp_path / "reports/stage0.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")
    paths = ProjectPaths.discover(
        project_root=REPOSITORY_ROOT,
        reports_root=tmp_path / "reports",
    )
    config = replace(
        FormalAdmissionConfig(),
        stage0_admission_ref="reports/stage0.jsonl",
        expected_stage0_admission_sha256=sha256_file(ledger),
    )
    with pytest.raises(FormalAdmissionError) as exc:
        validate_stage0_admission_ledger(paths, config)
    assert exc.value.code == "stage0_ledger_identity_mismatch"


def test_stage0_ledger_rejects_malformed_authorization_state(tmp_path: Path) -> None:
    source = (
        REPOSITORY_ROOT
        / "experiments/p2_design_baseline/stage0/"
        "stage0_intervention_admission_v1.jsonl"
    )
    records = [json.loads(line) for line in source.read_text().splitlines()]
    records[0]["authorized_sequence_variants"] = ["A1V"]
    ledger = tmp_path / "reports/stage0.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        "\n".join(
            json.dumps(record, sort_keys=True, separators=(",", ":"))
            for record in records
        )
        + "\n",
        encoding="utf-8",
    )
    paths = ProjectPaths.discover(
        project_root=REPOSITORY_ROOT,
        reports_root=tmp_path / "reports",
    )
    config = replace(
        FormalAdmissionConfig(),
        stage0_admission_ref="reports/stage0.jsonl",
        expected_stage0_admission_sha256=sha256_file(ledger),
    )
    with pytest.raises(FormalAdmissionError) as exc:
        validate_stage0_admission_ledger(paths, config)
    assert exc.value.code == "invalid_variant_authorization"


def test_common_mask_size_is_not_an_admission_predicate() -> None:
    first = classify_identity_admission(
        mismatch_count=0,
        paired_identity=1.0,
        observed_variants=(),
        authorized_variants=None,
        mismatch_budget=3,
        identity_threshold=0.99,
    )
    second = classify_identity_admission(
        mismatch_count=0,
        paired_identity=1.0,
        observed_variants=(),
        authorized_variants=None,
        mismatch_budget=3,
        identity_threshold=0.99,
    )
    assert first == second
    assert not hasattr(first, "common_mask_count")


def test_frozen_common_mask_helper_disagreement_is_a_global_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = ProjectPaths.discover(project_root=REPOSITORY_ROOT)
    config = FormalAdmissionConfig()
    scale1_gate = validate_a0_gate(paths, config)
    stage0_gate = validate_stage0_admission_ledger(paths, config)
    row = scale1_gate.evaluated_frame.loc[
        scale1_gate.evaluated_frame["pair_id"].eq("5gv8_A__P83686")
    ].iloc[0]
    monkeypatch.setattr(
        formal_admission,
        "paired_common_backbone_positions",
        lambda *args, **kwargs: (-1,),
    )

    with pytest.raises(FormalAdmissionError) as exc:
        formal_admission._evaluate_candidate(
            row,
            paths=paths,
            config=config,
            authorizations=formal_admission._load_authorizations(
                stage0_gate.records
            ),
        )
    assert exc.value.code == "blocked_admission_regression"


def test_repository_census_preserves_72_and_reproduces_stage0(
    repository_result,
) -> None:
    result = repository_result
    assert len(result.census) == 72
    assert result.census["sampling_frame_index"].tolist() == sorted(
        result.census["sampling_frame_index"].tolist()
    )
    assert result.census["pair_id"].is_unique
    assert result.census["admission_status"].notna().all()

    regression = result.manifest["stage0_regression"]
    assert regression["status"] == "PASS"
    assert regression["expected_counts"] == {
        FORMALLY_ADMITTED: 8,
        PENDING_HUMAN_VARIANT_REVIEW: 2,
        IDENTITY_CONTRACT_FAIL: 1,
    }
    assert regression["observed_counts"] == regression["expected_counts"]
    assert regression["disagreements"] == []

    by_id = result.census.set_index("pair_id")
    assert by_id.loc["6jgj_A__P42212", "admission_status"] == IDENTITY_CONTRACT_FAIL
    assert (
        by_id.loc["2ykz_A__P00138", "admission_status"]
        == PENDING_HUMAN_VARIANT_REVIEW
    )
    assert (
        by_id.loc["1ix9_A__P00448", "admission_status"]
        == PENDING_HUMAN_VARIANT_REVIEW
    )


def test_census_outputs_are_scope_safe_and_use_one_terminal_status(
    repository_result,
) -> None:
    result = repository_result
    allowed = set(result.manifest["status_vocabulary"])
    assert set(result.census["admission_status"]).issubset(allowed)
    assert result.summary["sampling_frame_count"] == 213
    assert result.summary["evaluated_count"] == 72
    assert result.summary["unevaluated_count"] == 141
    assert result.summary["unevaluated_status"] == (
        "NOT_EVALUATED_LOCAL_DATA_INCOMPLETE"
    )
    assert result.manifest["ADMISSION_RATE_SCOPE"] == (
        "CONDITIONAL_ON_LOCAL_READINESS"
    )
    assert result.manifest["SCALE1B_ELIGIBILITY_NOT_DEFINED"] is True
    assert result.manifest["SCALE1_SCORING_NOT_STARTED"] is True
    assert result.manifest["scientific_contract"]["common_mask_minimum"] is None
    assert len(result.manifest["candidate_input_bindings"]) == 72
    for binding in result.manifest["candidate_input_bindings"]:
        for asset in (
            "canonical_sequence_source",
            "mapping_source",
            "pdb_mmcif",
            "afdb_structure",
        ):
            assert not Path(binding[asset]["path"]).is_absolute()
            assert len(binding[asset]["sha256"]) == 64
    assert not {
        "rmsd",
        "tm_score",
        "plddt",
        "pae",
        "sdfi",
        "top1",
        "regret",
        "structural_excess",
    }.intersection({column.lower() for column in result.census.columns})

    serialized = json.dumps(
        {
            "summary": result.summary,
            "manifest": result.manifest,
            "census": result.census.astype(object)
            .where(result.census.notna(), None)
            .to_dict(orient="records"),
        },
        allow_nan=False,
    )
    assert str(REPOSITORY_ROOT) not in serialized
    assert "/home/" not in serialized


def test_common_mask_table_is_canonical_position_auditable(repository_result) -> None:
    masks = repository_result.common_masks
    required = {
        "sampling_frame_index",
        "pair_id",
        "canonical_position",
        "canonical_aa",
        "pdb_mapped",
        "afdb_mapped",
        "pdb_backbone_complete",
        "afdb_backbone_complete",
        "common_mask",
    }
    assert required.issubset(masks.columns)
    assert not masks.duplicated(
        ["sampling_frame_index", "canonical_position"]
    ).any()
    assert (
        masks.loc[masks["observability_evaluation_status"].eq("EVALUATED"), "common_mask"]
        == (
            masks.loc[
                masks["observability_evaluation_status"].eq("EVALUATED"),
                "pdb_backbone_complete",
            ]
            & masks.loc[
                masks["observability_evaluation_status"].eq("EVALUATED"),
                "afdb_backbone_complete",
            ]
        )
    ).all()
    assert len(masks) == 22_200
    canonical_lengths = repository_result.census.set_index("pair_id")[
        "canonical_sequence_length"
    ]
    for pair_id, candidate_masks in masks.groupby("pair_id", sort=False):
        assert candidate_masks["canonical_position"].tolist() == list(
            range(1, int(canonical_lengths[pair_id]) + 1)
        )

    census = repository_result.census.set_index("pair_id")
    late_failures = census.loc[
        census["admission_status"].eq("PROVENANCE_FAIL")
        & census["mapped_residue_count"].notna()
    ]
    assert set(late_failures.index) == {"1pq7_A__P35049", "1n9b_A__P30896"}
    for pair_id, row in late_failures.iterrows():
        candidate_masks = masks.loc[masks["pair_id"].eq(pair_id)]
        assert int(candidate_masks["mapping_present"].sum()) == int(
            row["mapped_residue_count"]
        )
        assert set(candidate_masks["observability_evaluation_status"]) == {
            "NOT_EVALUATED_AFTER_PROVENANCE_FAILURE"
        }
        assert candidate_masks["common_mask"].isna().all()

    identity_failures = repository_result.census.loc[
        repository_result.census["admission_status"].eq(IDENTITY_CONTRACT_FAIL)
    ]
    assert set(identity_failures["terminal_reason_code"]) == {
        "sequence_variant_budget_exceeded"
    }
    assert all(
        json.loads(details)["failed_predicates"]
        == [
            "sequence_variant_budget_exceeded",
            "sequence_identity_below_threshold",
        ]
        for details in identity_failures["terminal_reason_details"]
    )


def test_immutable_materialization_reuses_identical_and_rejects_conflict(
    tmp_path: Path, repository_result
) -> None:
    paths = ProjectPaths.discover(
        project_root=REPOSITORY_ROOT,
        reports_root=tmp_path / "reports",
    )
    config = FormalAdmissionConfig(output_root_ref="reports/scale1a1")
    first = materialize_formal_admission(
        repository_result, paths, config=config
    )
    second = materialize_formal_admission(
        repository_result, paths, config=config
    )
    assert set(first["write_status"].values()) == {"created"}
    assert set(second["write_status"].values()) == {"reused_identical"}

    census = tmp_path / "reports/scale1a1/scale1_formal_admission_census.parquet"
    census.write_bytes(b"conflicting bytes")
    with pytest.raises(FormalAdmissionError, match="immutable"):
        materialize_formal_admission(repository_result, paths, config=config)
