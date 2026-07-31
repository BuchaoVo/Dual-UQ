from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest
import yaml

from dual_uq.a0_selection import (
    PANEL_COLUMNS,
    PRIMARY_CATEGORIES,
    select_final_a0_panel,
)

ROOT = Path(__file__).resolve().parents[1]


def _row(
    index: int,
    category: str,
    *,
    source: str = "screening_pool",
    quality: bool = True,
    eligible: bool = True,
    construct: bool = False,
    missing_coordinate: bool = False,
    unsupported: bool = False,
    extra_labels: tuple[str, ...] = (),
) -> dict[str, object]:
    labels = {
        "easy_control": "is_easy_control",
        "low_confidence_local": "is_low_conf_local",
        "high_pae_long_range": "is_high_pae_long_range",
        "high_confidence_state_disagreement": (
            "is_high_conf_state_disagreement"
        ),
    }
    row: dict[str, object] = {
        "screening_index": index,
        "source": source,
        "pdb_id": f"{index:04d}"[-4:],
        "chain_id": "A",
        "uniprot_id": f"P{index:05d}",
        "pair_name": f"{index:04d}_A__P{index:05d}",
        "preflight_status": (
            "unsupported_afdb_fragment"
            if unsupported
            else "pass_full_length"
        ),
        "classification_status": "complete_classification",
        "full_diagnostic_complete": quality,
        "main_quality_pass": quality,
        "selection_eligible": eligible,
        "primary_category": category,
        "is_easy_control": False,
        "is_low_conf_local": False,
        "is_high_pae_long_range": False,
        "is_high_conf_state_disagreement": False,
        "is_construct_difference": construct,
        "is_missing_coordinate_stress": missing_coordinate,
    }
    if category in labels:
        row[labels[category]] = True
    for label in extra_labels:
        row[label] = True
    return row


def _current_blocked_fixture() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    index = 1
    for category, count in (
        ("easy_control", 15),
        ("low_confidence_local", 1),
        ("high_pae_long_range", 2),
        ("high_confidence_state_disagreement", 4),
    ):
        for _ in range(count):
            rows.append(_row(index, category))
            index += 1
    rows.append(
        _row(
            index,
            "ordinary_or_unclassified",
            eligible=False,
        )
    )
    return pd.DataFrame(rows)


def _passing_fixture() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    index = 1
    for category in PRIMARY_CATEGORIES:
        for _ in range(3):
            rows.append(_row(index, category))
            index += 1
    for _ in range(12):
        rows.append(
            _row(
                index,
                "ordinary_or_unclassified",
                eligible=False,
            )
        )
        index += 1
    return pd.DataFrame(rows)


def test_current_counts_block_both_diagnostic_and_category_gates() -> None:
    result = select_final_a0_panel(_current_blocked_fixture())

    assert result.panel.empty
    assert list(result.panel.columns) == list(PANEL_COLUMNS)
    assert result.audit["complete_quality_pass_diagnostics"] == 23
    assert result.audit["minimum_complete_quality_pass_diagnostics"] == 24
    assert result.audit["diagnostic_gate_pass"] is False
    assert result.audit["category_gate_pass"] is False
    assert result.audit["audit_pass"] is False
    assert result.audit["blocked_reasons"] == [
        "fewer_than_24_complete_quality_pass_diagnostics",
        "insufficient_low_confidence_local_candidates",
        "insufficient_high_pae_long_range_candidates",
    ]


def test_ordinary_complete_counts_for_diagnostic_gate_not_quota() -> None:
    summary = _passing_fixture()

    result = select_final_a0_panel(summary)

    assert result.audit["complete_quality_pass_diagnostics"] == 24
    assert result.audit["ordinary_complete_count"] == 12
    assert result.audit["diagnostic_gate_pass"] is True
    assert result.audit["category_gate_pass"] is True
    assert len(result.panel) == 12
    assert (
        result.panel["primary_category"].value_counts().to_dict()
        == {category: 3 for category in PRIMARY_CATEGORIES}
    )
    assert "ordinary_or_unclassified" not in set(
        result.panel["primary_category"]
    )


def test_multilabel_candidate_only_consumes_primary_category_quota() -> None:
    summary = _passing_fixture()
    target = summary.loc[
        summary["primary_category"].eq("high_pae_long_range")
    ].index[0]
    summary.loc[target, "is_low_conf_local"] = True

    result = select_final_a0_panel(summary)

    assert result.audit["eligible_primary_category_counts"][
        "low_confidence_local"
    ] == 3
    assert result.audit["eligible_primary_category_counts"][
        "high_pae_long_range"
    ] == 3
    assert result.panel["primary_category"].value_counts().to_dict() == {
        category: 3 for category in PRIMARY_CATEGORIES
    }


@pytest.mark.parametrize(
    "excluded",
    ["reference", "construct", "missing_coordinate", "unsupported"],
)
def test_ineligible_sample_types_never_enter_panel(excluded: str) -> None:
    summary = _passing_fixture()
    candidate = _row(999, "easy_control")
    if excluded == "reference":
        candidate["source"] = "reference_pair"
    elif excluded == "construct":
        candidate["is_construct_difference"] = True
    elif excluded == "missing_coordinate":
        candidate["is_missing_coordinate_stress"] = True
    else:
        candidate["preflight_status"] = "unsupported_afdb_fragment"
    summary = pd.concat([pd.DataFrame([candidate]), summary], ignore_index=True)

    result = select_final_a0_panel(summary)

    assert 999 not in set(result.panel["screening_index"])
    assert len(result.panel) == 12


def test_selection_is_deterministic_and_input_order_independent() -> None:
    summary = _passing_fixture()
    forward = select_final_a0_panel(summary)
    reverse = select_final_a0_panel(
        summary.sample(frac=1.0, random_state=73).reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(forward.panel, reverse.panel)
    assert forward.audit == reverse.audit


@pytest.fixture
def selection_script() -> ModuleType:
    path = ROOT / "scripts" / "23_select_final_a0_panel.py"
    spec = spec_from_file_location("select_final_a0_panel_script", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_writes_header_only_panel_and_blocked_audit(
    selection_script: ModuleType,
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "summary.csv"
    panel_path = tmp_path / "panel.tsv"
    audit_path = tmp_path / "audit.json"
    config_path = tmp_path / "selection.yaml"
    _current_blocked_fixture().to_csv(summary_path, index=False)
    config_path.write_text(
        yaml.safe_dump(
            {
                "selection": {
                    "target_total": 12,
                    "target_per_primary_category": {
                        category: 3 for category in PRIMARY_CATEGORIES
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    selection_script.main(
        [
            "--project-root",
            str(tmp_path),
            "--summary",
            str(summary_path),
            "--config",
            str(config_path),
            "--panel",
            str(panel_path),
            "--audit",
            str(audit_path),
        ]
    )

    panel = pd.read_csv(panel_path, sep="\t")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert panel.empty
    assert list(panel.columns) == list(PANEL_COLUMNS)
    assert audit["audit_pass"] is False
    assert audit["selected_panel_rows"] == 0
