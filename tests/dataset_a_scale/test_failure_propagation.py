from __future__ import annotations

import pytest

from dual_uq.dataset_a_scale.lifecycle import evaluate_upstream_dependencies
from dual_uq.dataset_a_scale.schema import LifecycleStatus


def test_complete_and_skipped_validated_are_successful_upstreams() -> None:
    decision = evaluate_upstream_dependencies(
        {
            "P1": LifecycleStatus.COMPLETE,
            "P2": LifecycleStatus.SKIPPED_VALIDATED,
        }
    )
    assert decision.ready is True
    assert decision.target_status is None
    assert decision.reason_code == "upstream_ready"
    assert decision.blockers == ()


@pytest.mark.parametrize(
    "status",
    [
        LifecycleStatus.FAILED_VALIDATION,
        LifecycleStatus.FAILED_RUNTIME,
        LifecycleStatus.BLOCKED_INPUT_DRIFT,
        LifecycleStatus.BLOCKED_UPSTREAM,
        LifecycleStatus.BLOCKED_INSUFFICIENT_SITES,
        LifecycleStatus.NOT_RUN_BY_TIER,
        LifecycleStatus.PLANNED,
        LifecycleStatus.RUNNING,
    ],
)
def test_unsuccessful_upstream_blocks_downstream(status: LifecycleStatus) -> None:
    decision = evaluate_upstream_dependencies({"P2": status})
    assert decision.ready is False
    assert decision.target_status is LifecycleStatus.BLOCKED_UPSTREAM
    assert decision.reason_code == "required_upstream_not_ready"
    assert decision.blockers[0].stage == "P2"
    assert decision.blockers[0].status is status


def test_blocker_order_is_deterministic() -> None:
    first = evaluate_upstream_dependencies(
        {
            "P3": LifecycleStatus.FAILED_RUNTIME,
            "P2": LifecycleStatus.FAILED_VALIDATION,
        }
    )
    second = evaluate_upstream_dependencies(
        {
            "P2": LifecycleStatus.FAILED_VALIDATION,
            "P3": LifecycleStatus.FAILED_RUNTIME,
        }
    )
    assert first == second
    assert [blocker.stage for blocker in first.blockers] == ["P2", "P3"]


def test_invalid_upstream_status_is_rejected() -> None:
    with pytest.raises(ValueError, match="status"):
        evaluate_upstream_dependencies({"P2": "done"})
