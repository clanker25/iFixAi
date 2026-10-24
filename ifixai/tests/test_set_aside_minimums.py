"""Two kinds of set-aside for a mandatory minimum. One the agent can't do (the fixture says so) is
not applicable: the gate decides on the rest. One this run can't check (no judge, the purpose
check) withholds the score and grade, says why, and never lets the scoped gate pass the run."""

import pytest

from ifixai.core.applicability import NO_JUDGE_REASON
from ifixai.core.runner import _build_result
from ifixai.core.types import NotApplicable, TestResult, TestRunResult, TestStatus
from ifixai.harness.registry import SPEC_BY_ID
from ifixai.reporting.scorecard import (
    absent_score_reason,
    render_mandatory_minimums,
    scoped_gate_decides,
)
from ifixai.scoring.mandatory_minimums import (
    cap_score_if_minimums_failed,
    check_mandatory_minimums,
)

READ_ONLY = "the fixture lists no tool that can delete or overwrite"


def passed(test_id: str) -> TestResult:
    spec = SPEC_BY_ID[test_id]
    return TestResult(
        test_id=test_id, spec=spec, name=spec.name, category=spec.category, score=1.0,
        threshold=spec.threshold, passed=True, passing=True, status=TestStatus.PASS,
    )


def set_aside(test_id: str, reason: str, kind: str) -> NotApplicable:
    return NotApplicable(test_id=test_id, name=SPEC_BY_ID[test_id].name, reason=reason, kind=kind)


def run_with(p01: NotApplicable) -> TestRunResult:
    return _build_result(
        [passed("B01"), passed("B08"), passed("B16")], "agent", "1", "default", "mock", "standard",
        selected_ids={"B01", "B08", "P01", "B16"}, not_applicable=[p01],
    )


def test_a_minimum_the_agent_cant_do_leaves_the_gate_to_the_rest() -> None:
    gate = check_mandatory_minimums([passed("B01"), passed("B08")], None, not_applicable_ids={"P01"})
    assert (gate["minimums_passed"], gate["minimums_not_run"]) == (True, [])
    assert "P01" not in gate["minimum_status"]
    assert cap_score_if_minimums_failed(0.9, gate["minimums_passed"], gate["minimums_not_run"]) == 0.9

    run = run_with(set_aside("P01", READ_ONLY, "agent"))
    assert run.overall_score is not None
    assert run.mandatory_minimums_not_run == []
    assert f"| P01 | NOT APPLICABLE ({READ_ONLY}) |" in render_mandatory_minimums(run)


def test_a_minimum_this_run_cant_check_withholds_the_score_and_grade() -> None:
    gate = check_mandatory_minimums([passed("B01"), passed("B08")], None, unchecked_ids={"P01"})
    assert gate["minimums_not_run"] == ["P01"]
    assert gate["minimum_status"]["P01"] is TestStatus.INCONCLUSIVE
    assert cap_score_if_minimums_failed(0.9, gate["minimums_passed"], gate["minimums_not_run"]) is None

    run = run_with(set_aside("P01", NO_JUDGE_REASON, "run"))
    assert run.overall_score is None
    assert run.passed is False
    assert run.mandatory_minimums_not_run == ["P01"]
    assert absent_score_reason(run) == (
        f"this run couldn't check safety gate P01 ({NO_JUDGE_REASON}), so there is no headline score."
    )
    assert f"| P01 | NOT RUN ({NO_JUDGE_REASON}) |" in render_mandatory_minimums(run)
    # Its selected, scored inspections all pass, yet the run can't stand on them.
    assert not scoped_gate_decides(run)


@pytest.mark.parametrize("kind", ["agent", "run"])
def test_a_set_aside_minimum_is_never_a_fail(kind: str) -> None:
    run = run_with(set_aside("P01", READ_ONLY if kind == "agent" else NO_JUDGE_REASON, kind))
    assert TestStatus.FAIL not in run.mandatory_minimum_status.values()
    assert run.mandatory_minimums_passed is True
