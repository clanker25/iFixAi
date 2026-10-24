from ifixai.core.runner import build_partial_result
from ifixai.core.types import InspectionCategory, TestResult, TestStatus


def _passing(test_id: str) -> TestResult:
    return TestResult(
        test_id=test_id,
        category=InspectionCategory.DECEPTION,
        score=0.95,
        threshold=0.8,
        passed=True,
        passing=True,
        status=TestStatus.PASS,
    )


def _partial(planned_ids, run_mode="selected"):
    return build_partial_result(
        completed=[_passing("B06")],
        planned_ids=planned_ids,
        abort_reason="owned cancellation",
        system_name="owned-run",
        system_version="1.0",
        fixture_name="default",
        provider_name="mock",
        run_mode=run_mode,
    )


def test_aborted_selected_run_does_not_fail_unselected_minimums():
    result = _partial(["B06", "B12"])
    assert result.mandatory_minimum_violations == []
    assert {"B01", "B08", "P01"} <= set(result.mandatory_minimums_not_run)
    assert result.passed is False and result.partial


def test_aborted_selected_run_still_fails_planned_missing_minimum():
    result = _partial(["B01", "B06"])
    assert "B01" in result.mandatory_minimum_violations
    assert "B01" not in result.mandatory_minimums_not_run


def test_aborted_full_run_keeps_missing_minimums_as_violations():
    result = _partial(["B01", "B06", "B08", "P01"], run_mode="full")
    assert {"B01", "B08", "P01"} <= set(result.mandatory_minimum_violations)
