import json

from ifixai.core.runner import _build_result
from ifixai.core.types import InspectionCategory, TestResult, TestStatus
from ifixai.reporting.scorecard import generate_json_report


def _report(selected_ids):
    passing = TestResult(
        test_id="B12",
        category=InspectionCategory.MANIPULATION,
        score=1.0,
        threshold=0.9,
        passed=True,
        passing=True,
        status=TestStatus.PASS,
    )
    result = _build_result([passing], "sys", "1", "fx", "mock", "api", selected_ids=selected_ids)
    return result, json.loads(generate_json_report(result))


def test_unscored_run_exports_no_grade():
    # A scoped run that leaves mandatory minimums unselected has no overall score.
    result, report = _report({"B12"})
    assert result.overall_score is None
    assert report["overall"]["score"] is None
    assert report["overall"]["grade"] == "n/a"
    assert report["insights"]["grade"] == "n/a"


def test_scored_run_keeps_its_grade():
    result, report = _report(None)
    assert result.overall_score is not None
    assert report["overall"]["grade"] == result.grade.value
    assert report["insights"]["grade"] == result.grade.value
