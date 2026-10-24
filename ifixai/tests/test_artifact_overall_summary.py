"""The HTML artifact's headline score agrees with the JSON and Markdown reports."""

import json

from ifixai.core.runner import _build_result
from ifixai.core.types import TestResult, TestStatus
from ifixai.harness.registry import SPEC_BY_ID
from ifixai.reporting.artifact import _build_payload
from ifixai.reporting.scorecard import absent_score_reason, generate_json_report


def _result(test_id: str, score: float) -> TestResult:
    spec = SPEC_BY_ID[test_id]
    status = TestStatus.PASS if score >= spec.threshold else TestStatus.FAIL
    return TestResult(test_id=test_id, spec=spec, name=spec.name, category=spec.category,
                      score=score, threshold=spec.threshold, status=status,
                      passed=status is TestStatus.PASS, passing=status is TestStatus.PASS)


def _run(scores: dict[str, float], selected_ids=None):
    run_mode = "full" if selected_ids is None else "selected"
    result = _build_result([_result(t, s) for t, s in scores.items()], "agent", "1", "fx",
                           "mock", run_mode, selected_ids=selected_ids)
    summary = _build_payload(result, live=False, transport="mock", sut_model=None,
                             judge_model=None, honesty_note="", previous=None)["summary"]
    return result, summary, json.loads(generate_json_report(result))["overall"]


def test_no_capped_from_when_the_cap_did_not_lower_the_score():
    # A failed minimum with a raw score already under the 60% cap: nothing was capped.
    result, summary, overall = _run({t: 0.4 for t in ("B01", "B08", "P01", "B07", "B12")})
    assert result.overall_score_before_cap is not None
    assert overall["cap_applied"] is False
    assert summary["score_before_cap"] is None


def test_capped_from_shown_when_the_cap_bound():
    result, summary, overall = _run({"B01": 0.5, "B08": 1.0, "P01": 1.0, "B07": 1.0, "B12": 1.0})
    assert overall["cap_applied"] is True
    assert summary["score_before_cap"] == f"{result.overall_score_before_cap * 100:.1f}%"


def test_withheld_score_gives_the_shared_reason():
    # A scoped run that never selected P01 has no headline score; evidence was not lacking.
    ids = {"B01", "B08", "B07", "B12"}
    result, summary, _ = _run({t: 1.0 for t in ids}, selected_ids=ids)
    assert result.overall_score is None
    reason = absent_score_reason(result)
    assert "P01" in reason
    assert summary["overall_pct"] == "n/a"
    assert summary["stability"].lower() == reason.lower()
    assert "insufficient evidence" not in summary["stability"]
    assert summary["score_before_cap"] is None
