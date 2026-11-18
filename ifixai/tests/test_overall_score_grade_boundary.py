"""An overall score exactly on a grade or pass boundary keeps that grade."""

import pytest

from ifixai.core.runner import _build_result
from ifixai.core.types import (
    CategoryScore,
    EvidenceItem,
    TestGrade,
    TestResult,
    TestStatus,
)
from ifixai.harness.registry import SPEC_BY_ID
from ifixai.scoring.category_weights import GRADED_CATEGORY_WEIGHTS
from ifixai.scoring.engine import compute_overall_score
from ifixai.scoring.mandatory_minimums import PASS_THRESHOLD

_ITEMS = 20


def _result(test_id: str, passed_items: int) -> TestResult:
    spec = SPEC_BY_ID[test_id]
    score = passed_items / _ITEMS
    status = TestStatus.PASS if score >= spec.threshold else TestStatus.FAIL
    evidence = [
        EvidenceItem(test_case_id=f"{test_id}-{index}", description="d", expected="e",
                     actual="a", passed=index < passed_items)
        for index in range(_ITEMS)
    ]
    return TestResult(test_id=test_id, spec=spec, name=spec.name, category=spec.category,
                      score=score, threshold=spec.threshold, status=status,
                      passed=status is TestStatus.PASS, passing=status is TestStatus.PASS,
                      evidence=evidence)


def test_exact_ninety_percent_is_an_a():
    # Mandatory minimums are perfect; the weighted mean is exactly 0.9 in exact arithmetic.
    passed = {"B01": 20, "B08": 20, "P01": 20, "B05": 10, "B07": 20, "B10": 14, "B19": 19, "B24": 19}
    result = _build_result([_result(t, k) for t, k in passed.items()],
                           "sys", "1", "fx", "mock", "full")
    assert result.overall_score == pytest.approx(0.9)
    assert result.overall_score >= 0.9
    assert result.grade is TestGrade.A
    assert result.passed is True


@pytest.mark.parametrize("pillar_score", [PASS_THRESHOLD, 0.9, 0.8, 0.7, 0.6])
def test_uniform_pillars_on_a_boundary_stay_on_it(pillar_score):
    pillars = [
        CategoryScore(category=category, score=pillar_score, weight=weight,
                      test_count=1, tests_passed=1, test_ids=[])
        for category, weight in GRADED_CATEGORY_WEIGHTS.items()
    ]
    assert compute_overall_score(pillars, GRADED_CATEGORY_WEIGHTS) >= pillar_score
