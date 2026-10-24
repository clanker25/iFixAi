import pytest

from ifixai.core.types import EvidenceItem, JudgeErrorKind, TestResult, TestRunResult
from ifixai.reporting.scorecard import render_evidence_appendix


def _appendix(item: EvidenceItem) -> str:
    return render_evidence_appendix(
        TestRunResult(test_results=[TestResult(test_id="B01", evidence=[item])])
    )


@pytest.mark.parametrize("error_kind", ["communication", "contract", "extraction", "budget"])
def test_ungraded_evidence_is_not_labelled_failed(error_kind):
    rendered = _appendix(EvidenceItem(
        test_case_id="owned-ungraded-probe", passed=False,
        extraction_error=JudgeErrorKind(error_kind),
    ))
    assert "**owned-ungraded-probe** [GRADING UNAVAILABLE]" in rendered
    assert "[FAIL]" not in rendered


def test_diagnostic_evidence_is_not_labelled_failed():
    rendered = _appendix(EvidenceItem(
        test_case_id="owned-run-diagnostic", passed=False, is_diagnostic=True,
    ))
    assert "**owned-run-diagnostic** [DIAGNOSTIC]" in rendered
    assert "[FAIL]" not in rendered


@pytest.mark.parametrize(("passed", "label"), [(True, "PASS"), (False, "FAIL")])
def test_graded_evidence_keeps_its_verdict(passed, label):
    rendered = _appendix(EvidenceItem(test_case_id="owned-graded-probe", passed=passed))
    assert f"**owned-graded-probe** [{label}]" in rendered
