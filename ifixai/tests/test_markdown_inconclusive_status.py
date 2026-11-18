import json

from ifixai.core.types import (
    InspectionCategory,
    TestGrade,
    TestResult,
    TestRunResult,
    TestStatus,
)
from ifixai.reporting.scorecard import generate_json_report, generate_markdown_report


def _run(result: TestResult) -> TestRunResult:
    return TestRunResult(
        system_name="demo",
        provider="mock",
        fixture_name="default",
        overall_score=None,
        grade=TestGrade.F,
        strategic_score=0.0,
        test_results=[result],
        category_scores=[],
    )


def _b01_row(run: TestRunResult) -> str:
    return next(
        line for line in generate_markdown_report(run).splitlines() if line.startswith("| B01 ")
    )


def _result(status: TestStatus, **extra) -> TestResult:
    return TestResult(
        test_id="B01",
        name="Tool Invocation Governance",
        category=InspectionCategory.FABRICATION,
        score=0.0,
        threshold=1.0,
        passed=False,
        passing=False,
        status=status,
        evidence=[],
        error="provider returned empty content",
        error_message="provider returned empty content",
        **extra,
    )


def test_empty_content_inconclusive_is_not_labelled_error():
    # Same shape BaseTest.run returns for ProviderEmptyContentError.
    run = _run(_result(TestStatus.INCONCLUSIVE, insufficient_evidence=True))
    row = _b01_row(run)
    assert row.endswith("| INCONCLUSIVE |")
    assert "ERROR" not in row
    assert json.loads(generate_json_report(run))["test_results"][0]["status"] == "inconclusive"


def test_errored_inspection_is_still_labelled_error():
    assert _b01_row(_run(_result(TestStatus.FAIL))).endswith("| **ERROR** |")
    assert _b01_row(_run(_result(TestStatus.ERROR))).endswith("| **ERROR** |")
