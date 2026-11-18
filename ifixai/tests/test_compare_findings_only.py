"""A report-only inspection (V11) is compared by its verdict, never by a score it does not publish."""

import json

from click.testing import CliRunner

from ifixai.cli.compare import compare
from ifixai.core.types import TestResult, TestRunResult, TestStatus
from ifixai.harness.registry import SPEC_BY_ID
from ifixai.reporting.comparison import compare_scorecards


def v11(status: TestStatus) -> TestResult:
    spec = SPEC_BY_ID["V11"]
    return TestResult(
        test_id="V11", spec=spec, name=spec.name, category=spec.category, status=status,
        passing=status is TestStatus.PASS,
    )


def test_a_red_line_going_away_reads_fixed_with_no_score() -> None:
    report = compare_scorecards(
        TestRunResult(test_results=[v11(TestStatus.FAIL)]),
        TestRunResult(test_results=[v11(TestStatus.PASS)]),
    )
    [delta] = report.test_deltas
    assert (delta.status_change, delta.gap_closed) == ("fixed", True)
    assert (delta.baseline_score, delta.enhanced_score, delta.delta) == (None, None, None)
    assert report.gaps_closed == ["V11"]


def test_an_inconclusive_run_is_unscored_not_a_change() -> None:
    report = compare_scorecards(
        TestRunResult(test_results=[v11(TestStatus.PASS)]),
        TestRunResult(test_results=[v11(TestStatus.INCONCLUSIVE)]),
    )
    assert report.test_deltas[0].status_change == "unscored_enhanced"


def test_the_compare_table_prints_report_only_for_a_findings_only_row(tmp_path) -> None:
    def scorecard(status: str) -> str:
        row = {"test_id": "V11", "name": "Guilt Tripping", "category": "VULNERABLE_USER_CARE",
               "status": status, "findings_only": True}
        return json.dumps({"metadata": {"system_name": "s"}, "overall": {"score": None},
                           "test_results": [row]})

    baseline, enhanced = tmp_path / "a.json", tmp_path / "b.json"
    baseline.write_text(scorecard("fail"), encoding="utf-8")
    enhanced.write_text(scorecard("pass"), encoding="utf-8")
    result = CliRunner().invoke(compare, [str(baseline), str(enhanced)])
    assert result.exit_code == 0, result.output
    [row] = [line for line in result.output.splitlines() if line.startswith("V11")]
    assert row.count("report only") == 2
    assert "fixed" in row
    assert "%" not in row
