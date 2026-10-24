"""An invalid measurement cannot retain an upstream grade in the CLI report."""

import importlib
import json

import pytest
from click.testing import CliRunner

from ifixai.cli.main import ifixai_cli
from ifixai.core.types import (
    EvidenceItem,
    JudgeErrorKind,
    TestGrade,
    TestResult,
    TestRunResult,
    TestStatus,
)
from ifixai.harness.registry import SPEC_BY_ID


@pytest.mark.parametrize(
    "unanswered", [True, False], ids=["invalid-measurement", "healthy-control"]
)
def test_cli_report_refuses_grade_for_an_invalid_measurement(
    tmp_path, monkeypatch, unanswered
):
    module = importlib.import_module("ifixai.cli.run")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("IFIXAI_TELEMETRY", "0")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    evidence = EvidenceItem(
        test_case_id="B19-owned-probe",
        prompt_sent="Owned question",
        actual_response="" if unanswered else "Owned answer",
        passed=not unanswered,
        extraction_error=JudgeErrorKind.COMMUNICATION if unanswered else None,
        details={"comm_failure": True} if unanswered else {},
    )
    initial_grade = TestGrade.D if unanswered else TestGrade.A
    run = TestRunResult(
        system_name="owned-report-boundary",
        overall_score=0.6 if unanswered else 0.95,
        grade=initial_grade,
        mandatory_minimums_passed=True,
        test_results=[
            TestResult(
                test_id="B19",
                spec=SPEC_BY_ID["B19"],
                evidence=[evidence],
                status=TestStatus.INCONCLUSIVE if unanswered else TestStatus.PASS,
            )
        ],
    )
    calls = []

    async def execute(**kwargs):
        calls.append(kwargs)
        return run

    monkeypatch.setattr(module, "execute_tests", execute)
    result = CliRunner().invoke(
        ifixai_cli,
        [
            "run",
            "--provider",
            "mock",
            "--api-key",
            "unused",
            "--test",
            "B19",
            "--eval-mode",
            "self",
            "--format",
            "json",
            "--min-score",
            "0",
            "--no-telemetry",
            "--no-promo",
            "--no-parallel",
        ],
    )
    reports = list((tmp_path / "ifixai-results").glob("*.json"))
    assert len(reports) == len(calls) == 1, result.output
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["overall"]["grade"] == ("F" if unanswered else "A")
    assert (
        any(
            warning.startswith("run_invalid")
            for warning in report["validation_warnings"]
        )
        is unanswered
    )
