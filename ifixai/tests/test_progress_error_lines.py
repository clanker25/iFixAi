"""An inspection that errored shows as ERROR with a short why on its progress line, never as FAIL (0%)."""

import re

import pytest

from ifixai.cli.orchestrator import BenchmarkProgressDisplay, _progress_callback_plain
from ifixai.core.types import TestResult, TestStatus

_ANSI = re.compile(r"\033\[[0-9;]*m")
LONG = "the agent's adapter raised ValueError: " + "x" * 120


def errored(message: str) -> TestResult:
    return TestResult(
        test_id="B16", name="Silent Failure", score=0.0, passed=False, passing=False,
        status=TestStatus.ERROR, error=message, error_message=message,
    )


def tty_line(result: TestResult) -> str:
    display = BenchmarkProgressDisplay([("B16", "Silent Failure")], [])
    display._results["B16"] = result
    [line] = display._build_lines(final=True)
    return _ANSI.sub("", line)


def plain_line(result: TestResult, capsys: pytest.CaptureFixture[str]) -> str:
    _progress_callback_plain("B16", 1, 1, result)
    return _ANSI.sub("", capsys.readouterr().out.strip())


@pytest.mark.parametrize(
    ("message", "note"),
    [
        ("inspection crashed\nTraceback (most recent call last): ...", "inspection crashed"),
        (LONG, LONG[:77] + "..."),
        ("", "no detail available"),
    ],
)
def test_an_errored_inspection_shows_error_and_why_in_both_paths(
    capsys: pytest.CaptureFixture[str], message: str, note: str
) -> None:
    result = errored(message)
    for line in (tty_line(result), plain_line(result, capsys)):
        assert line.endswith(f"B16 Silent Failure ... ERROR ({note})"), line
        assert "FAIL" not in line and "0%" not in line


def test_a_failed_inspection_still_shows_fail_with_its_score(capsys: pytest.CaptureFixture[str]) -> None:
    result = TestResult(test_id="B16", name="Silent Failure", score=0.4, status=TestStatus.FAIL)
    for line in (tty_line(result), plain_line(result, capsys)):
        assert line.endswith("B16 Silent Failure ... FAIL (40%)"), line
