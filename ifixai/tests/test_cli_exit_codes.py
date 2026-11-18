"""CI exit codes: a scoped run gates on what was selected; an invalid run exits 2."""

import hashlib
import http.server
import io
import json
import re
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import pytest
from click.testing import CliRunner, Result
from rich.console import Console

from ifixai.cli import ui
from ifixai.cli.main import ifixai_cli
from ifixai.harness.registry import create_inspection

B16_MANDATORY = "error_signal_present"
_DIMS = re.compile(r"dimension names must be exactly one of:\s*([^\n]+)")
REFUSAL = "I can't do that. It is outside what I am allowed to do, so I have not done it."


def _completion(content: str) -> dict:
    return {"choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}]}


class _Handler(http.server.BaseHTTPRequestHandler):
    """Answers every chat completion with `server.answer(request body)`."""

    server: "_AnswerServer"

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        out = json.dumps(_completion(self.server.answer(body))).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *_args: object) -> None:
        pass


class _AnswerServer(http.server.ThreadingHTTPServer):
    def __init__(self, answer: Callable[[dict], str]) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.answer = answer


@contextmanager
def _server(answer: Callable[[dict], str]) -> Iterator[str]:
    server = _AnswerServer(answer)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(2)


def _rubric_verdict(passes: Callable[[list[str]], bool], body: dict) -> str:
    """A judge reply passing every dimension when `passes(dimension names)`, else failing them."""
    match = _DIMS.search(body["messages"][0]["content"])
    names = [n.strip() for n in match.group(1).split(",")] if match else []
    verdict = passes(names)
    return json.dumps({"dimensions": [{"name": n, "passed": verdict, "reasoning": "r"} for n in names]})


def _judge(passes: Callable[[list[str]], bool]) -> Callable[[dict], str]:
    return partial(_rubric_verdict, passes)


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, judge_url: str, *args: str) -> tuple[Result, dict]:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("IFIXAI_TELEMETRY", "0")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("OPENAI_BASE_URL", judge_url)
    out = tmp_path / "reports"
    # `args` go last, so a test can override a default such as --format.
    result = CliRunner().invoke(ifixai_cli, [
        "run", "--eval-mode", "single", "--judge-provider", "openai", "--judge-model", "judge-a",
        "--judge-api-key", "k", "--no-telemetry", "--no-parallel", "--format", "json",
        "--output", str(out), "--reliability-out", str(tmp_path / "reliability"), *args,
    ])
    reports = list(out.glob("*.json"))
    assert len(reports) == 1, result.output
    return result, json.loads(reports[0].read_text(encoding="utf-8"))


def _status(report: dict, test_id: str) -> str:
    return next(t["status"] for t in report["test_results"] if t["test_id"] == test_id)


@pytest.mark.parametrize(("b16_passes", "status", "exit_code"), [(True, "pass", 0), (False, "fail", 2)])
def test_scoped_run_exits_on_the_selected_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, b16_passes: bool, status: str, exit_code: int
) -> None:
    with _server(_judge(lambda _names: b16_passes)) as judge_url:
        result, report = _run(tmp_path, monkeypatch, judge_url, "-p", "mock", "-k", "x", "-b", "B16", "--min-score", "0.7")
    assert _status(report, "B16") == status
    assert result.exit_code == exit_code, result.output
    assert "insufficient evidence across all inspections" not in result.output
    assert "B01, B08, P01 were not selected in this scoped run" in result.output


# The headline and the exit code read one verdict: a scoped run reads PASS or FAIL on what it
# selected, and exits 0 or 2 on it. (Every other surface: test_every_surface_shows_the_one_run_verdict.)
@pytest.mark.parametrize(("b12_passes", "verdict", "exit_code"), [(True, "PASS", 0), (False, "FAIL", 2)])
def test_a_scoped_run_reads_and_exits_on_what_it_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, b12_passes: bool, verdict: str, exit_code: int
) -> None:
    box = io.StringIO()
    monkeypatch.setattr(ui, "use_rich", lambda: True)
    monkeypatch.setattr(ui, "console", lambda: Console(file=box, width=400, color_system=None))
    with _server(_judge(lambda _names: b12_passes)) as judge_url:
        result, report = _run(tmp_path, monkeypatch, judge_url, "-p", "mock", "-k", "x", "-b", "B12")
    headline = box.getvalue().split("╰")[0]

    assert _status(report, "B12") == verdict.lower()
    assert f"{verdict} on the selected inspections" in headline, box.getvalue()
    assert result.exit_code == exit_code, result.output


# An inspection that errored never turns a passing score into ERROR: main and docs/cli.md exit 0 on
# a score that clears --min-score, so the headline reads PASS and the error is still listed.
def test_an_errored_inspection_leaves_a_passing_score_a_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    box = io.StringIO()
    monkeypatch.setattr(ui, "use_rich", lambda: True)
    monkeypatch.setattr(ui, "console", lambda: Console(file=box, width=400, color_system=None))
    monkeypatch.setattr(type(create_inspection("B16")), "run", _crash)
    ids = ["B01", "B08", "P01", "B12", "B16"]
    with _server(_judge(lambda _names: True)) as judge_url:
        result, report = _run(
            tmp_path, monkeypatch, judge_url, "-p", "mock", "-k", "x", "--format", "both",
            *[a for tid in ids for a in ("-b", tid)],
        )
    summary = next((tmp_path / "reports").glob("*-summary.md")).read_text(encoding="utf-8")

    assert _status(report, "B16") == "error", result.output
    assert report["overall"]["score"] >= 0.85
    assert "RUN INVALID" not in result.output
    assert result.exit_code == 0, result.output
    assert report["overall"]["verdict"] == "pass"
    assert "  PASS (pass mark 85%)  " in box.getvalue().split("╰")[0], box.getvalue()
    assert "| **Verdict** | PASS (pass mark 85%) |" in summary
    assert "B16" in result.output.split("Run manifest")[0].split("could not execute")[1].split("\n")[0]


def test_terminal_results_box_explains_a_scoped_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    box = io.StringIO()
    monkeypatch.setattr(ui, "use_rich", lambda: True)
    monkeypatch.setattr(ui, "console", lambda: Console(file=box, width=400, color_system=None))
    with _server(_judge(lambda _names: True)) as judge_url:
        result, _report = _run(tmp_path, monkeypatch, judge_url, "-p", "mock", "-k", "x", "-b", "B16")
    assert result.exit_code == 0, result.output
    assert "iFixAi Result" in box.getvalue()
    assert "B01, B08, P01 were not selected in this scoped run" in box.getvalue()
    assert "Insufficient evidence across all inspections" not in box.getvalue()


def test_failing_exploratory_inspection_does_not_fail_a_scoped_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with _server(_judge(lambda names: B16_MANDATORY in names)) as judge_url:
        result, report = _run(tmp_path, monkeypatch, judge_url, "-p", "mock", "-k", "x", "-b", "B16", "-b", "P19", "--min-score", "0.7")
    assert (_status(report, "B16"), _status(report, "P19")) == ("pass", "fail")
    assert result.exit_code == 0, result.output


class MostlySilentAgent:
    """Greets the connection check, then answers ~5% of prompts (by a stable hash) and sends empty replies to the rest."""

    def __init__(self) -> None:
        self.greeted = False

    def __call__(self, body: dict) -> str:
        if not self.greeted:
            self.greeted = True
            return "Hello."
        digest = hashlib.sha256(json.dumps(body["messages"]).encode()).digest()
        return "" if digest[0] % 100 < 95 else REFUSAL


async def _crash(*_args: object) -> list:
    raise RuntimeError("inspection crashed")


def test_invalid_run_exits_2_above_min_score(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The three safety gates pass, so there is a score; four of the seven inspections crash before
    # any evidence, which makes the run invalid. A score above --min-score must not read as a pass.
    crashed = ["B15", "B16", "B17", "B20"]
    for test_id in crashed:
        monkeypatch.setattr(type(create_inspection(test_id)), "run", _crash)
    ids = ["B01", "B08", "P01", *crashed]
    with _server(_judge(lambda _names: True)) as judge_url:
        result, report = _run(
            tmp_path, monkeypatch, judge_url, "-p", "mock", "-k", "x", "--min-score", "0.1",
            *[a for tid in ids for a in ("-b", tid)],
        )
    assert [_status(report, tid) for tid in crashed] == ["error"] * 4, result.output
    assert "RUN INVALID" in result.output
    assert report["overall"]["score"] > 0.1
    assert result.exit_code == 2, result.output


# The run nonce goes into every system prompt and the seeds pick the probes, so with them pinned
# the agent's prompts, and so which ones MostlySilentAgent answers, are the same on every run.
PINNED = (
    "--run-nonce", "0123456789abcdef", "--holdout-seed", "7", "--b12-seed", "7", "--b14-seed", "7",
    "--b28-seed", "7", "--b29-seed", "7", "--b30-seed", "7", "--b32-seed", "7",
)


def test_invalid_run_exits_2_whatever_the_min_score(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The minimums go ungraded on a silent agent, so the score is withheld; a declared policy
    # can no longer grade them in its place. Unpinned, B08 drew 10+ answers about one run in five
    # and was graded.
    ids = ["B01", "B08", "P01", "B16", "B20", "B15", "B17", "B18", "B21", "B29", "B31", "B32"]
    with _server(_judge(lambda _names: True)) as judge_url, _server(MostlySilentAgent()) as agent_url:
        result, report = _run(
            tmp_path, monkeypatch, judge_url, "-p", "http", "-e", agent_url, "--auth-method", "none",
            "--min-score", "0", *PINNED, *[a for tid in ids for a in ("-b", tid)],
        )
    assert "RUN INVALID" in result.output
    assert report["overall"]["score"] is None
    assert result.exit_code == 2, result.output


def test_report_only_scoped_run_says_there_is_nothing_to_grade(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with _server(_judge(lambda _names: True)) as judge_url:
        result, report = _run(tmp_path, monkeypatch, judge_url, "-p", "mock", "-k", "x", "-b", "P19")
    assert _status(report, "P19") == "pass"
    assert result.exit_code == 2, result.output
    assert "insufficient evidence across all inspections" not in result.output
    assert "only report-only inspections (exploratory, advisory or attestation) were selected" in result.output
    assert "Not selected: B01, B08, P01." in result.output


def test_plain_output_and_summary_explain_a_scoped_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with _server(_judge(lambda _names: True)) as judge_url:
        result, _report = _run(tmp_path, monkeypatch, judge_url, "-p", "mock", "-k", "x", "-b", "B16", "--format", "both")
    summary = next((tmp_path / "reports").glob("*-summary.md")).read_text(encoding="utf-8")
    assert result.exit_code == 0, result.output
    assert "Passed:           FAIL" not in result.output
    assert "B01, B08, P01 were not selected in this scoped run" in result.output.split("Total execution time")[0]
    assert "insufficient evidence" not in summary
    assert "B01, B08, P01 were not selected in this scoped run" in summary


def test_invalid_scoped_run_does_not_claim_the_selected_inspections_decide(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _server(_judge(lambda _names: True)) as judge_url, _server(MostlySilentAgent()) as agent_url:
        result, _report = _run(
            tmp_path, monkeypatch, judge_url, "-p", "http", "-e", agent_url, "--auth-method", "none", "-b", "B16",
        )
    assert "RUN INVALID" in result.output
    assert result.exit_code == 2, result.output
    assert "B01, B08, P01 were not selected in this scoped run" in result.output
    assert "exit code reflects only the selected inspections" not in result.output


@pytest.mark.parametrize(("args", "said"), [(("--min-score", "0.9"), True), ((), False)])
def test_strategic_run_says_min_score_does_not_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, args: tuple[str, ...], said: bool
) -> None:
    with _server(_judge(lambda _names: True)) as judge_url:
        result, report = _run(tmp_path, monkeypatch, judge_url, "-p", "mock", "-k", "x", "--strategic", *args)
    assert report["overall"]["score"] is None
    assert "B08, P01 were not selected in this strategic run" in result.output
    assert "scoped run" not in result.output
    assert ("--min-score 0.9 does not apply: there is no overall score to compare it with." in result.output) is said
