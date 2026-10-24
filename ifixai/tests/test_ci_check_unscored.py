"""CI on the public API never reads an ungradeable or cut-short run as a pass (#136): a run with no
overall score says so and does not pass, a judge that refuses its key stops the run instead of
returning a result, and `ifixai run` exits 1 on a partial run."""

import http.server
import json
import re
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from click.testing import CliRunner

from ifixai.api import run_inspections, run_selected
from ifixai.cli.main import ifixai_cli
from ifixai.core.types import EvaluationMode, EvaluationPipelineConfig
from ifixai.evaluation.errors import JudgeUnavailableError
from ifixai.judge.config import JudgeConfig

_DIMS = re.compile(r"dimension names must be exactly one of:\s*([^\n]+)")
REJECTED = {"error": {"message": "Incorrect API key provided", "code": 401}}


class _JudgeUntilB20(http.server.BaseHTTPRequestHandler):
    """A judge that passes every grade, and refuses its key once any request is about B20 (all of
    them when the server's `refuse_all` is set)."""

    server: "_JudgeServer"

    def do_POST(self) -> None:
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        system = str(request["messages"][0]["content"])
        if self.server.refuse_all or "B20" in system:
            status, answer = 401, REJECTED
        else:
            match = _DIMS.search(system)
            names = [n.strip() for n in match.group(1).split(",")] if match else []
            verdict = {"dimensions": [{"name": n, "passed": True, "reasoning": "r"} for n in names]}
            status = 200
            answer = {"choices": [{"index": 0, "finish_reason": "stop",
                                   "message": {"role": "assistant", "content": json.dumps(verdict)}}]}
        body = json.dumps(answer).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        pass


class _JudgeServer(http.server.ThreadingHTTPServer):
    refuse_all = True


def _serve(refuse_all: bool) -> Iterator[str]:
    server = _JudgeServer(("127.0.0.1", 0), _JudgeUntilB20)
    server.refuse_all = refuse_all
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(2)


@pytest.fixture
def rejecting_judge() -> Iterator[str]:
    yield from _serve(refuse_all=True)


@pytest.fixture
def judge_refusing_at_b20() -> Iterator[str]:
    yield from _serve(refuse_all=False)


async def test_a_run_that_cannot_grade_has_no_overall_score_and_does_not_pass() -> None:
    # No judge: the safety gates B01 and P01 can't be checked, so there is nothing to grade.
    result = await run_inspections(provider="mock", api_key="x", fixture="default")
    assert result.overall_score is None
    assert result.passed is False
    assert not result.partial
    assert {"B01", "P01"} <= set(result.mandatory_minimums_not_run)


async def test_a_complete_scored_run_still_has_a_score() -> None:
    result = await run_selected(
        {"B01", "B08", "P01", "B16"}, provider="mock", api_key="x", fixture="default",
        judge_config=JudgeConfig(provider="mock"),
        pipeline_config=EvaluationPipelineConfig(mode=EvaluationMode.SINGLE, judge_max_calls=0),
    )
    assert result.overall_score is not None
    assert not result.partial
    assert result.mandatory_minimums_not_run == []


async def test_a_judge_that_refuses_its_key_stops_the_run_instead_of_returning_one(
    rejecting_judge: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("IFIXAI_JUDGE_FAIL_FAST", raising=False)
    with pytest.raises(JudgeUnavailableError, match="refused its API key"):
        await run_inspections(
            provider="mock", api_key="x", fixture="default",
            judge_config=JudgeConfig(provider="openai", model="judge", api_key="k", endpoint=rejecting_judge),
            pipeline_config=EvaluationPipelineConfig(mode=EvaluationMode.SINGLE),
        )


def test_ifixai_run_exits_1_on_a_partial_run(
    judge_refusing_at_b20: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # B16 is graded, then the judge refuses its key on B20: the run stops with what it finished.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("IFIXAI_TELEMETRY", "0")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("OPENAI_BASE_URL", judge_refusing_at_b20)
    monkeypatch.delenv("IFIXAI_JUDGE_FAIL_FAST", raising=False)
    result = CliRunner().invoke(ifixai_cli, [
        "run", "-p", "mock", "-k", "x", "--eval-mode", "single", "--judge-provider", "openai",
        "--judge-model", "judge", "--judge-api-key", "k", "-b", "B16", "-b", "B20", "--min-score", "0",
        "--no-telemetry", "--no-promo", "--no-parallel", "--format", "json", "--output", "out",
    ])
    assert result.exit_code == 1, result.output
    [report] = [json.loads(f.read_text(encoding="utf-8")) for f in (tmp_path / "out").glob("*.json")]
    assert report["partial"] is True
    assert "refused its API key" in report["abort_reason"]
    assert [t["test_id"] for t in report["test_results"]] == ["B16"]
    assert report["not_run_test_ids"] == ["B20"]
