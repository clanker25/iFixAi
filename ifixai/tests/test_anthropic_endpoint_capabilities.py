"""Native SDK requests must learn temperature support per endpoint and model."""
import json
import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import ThreadingHTTPServer

import pytest

from ifixai.core.types import ChatMessage, ProviderConfig
from ifixai.judge.config import JudgeConfig
from ifixai.judge.evaluator import JudgeEvaluator
from ifixai.tests.test_anthropic_temperature_native import (
    ACCEPTS,
    REJECTS,
    AnthropicProvider,
    FakeMessagesApi,
    adapter,
    message,
    send,
)


@contextmanager
def local_api() -> Iterator[ThreadingHTTPServer]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeMessagesApi)
    server.bodies = []
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        worker.join(2)


@pytest.mark.parametrize("same_provider", [True, False])
async def test_temperature_rejection_at_one_endpoint_does_not_disable_another(
    monkeypatch, same_provider,
):
    monkeypatch.setattr(adapter, "_NO_TEMPERATURE_MODELS", set())
    # Same model name resolves to different capabilities behind the two hosts.
    with local_api() as rejects, local_api() as accepts:
        original_handler = accepts.RequestHandlerClass
        class AcceptAll(original_handler):
            def do_POST(self):
                import json

                from ifixai.tests.test_anthropic_temperature_native import message
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.server.bodies.append(body)
                self.reply(200, message(body["model"]))
        accepts.RequestHandlerClass = AcceptAll
        first = AnthropicProvider()
        second = first if same_provider else AnthropicProvider()
        try:
            assert await send(first, rejects, REJECTS, .25) == "fake reply"
            assert await send(second, accepts, REJECTS, .75) == "fake reply"
            assert await send(second, rejects, REJECTS, .25) == "fake reply"
            assert await send(first, accepts, ACCEPTS, .5) == "fake reply"
        finally:
            await first.aclose()
            if second is not first:
                await second.aclose()
        assert [b.get("temperature") for b in rejects.bodies] == [.25, None, None]
        assert [b.get("temperature") for b in accepts.bodies] == [.75, .5]


async def test_default_api_and_equivalent_trailing_slash_share_learning(monkeypatch):
    import json

    import anthropic
    from anthropic import _base_client
    httpx = getattr(_base_client, "httpx2", None) or _base_client.httpx
    from ifixai.tests.test_anthropic_temperature_native import error, message

    monkeypatch.setattr(adapter, "_NO_TEMPERATURE_MODELS", set())
    bodies = []
    def respond(request):
        assert request.url.host == "api.anthropic.com"
        body = json.loads(request.content)
        bodies.append(body)
        if body["model"] == REJECTS and "temperature" in body:
            return httpx.Response(400, json=error("temperature not supported"))
        return httpx.Response(200, json=message(body["model"]))

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as transport:
        async with anthropic.AsyncAnthropic(api_key="owned-fixture", http_client=transport) as default:
            await adapter._create_message(default, {"model": REJECTS, "max_tokens": 4, "messages": [{"role": "user", "content": "owned"}]}, .25)
            await adapter._create_message(default, {"model": ACCEPTS, "max_tokens": 4, "messages": [{"role": "user", "content": "owned"}]}, .75)
            async with anthropic.AsyncAnthropic(api_key="owned-fixture", base_url="https://api.anthropic.com/", http_client=transport) as equivalent:
                await adapter._create_message(equivalent, {"model": REJECTS, "max_tokens": 4, "messages": [{"role": "user", "content": "owned"}]}, .25)
    assert [body.get("temperature") for body in bodies] == [.25, None, .75, None]


class AcceptsAnyTemperature(FakeMessagesApi):
    """A gateway in front of the same model that takes temperature."""

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.bodies.append(body)
        self.reply(200, message(body["model"]))


async def send_to(provider: AnthropicProvider, endpoint: str | None, model: str) -> str:
    return await provider.send_message(
        [ChatMessage(role="user", content="hi")],
        ProviderConfig(provider="anthropic", endpoint=endpoint, api_key="synthetic-local-key",
                       model=model, timeout=5, max_retries=0, temperature=0.0),
    )


@pytest.mark.parametrize("judge_endpoint_from_env", [False, True])
async def test_accepts_temperature_answers_per_endpoint_for_the_same_model(
    monkeypatch, judge_endpoint_from_env,
):
    monkeypatch.setattr(adapter, "_NO_TEMPERATURE_MODELS", set())
    monkeypatch.setattr(adapter, "_API_ROOTS", {})
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    with local_api() as rejects, local_api() as accepts:
        accepts.RequestHandlerClass = AcceptsAnyTemperature
        agent_endpoint = f"http://127.0.0.1:{rejects.server_port}"
        judge_endpoint = f"http://127.0.0.1:{accepts.server_port}"
        if judge_endpoint_from_env:
            # A judge with no endpoint goes wherever the SDK sends it.
            monkeypatch.setenv("ANTHROPIC_BASE_URL", judge_endpoint)
            judge_endpoint = None
        agent, judge = AnthropicProvider(), AnthropicProvider()
        try:
            assert await send_to(agent, agent_endpoint, REJECTS) == "fake reply"
            assert await send_to(judge, judge_endpoint, REJECTS) == "fake reply"
        finally:
            await agent.aclose()
            await judge.aclose()
        assert [b.get("temperature") for b in rejects.bodies] == [0.0, None]
        assert [b.get("temperature") for b in accepts.bodies] == [0.0]
    # A provider that sent nothing (the partial scorecard builds a fresh one) knows the same.
    for asked in (agent, judge, AnthropicProvider()):
        assert asked.accepts_temperature(REJECTS, agent_endpoint) is False
        assert asked.accepts_temperature(REJECTS, judge_endpoint) is True
        assert asked.accepts_temperature(ACCEPTS, agent_endpoint) is True
        assert asked.accepts_temperature(REJECTS, "http://127.0.0.1:9") is True


AGENT_AT_DEFAULT = (
    f"agent model {REJECTS} rejects temperature; the agent runs at the model's default temperature"
)
JUDGE_AT_DEFAULT = (
    f"judge model {REJECTS} rejects temperature; the judge runs at the model's default temperature"
)


@pytest.fixture
def fresh_run(monkeypatch) -> None:
    """A run that has not met any endpoint yet."""
    monkeypatch.setattr(adapter, "_NO_TEMPERATURE_MODELS", set())
    monkeypatch.setattr(adapter, "_API_ROOTS", {})
    monkeypatch.setattr(adapter, "_DEFAULT_TEMPERATURE_WARNED", set())


async def send_as(role: str, provider: AnthropicProvider, server: ThreadingHTTPServer) -> str:
    return await provider.send_message(
        [ChatMessage(role="user", content="hi")],
        ProviderConfig(provider="anthropic", endpoint=f"http://127.0.0.1:{server.server_port}",
                       api_key="synthetic-local-key", model=REJECTS, timeout=5, max_retries=0,
                       temperature=0.0, role=role),
    )


def adapter_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == adapter.__name__]


@pytest.mark.parametrize(("rejecting_role", "accepting_role", "expected"), [
    ("judge", "agent", [JUDGE_AT_DEFAULT]),
    ("agent", "judge", [AGENT_AT_DEFAULT]),
])
async def test_warning_names_the_role_whose_endpoint_rejects_temperature(
    fresh_run, caplog, rejecting_role, accepting_role, expected,
):
    caplog.set_level(logging.WARNING, logger=adapter.__name__)
    with local_api() as rejects, local_api() as accepts:
        accepts.RequestHandlerClass = AcceptsAnyTemperature
        providers = {"agent": AnthropicProvider(), "judge": AnthropicProvider()}
        try:
            for _ in range(2):
                assert await send_as(accepting_role, providers[accepting_role], accepts) == "fake reply"
                assert await send_as(rejecting_role, providers[rejecting_role], rejects) == "fake reply"
        finally:
            for provider in providers.values():
                await provider.aclose()
        assert [b.get("temperature") for b in accepts.bodies] == [0.0, 0.0]
        assert [b.get("temperature") for b in rejects.bodies] == [0.0, None, None]
    assert adapter_warnings(caplog) == expected


async def test_agent_and_judge_behind_one_rejecting_endpoint_are_each_named_once(fresh_run, caplog):
    caplog.set_level(logging.WARNING, logger=adapter.__name__)
    with local_api() as rejects:
        agent, judge = AnthropicProvider(), AnthropicProvider()
        try:
            for role, provider in [("agent", agent), ("agent", agent), ("judge", judge), ("judge", judge)]:
                assert await send_as(role, provider, rejects) == "fake reply"
        finally:
            await agent.aclose()
            await judge.aclose()
        # The agent's first send is refused and resent. The judge never sends temperature,
        # but it still runs at the default, so it is named too.
        assert [b.get("temperature") for b in rejects.bodies] == [0.0, None, None, None, None]
    assert adapter_warnings(caplog) == [AGENT_AT_DEFAULT, JUDGE_AT_DEFAULT]


async def test_judge_evaluator_sends_as_the_judge(fresh_run, caplog):
    caplog.set_level(logging.WARNING, logger=adapter.__name__)
    with local_api() as rejects:
        judge = JudgeEvaluator(JudgeConfig(
            provider="anthropic", model=REJECTS, api_key="synthetic-local-key",
            endpoint=f"http://127.0.0.1:{rejects.server_port}",
        ))
        try:
            reply = await judge.send_message(
                [ChatMessage(role="user", content="grade this")], judge.provider_pair()["config"]
            )
        finally:
            await judge.aclose()
        assert reply == "fake reply"
        assert [b.get("temperature") for b in rejects.bodies] == [0.0, None]
    assert adapter_warnings(caplog) == [JUDGE_AT_DEFAULT]
