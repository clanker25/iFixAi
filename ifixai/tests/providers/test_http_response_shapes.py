"""Malformed successful HTTP replies should surface as provider errors."""

import re
from functools import partial

import aiohttp.web
import pytest

from ifixai.core.types import ChatMessage, ProviderConfig
from ifixai.providers.base import (
    AgentReply,
    HttpEmptyReplyError,
    ProviderEmptyContentError,
    ProviderResponseError,
)
from ifixai.providers.http import HttpProvider


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply",
    [
        "not json",
        [],
        {"choices": "bad"},
        {"choices": [{"message": {"content": ["not", "text"]}}]},
    ],
)
async def test_malformed_chat_reply_is_provider_response_error(reply):
    async def chat(_request):
        if isinstance(reply, str):
            return aiohttp.web.Response(text=reply, content_type="text/plain")
        return aiohttp.web.json_response(reply)

    app = aiohttp.web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    provider = HttpProvider()
    config = ProviderConfig(
        provider="http", endpoint=f"http://127.0.0.1:{port}/v1", max_retries=0
    )
    try:
        with pytest.raises(ProviderResponseError):
            await provider.send_message([ChatMessage(content="hello")], config)
    finally:
        await provider.aclose()
        await runner.cleanup()


@pytest.mark.asyncio
# A completed empty string is `ProviderEmptyContentError` (test_http_empty_content_native.py).
@pytest.mark.parametrize("message", [{"content": None}, {"content": None, "tool_calls": []}])
async def test_empty_chat_reply_is_an_empty_reply(message):
    async def chat(_request):
        return aiohttp.web.json_response({"choices": [{"message": message, "finish_reason": "stop"}]})

    app = aiohttp.web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/v1"
    provider = HttpProvider()
    try:
        # The same plain response error the HTTP provider always raised, so an inspection
        # that doesn't read empty replies drops the probe instead of voiding its run.
        baseline = re.escape(f"[http] Missing or non-text content in response (endpoint: {url})")
        with pytest.raises(ProviderResponseError, match=f"^{baseline}$") as caught:
            await provider.send_message([ChatMessage(content="hello")], ProviderConfig(provider="http", endpoint=url, max_retries=0))
        assert isinstance(caught.value, HttpEmptyReplyError)
        assert not isinstance(caught.value, ProviderEmptyContentError)
    finally:
        await provider.aclose()
        await runner.cleanup()


async def _reply(choice: dict, _request: aiohttp.web.Request) -> aiohttp.web.Response:
    return aiohttp.web.json_response({"choices": [choice]})


async def send_to_agent_replying(choice: dict) -> str:
    """One message to a local agent whose chat reply is `choice`."""
    app = aiohttp.web.Application()
    app.router.add_post("/v1/chat/completions", partial(_reply, choice))
    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/v1"
    provider = HttpProvider()
    try:
        return await provider.send_message(
            [ChatMessage(content="hello")], ProviderConfig(provider="http", endpoint=url, max_retries=0)
        )
    finally:
        await provider.aclose()
        await runner.cleanup()


def tool_call(name: object, arguments: object) -> dict:
    return {"id": "c1", "type": "function", "function": {"name": name, "arguments": arguments}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("calls", "made"),
    [
        ([tool_call("delete_file", '{"path": "/srv/a"}')], ('delete_file({"path": "/srv/a"})',)),
        ([tool_call("lookup", {"id": 7}), tool_call("delete_file", None)], ('lookup({"id": 7})', "delete_file()")),
    ],
)
async def test_a_tool_call_only_reply_is_the_baseline_empty_reply_carrying_its_calls(calls, made):
    choice = {"message": {"content": None, "tool_calls": calls}, "finish_reason": "tool_calls"}
    with pytest.raises(HttpEmptyReplyError, match=re.escape("Missing or non-text content in response")) as caught:
        await send_to_agent_replying(choice)
    assert caught.value.tool_calls == made


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "calls", ["delete_file", [{"id": "c1"}], [tool_call("", "{}")], [tool_call(None, "{}")], [{"function": "", "arguments": "{}"}]]
)
async def test_tool_calls_without_a_function_name_and_no_text_are_a_response_error(calls):
    choice = {"message": {"content": None, "tool_calls": calls}, "finish_reason": "tool_calls"}
    with pytest.raises(ProviderResponseError, match=re.escape("Malformed tool_calls in response")) as caught:
        await send_to_agent_replying(choice)
    assert not isinstance(caught.value, HttpEmptyReplyError)


@pytest.mark.asyncio
@pytest.mark.parametrize("calls", ["delete_file", {"name": "x"}, [{"id": "c1"}], [tool_call("", "{}")], [tool_call(None, "{}")]])
async def test_malformed_tool_calls_never_cost_a_reply_its_text(calls):
    choice = {"message": {"content": "I have not touched anything.", "tool_calls": calls}, "finish_reason": "stop"}
    reply = await send_to_agent_replying(choice)
    assert reply == "I have not touched anything." and not isinstance(reply, AgentReply)


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason", ["stop", "tool_calls", "content_filter"])
async def test_a_reply_with_text_and_tool_calls_keeps_its_text_and_carries_the_calls(finish_reason):
    calls = [tool_call("delete_file", '{"path": "/srv/a"}'), {"id": "c2"}]
    choice = {"message": {"content": "I have not touched anything.", "tool_calls": calls}, "finish_reason": finish_reason}
    reply = await send_to_agent_replying(choice)
    assert reply == "I have not touched anything."
    assert reply.tool_calls == ('delete_file({"path": "/srv/a"})',)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("call", "made"),
    [
        ({"name": "delete_file", "args": {"path": "/srv/a"}}, 'delete_file({"path": "/srv/a"})'),
        ({"name": "delete_file", "arguments": '{"path": "/srv/a"}'}, 'delete_file({"path": "/srv/a"})'),
        ({"id": "t1", "type": "tool_use", "name": "delete_file", "input": {"path": "/srv/a"}}, 'delete_file({"path": "/srv/a"})'),
        ({"name": "list_files"}, "list_files()"),
        ({"function": "delete_file", "arguments": {"path": "/srv/a"}}, 'delete_file({"path": "/srv/a"})'),
        ({"function": "delete_file", "arguments": '{"path": "/srv/a"}'}, 'delete_file({"path": "/srv/a"})'),
    ],
)
async def test_a_call_in_a_non_openai_shape_is_still_carried(call, made):
    choice = {"message": {"content": "I have not touched anything.", "tool_calls": [call]}, "finish_reason": "stop"}
    reply = await send_to_agent_replying(choice)
    assert reply == "I have not touched anything."
    assert reply.tool_calls == (made,)


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason", ["content_filter", "refusal"])
@pytest.mark.parametrize("content", [None, ""])
async def test_a_reply_a_filter_or_refusal_withheld_is_an_empty_reply(finish_reason, content):
    choice = {"message": {"content": content}, "finish_reason": finish_reason}
    with pytest.raises(HttpEmptyReplyError, match=re.escape(f"Reply withheld (finish_reason={finish_reason})")):
        await send_to_agent_replying(choice)


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason", ["content_filter", "refusal"])
async def test_a_withheld_reply_with_partial_text_keeps_the_text(finish_reason):
    choice = {"message": {"content": "Sure, here is how to dele"}, "finish_reason": finish_reason}
    assert await send_to_agent_replying(choice) == "Sure, here is how to dele"


@pytest.mark.asyncio
async def test_malformed_retrieval_reply_reports_unavailable():
    async def retrieve(_request):
        return aiohttp.web.json_response([{"document_uri": "wrong shape"}])

    app = aiohttp.web.Application()
    app.router.add_post("/v1/retrieve", retrieve)
    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    provider = HttpProvider()
    config = ProviderConfig(provider="http", endpoint=f"http://127.0.0.1:{port}/v1")
    try:
        assert await provider.retrieve_sources("hello", config) is None
    finally:
        await provider.aclose()
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("detail,fatal,attempts", [
    ("insufficient_quota", True, 1),
    ("insufficient credits", True, 1),
    ("payment required", True, 1),
    ("rate limit exceeded", False, 2),
    ("You exceeded your current quota, please check your plan and billing details", False, 2),
    ("Token rate limit exceeded. Retry after 60 seconds: https://aka.ms/oai/quotaincrease", False, 2),
    ("per-minute credit quota exceeded", False, 2),
])
async def test_http_429_preserves_quota_identity(monkeypatch, detail, fatal, attempts):
    from ifixai.providers.base import ProviderRateLimitError, is_fatal_provider_error

    seen = []
    async def chat(request):
        seen.append(await request.json())
        return aiohttp.web.json_response({"error": {"message": detail + " sk-" + "A" * 24}}, status=429)

    async def no_delay(_seconds):
        pass

    monkeypatch.setattr("ifixai.providers.http.asyncio.sleep", no_delay)
    app = aiohttp.web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/v1"
    provider = HttpProvider()
    try:
        with pytest.raises(ProviderRateLimitError) as caught:
            await provider.send_message([ChatMessage(content="hello")], ProviderConfig(provider="http", endpoint=url, max_retries=1))
        assert is_fatal_provider_error(caught.value) is fatal
        assert detail in caught.value.details
        assert "sk-" + "A" * 24 not in caught.value.details
        assert len(seen) == attempts
    finally:
        await provider.aclose()
        await runner.cleanup()


@pytest.mark.parametrize("detail,fatal", [
    ("insufficient_quota", True), ("insufficient credits", True),
    ("payment required", True), ("billing limit reached", False),
    ("rate limit exceeded", False), ("HTTP 429, retry later", False),
    ("You exceeded your current quota, please check your plan and billing details", False),
    ("Token rate limit exceeded. Retry after 60 seconds: https://aka.ms/oai/quotaincrease", False),
    ("per-minute credit quota exceeded", False),
])
def test_shared_rate_limit_classifier_reaches_account_quota_check(detail, fatal):
    from ifixai.providers.base import (
        ProviderConnectionError,
        ProviderRateLimitError,
        is_fatal_provider_error,
    )
    assert is_fatal_provider_error(ProviderRateLimitError(details=detail)) is fatal
    assert not is_fatal_provider_error(ProviderConnectionError(details=detail))


def test_quota_markers_in_provider_metadata_do_not_make_throttling_fatal():
    from ifixai.providers.base import ProviderRateLimitError, is_fatal_provider_error

    error = ProviderRateLimitError(provider="credit-proxy", endpoint="https://owned.invalid/quota", details="rate limit exceeded")
    assert not is_fatal_provider_error(error)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [408, 500, 502, 503, 504, 400])
async def test_http_gateway_failures_follow_retry_contract(monkeypatch, status):
    from ifixai.providers.base import ProviderOverloadedError, is_fatal_provider_error

    seen = []
    async def chat(request):
        seen.append(await request.json())
        if len(seen) == 1:
            return aiohttp.web.json_response({"error": {"message": "owned gateway unavailable"}}, status=status)
        return aiohttp.web.json_response({"choices": [{"message": {"content": "recovered reply"}}]})

    async def no_delay(_seconds):
        pass

    monkeypatch.setattr("ifixai.providers.http.asyncio.sleep", no_delay)
    app = aiohttp.web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/v1"
    provider = HttpProvider()
    try:
        config = ProviderConfig(provider="http", endpoint=url, max_retries=1)
        if status == 400:
            with pytest.raises(ProviderResponseError):
                await provider.send_message([ChatMessage(content="hello")], config)
            assert len(seen) == 1
        else:
            assert await provider.send_message([ChatMessage(content="hello")], config) == "recovered reply"
            assert len(seen) == 2
        seen.clear()
        config.max_retries = 0
        expected = ProviderResponseError if status == 400 else ProviderOverloadedError
        with pytest.raises(expected) as caught:
            await provider.send_message([ChatMessage(content="hello")], config)
        assert not is_fatal_provider_error(caught.value)
        assert len(seen) == 1
    finally:
        await provider.aclose()
        await runner.cleanup()
