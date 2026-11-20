import asyncio

import pytest

from ifixai.core import connection
from ifixai.core.types import ProviderConfig
from ifixai.providers.base import ChatProvider


class HungCapabilities(ChatProvider):
    async def send_message(self, messages, config):
        return "Hello"

    async def list_tools(self, config):
        await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_connection_probe_bounds_capability_discovery(monkeypatch):
    monkeypatch.setattr(connection, "CONNECTION_TEST_TIMEOUT", 0.01)
    result = await asyncio.wait_for(
        connection.test_connection(HungCapabilities(), ProviderConfig(provider="mock", timeout=1)),
        timeout=2,
    )
    assert result.success
    assert result.capabilities is None


class SlowCapabilities(HungCapabilities):
    async def list_tools(self, config):
        await asyncio.sleep(0.03)
        return []


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [0, -1, 2])
async def test_capability_probe_honors_configured_timeout(monkeypatch, timeout):
    monkeypatch.setattr(connection, "CONNECTION_TEST_TIMEOUT", 0.01)
    result = await connection.test_connection(
        SlowCapabilities(), ProviderConfig(provider="mock", timeout=timeout)
    )
    assert result.success
    assert result.capabilities is not None


@pytest.mark.asyncio
async def test_slow_http_retrieval_uses_provider_timeout(monkeypatch):
    from aiohttp import web

    from ifixai.providers.http import HttpProvider

    async def chat(request):
        return web.json_response({"choices": [{"message": {"content": "Hello"}}]})

    async def retrieval(request):
        await asyncio.sleep(0.3)
        return web.json_response({"sources": []})

    app = web.Application()
    app.router.add_post("/chat/completions", chat)
    app.router.add_post("/retrieve", retrieval)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    provider = HttpProvider()
    monkeypatch.setattr(connection, "CONNECTION_TEST_TIMEOUT", 0.2)
    try:
        result = await connection.test_connection(
            provider,
            ProviderConfig(
                provider="http", timeout=2,
                endpoint=f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}",
            ),
        )
        assert result.success
        assert result.capabilities is not None
        assert result.capabilities.has_retrieval
    finally:
        await provider.aclose()
        await runner.cleanup()
