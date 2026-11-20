"""Exercise setup defaults through the real HTTP serialization boundary."""

from contextlib import asynccontextmanager

import aiohttp
import pytest
from aiohttp import web
from yarl import URL

from ifixai.cli.model_catalog import default_model, suggestions
from ifixai.cli.setup_cmd import _CUSTOM_MODEL, _pick_model
from ifixai.core.types import ChatMessage, ProviderConfig
from ifixai.providers.minimax import MODEL_METADATA, REGIONAL_ENDPOINTS
from ifixai.providers.resolver import resolve_provider


@asynccontextmanager
async def capture_requests(monkeypatch):
    captured = []

    async def reply(request):
        captured.append((request.path, await request.json(), dict(request.headers)))
        if request.path.endswith("/messages"):
            return web.json_response({"content": [{"type": "text", "text": "ok"}]})
        return web.json_response({"choices": [{"message": {"content": "ok"}}]})

    app = web.Application()
    app.router.add_post("/{path:.*}", reply)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    original_request = aiohttp.ClientSession._request
    destinations = []

    async def local_request(session, method, url, **kwargs):
        destinations.append(str(url))
        local_url = URL(url).with_scheme("http").with_host("127.0.0.1").with_port(port)
        return await original_request(session, method, local_url, **kwargs)

    monkeypatch.setattr(aiohttp.ClientSession, "_request", local_request)
    try:
        yield destinations, captured
    finally:
        await runner.cleanup()


@pytest.mark.parametrize(
    "endpoint",
    [None]
    + [
        region[style]
        for region in REGIONAL_ENDPOINTS.values()
        for style in ("openai_base_url", "anthropic_base_url")
    ]
    + ["https://custom.invalid/v1", "https://custom.invalid/anthropic/v1"],
)
@pytest.mark.parametrize(
    "model",
    [None, "MiniMax-M3.1-Flash", "MiniMax-M3", "MiniMax-M2.7", "saved-custom-id"],
)
async def test_default_and_saved_models_reach_http_unchanged(
    monkeypatch, endpoint, model
):
    # Round-trip saved configuration before resolving the runtime provider.
    config = ProviderConfig.model_validate_json(
        ProviderConfig(
            provider="minimax",
            endpoint=endpoint,
            model=model,
            api_key="offline-fixture",
            max_tokens=128,
            temperature=0.5,
            seed=17,
            json_output=True,
            max_retries=0,
            extra_headers={"X-Test-Header": "preserved"},
        ).model_dump_json()
    )
    provider = resolve_provider(config.provider)
    messages = [
        ChatMessage(role="system", content="Be helpful."),
        ChatMessage(content="Hello"),
    ]
    async with capture_requests(monkeypatch) as (destinations, captured):
        try:
            assert await provider.send_message(messages, config) == "ok"
        finally:
            await provider.aclose()
    base = endpoint or REGIONAL_ENDPOINTS["global_en"]["openai_base_url"]
    is_messages = "/anthropic" in base
    expected_url = (
        f"{base.removesuffix('/v1')}/v1/messages"
        if is_messages
        else f"{base}/chat/completions"
    )
    assert destinations == [expected_url]
    assert len(captured) == 1
    path, payload, headers = captured[0]
    assert path == URL(expected_url).path
    expected_payload = {
        "model": model or "MiniMax-M3.1-Flash",
        "max_tokens": 128,
        "temperature": 0.5,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    if is_messages:
        expected_payload["system"] = "Be helpful."
        assert headers["anthropic-version"] == "2023-06-01"
    else:
        expected_payload["messages"].insert(
            0, {"role": "system", "content": "Be helpful."}
        )
        expected_payload.update(seed=17, response_format={"type": "json_object"})
    # Exact equality also guards against unsupported thinking/effort/tier overrides.
    assert payload == expected_payload
    assert headers["Authorization"] == "Bearer offline-fixture"
    assert headers["X-Test-Header"] == "preserved"


def test_active_catalog_and_pricing():
    assert default_model("minimax") == "MiniMax-M3.1-Flash"
    models = suggestions("minimax")
    assert [model for model, _ in models] == ["MiniMax-M3.1-Flash", "MiniMax-M3"]
    assert "$0.15/$0.60" in models[0][1]
    assert "$0.30/$1.20 up to 512k input; $0.60/$2.40 above 512k input" in models[1][1]
    assert MODEL_METADATA[models[0][0]] == {
        "context_window": 1000000,
        "pricing_usd_per_million_tokens": {
            "input": 0.15,
            "output": 0.6,
            "cache_read": 0.015,
            "cache_write": None,
        },
        "input_modalities": ["text", "image", "video"],
        "thinking": ["adaptive"],
    }
    assert MODEL_METADATA["MiniMax-M3"]["thinking"] == ["adaptive", "disabled"]


@pytest.mark.parametrize("custom", [None, "MiniMax-M2.7", "saved-custom-id"])
def test_setup_default_and_custom_models(monkeypatch, custom):
    def select(prompt, labels, **kwargs):
        assert labels == [
            "Provider default (MiniMax-M3.1-Flash)",
            "MiniMax-M3.1-Flash",
            "MiniMax-M3",
            _CUSTOM_MODEL,
        ]
        return _CUSTOM_MODEL if custom else kwargs["default"]

    monkeypatch.setattr("ifixai.cli.setup_cmd.ui.select", select)
    monkeypatch.setattr("ifixai.cli.setup_cmd.ui.text", lambda *args, **kwargs: custom)
    assert _pick_model("minimax", role="agent") == custom
