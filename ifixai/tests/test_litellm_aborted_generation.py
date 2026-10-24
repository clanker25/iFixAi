"""LiteLLM must surface an upstream mid-generation abort like the OpenAI adapter."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("litellm")

from ifixai.core.types import ChatMessage, ProviderConfig
from ifixai.providers.base import (
    ProviderOverloadedError,
    ProviderRateLimitError,
)
from ifixai.providers.litellm import LiteLLMProvider


def _serve(content, error):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            body = json.dumps({
                "id": "owned", "object": "chat.completion", "created": 1, "model": "gpt-4o",
                "choices": [{"index": 0, "finish_reason": "error", "error": error,
                             "message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.mark.parametrize("content", ["partial answer", ""])
@pytest.mark.parametrize("error, expected", [
    ({"code": 429, "message": "Rate limit exceeded upstream"}, ProviderRateLimitError),
    ({"code": 503, "message": "Upstream unavailable"}, ProviderOverloadedError),
])
def test_aborted_generation_is_not_graded_as_a_reply(monkeypatch, content, error, expected):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    server = _serve(content, error)
    try:
        config = ProviderConfig(
            provider="litellm", model="openai/gpt-4o", api_key="owned-key", timeout=5, max_retries=0,
            endpoint=f"http://127.0.0.1:{server.server_port}/v1",
        )
        with pytest.raises(expected, match="finish_reason=error"):
            asyncio.run(LiteLLMProvider().send_message([ChatMessage(content="hi")], config))
    finally:
        server.shutdown()
        server.server_close()
