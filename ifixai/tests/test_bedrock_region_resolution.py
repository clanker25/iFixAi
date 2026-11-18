"""Bedrock follows the standard AWS region chain unless a region is passed."""

import asyncio
import json

import pytest

boto3 = pytest.importorskip("boto3")
from botocore.awsrequest import AWSResponse  # noqa: E402

from ifixai.core.types import ChatMessage, ProviderConfig  # noqa: E402
from ifixai.providers import bedrock  # noqa: E402


class _Body:
    def __init__(self, data):
        self._data = data

    def stream(self, **_kwargs):
        yield self._data

    def read(self):
        return self._data


def _send(monkeypatch, provider):
    seen = []

    def reply(request, **_kwargs):
        seen.append((request.url, request.headers["Authorization"].decode()
                     if isinstance(request.headers["Authorization"], bytes)
                     else request.headers["Authorization"]))
        body = json.dumps({
            "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        }).encode()
        return AWSResponse(request.url, 200, {"Content-Type": "application/json"}, _Body(body))

    real_session = boto3.Session

    class HookedSession(real_session):
        def client(self, *args, **kwargs):
            client = super().client(*args, **kwargs)
            client.meta.events.register("before-send.bedrock-runtime.Converse", reply)
            return client

    monkeypatch.setattr(bedrock.boto3, "Session", HookedSession)
    config = ProviderConfig(provider="bedrock", model="anthropic.claude-3-haiku-20240307-v1:0", timeout=5, max_retries=0)
    assert asyncio.run(provider.send_message([ChatMessage(role="user", content="hi")], config)) == "ok"
    assert len(seen) == 1
    return seen[0]


@pytest.fixture
def aws_env(monkeypatch, tmp_path):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLEEXAMPLE00")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "x" * 40)
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "credentials"))
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)


def test_environment_region_is_honored(monkeypatch, aws_env):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
    url, auth = _send(monkeypatch, bedrock.BedrockProvider())
    assert "bedrock-runtime.eu-west-1.amazonaws.com" in url
    assert "/eu-west-1/bedrock/" in auth


def test_explicit_region_overrides_environment(monkeypatch, aws_env):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
    url, auth = _send(monkeypatch, bedrock.BedrockProvider(region_name="ap-south-1"))
    assert "bedrock-runtime.ap-south-1.amazonaws.com" in url
    assert "/ap-south-1/bedrock/" in auth


def test_us_east_1_remains_the_default_without_a_configured_region(monkeypatch, aws_env):
    url, auth = _send(monkeypatch, bedrock.BedrockProvider())
    assert "bedrock-runtime.us-east-1.amazonaws.com" in url
    assert "/us-east-1/bedrock/" in auth
