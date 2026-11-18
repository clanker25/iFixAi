"""Bedrock 403 credential failures must stop the run as fatal auth errors."""

import asyncio
import json

import pytest

boto3 = pytest.importorskip("boto3")
from botocore.awsrequest import AWSResponse  # noqa: E402

from ifixai.core.types import ChatMessage, ProviderConfig  # noqa: E402
from ifixai.providers import bedrock  # noqa: E402
from ifixai.providers.base import ProviderAuthError, is_fatal_provider_error  # noqa: E402


class _Body:
    def __init__(self, data):
        self._data = data

    def stream(self, **_kwargs):
        yield self._data

    def read(self):
        return self._data


@pytest.mark.parametrize("code", [
    "UnrecognizedClientException",  # wrong access key id (already fatal)
    "InvalidSignatureException",  # right key id, wrong secret
    "ExpiredTokenException",  # expired STS / SSO session
])
def test_bedrock_credential_rejections_are_fatal_auth_errors(monkeypatch, code):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLEEXAMPLE00")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "x" * 40)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    requests = []

    def reject(request, **_kwargs):
        requests.append(request.url)
        body = json.dumps({"message": "credential rejected"}).encode()
        headers = {"Content-Type": "application/json", "x-amzn-ErrorType": f"{code}:http://internal.amazon.com/"}
        return AWSResponse(request.url, 403, headers, _Body(body))

    real_session = boto3.Session

    class HookedSession(real_session):
        def client(self, *args, **kwargs):
            client = super().client(*args, **kwargs)
            client.meta.events.register("before-send.bedrock-runtime.Converse", reject)
            return client

    monkeypatch.setattr(bedrock.boto3, "Session", HookedSession)
    config = ProviderConfig(provider="bedrock", model="anthropic.claude-3-haiku-20240307-v1:0", timeout=5, max_retries=2)
    with pytest.raises(ProviderAuthError) as raised:
        asyncio.run(bedrock.BedrockProvider().send_message([ChatMessage(role="user", content="hi")], config))
    assert is_fatal_provider_error(raised.value)
    assert len(requests) == 1
