import asyncio

from ifixai.core.types import (
    EvaluationMethod,
    EvaluationPipelineConfig,
    Fixture,
    FixtureMetadata,
    PipelineResult,
    ProviderConfig,
    User,
)
from ifixai.inspections.b27_session_integrity.runner import B27SessionIntegrity
from ifixai.providers.base import ChatProvider, ProviderConnectionError


class _FailAliceProbe(ChatProvider):
    async def send_message(self, messages, config):
        # Fail only the probe turn of the conversation where Alice shared her secret.
        if len(messages) >= 3 and "secret-code-alice-" in messages[-3].content:
            raise ProviderConnectionError("simulated network drop")
        return "I can't share another user's data."


class _PassingPipeline:
    async def evaluate(self, **kwargs):
        return PipelineResult(
            passed=True, evaluation_result="ok", evaluation_method=EvaluationMethod.JUDGE
        )


def test_failed_probe_does_not_take_the_reverse_pair_case_id():
    fixture = Fixture(
        metadata=FixtureMetadata(name="b27"),
        users=[
            User(user_id="alice", name="Alice", roles=["user"]),
            User(user_id="bob", name="Bob", roles=["user"]),
        ],
    )
    result = asyncio.run(
        B27SessionIntegrity().execute(
            _FailAliceProbe(),
            ProviderConfig(provider="synthetic"),
            fixture,
            pipeline=_PassingPipeline(),
            pipeline_config=EvaluationPipelineConfig(),
        )
    )
    ids = [item.test_case_id for item in result.evidence]
    assert len(ids) == len(set(ids)) == 2
    failed = next(item for item in result.evidence if item.details.get("comm_failure"))
    assert failed.test_case_id == "B27_bob_vs_alice"
    assert failed.details["inspecting_user"] == "bob"
    assert failed.details["target_user"] == "alice"
