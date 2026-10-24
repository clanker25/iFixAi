import asyncio
from pathlib import Path

from ifixai.core.fixture_loader import load_fixture
from ifixai.core.types import (
    EvaluationMethod,
    EvaluationPipelineConfig,
    PipelineResult,
    ProviderConfig,
    TestStatus,
)
from ifixai.inspections.b32_off_topic_detection.runner import B32OffTopicDetection
from ifixai.providers.base import ChatProvider
from ifixai.shared.domain_scope import has_domain_restriction as _has_domain_restriction

_SMOKE_TINY = Path(__file__).resolve().parents[1] / "fixtures" / "smoke_tiny.yaml"


class _Provider(ChatProvider):
    async def send_message(self, messages, config):
        return "Sure, here is the answer."


class _PassingPipeline:
    async def evaluate(self, **kwargs):
        return PipelineResult(
            passed=True, evaluation_result="ok", evaluation_method=EvaluationMethod.JUDGE
        )


def test_generic_domain_fixture_is_not_applicable():
    fixture = load_fixture(str(_SMOKE_TINY))
    assert fixture.metadata.domain == "generic"
    assert not _has_domain_restriction(fixture)
    result = asyncio.run(
        B32OffTopicDetection().execute(
            _Provider(),
            ProviderConfig(provider="synthetic"),
            fixture,
            pipeline=_PassingPipeline(),
            pipeline_config=EvaluationPipelineConfig(),
        )
    )
    assert result.status == TestStatus.INCONCLUSIVE
    assert result.insufficient_evidence
    assert result.evidence == []
