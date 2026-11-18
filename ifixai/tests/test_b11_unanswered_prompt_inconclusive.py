"""B11 never passes on governance hooks alone when its one prompt got no usable reply."""

import pytest

from ifixai.core.fixture_loader import load_fixture
from ifixai.core.types import (
    ChatMessage,
    EvaluationMethod,
    PipelineResult,
    ProviderConfig,
    TestStatus,
)
from ifixai.inspections.b11_system_controllability.runner import (
    B11SystemControllability,
)
from ifixai.providers.base import (
    ProviderConnectionError,
    ProviderError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from ifixai.providers.mock_governance import MockGovernanceProvider


class PassingGrader:
    async def evaluate(self, **kwargs: object) -> PipelineResult:
        return PipelineResult(
            passed=True,
            evaluation_result="describes its configuration controls",
            evaluation_method=EvaluationMethod.JUDGE,
        )


def failing_provider(error: ProviderError) -> type[MockGovernanceProvider]:
    class Unanswered(MockGovernanceProvider):
        async def send_message(
            self, messages: list[ChatMessage], config: ProviderConfig
        ) -> str:
            raise error

    return Unanswered


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        ProviderResponseError(provider="http", details="no choices in response"),
        ProviderResponseError(provider="http", details="HTTP 422"),
        ProviderTimeoutError(provider="http", details="timed out"),
        ProviderConnectionError(provider="http", details="connection refused"),
    ],
    ids=["no-choices", "http-4xx", "timeout", "connection"],
)
async def test_b11_with_unanswered_prompt_is_inconclusive(error: ProviderError) -> None:
    fixture = load_fixture("customer_support")
    # Without a fixture `governance:` block the mock's hooks answer as the agent's own and score;
    # answers from the fixture's block would be notes.
    result = await B11SystemControllability().execute(
        failing_provider(error)(fixture.governance),
        ProviderConfig(provider="mock"),
        fixture.model_copy(update={"governance": None}),
        pipeline=PassingGrader(),
    )
    structural = [
        item for item in result.evidence
        if item.evaluation_method is EvaluationMethod.STRUCTURAL
    ]
    unanswered = [item for item in result.evidence if item.details.get("comm_failure")]
    # Every hook passes and scores 1.0, so only the missing answer can stop a PASS.
    assert len(structural) == 5 and all(item.passed for item in structural)
    assert result.score == 1.0
    assert len(unanswered) == 1 and unanswered[0].prompt_sent
    assert result.status is TestStatus.INCONCLUSIVE
    assert result.insufficient_evidence is True
