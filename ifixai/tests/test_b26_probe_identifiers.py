import asyncio
import json
from pathlib import Path
from urllib.parse import quote

from ifixai.core.fixture_loader import load_fixture
from ifixai.core.types import (
    EvaluationMethod,
    EvaluationPipelineConfig,
    PipelineResult,
    ProviderConfig,
    TestRunResult,
)
from ifixai.inspections.b26_rate_limiting.runner import B26RateLimiting
from ifixai.providers.base import ChatProvider
from ifixai.reporting.scorecard import generate_json_report

_SMOKE_TINY = Path(__file__).resolve().parents[1] / "fixtures" / "smoke_tiny.yaml"


class _Provider(ChatProvider):
    surfaces_rate_limit_errors = True

    async def send_message(self, messages, config):
        return "60 requests per minute; excess requests are throttled."


class _PassingPipeline:
    async def evaluate(self, **kwargs):
        return PipelineResult(
            passed=True, evaluation_result="ok", evaluation_method=EvaluationMethod.JUDGE
        )


def test_each_tool_probe_exports_its_own_case_id():
    fixture = load_fixture(str(_SMOKE_TINY))
    assert len(fixture.tools) > 1
    result = asyncio.run(
        B26RateLimiting().execute(
            _Provider(),
            ProviderConfig(provider="synthetic"),
            fixture,
            pipeline=_PassingPipeline(),
            pipeline_config=EvaluationPipelineConfig(),
        )
    )
    exported = json.loads(generate_json_report(TestRunResult(test_results=[result])))
    ids = [item["test_case_id"] for item in exported["test_results"][0]["evidence"]]
    tool_ids = [i for i in ids if i.startswith("B26_step")]
    assert len(tool_ids) == len(set(tool_ids))
    assert set(tool_ids) == {
        f"B26_step1_tool={quote(tool.tool_id, safe='')}" for tool in fixture.tools
    }
