import json

import pytest

from ifixai.api import run_single
from ifixai.core.fixture_loader import load_fixture
from ifixai.core.types import (
    AuditRecord,
    EvaluationMethod,
    EvaluationMode,
    EvaluationPipelineConfig,
    TestRunResult,
    TestStatus,
)
from ifixai.judge.config import JudgeConfig
from ifixai.providers.base import ChatProvider
from ifixai.reporting.scorecard import generate_json_report


class TimestampProvider(ChatProvider):
    """An agent with its own audit-trail hook (an adapter, not the fixture's declared one), so
    B13 scores the trail it returns."""

    def __init__(self, timestamps):
        super().__init__()
        self.timestamps = timestamps

    async def send_message(self, messages, config):
        return "ACK"

    async def get_audit_trail(self, request_id, config):
        return [AuditRecord(timestamp=timestamp, actor="analyst", action="invoke",
                            decision="allow", request_id=request_id)
                for timestamp in self.timestamps]


@pytest.mark.asyncio
@pytest.mark.parametrize("timestamps", [
    ["not a timestamp"], [""],
    ["2026-10-10T12:00:00", "2026-10-10T12:01:00Z"],
    ["2026-10-10T12:00:00Z", "99999999999999999999999999"],
])
async def test_native_b13_rejects_invalid_or_incomparable_timestamps(timestamps):
    fixture = load_fixture("customer_support")
    fixture = fixture.model_copy(update={"users": fixture.users[:1], "tools": fixture.tools[:1], "governance": None})
    result = await run_single("B13", TimestampProvider(timestamps), fixture=fixture,
                              judge_config=JudgeConfig(provider="mock"),
                              pipeline_config=EvaluationPipelineConfig(mode=EvaluationMode.SINGLE))
    assert result.status == TestStatus.FAIL, result.error_message
    structural = [e for e in result.evidence if e.evaluation_method == EvaluationMethod.STRUCTURAL]
    assert structural
    assert structural[0].passed is False
    assert structural[0].details["is_monotonic"] is False
    exported = json.loads(generate_json_report(TestRunResult(test_results=[result])))["test_results"][0]
    assert exported["status"] == "fail"
    assert exported["score"] is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("timestamps,expected", [
    ([], TestStatus.FAIL),
    (["2026-10-10T12:00:00Z"], TestStatus.PASS),
    (["2026-10-10T12:00:00Z", "2026-10-10T12:01:00Z"], TestStatus.PASS),
    (["2026-10-10T12:01:00Z", "2026-10-10T12:00:00Z"], TestStatus.FAIL),
    (["2026-10-10T12:00:00Z", "not a timestamp"], TestStatus.FAIL),
])
async def test_native_b13_timestamp_controls(timestamps, expected):
    fixture = load_fixture("customer_support")
    fixture = fixture.model_copy(update={"users": fixture.users[:1], "tools": fixture.tools[:1], "governance": None})
    result = await run_single("B13", TimestampProvider(timestamps), fixture=fixture,
                              judge_config=JudgeConfig(provider="mock"),
                              pipeline_config=EvaluationPipelineConfig(mode=EvaluationMode.SINGLE))
    assert result.status == expected, result.error_message
