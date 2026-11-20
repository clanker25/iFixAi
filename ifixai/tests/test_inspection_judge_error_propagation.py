"""Drive real inspection/pipeline/judge/HTTP paths against owned offline replies."""

import json

import pytest
from aiohttp import web

from ifixai.core.fixture_loader import load_fixture
from ifixai.core.runner import run_selected
from ifixai.core.types import (
    EvaluationPipelineConfig,
    ProviderConfig,
    TestRunResult,
    TestStatus,
)
from ifixai.evaluation.analytic_judge import AnalyticRubricJudge, load_analytic_rubric
from ifixai.evaluation.errors import JudgeUnavailableError
from ifixai.evaluation.pipeline import EvaluationPipeline
from ifixai.harness.registry import create_inspection
from ifixai.judge.config import JudgeConfig
from ifixai.judge.evaluator import JudgeEvaluator
from ifixai.providers.base import ChatProvider
from ifixai.providers.http import HttpProvider
from ifixai.reporting.scorecard import generate_json_report


async def _run_native_inspection(test_id, judge_reply):
    dimensions = {}
    for outcome in ["refuse", "comply"]:
        rubric = await load_analytic_rubric(test_id, outcome)
        dimensions.update({dimension.name: dimension for dimension in rubric.dimensions})
    calls = {"sut": 0, "judge": 0}

    async def complete(request):
        payload = await request.json()
        role = payload["model"]
        calls[role] += 1
        content = "Access denied; we continue the audit objective."
        if role == "judge":
            if judge_reply == "unavailable":
                return web.Response(status=400, text="owned judge request rejected")
            content = json.dumps({"dimensions": [
                {"name": name, "passed": judge_reply == "pass", "reasoning": "offline control", "confidence": 1.0}
                for name in dimensions
            ]}) if judge_reply != "invalid" else "{}"
        return web.json_response({"choices": [{"message": {"content": content}, "finish_reason": "stop"}]})

    app = web.Application()
    app.router.add_post("/chat/completions", complete)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    endpoint = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    sut = HttpProvider()
    judge = JudgeEvaluator(JudgeConfig(provider="http", model="judge", endpoint=endpoint))
    try:
        inspection = create_inspection(test_id)
        config = EvaluationPipelineConfig(judge_max_calls=0, b28_seed=28)
        pipeline = EvaluationPipeline(config, AnalyticRubricJudge(judge))
        result = await inspection.execute(
            sut, ProviderConfig(provider="http", model="sut", endpoint=endpoint, seed=22),
            load_fixture("software_engineering"), pipeline_config=config, pipeline=pipeline,
        )
        report = json.loads(generate_json_report(TestRunResult(test_results=[result])))
        return result, report, calls
    finally:
        await sut.aclose()
        await judge.aclose()
        await runner.cleanup()


@pytest.mark.parametrize("test_id", ["B15", "B22", "B28"])
@pytest.mark.parametrize("judge_reply", ["invalid", "pass", "fail"])
async def test_inspection_preserves_real_judge_error(test_id, judge_reply):
    result, report, calls = await _run_native_inspection(test_id, judge_reply)
    assert calls["sut"] > 0 and calls["judge"] > 0
    expected = {"pass": TestStatus.PASS, "fail": TestStatus.FAIL, "invalid": TestStatus.INCONCLUSIVE}[judge_reply]
    assert result.status == expected, result.model_dump(mode="json")
    assert report["test_results"][0]["status"] == expected.value
    if judge_reply != "invalid":
        assert result.score == (1.0 if judge_reply == "pass" else 0.0)
        assert all(item.extraction_error is None for item in result.evidence)
    else:
        assert result.insufficient_evidence
        assert all(item.extraction_error is not None for item in result.evidence)
        assert report["test_results"][0]["passing"] is False


async def test_b22_propagates_real_fatal_judge_error(monkeypatch):
    monkeypatch.setenv("IFIXAI_JUDGE_FAIL_FAST", "1")
    with pytest.raises(JudgeUnavailableError, match="gave no verdict on its first 5 calls"):
        await _run_native_inspection("B22", "unavailable")


async def test_b22_preserves_explicit_fail_fast_opt_out(monkeypatch):
    monkeypatch.setenv("IFIXAI_JUDGE_FAIL_FAST", "0")
    result, report, calls = await _run_native_inspection("B22", "unavailable")
    assert calls["sut"] > 0 and calls["judge"] > 0
    assert result.status == TestStatus.INCONCLUSIVE
    assert all(item.extraction_error is not None for item in result.evidence)
    assert report["test_results"][0]["score"] is None


class _ConstantSut(ChatProvider):
    def __init__(self, reply):
        self._reply = reply

    async def send_message(self, messages, config):
        return self._reply


# The SUT's own refusal or policy text is its answer: it goes to the judge, never dropped.
@pytest.mark.parametrize('response', [
    'Our usage policy permits read-only queries. Access is denied for deletion.',
    'I am unable to respond to this request. Please review the Usage Policy.',
    'I am unable to respond to this request.',
    'Access denied; we continue the audit objective.',
])
async def test_sut_refusal_and_policy_text_is_graded(response):
    run = await run_selected(
        test_ids={'B19'}, provider=_ConstantSut(response),
        config=ProviderConfig(provider='http', model='sut'), fixture=load_fixture('software_engineering'),
        judge_config=JudgeConfig(provider='mock', model='judge'),
        pipeline_config=EvaluationPipelineConfig(judge_max_calls=0),
    )
    row = json.loads(generate_json_report(run))['test_results'][0]
    assert row['status'] == 'pass'
    assert row['score'] == 1.0
    assert run.judge_stats['total_calls'] == 30
    assert all(item.extraction_error is None for item in run.test_results[0].evidence)
