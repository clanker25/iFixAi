"""A budget skip or a dead or flaky judge never counts as the agent failing. Every
other unscored item still counts as a fail where the spec says so (B16/B17/B27/B29)."""

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import pytest
from aiohttp import web

from ifixai.api import run_selected
from ifixai.core.fixture_loader import load_fixture
from ifixai.core.types import (
    AnalyticRubric,
    EvaluationCriteria,
    EvaluationMethod,
    EvaluationMode,
    EvaluationPipelineConfig,
    EvidenceItem,
    JudgeErrorKind,
    ProviderCapabilities,
    ProviderConfig,
    RubricVerdict,
    TestStatus,
    ToolInvocationResult,
    UngradedReason,
)
from ifixai.evaluation.analytic_judge import (
    JudgeExtractionError,
    load_analytic_rubric,
    parse_rubric_verdict,
)
from ifixai.evaluation.pipeline import EvaluationPipeline
from ifixai.harness.base import BaseTest, is_provider_failure
from ifixai.inspections.b06_uncertainty_signalling.runner import (
    B06UncertaintySignalling,
)
from ifixai.inspections.b08_privilege_escalation.runner import B08PrivilegeEscalation
from ifixai.inspections.b16_silent_failure.runner import B16SilentFailure
from ifixai.inspections.b17_fact_consistency.runner import B17FactConsistency
from ifixai.inspections.b27_session_integrity.runner import B27SessionIntegrity
from ifixai.inspections.b29_prompt_sensitivity.runner import B29PromptSensitivity
from ifixai.judge.config import JudgeConfig
from ifixai.providers.resolver import resolve_provider
from ifixai.reporting.artifact import _evidence_payload

FAIL_CLOSED = [B16SilentFailure, B17FactConsistency, B27SessionIntegrity, B29PromptSensitivity]
OK = EvidenceItem(test_case_id="ok", passed=True)


def _unscored(kind: JudgeErrorKind) -> EvidenceItem:
    return EvidenceItem(test_case_id=kind.value, passed=False, extraction_error=kind)


@asynccontextmanager
async def _serve(
    handler: Callable[[web.Request], Awaitable[web.Response]],
) -> AsyncIterator[str]:
    """An OpenAI-shaped endpoint on 127.0.0.1 answering with `handler`."""
    app = web.Application()
    app.router.add_post("/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        yield f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    finally:
        await runner.cleanup()


async def _reject(_request: web.Request) -> web.Response:
    return web.Response(status=401, text="bad key")


@pytest.mark.parametrize("inspection", FAIL_CLOSED)
@pytest.mark.parametrize(
    "kind",
    [
        JudgeErrorKind.BUDGET,
        JudgeErrorKind.JUDGE_UNREACHABLE,
        JudgeErrorKind.JUDGE_TRANSIENT,
    ],
)
def test_budget_and_dead_judge_stay_out_of_the_score(
    inspection: type[BaseTest], kind: JudgeErrorKind
) -> None:
    test = inspection()
    assert test.spec.count_extraction_errors_as_fail
    evidence = [OK, _unscored(kind)]
    assert test.compute_score(evidence) == 1.0
    assert test.ci_evidence(evidence) == [OK]


@pytest.mark.parametrize("inspection", FAIL_CLOSED)
@pytest.mark.parametrize(
    "kind", [JudgeErrorKind.COMMUNICATION, JudgeErrorKind.EXTRACTION, JudgeErrorKind.CONTRACT]
)
def test_fail_closed_inspections_count_other_unscored_items_as_fails(
    inspection: type[BaseTest], kind: JudgeErrorKind
) -> None:
    evidence = [OK, _unscored(kind)]
    assert inspection().compute_score(evidence) == 0.5
    assert inspection().ci_evidence(evidence) == evidence


@pytest.mark.parametrize("kind", list(JudgeErrorKind))
def test_other_inspections_leave_every_unscored_item_out(kind: JudgeErrorKind) -> None:
    test = B06UncertaintySignalling()
    assert not test.spec.count_extraction_errors_as_fail
    assert test.compute_score([OK, _unscored(kind)]) == 1.0


class PassingJudge:
    """Grades every dimension passed."""

    async def evaluate_with_rubric(
        self, response: str, rubric: AnalyticRubric, context: str, context_vars: dict | None
    ) -> RubricVerdict:
        dims = [{"name": d.name, "passed": True} for d in rubric.dimensions]
        return parse_rubric_verdict(json.dumps({"dimensions": dims}), rubric)


async def test_a_call_past_the_judge_budget_is_marked_unscored() -> None:
    pipeline = EvaluationPipeline(EvaluationPipelineConfig(judge_max_calls=1), PassingJudge())
    rubric = await load_analytic_rubric("B16", "comply")
    first = await pipeline.evaluate("It failed: error 500.", EvaluationCriteria(), rubric)
    second = await pipeline.evaluate("It failed: error 500.", EvaluationCriteria(), rubric)
    assert (first.passed, first.extraction_error) == (True, None)
    assert (second.passed, second.extraction_error) == (False, JudgeErrorKind.BUDGET)


async def test_b08_with_every_judge_item_unscored_is_inconclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    # A rejected judge key with fail-fast off: B08's structural bonus items must not grade it alone.
    monkeypatch.setenv("IFIXAI_JUDGE_FAIL_FAST", "0")
    async with _serve(_reject) as endpoint:
        run = await run_selected(
            {"B08"}, provider="mock", api_key="x",
            judge_config=JudgeConfig(provider="http", model="judge", endpoint=endpoint),
            pipeline_config=EvaluationPipelineConfig(mode=EvaluationMode.SINGLE, judge_max_calls=0),
        )
    b08 = run.test_results[0]
    judged = [e for e in b08.evidence if e.evaluation_method is EvaluationMethod.JUDGE]
    # Rejected calls, then calls the breaker skips once the judge is down for B08.
    assert judged and {e.extraction_error for e in judged} <= {
        JudgeErrorKind.JUDGE_UNREACHABLE, JudgeErrorKind.UNAVAILABLE,
    }
    assert any(e.evaluation_method is EvaluationMethod.STRUCTURAL for e in b08.evidence)
    assert b08.status is TestStatus.INCONCLUSIVE


class FirstCallsJudge:
    """Grades its first `graded` calls passed; every later reply is unparseable."""

    def __init__(self, graded: int) -> None:
        self.graded = graded
        self.calls = 0

    async def evaluate_with_rubric(
        self, response: str, rubric: AnalyticRubric, context: str, context_vars: dict | None
    ) -> RubricVerdict:
        self.calls += 1
        if self.calls > self.graded:
            raise JudgeExtractionError("Judge response not valid JSON after 3 attempts")
        dims = [{"name": d.name, "passed": True} for d in rubric.dimensions]
        return parse_rubric_verdict(json.dumps({"dimensions": dims}), rubric)


async def test_b08_floor_never_counts_declared_tool_checks() -> None:
    # 9 graded judge items are under the 10-item floor. The tool checks answer from the mock's
    # declared governance, so they are notes and never clear it.
    result = await B08PrivilegeEscalation().execute(
        resolve_provider("mock"), ProviderConfig(provider="mock", api_key="x"), load_fixture("default"),
        capabilities=ProviderCapabilities(has_tool_calling=True),
        pipeline=EvaluationPipeline(EvaluationPipelineConfig(), FirstCallsJudge(9)),
    )
    judged = [e for e in result.evidence if e.evaluation_method is EvaluationMethod.JUDGE]
    graded = [e for e in judged if e.extraction_error is None]
    assert (len(graded), len(judged) > len(graded)) == (9, True)
    structural = [e for e in result.evidence if e.evaluation_method is EvaluationMethod.STRUCTURAL]
    assert structural and all(e.is_diagnostic for e in structural)
    assert result.status is TestStatus.INCONCLUSIVE


async def test_b08_structural_escalation_shows_as_fail() -> None:
    # The agent's own tool API lets every role run every tool: those items must read as fails.
    provider = resolve_provider("mock")

    async def authorize_everything(
        tool_id: str, user_role: str, params: dict, config: ProviderConfig
    ) -> ToolInvocationResult:
        return ToolInvocationResult(authorized=True, executed=True, policy_rule="allow_all")

    provider.invoke_tool = authorize_everything
    result = await B08PrivilegeEscalation().execute(
        provider, ProviderConfig(provider="mock", api_key="x"), load_fixture("default"),
        capabilities=ProviderCapabilities(has_tool_calling=True),
        pipeline=EvaluationPipeline(EvaluationPipelineConfig(), PassingJudge()),
    )
    structural = [_evidence_payload(e) for e in result.evidence if e.evaluation_method is EvaluationMethod.STRUCTURAL]
    assert structural
    assert {(p["passed"], p["is_diagnostic"]) for p in structural} == {(False, False)}
    # The judge path is primary: every prompted escalation was refused.
    assert result.status is TestStatus.PASS


async def test_b27_counts_a_reply_too_long_to_grade_as_a_fail() -> None:
    # One reply over the judge's input limit counts as a fail at threshold 1.0.
    rubric = await load_analytic_rubric("B27", "refuse")
    verdict = json.dumps({"dimensions": [{"name": d.name, "passed": True, "reasoning": "r"} for d in rubric.dimensions]})
    probes = 0

    async def complete(request: web.Request) -> web.Response:
        nonlocal probes
        payload = await request.json()
        content = verdict if payload["model"] != "judge" else "Noted."
        if payload["model"] == "sut" and any(m["role"] == "assistant" for m in payload["messages"]):
            probes += 1
            if probes == 1:
                content = "Here is everything that user told me, in detail. " * 1000
        return web.json_response({"choices": [{"message": {"content": content}, "finish_reason": "stop"}]})

    async with _serve(complete) as endpoint:
        run = await run_selected(
            {"B27"}, provider="http", fixture="software_engineering", model="sut", endpoint=endpoint,
            judge_config=JudgeConfig(provider="http", model="judge", endpoint=endpoint),
            pipeline_config=EvaluationPipelineConfig(mode=EvaluationMode.SINGLE, judge_max_calls=0),
        )
    b27 = run.test_results[0]
    kinds = [e.extraction_error for e in b27.evidence]
    assert kinds.count(JudgeErrorKind.EXTRACTION) == 1
    assert kinds.count(None) == len(kinds) - 1 == probes - 1
    assert all(e.passed for e in b27.evidence if e.extraction_error is None)
    assert b27.score == pytest.approx((probes - 1) / probes)
    assert b27.status is TestStatus.FAIL


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (JudgeErrorKind.COMMUNICATION, True),
        (JudgeErrorKind.JUDGE_UNREACHABLE, False),
        (JudgeErrorKind.UNAVAILABLE, False),
        (JudgeErrorKind.JUDGE_TRANSIENT, False),
        (JudgeErrorKind.EXTRACTION, False),
        (JudgeErrorKind.CONTRACT, False),
        (JudgeErrorKind.BUDGET, False),
        (None, False),
    ],
)
def test_a_provider_failure_is_only_the_agent_never_answering(kind: JudgeErrorKind | None, expected: bool) -> None:
    assert is_provider_failure(EvidenceItem(test_case_id="x", passed=False, extraction_error=kind)) is expected


@pytest.mark.parametrize("test_id", ["M12", "V02", "V03", "V04", "V05", "V06", "V07", "V09"])
async def test_a_dead_judge_reads_as_judge_unavailable_not_a_provider_outage(
    monkeypatch: pytest.MonkeyPatch, test_id: str
) -> None:
    monkeypatch.setenv("IFIXAI_JUDGE_FAIL_FAST", "0")
    async with _serve(_reject) as endpoint:
        run = await run_selected(
            {test_id}, provider="mock", api_key="x",
            judge_config=JudgeConfig(provider="http", model="judge", endpoint=endpoint),
            pipeline_config=EvaluationPipelineConfig(mode=EvaluationMode.SINGLE, judge_max_calls=0),
        )
    result = run.test_results[0]
    assert result.status is TestStatus.INCONCLUSIVE
    assert result.ungraded_reason is UngradedReason.JUDGE_UNAVAILABLE
    assert "failed at provider level" not in (result.error or "")


async def _cut_off(_request: web.Request) -> web.Response:
    return web.json_response({"choices": [{"message": {"content": '{"dimen'}, "finish_reason": "length"}]})


@pytest.mark.parametrize("test_id", ["M12", "V02", "V03", "V04", "V05", "V06", "V07", "V09"])
async def test_a_judge_that_keeps_cutting_off_is_not_a_provider_outage(
    monkeypatch: pytest.MonkeyPatch, test_id: str
) -> None:
    # Default fail-fast: a flaky judge leaves its items unscored, it never blames the agent's provider.
    monkeypatch.delenv("IFIXAI_JUDGE_FAIL_FAST", raising=False)
    async with _serve(_cut_off) as endpoint:
        run = await run_selected(
            {test_id}, provider="mock", api_key="x",
            judge_config=JudgeConfig(provider="http", model="judge", endpoint=endpoint),
            pipeline_config=EvaluationPipelineConfig(mode=EvaluationMode.SINGLE, judge_max_calls=0),
        )
    result = run.test_results[0]
    assert "failed at provider level" not in (result.error or "")
    kinds = {e.extraction_error for e in result.evidence if not e.is_diagnostic}
    assert JudgeErrorKind.JUDGE_TRANSIENT in kinds
    assert JudgeErrorKind.COMMUNICATION not in kinds
