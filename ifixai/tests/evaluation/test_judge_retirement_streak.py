"""One bad judge reply never ends grading: only 3 failures in a row retire a judge."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from ifixai.core.types import (
    AnalyticRubric,
    ChatMessage,
    EvaluationCriteria,
    EvaluationPipelineConfig,
    EvidenceItem,
    ExpectedClaim,
    JudgeErrorKind,
    PipelineResult,
    ProviderConfig,
    RubricDimension,
)
from ifixai.evaluation import analytic_judge
from ifixai.evaluation.analytic_judge import (
    AnalyticRubricJudge,
    EnsembleAnalyticRubricJudge,
)
from ifixai.evaluation.atomic_claims import (
    score_atomic_claims,
    score_atomic_claims_with_ground_truth,
)
from ifixai.evaluation.errors import JudgeUnavailableError
from ifixai.evaluation.pipeline import EvaluationPipeline
from ifixai.inspections.b16_silent_failure.runner import B16SilentFailure
from ifixai.judge.config import JudgeConfig
from ifixai.judge.evaluator import JudgeEvaluator
from ifixai.judge.fallbacks import FALLBACKS_PATH_ENV_VAR
from ifixai.providers.base import (
    ProviderAuthError,
    ProviderConnectionError,
    ProviderOverloadedError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderTruncatedError,
)

RUBRIC = AnalyticRubric(test_id="B16", outcome_type="refuse", dimensions=[
    RubricDimension(name="no_leak", description="Leaks nothing.", weight=0.5, mandatory=True),
    RubricDimension(name="clear", description="Says why.", weight=0.5),
])
VERDICT = json.dumps({"dimensions": [
    {"name": "no_leak", "passed": True, "reasoning": "r"},
    {"name": "clear", "passed": True, "reasoning": "r"},
]})
# What a provider raises for each failure, worded as the real adapters word it.
FAILURES: dict[str, Callable[[], Exception]] = {
    "drop": lambda: ProviderConnectionError(provider="mock", endpoint="loopback", details="connection dropped"),
    "timeout": lambda: ProviderTimeoutError(provider="mock", endpoint="loopback", details="read timed out"),
    "too_long": lambda: ProviderResponseError(
        provider="mock", endpoint="loopback", details="HTTP 400: prompt is too long for this model"),
    "gateway": lambda: ProviderOverloadedError(provider="mock", endpoint="loopback", details=(
        "Upstream aborted the generation (finish_reason=error, code=502, 0 chars returned): upstream down")),
    "cut": lambda: ProviderTruncatedError(provider="mock", endpoint="loopback", details="finish_reason=length"),
    "no_choices": lambda: ProviderResponseError(
        provider="mock", endpoint="loopback", details="No choices in response (id=x)"),
    "dead_model": lambda: ProviderResponseError(provider="gemini", endpoint="", details=(
        "404 models/gemini-dead is not found for API version v1beta, or is not supported for generateContent.")),
    "quota": lambda: ProviderResponseError(provider="bedrock", endpoint="", details=(
        "An error occurred (ServiceQuotaExceededException) when calling the Converse operation: "
        "Your account's quota for this model is exhausted.")),
    "throttle": lambda: ProviderRateLimitError(
        provider="minimax", endpoint="", details='HTTP 429: {"error": {"code": 1002, "message": "rate limit exceeded"}}'),
    "spent": lambda: ProviderRateLimitError(
        provider="minimax", endpoint="", details='HTTP 429: {"error": {"code": 1008, "message": "insufficient balance"}}'),
    "auth": lambda: ProviderAuthError(
        provider="mock", endpoint="loopback", details="HTTP 401: Incorrect API key provided"),
}


class ScriptedJudge(JudgeEvaluator):
    """A judge whose provider send does what `plan(item, model)` says: ok, junk or a FAILURES key."""

    def __init__(self, plan: Callable[[int, str | None], str], model: str = "judge-a") -> None:
        super().__init__(JudgeConfig(provider="mock", model=model))
        self.plan = plan
        self.seen: list[int] = []
        self.sends: list[tuple[str | None, int | None]] = []
        self.calls: dict[int, int] = {}

    async def send_message(self, messages: list[ChatMessage], config: ProviderConfig) -> str:
        await super().send_message(messages, config)
        item = int(messages[-1].content.split("item-")[1].split()[0])
        if item not in self.seen:
            self.seen.append(item)
        self.sends.append((config.model, config.max_tokens))
        self.calls[item] = self.calls.get(item, 0) + 1
        step = self.plan(item, config.model)
        if step == "ok":
            return VERDICT
        if step == "junk":
            return "not a verdict"
        raise FAILURES[step]()


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analytic_judge, "_BACKOFF_BASE", 0.0)


@pytest.fixture
def backup_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Gives the mock judge a fallback chain: judge-a, then judge-b."""
    chain = tmp_path / "fallbacks.json"
    chain.write_text(json.dumps({"providers": {"mock": {"models": [{"model": "judge-b"}]}}}), encoding="utf-8")
    monkeypatch.setenv(FALLBACKS_PATH_ENV_VAR, str(chain))


async def _grade_items(judge: ScriptedJudge, count: int) -> list[PipelineResult]:
    pipeline = EvaluationPipeline(EvaluationPipelineConfig(judge_max_calls=0), AnalyticRubricJudge(judge))
    return [
        await pipeline.evaluate(f"I won't. item-{n} end", EvaluationCriteria(), RUBRIC)
        for n in range(1, count + 1)
    ]


def _graded(results: list[PipelineResult]) -> int:
    return sum(1 for r in results if r.extraction_error is None and r.passed)


def _unscored(results: list[PipelineResult]) -> int:
    return sum(1 for r in results if r.extraction_error is not None)


@pytest.mark.parametrize("failure", ["cut", "drop", "gateway", "throttle"])
async def test_an_item_no_judge_model_answered_stays_out_of_fail_closed_scores(failure: str) -> None:
    judge = ScriptedJudge(lambda item, _model: failure if item == 1 else "ok")
    results = await _grade_items(judge, 2)
    assert [r.extraction_error for r in results] == [JudgeErrorKind.JUDGE_TRANSIENT, None]
    evidence = [
        EvidenceItem(test_case_id=str(n), passed=r.passed, extraction_error=r.extraction_error)
        for n, r in enumerate(results)
    ]
    # B16 counts other unscored items as fails; a judge that never answered is not one.
    assert B16SilentFailure().compute_score(evidence) == 1.0


async def test_an_unparseable_verdict_stays_an_extraction_error() -> None:
    judge = ScriptedJudge(lambda item, _model: "junk" if item == 1 else "ok")
    results = await _grade_items(judge, 2)
    assert [r.extraction_error for r in results] == [JudgeErrorKind.EXTRACTION, None]


async def test_three_failures_in_a_row_retire_the_judge() -> None:
    judge = ScriptedJudge(lambda item, _model: "drop" if item >= 3 else "ok")
    results = await _grade_items(judge, 5)
    assert judge.seen == [1, 2, 3]
    assert (_graded(results), _unscored(results)) == (0, 5)


async def test_a_transient_gateway_failure_without_a_status_is_retried() -> None:
    judge = ScriptedJudge(lambda item, _model: "gateway" if judge.calls[item] == 1 else "ok")
    results = await _grade_items(judge, 2)
    assert (_graded(results), _unscored(results)) == (2, 0)
    assert judge.calls == {1: 2, 2: 2}


# A refused key or spent account stops the run with the plain stop line (credit or key, and
# how to fix it); so does a dead model, with its own.
@pytest.mark.parametrize(("failure", "sends", "stop_line"), [
    ("dead_model", 3, r"the judge's provider \(gemini\) has no such model"), ("quota", 1, "ran out of credit or quota"),
    ("spent", 1, "ran out of credit or quota"), ("auth", 1, "refused its API key"),
])
async def test_a_dead_model_or_rejected_key_stops_the_run_on_the_first_item(
    monkeypatch: pytest.MonkeyPatch, failure: str, sends: int, stop_line: str
) -> None:
    monkeypatch.delenv("IFIXAI_JUDGE_FAIL_FAST", raising=False)
    judge = ScriptedJudge(lambda _item, _model: failure)
    with pytest.raises(JudgeUnavailableError, match=stop_line):
        await _grade_items(judge, 5)
    assert judge.seen == [1]
    assert len(judge.sends) == sends


@pytest.mark.parametrize(("failure", "stop_line"), [
    ("quota", "ran out of credit or quota"), ("spent", "ran out of credit or quota"),
    ("auth", "refused its API key"),
])
async def test_a_spent_or_rejected_key_stops_the_run_after_a_verdict_too(
    monkeypatch: pytest.MonkeyPatch, failure: str, stop_line: str
) -> None:
    # The key graded item 1, then the account ran out: every later call would be refused.
    monkeypatch.delenv("IFIXAI_JUDGE_FAIL_FAST", raising=False)
    judge = ScriptedJudge(lambda item, _model: "ok" if item == 1 else failure)
    with pytest.raises(JudgeUnavailableError, match=stop_line):
        await _grade_items(judge, 5)
    assert judge.seen == [1, 2]
    assert len(judge.sends) == 2


# Before any verdict, an empty reply stops the run on the fifth item. It walks the chain and
# retires the model after three in a row, so items 4 and 5 fail without a send.
async def test_an_empty_reply_stops_the_run_after_five_items(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IFIXAI_JUDGE_FAIL_FAST", raising=False)
    judge = ScriptedJudge(lambda _item, _model: "no_choices")
    with pytest.raises(JudgeUnavailableError, match="gave no verdict on its first 5 calls"):
        await _grade_items(judge, 6)
    assert judge.seen == [1, 2, 3]


async def test_rejected_key_with_fail_fast_off_unscores_each_item_as_judge_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IFIXAI_JUDGE_FAIL_FAST", "0")
    judge = ScriptedJudge(lambda _item, _model: "auth")
    results = await _grade_items(judge, 5)
    # A rejected key is not a failure that walks the chain or retires.
    assert judge.seen == [1, 2, 3, 4, 5]
    assert [r.extraction_error for r in results] == [JudgeErrorKind.JUDGE_UNREACHABLE] * 5


async def test_an_answer_resets_the_failure_streak() -> None:
    judge = ScriptedJudge(lambda item, _model: "drop" if item in (1, 2, 4, 5) else "ok")
    results = await _grade_items(judge, 7)
    assert judge.seen == [1, 2, 3, 4, 5, 6, 7]
    assert (_graded(results), _unscored(results)) == (3, 4)


async def test_first_judge_call_keeps_the_rubric_budget() -> None:
    judge = ScriptedJudge(lambda _item, _model: "ok")
    await _grade_items(judge, 1)
    # 2 dims size a 320-token verdict, raised to the 512 floor.
    assert judge.sends == [("judge-a", 512)]


@pytest.mark.parametrize(("model", "max_tokens"), [
    ("o4-mini", 512 + 4096),
    ("o3", 512 + 4096),
    ("o1-preview", 512 + 4096),
    ("openai/gpt-5-mini", 512 + 4096),
    ("deepseek/deepseek-r1", 512 + 4096),
    ("claude-3-7-sonnet-thinking", 512 + 4096),
    ("gemini-2.5-pro", 512 + 4096),
    ("google/gemini-3-pro-preview", 512 + 4096),
    ("qwen/qwq-32b", 512 + 4096),
    ("deepseek-reasoner", 512 + 4096),
    ("x-ai/grok-3-mini", 512 + 4096),
    ("grok-3-mini-fast", 512 + 4096),
    ("openai/gpt-oss-120b", 512 + 4096),
    ("gpt-3.5-turbo", 512),
    ("x-ai/grok-3", 512),
    ("gpt-4.1-mini", 512),
    ("gpt-4o-mini", 512),
    ("deepseek/deepseek-chat", 512),
    ("gemini-2.5-flash", 512),
    ("claude-sonnet-4-6", 512),
])
async def test_reasoning_judges_get_headroom_on_every_call(model: str, max_tokens: int) -> None:
    judge = ScriptedJudge(lambda item, _model: "cut" if item == 2 else "ok", model=model)
    await _grade_items(judge, 3)
    assert judge.sends == [(model, max_tokens)] * 3


class AtomicJudge(JudgeEvaluator):
    """Answers every atomic-claims call with one supported, correct claim."""

    def __init__(self, model: str) -> None:
        super().__init__(JudgeConfig(provider="mock", model=model))
        self.max_tokens: list[int | None] = []

    async def send_message(self, messages: list[ChatMessage], config: ProviderConfig) -> str:
        self.max_tokens.append(config.max_tokens)
        return '[{"claim":"c","supported":true,"response_correct":true,"reason":"r"}]'


@pytest.mark.parametrize(("model", "max_tokens"), [
    ("deepseek-reasoner", 2048 + 4096),
    ("openai/gpt-5-mini", 2048 + 4096),
    ("gpt-3.5-turbo", 2048),
    ("gpt-4o-mini", 2048),
])
async def test_atomic_claim_calls_get_the_same_headroom(model: str, max_tokens: int) -> None:
    judge = AtomicJudge(model)
    await score_atomic_claims("The sky is blue.", "knowledge_base", "grounding", judge)
    await score_atomic_claims_with_ground_truth(
        "The sky is blue.", [ExpectedClaim(claim="c", supported=False, reason="no source")], judge
    )
    assert judge.max_tokens == [max_tokens, max_tokens]


@pytest.mark.usefixtures("backup_model")
async def test_a_failed_item_goes_to_the_next_model() -> None:
    judge = ScriptedJudge(lambda _item, model: "drop" if model == "judge-a" else "ok")
    results = await _grade_items(judge, 2)
    assert judge.sends == ([("judge-a", 512)] * 3 + [("judge-b", 512)]) * 2
    assert _graded(results) == 2


@pytest.mark.usefixtures("backup_model")
@pytest.mark.parametrize("fail_fast", ["1", "0"])
@pytest.mark.parametrize("blip", ["timeout", "cut"])
async def test_an_earlier_recovered_hard_failure_does_not_make_a_later_blip_hard(
    monkeypatch: pytest.MonkeyPatch, fail_fast: str, blip: str
) -> None:
    # Item 1: judge-a rejects the prompt (400), judge-b grades it. Items 2-3: judge-a grades.
    # Item 4: both models only time out or cut off, so it is a transient miss, not a dead judge.
    monkeypatch.setenv("IFIXAI_JUDGE_FAIL_FAST", fail_fast)
    plan = {(1, "judge-a"): "too_long", (4, "judge-a"): blip, (4, "judge-b"): blip}
    judge = ScriptedJudge(lambda item, model: plan.get((item, model), "ok"))
    results = await _grade_items(judge, 5)
    assert judge.seen == [1, 2, 3, 4, 5]
    assert [r.extraction_error for r in results] == [None, None, None, JudgeErrorKind.JUDGE_TRANSIENT, None]
    assert _graded(results) == 4


@pytest.mark.parametrize(("failure", "kind"), [
    ("dead_model", JudgeErrorKind.JUDGE_UNREACHABLE),
    ("timeout", JudgeErrorKind.JUDGE_TRANSIENT),
])
async def test_items_after_retirement_keep_the_kind_that_retired_the_judge(
    monkeypatch: pytest.MonkeyPatch, failure: str, kind: JudgeErrorKind
) -> None:
    monkeypatch.setenv("IFIXAI_JUDGE_FAIL_FAST", "0")
    judge = ScriptedJudge(lambda _item, _model: failure)
    results = await _grade_items(judge, 5)
    assert judge.seen == [1, 2, 3]
    assert [r.extraction_error for r in results] == [kind] * 5


class TwoJudges:
    """Holds two scripted judges in order, the way an ensemble does."""

    def __init__(self, first: ScriptedJudge, second: ScriptedJudge) -> None:
        self.evaluators = [first, second]


@pytest.mark.parametrize(("one", "other", "kind"), [
    ("junk", "cut", JudgeErrorKind.EXTRACTION),
    ("auth", "junk", JudgeErrorKind.JUDGE_UNREACHABLE),
    ("auth", "cut", JudgeErrorKind.JUDGE_UNREACHABLE),
])
async def test_when_every_judge_fails_the_order_does_not_matter(
    monkeypatch: pytest.MonkeyPatch, one: str, other: str, kind: JudgeErrorKind
) -> None:
    monkeypatch.setenv("IFIXAI_JUDGE_FAIL_FAST", "0")
    for first, second in ((one, other), (other, one)):
        judges = TwoJudges(
            ScriptedJudge(lambda _item, _model, step=first: step, model="judge-a"),
            ScriptedJudge(lambda _item, _model, step=second: step, model="judge-b"),
        )
        pipeline = EvaluationPipeline(EvaluationPipelineConfig(judge_max_calls=0), EnsembleAnalyticRubricJudge(judges))
        result = await pipeline.evaluate("I won't. item-1 end", EvaluationCriteria(), RUBRIC)
        assert (first, second, result.extraction_error) == (first, second, kind)
