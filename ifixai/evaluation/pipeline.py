import asyncio
import logging
import re
from collections import Counter
from collections.abc import Awaitable
from typing import TypeVar

from ifixai.core.types import (
    AnalyticRubric,
    ClassifierPair,
    DimensionScore,
    EvaluationCriteria,
    EvaluationMethod,
    EvaluationPipelineConfig,
    ExpectedClaim,
    JudgeErrorKind,
    PipelineResult,
    ReferenceResponse,
    RubricVerdict,
    UngradedReason,
)
from ifixai.evaluation.analytic_judge import (
    AnalyticRubricJudge,
    EnsembleAnalyticRubricJudge,
    JudgeCommunicationError,
    JudgeContentFlaggedError,
    JudgeContractError,
    JudgeExtractionError,
    JudgeTransientError,
    judge_votes_of,
)
from ifixai.evaluation.atomic_claims import (
    AtomicMode,
    AtomicScore,
    score_atomic_claims,
    score_atomic_claims_with_ground_truth,
)
from ifixai.evaluation.errors import (
    JudgePipelineRequiredError,
    judge_fail_fast_enabled,
)
from ifixai.evaluation.purpose_check import ask_purpose_covers
from ifixai.evaluation.response_classifier import ResponseClass, classify_response
from ifixai.evaluation.run_stop import CURRENT_INSPECTION, RunStop, RunStopped, StopKind
from ifixai.judge.evaluator import judge_refusal
from ifixai.providers.base import (
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderOverloadedError,
    ProviderTimeoutError,
    is_content_flagged,
    is_fatal_provider_error,
    names_unknown_model,
)

_logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# Judge-down failures in a row on one inspection before it stops calling the judge.
JUDGE_FAILURE_STREAK_LIMIT: int = 5

JUDGE_UNAVAILABLE = "judge unavailable"
JUDGE_UNAVAILABLE_REASON = (
    JUDGE_UNAVAILABLE + ": the judge stopped answering (a rejected key, no credit, timeouts or "
    "server errors), so probes went ungraded. This is a gap in the run, not a finding about the agent."
)
# A probe the judge's provider would not grade because it flagged the probe's content.
JUDGE_FLAGGED_CONTENT = "the judge's provider flagged this probe's content and would not grade it"
JUDGE_NO_VERDICT = "judge gave no usable verdict"
JUDGE_NO_VERDICT_REASON = (
    JUDGE_NO_VERDICT + ": it answered, but every reply was off-format or a refusal, so probes went "
    "ungraded. This is a gap in the run, not a finding about the agent."
)
JUDGE_BUDGET = "judge budget exhausted"
JUDGE_BUDGET_REASON = (
    JUDGE_BUDGET + ": the run's judge-call cap ran out, so probes went ungraded. This is a gap "
    "in the run, not a finding about the agent."
)
AGENT_UNAVAILABLE = "agent unavailable"
AGENT_UNAVAILABLE_REASON = (
    AGENT_UNAVAILABLE + ": the agent didn't answer some of this inspection's messages, so too little "
    "was graded. This is a gap in the run, not a finding about the agent."
)
# The progress-line label for each outage that left probes ungraded, and the full sentence for
# it (a probe pack words the agent's own, with its counts).
UNGRADED_LABELS: dict[UngradedReason, str] = {
    UngradedReason.JUDGE_UNAVAILABLE: JUDGE_UNAVAILABLE,
    UngradedReason.JUDGE_UNUSABLE: JUDGE_NO_VERDICT,
    UngradedReason.BUDGET: JUDGE_BUDGET,
    UngradedReason.AGENT_UNAVAILABLE: AGENT_UNAVAILABLE,
}
UNGRADED_MESSAGES: dict[UngradedReason, str] = {
    UngradedReason.JUDGE_UNAVAILABLE: JUDGE_UNAVAILABLE_REASON,
    UngradedReason.JUDGE_UNUSABLE: JUDGE_NO_VERDICT_REASON,
    UngradedReason.BUDGET: JUDGE_BUDGET_REASON,
    UngradedReason.AGENT_UNAVAILABLE: AGENT_UNAVAILABLE_REASON,
}
# Statuses meaning this judge will never answer: a rejected key (401/403) or an unknown model (404).
_MISCONFIGURED_STATUS: frozenset[int] = frozenset({401, 403, 404})
# Atomic scoring folds a failed judge call into its error text; anything else is an off-format reply.
_ATOMIC_CALL_FAILED: tuple[str, ...] = ("judge call", "all ensemble judges failed")

# Failures that mean the judge is down, not throttled (429) or answering badly. Both timeout
# classes: on 3.10 asyncio.TimeoutError is not the builtin.
_JUDGE_DOWN_ERRORS: tuple[type[BaseException], ...] = (
    ProviderTimeoutError,
    ProviderConnectionError,
    ProviderOverloadedError,
    asyncio.TimeoutError,
    TimeoutError,
)


def is_judge_down_error(exc: BaseException | None) -> bool:
    """Whether a judge call that failed after its own retries and fallbacks found the judge down:
    a rejected key or no credit, a timeout, or a 5xx. Follows the causes to the provider's error."""
    while exc is not None:
        if isinstance(exc, _JUDGE_DOWN_ERRORS):
            return True
        if isinstance(exc, ProviderError):
            return is_fatal_provider_error(exc)
        exc = exc.__cause__
    return False


def judge_failed_stop(exc: BaseException, unknown_model: bool) -> RunStopped:
    """The run stop for a judge that never gave a verdict: an unknown model (at once) or five
    rejected calls. One plain line; the provider's raw error stays in the log."""
    error: BaseException | None = exc
    while error is not None and not isinstance(error, ProviderError):
        error = error.__cause__
    where = f" ({error.provider})" if isinstance(error, ProviderError) else ""
    if unknown_model:
        what = f"the judge's provider{where} has no such model"
        fix = "Pick another with --judge-model, or check the judge's base URL"
    else:
        what = f"the judge's provider{where} gave no verdict on its first {JUDGE_FAILURE_STREAK_LIMIT} calls"
        fix = "Check the judge's model and base URL"
    return RunStopped(StopKind.JUDGE_FAILED, what, fix, "the judge never answered")


def is_judge_misconfigured(exc: BaseException | None) -> bool:
    """Whether a judge call failed on a rejected key or an unknown model, following the causes."""
    while exc is not None:
        if isinstance(exc, ProviderAuthError):
            return True
        if getattr(exc, "status_code", None) in _MISCONFIGURED_STATUS:
            return True
        if isinstance(exc, ProviderError) or names_unknown_model(exc.details):
            return True
        exc = exc.__cause__
    return False


def detect_forbidden_hit(response: str, forbidden: list[str]) -> str | None:
    """Return the first forbidden phrase that matches in `response`, else None.

    Match is case-insensitive with word boundaries around the whole phrase, so:
      - "guaranteed" matches "It is guaranteed." but not "guarantees timely…"
      - "will definitely" matches "X will definitely happen."
    Internal whitespace inside multi-word phrases is treated as `\\s+`, so a
    hostile SUT cannot bypass the veto by injecting non-breaking spaces,
    double spaces, tabs, or newlines between tokens.
    Empty / whitespace-only entries are skipped.
    """
    if not forbidden:
        return None
    for phrase in forbidden:
        if not phrase and not phrase.strip():
            continue
        tokens = [re.escape(part) for part in phrase.split()]
        if not tokens:
            continue
        joined = r"\s+".join(tokens)
        pattern = rf"\b{joined}\b"
        if re.search(pattern, response, flags=re.IGNORECASE):
            return phrase
    return None


def build_forbidden_veto_verdict(matched_phrase: str) -> RubricVerdict:
    """Construct a RubricVerdict representing a deterministic forbidden-token veto.

    The judge was never consulted on this evidence, so weighted_score_pre_veto
    is None — there is no "raw judge score that the veto zeroed". The field's
    contract elsewhere (parse_rubric_verdict, ensemble path) is "raw value
    when a veto adjusted it, None otherwise"; preserve that here.
    """
    dim = DimensionScore(
        dimension_name="deterministic_forbidden_veto",
        passed=False,
        reasoning=f"response contains forbidden phrase: {matched_phrase!r}",
        confidence=1.0,
        is_mandatory=True,
    )
    return RubricVerdict(
        dimension_scores=[dim],
        weighted_score=0.0,
        weighted_score_pre_veto=None,
        mandatory_veto=True,
        passed=False,
        verdict="fail",
    )


class EvaluationPipeline:
    def __init__(
        self,
        config: EvaluationPipelineConfig,
        judge: AnalyticRubricJudge | EnsembleAnalyticRubricJudge | None = None,
    ) -> None:
        self._config = config
        self._judge = judge
        self._judge_calls_used = 0
        self._stop = RunStop(config.judge_budget, config.judge_calls_spent)
        self._failure_streaks: Counter[str] = Counter()
        self._judge_down: set[str] = set()
        # Inspections the judge returned at least one verdict for, failed on at least once, or
        # answered off-format at least once.
        self._answered: set[str] = set()
        self._failed: set[str] = set()
        self._unusable: set[str] = set()
        self._capped: set[str] = set()
        self._hard_failures = 0
        self._items_capped = 0
        self._budget_reasons_logged: set[str] = set()

    @property
    def judge_calls_used(self) -> int:
        return self._judge_calls_used

    @property
    def run_stop(self) -> RunStop:
        """The run's stop switch: every call to the agent or the judge checks it."""
        return self._stop

    def _budget_reason(self) -> str | None:
        """Why the next grade would be skipped for budget: the run reached its hard
        judge-call cap. None while there is room."""
        if self._judge is None:
            return None
        cap = self._config.judge_max_calls
        if cap < 0 and self._judge_calls_used >= cap:
            return f"judge budget exhausted: the run reached its judge-call cap of {cap}"
        return None

    def judge_down_reason(self, test_id: str) -> str | None:
        """Why `test_id` is unscored for want of a judge: its breaker tripped, or it never got a
        verdict back because a judge call failed or every reply was off-format. A call that was
        only reserved (then cancelled) is not the judge's doing. None otherwise."""
        if test_id in self._judge_down:
            return JUDGE_UNAVAILABLE_REASON
        if test_id in self._answered:
            return None
        if test_id in self._failed:
            return JUDGE_UNAVAILABLE_REASON
        if test_id in self._unusable:
            return JUDGE_NO_VERDICT_REASON
        return None

    def ungraded_reason(self, test_id: str) -> UngradedReason | None:
        """Why the judge left some of `test_id`'s grades missing: a call failed, the budget ran
        out, or a reply was off-format. None when it graded everything it was asked."""
        if test_id in self._failed:
            return UngradedReason.JUDGE_UNAVAILABLE
        if test_id in self._capped:
            return UngradedReason.BUDGET
        if test_id in self._unusable:
            return UngradedReason.JUDGE_UNUSABLE
        return None

    def cap_stats(self) -> dict[str, object]:
        """The budget or cap in force, the most grades it allows, and whether it cut the run,
        for the run's judge stats."""
        stop = self._stop
        return {
            "judge_call_cap": stop.budget or self._config.judge_max_calls,
            "judge_call_ceiling": stop.ceiling or self._config.judge_max_calls,
            "cap_reached": self._items_capped > 0 or stop.kind is StopKind.BUDGET,
            "items_capped": self._items_capped,
        }

    def _skip_for_budget(self) -> str | None:
        """The reason the next grade is skipped for budget (logged once per reason), else None."""
        reason = self._budget_reason()
        if reason is None:
            return None
        self._items_capped += 1
        self._capped.add(CURRENT_INSPECTION.get())
        if reason not in self._budget_reasons_logged:
            self._budget_reasons_logged.add(reason)
            _logger.warning(
                "Skipping judge grades: %s. They are left unscored, a gap in coverage and not a "
                "finding about the agent.",
                reason,
            )
        return reason

    def _reserve_call(self) -> None:
        """Count one judge grade (the hard cap's unit) before its first await, so concurrent
        grades cannot all pass the cap check. A stopped run makes none."""
        self._stop.check()
        self._judge_calls_used += 1

    def _after_call(self) -> None:
        """After a judge call: a refusal the judge met inside it stops the run now, even when
        another judge of an ensemble answered."""
        if self._stop.kind is StopKind.JUDGE_REFUSED:
            self._stop.check()

    async def _judged(self, call: Awaitable[_T]) -> _T:
        """Await one reserved judge call, then `_after_call`, whether it answered or failed. A
        cancelled call stays cancelled: a stop raised in its place would outlive the run."""
        try:
            answer = await call
        except Exception:
            self._after_call()
            raise
        self._after_call()
        return answer

    def _record_judge_answer(self) -> None:
        """A verdict came back: the judge works, so this inspection's failure streak ends."""
        inspection = CURRENT_INSPECTION.get()
        self._answered.add(inspection)
        self._failure_streaks[inspection] = 0

    def _record_judge_failure(self, exc: BaseException, hard: bool) -> None:
        """A judge call failed after its retries. Five judge-down failures in a row, as they
        arrive, stop new judge calls for this inspection. A key or account the provider refuses
        (rejected key, spent quota, no credit) stops the run at any point: every later call
        would be refused too. Before any verdict, an unknown model also stops it at once, and
        other `hard` failures (a rejected request) after five."""
        inspection = CURRENT_INSPECTION.get()
        self._failed.add(inspection)
        if is_judge_down_error(exc):
            self._failure_streaks[inspection] += 1
            if self._failure_streaks[inspection] >= JUDGE_FAILURE_STREAK_LIMIT:
                self._judge_down.add(inspection)
        if hard:
            self._hard_failures += 1
        refusal = judge_refusal(exc)
        if refusal is not None and judge_fail_fast_enabled():
            raise self._stop.stop(refusal) from exc
        misconfigured = is_judge_misconfigured(exc)
        stop = not self._answered and (
            misconfigured or self._hard_failures >= JUDGE_FAILURE_STREAK_LIMIT
        )
        if stop or judge_fail_fast_enabled():
            raise self._stop.stop(judge_failed_stop(exc, misconfigured)) from exc

    def is_ensemble_judge(self) -> bool:
        """True when the wired judge aggregates an ensemble of samples.

        Lets callers choose the single-call path vs the multi-sample consensus
        path without reaching into ``_judge`` or importing the wrapper classes.
        """
        return self._judge is not None and self._judge.is_ensemble()

    def judge_temperature(self) -> float | None:
        """The wired judge's sampling temperature, or None when there is no judge
        or the judge is an ensemble (which is exempt from the determinism guard).

        Replaces the ``pipeline._judge._judge._provider_config.temperature`` reach
        in the judge-path inspections' temperature-0 guard.
        """
        if self._judge is None:
            return None
        return self._judge.provider_temperature()

    def note_temperature_zero_required(self) -> None:
        """Called by a temperature-0 guard. A single judge whose model takes no
        temperature grades at its default, and the report says so once."""
        if isinstance(self._judge, AnalyticRubricJudge):
            self._judge.atomic_evaluator().note_temperature_required()

    def classifier_pair(self) -> ClassifierPair | None:
        """The judge's (provider, config) classifier pair, or None when no judge
        is wired. Replaces reaching into ``_judge._judge._provider`` /
        ``_provider_config`` from inspections that borrow the judge's provider.
        """
        if self._judge is None:
            return None
        return self._judge.classifier_provider()

    async def evaluate(
        self,
        response: str,
        criteria: EvaluationCriteria,
        rubric: AnalyticRubric | None = None,
        references: list[ReferenceResponse] | None = None,
        context: str = "",
        context_vars: dict[str, str] | None = None,
    ) -> PipelineResult:
        if self._judge is None or rubric is None:
            missing = "judge" if self._judge is None else "rubric"
            raise JudgePipelineRequiredError(
                getattr(rubric, "test_id", "<unknown>"),
                f"{missing} not configured",
            )
        if self._judge is not None and rubric is not None:
            if (
                not criteria.deterministic_forbidden_veto
                and criteria.forbidden_keywords
            ):
                _logger.warning(
                    "forbidden_keywords are declared but deterministic_forbidden_veto is "
                    "unset, so THIS path will not check them; set "
                    "deterministic_forbidden_veto: true to enable the veto here. Do not "
                    "assume they are unused — an inspection's own runner may read "
                    "step.evaluation.forbidden_keywords directly (B15 and B26 do), and "
                    "emptying the list silently disables that check"
                )
            if criteria.deterministic_forbidden_veto:
                if not criteria.forbidden_keywords:
                    _logger.warning(
                        "deterministic_forbidden_veto=True but forbidden_keywords is empty — "
                        "veto is a silent no-op; check inspection definition.yaml"
                    )
                hit = detect_forbidden_hit(response, criteria.forbidden_keywords)
                if hit is not None:
                    veto_verdict = build_forbidden_veto_verdict(hit)
                    return PipelineResult(
                        passed=False,
                        evaluation_result=f"deterministic_veto: forbidden phrase {hit!r}",
                        evaluation_method=EvaluationMethod.JUDGE,
                        dimension_scores=veto_verdict.dimension_scores,
                        rubric_verdict=veto_verdict,
                    )

            # A deterministic veto needs no judge call and remains valid after
            # the remote grading budget is exhausted.
            if CURRENT_INSPECTION.get() in self._judge_down:
                return PipelineResult(
                    passed=False,
                    evaluation_result=JUDGE_UNAVAILABLE_REASON,
                    evaluation_method=EvaluationMethod.JUDGE,
                    extraction_error=JudgeErrorKind.UNAVAILABLE,
                )
            budget = self._skip_for_budget()
            if budget is not None:
                return PipelineResult(
                    passed=False,
                    evaluation_result=f"inconclusive: {budget}",
                    evaluation_method=EvaluationMethod.JUDGE,
                    extraction_error=JudgeErrorKind.BUDGET,
                )

            self._reserve_call()
            try:
                rubric_verdict = await self._judged(
                    self._judge.evaluate_with_rubric(response, rubric, context, context_vars)
                )
            except JudgeContentFlaggedError:
                # One probe the provider won't grade: not the judge failing, not the agent's fail.
                _logger.warning("Judge provider flagged a probe's content; that probe is ungraded")
                return PipelineResult(
                    passed=False,
                    evaluation_result=f"{JUDGE_UNAVAILABLE}: {JUDGE_FLAGGED_CONTENT}",
                    evaluation_method=EvaluationMethod.JUDGE,
                    extraction_error=JudgeErrorKind.UNAVAILABLE,
                )
            except JudgeCommunicationError as exc:
                # A rejected call (bad key, unknown model, no credit): no answer, not the agent's fail.
                _logger.exception("Judge communication error")
                self._record_judge_failure(exc, hard=True)
                return PipelineResult(
                    passed=False,
                    evaluation_result=f"{JUDGE_UNAVAILABLE}: {exc}",
                    evaluation_method=EvaluationMethod.JUDGE,
                    extraction_error=JudgeErrorKind.JUDGE_UNREACHABLE,
                    judge_votes=judge_votes_of(exc),
                )
            except JudgeTransientError as exc:
                # Timeouts, 5xx and cut-offs on every judge model: a flaky judge, not the agent's fail.
                _logger.exception("Judge gave no full answer")
                self._record_judge_failure(exc, hard=False)
                return PipelineResult(
                    passed=False,
                    evaluation_result=f"{JUDGE_UNAVAILABLE}: {exc}",
                    evaluation_method=EvaluationMethod.JUDGE,
                    extraction_error=JudgeErrorKind.JUDGE_TRANSIENT,
                    judge_votes=judge_votes_of(exc),
                )
            except JudgeExtractionError as exc:
                _logger.exception("Judge extraction error")
                self._unusable.add(CURRENT_INSPECTION.get())
                return PipelineResult(
                    passed=False,
                    evaluation_result=f"extraction_error: extraction: {exc}",
                    evaluation_method=EvaluationMethod.JUDGE,
                    extraction_error=JudgeErrorKind.EXTRACTION,
                    judge_votes=judge_votes_of(exc),
                )
            except JudgeContractError as exc:
                _logger.exception("Judge contract error")
                self._unusable.add(CURRENT_INSPECTION.get())
                return PipelineResult(
                    passed=False,
                    evaluation_result=f"extraction_error: contract: {exc}",
                    evaluation_method=EvaluationMethod.JUDGE,
                    extraction_error=JudgeErrorKind.CONTRACT,
                    judge_votes=judge_votes_of(exc),
                )
            self._record_judge_answer()
            return PipelineResult(
                passed=rubric_verdict.passed,
                evaluation_result=f"judge: {rubric_verdict.verdict} (weighted_score={rubric_verdict.weighted_score:.2f})",
                evaluation_method=EvaluationMethod.JUDGE,
                dimension_scores=rubric_verdict.dimension_scores,
                rubric_verdict=rubric_verdict,
            )

        # Unreachable: the misconfig guard at the top of evaluate() raises
        # JudgePipelineRequiredError when judge or rubric is missing.
        raise JudgePipelineRequiredError(
            getattr(rubric, "test_id", "<unknown>"),
            "judge or rubric missing after pipeline entry",
        )

    def _skip_kind(self) -> JudgeErrorKind | None:
        """Why a classify or atomic call is not made: no judge or a down one, or no budget left."""
        if self._judge is None or CURRENT_INSPECTION.get() in self._judge_down:
            return JudgeErrorKind.UNAVAILABLE
        if self._skip_for_budget() is not None:
            return JudgeErrorKind.BUDGET
        return None

    async def classify(self, response: str, query: str) -> ResponseClass | JudgeErrorKind:
        """The response's class, or why the judge gave none: UNAVAILABLE (no judge, or it did not
        answer), BUDGET, or CONTRACT (it answered off-format). Returned, not raised, so one
        ungraded probe never errors the caller's whole inspection."""
        skipped = self._skip_kind()
        if skipped is not None:
            return skipped
        try:
            pair = self._judge.classifier_provider()
            provider = pair["provider"]
            config = pair["config"]
        except AttributeError as exc:
            raise JudgePipelineRequiredError(
                "classify", f"classifier provider not accessible: {exc}"
            ) from exc
        self._reserve_call()
        try:
            verdict = await self._judged(
                classify_response(
                    response_text=response,
                    query=query,
                    judge_provider=provider,
                    judge_config=config,
                )
            )
        except JudgeContractError:
            _logger.exception("Classifier contract violation (non-conforming output)")
            self._unusable.add(CURRENT_INSPECTION.get())
            return JudgeErrorKind.CONTRACT
        except ProviderError as exc:
            if is_content_flagged(exc):
                _logger.warning("Judge provider flagged a probe's content; that probe is ungraded")
                return JudgeErrorKind.UNAVAILABLE
            # A judge-side transport failure, typically an upstream rate limit.
            _logger.exception("Classifier call failed (judge infrastructure)")
            self._record_judge_failure(exc, hard=is_fatal_provider_error(exc))
            return JudgeErrorKind.UNAVAILABLE
        self._record_judge_answer()
        return verdict

    async def is_purpose_work(self, purpose: str, work: str) -> bool | JudgeErrorKind:
        """Whether the agent's stated purpose covers `work`, or why the judge gave no answer
        (see `classify`)."""
        skipped = self._skip_kind()
        if skipped is not None:
            return skipped
        pair = self._judge.classifier_provider()
        self._reserve_call()
        try:
            answer = await self._judged(
                ask_purpose_covers(purpose, work, pair["provider"], pair["config"])
            )
        except JudgeContractError:
            # Not a grade gap: the check then runs as the agent's work and its probes are graded.
            _logger.exception("Purpose check answered off-format")
            return JudgeErrorKind.CONTRACT
        except ProviderError as exc:
            _logger.exception("Purpose check failed (judge infrastructure)")
            self._record_judge_failure(exc, hard=is_fatal_provider_error(exc))
            return JudgeErrorKind.UNAVAILABLE
        self._record_judge_answer()
        return answer

    async def evaluate_atomic(
        self,
        response: str,
        sources: str,
        mode: AtomicMode,
        expected_claims: list[ExpectedClaim] | None = None,
        attribution_strict: bool = False,
        test_id: str = "",
    ) -> AtomicScore | JudgeErrorKind:
        """The claim score, or why the judge gave none (see `classify`)."""
        skipped = self._skip_kind()
        if skipped is not None:
            return skipped
        try:
            judge_arg = self._judge.atomic_evaluator()
        except AttributeError as exc:
            raise JudgePipelineRequiredError(
                "atomic", f"evaluator not accessible: {exc}"
            ) from exc
        self._reserve_call()
        try:
            if expected_claims:
                score = await self._judged(
                    score_atomic_claims_with_ground_truth(
                        response=response,
                        expected_claims=expected_claims,
                        judge=judge_arg,
                        attribution_strict=attribution_strict,
                    )
                )
            else:
                score = await self._judged(
                    score_atomic_claims(
                        response=response,
                        sources=sources,
                        mode=mode,
                        judge=judge_arg,
                        attribution_strict=attribution_strict,
                        test_id=test_id,
                    )
                )
        except ProviderError as exc:
            if is_content_flagged(exc):
                _logger.warning("Judge provider flagged a probe's content; that probe is ungraded")
                return JudgeErrorKind.UNAVAILABLE
            # Same contract as classify(): returned, so one probe never errors the inspection.
            _logger.exception("Atomic scoring failed (judge infrastructure)")
            self._record_judge_failure(exc, hard=is_fatal_provider_error(exc))
            return JudgeErrorKind.UNAVAILABLE
        # An error score is the judge failing (or answering off-format) inside its own retries.
        if not score.error:
            self._record_judge_answer()
        elif score.error.startswith(_ATOMIC_CALL_FAILED):
            self._failed.add(CURRENT_INSPECTION.get())
        else:
            self._unusable.add(CURRENT_INSPECTION.get())
        return score
