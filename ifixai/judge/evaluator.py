import asyncio
from contextlib import nullcontext

from ifixai.core.types import ChatMessage, ClassifierPair, ProviderConfig
from ifixai.evaluation.errors import judge_fail_fast_enabled
from ifixai.evaluation.run_stop import RUN_STOP, RunStopped, StopKind
from ifixai.judge.config import JudgeConfig, JudgeProviderSpec
from ifixai.providers.base import (
    ChatProvider,
    ProviderAuthError,
    ProviderError,
    is_fatal_provider_error,
    names_spend_limit,
    refusal_status_code,
)
from ifixai.providers.resolver import resolve_provider


def judge_refusal(exc: BaseException | None) -> RunStopped | None:
    """The run stop for a call the judge's provider refused for its key or account (a rejected
    key, a spent quota or budget, no credit), following the causes. Any 403 counts, except one
    the provider sent because it flagged the request's content (`is_content_flagged`). None for
    any other failure: a call that can still succeed later never stops the run."""
    while exc is not None and not isinstance(exc, ProviderError):
        exc = exc.__cause__
    if exc is None or not is_fatal_provider_error(exc):
        return None
    status = refusal_status_code(exc.details)
    where = f" ({exc.provider}, HTTP {status})" if status else f" ({exc.provider})"
    spent = status == "403" and names_spend_limit(exc.details)
    if (isinstance(exc, ProviderAuthError) or status in {"401", "403"}) and not spent:
        what = f"the judge's provider refused its API key{where}"
        fix = "Swap in another key with --judge-api-key, or top up the judge's account"
    else:
        what = f"the judge's account ran out of credit or quota{where}"
        fix = "Top up the judge's account or swap in another key with --judge-api-key"
    return RunStopped(StopKind.JUDGE_REFUSED, what, fix, "the judge's key or credit ran out")


class _ClassifierJudgeProvider(ChatProvider):
    """Keep borrowed classifier sends on the evaluator's accounting boundary."""

    def __init__(self, evaluator: "JudgeEvaluator") -> None:
        self._evaluator = evaluator

    async def send_message(
        self, messages: list[ChatMessage], config: ProviderConfig
    ) -> str:
        return await self._evaluator.send_message(messages, config)

    async def aclose(self) -> None:
        await self._evaluator.aclose()


class JudgeEvaluator:

    def __init__(self, config: JudgeConfig) -> None:
        self._config = config
        self._provider = resolve_provider(config.provider)
        self._provider_config = ProviderConfig(
            provider=config.provider,
            api_key=config.api_key,
            model=config.model,
            endpoint=config.endpoint,
            timeout=config.timeout,
            temperature=config.temperature,
            role="judge",
        )
        self._call_count = 0
        self._cap_reached = False
        self._fallback_grades: dict[str, int] = {}
        self._transport_failures: dict[str, int] = {}
        self._temperature_required = False

    async def send_message(
        self, messages: list[ChatMessage], config: ProviderConfig
    ) -> str:
        """Count provider sends, including failed and retried grading attempts, each against
        the run's judge budget too.

        The retries the provider or its SDK makes inside one send count against
        the budget too (see `run_stop.count_request`), not in this count. A stopped
        run sends nothing, and neither does one whose budget's ceiling is spent; a
        refused key or account stops it at once.
        """
        stop = RUN_STOP.get()
        try:
            with nullcontext() if stop is None else stop.calling(judge=True):
                self._call_count += 1
                return await self._provider.send_message(messages, config)
        except ProviderError as exc:
            refusal = judge_refusal(exc)
            if stop is not None and refusal is not None and judge_fail_fast_enabled():
                stop.stop(refusal)
            raise

    def provider_pair(self) -> ClassifierPair:
        return ClassifierPair(
            provider=_ClassifierJudgeProvider(self), config=self._provider_config
        )

    @property
    def provider_name(self) -> str:
        """The judge provider's registry name (e.g. ``openrouter``).

        Public accessor so the fallback chain can be looked up without reaching
        into ``_provider_config``.
        """
        return self._provider_config.provider

    @property
    def panel_identity(self) -> str:
        """`provider:model` as configured: how a panel vote names this judge."""
        model = self._provider_config.model or "<provider default>"
        return f"{self._provider_config.provider}:{model}"

    def note_fallback_grade(self, model: str) -> None:
        """Record that `model` graded a probe in place of the configured judge.

        A run where a third of the verdicts came from a different model is a
        materially different measurement, so the substitution is counted and
        surfaced in the run stats rather than left in the logs.
        """
        self._fallback_grades[model] = self._fallback_grades.get(model, 0) + 1

    def note_transport_failure(self, model: str) -> None:
        """Record one judge call that never returned a verdict.

        A grade that only succeeded on the third attempt looks identical to a
        clean one in the scorecard. Counting the failed attempts is what lets
        the report say how hard the judge had to work for the number.
        """
        self._transport_failures[model] = self._transport_failures.get(model, 0) + 1

    def note_temperature_required(self) -> None:
        """A temperature-0 guard read this judge. A model that takes no temperature
        still grades, at its default sampling, and the report says so once."""
        self._temperature_required = True

    @property
    def temperature(self) -> float:
        """The judge provider's sampling temperature.

        Public accessor so callers (e.g. the judge-path inspections' determinism
        guard) can read the temperature without reaching into ``_provider_config``.
        """
        return self._provider_config.temperature

    @property
    def cap_reached(self) -> bool:
        return self._cap_reached

    def get_stats(self) -> dict[str, object]:
        """Provider-send counts; items_escalated remains the legacy count alias.

        These are not the pipeline's logical-grade budget or a count of SDK
        internal retries, which happen inside a single provider send.
        """
        return {
            "total_calls": self._call_count,
            "items_escalated": self._call_count,
            "cap_reached": self._cap_reached,
            "items_capped": 0,
            "judge_model": self._config.model or self._config.provider,
            "judge_provider": self._config.provider,
            "fallback_grades": dict(self._fallback_grades),
            "judge_transport_failures": dict(self._transport_failures),
            # Read at report time: Anthropic learns a model refuses temperature on its first call.
            "default_temperature": self._temperature_required
            and not self._provider.accepts_temperature(
                self._provider_config.model, self._provider_config.endpoint
            ),
        }

    async def aclose(self) -> None:
        await self._provider.aclose()


class EnsembleJudgeEvaluator:

    def __init__(self, config: JudgeConfig) -> None:
        if config.providers is None or len(config.providers) < 2:
            raise ValueError(
                f"EnsembleJudgeEvaluator requires >=2 providers, got "
                f"{len(config.providers) if config.providers else 0}"
            )
        self._config = config
        self._evaluators: list[JudgeEvaluator] = [
            JudgeEvaluator(_single_config_for(spec, config))
            for spec in config.providers
        ]

    @property
    def cap_reached(self) -> bool:
        return all(e.cap_reached for e in self._evaluators)

    @property
    def evaluators(self) -> list[JudgeEvaluator]:
        return list(self._evaluators)

    def get_stats(self) -> dict[str, object]:
        per_judge_stats = [e.get_stats() for e in self._evaluators]
        return {
            "total_calls": sum(s["total_calls"] for s in per_judge_stats),
            "items_escalated": sum(s["items_escalated"] for s in per_judge_stats),
            "cap_reached": self.cap_reached,
            "items_capped": sum(s["items_capped"] for s in per_judge_stats),
            "judge_model": "ensemble",
            "judge_provider": f"ensemble({len(self._evaluators)})",
            "fallback_grades": merge_counts(per_judge_stats, "fallback_grades"),
            "judge_transport_failures": merge_counts(
                per_judge_stats, "judge_transport_failures"
            ),
            "per_judge_stats": per_judge_stats,
        }

    async def aclose(self) -> None:
        outcomes = await asyncio.gather(
            *(evaluator.aclose() for evaluator in self._evaluators),
            return_exceptions=True,
        )
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome


def merge_counts(
    per_judge_stats: list[dict[str, object]], key: str
) -> dict[str, int]:
    """Sum each ensemble member's per-model counter under `key` into one tally."""
    merged: dict[str, int] = {}
    for stats in per_judge_stats:
        counts = stats.get(key) or {}
        if not isinstance(counts, dict):
            continue
        for model, count in counts.items():
            merged[model] = merged.get(model, 0) + int(count)
    return merged


def _single_config_for(
    spec: JudgeProviderSpec,
    parent: JudgeConfig,
) -> JudgeConfig:
    return JudgeConfig(
        provider=spec.provider,
        model=spec.model,
        api_key=spec.api_key,
        endpoint=parent.endpoint,
        temperature=parent.temperature,
        max_calls_per_run=parent.max_calls_per_run,
        timeout=parent.timeout,
    )
