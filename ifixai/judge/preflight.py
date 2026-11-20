"""One small call to each configured judge before the run's first request to the agent.

A judge that can't be reached, refuses its key or account, or has no such model would otherwise
leave the run sending hundreds of probes that no judge grades. The check is the judge-side twin of
the agent's connection test, and like it, it spends one call, which counts against the run's judge
budget like any other. Other failures (a 5xx, a rate limit) can clear by the time grading starts,
so they never stop the run here.
"""

import asyncio
from collections.abc import Awaitable, Callable

from ifixai.core.connection import ConnectionTestResult, test_connection
from ifixai.core.types import ProviderConfig
from ifixai.judge.config import JudgeConfig
from ifixai.judge.evaluator import JudgeEvaluator, judge_refusal
from ifixai.providers.base import (
    ChatProvider,
    ProviderConnectionError,
    ProviderError,
    ProviderTimeoutError,
    names_unknown_model,
    refusal_status_code,
)

# The same check the agent gets (`core.connection.test_connection`), one call per judge.
ConnectionCheck = Callable[[ChatProvider, ProviderConfig], Awaitable[ConnectionTestResult]]
_UNREACHABLE: tuple[type[BaseException], ...] = (
    ProviderConnectionError,
    ProviderTimeoutError,
    asyncio.TimeoutError,
    TimeoutError,
)


def needs_judge_preflight(eval_mode: str, judge_config: JudgeConfig | None) -> bool:
    """Whether the run has a judge of its own to check: not in self or deterministic mode, and not
    when every judge is the mock."""
    if judge_config is None or eval_mode == "self":
        return False
    return any(config.provider.lower() != "mock" for config in judge_configs(judge_config))


def judge_configs(judge_config: JudgeConfig) -> list[JudgeConfig]:
    """One config per judge: the single judge, or each member of a panel."""
    if judge_config.providers is None:
        return [judge_config]
    return [
        JudgeConfig(
            provider=spec.provider,
            model=spec.model,
            api_key=spec.api_key,
            endpoint=judge_config.endpoint,
            temperature=judge_config.temperature,
            timeout=judge_config.timeout,
        )
        for spec in judge_config.providers
    ]


def judge_preflight_calls(judge_config: JudgeConfig) -> int:
    """How many calls `preflight_judges` sends: one per judge that isn't the mock."""
    return sum(config.provider.lower() != "mock" for config in judge_configs(judge_config))


def preflight_problem(exc: BaseException, config: JudgeConfig) -> str | None:
    """The plain line that stops the run for this judge failure, or None to let the run go on."""
    refusal = judge_refusal(exc)
    if refusal is not None:
        return f"{refusal.what}. {refusal.fix}."
    if isinstance(exc, _UNREACHABLE):
        return (
            f"the judge's provider ({config.provider}) can't be reached. Check its base URL, the "
            "network and any proxy, then run again."
        )
    if isinstance(exc, ProviderError) and (
        refusal_status_code(exc.details, ("404",)) is not None or names_unknown_model(exc.details)
    ):
        # A 404 can also be a wrong base URL path, so the line names both.
        model = config.model or "its default model"
        return (
            f"the judge's provider ({config.provider}) has no model {model}. Pick another with "
            "--judge-model, or check the judge's base URL, then run again."
        )
    return None


async def preflight_judges(
    judge_config: JudgeConfig, check: ConnectionCheck = test_connection
) -> str | None:
    """Check every configured judge in turn; the first stop line, or None when all can grade."""
    for config in judge_configs(judge_config):
        if config.provider.lower() == "mock":
            continue
        try:
            evaluator = JudgeEvaluator(config)
        except ValueError as exc:
            # A judge whose SDK isn't installed can't grade either; the resolver says which
            # extra to install.
            return f"{exc}, then run again."
        pair = evaluator.provider_pair()
        try:
            result = await check(pair["provider"], pair["config"])
        finally:
            await evaluator.aclose()
        if not result.success and result.error is not None:
            problem = preflight_problem(result.error, config)
            if problem is not None:
                return problem
    return None
