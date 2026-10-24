"""The shared multi-sample judge majority — the part of the judge path that touches no SUT.

Split out of `judge_probe`, which re-exports every name here, because `judge_probe` also owns
`collect_trajectory` and therefore a send path to the customer's agent. An inspection declared
no-SUT-contact (V08) can grade through the shared majority by importing this module without
naming one that can reach the agent. Nothing here imports a provider or a transport; the only
outbound call is `pipeline.evaluate`.
"""

import asyncio
from collections.abc import Callable
from typing import TypedDict

from ifixai.core.types import (
    AnalyticRubric,
    EvaluationCriteria,
    JudgeErrorKind,
    PipelineResult,
)
from ifixai.inspections.dimension_majority import (
    MajorityRecord,
    majority_record,
    new_sample_tally,
    tally_sample,
)
from ifixai.inspections.fanout import raise_first_error

# Default judge samples for a single (non-ensemble) judge: a majority over an odd count stabilises the
# verdict against residual single-judge variance without masking a real split (recorded as dispersion for audit).
DEFAULT_JUDGE_SAMPLES: int = 3


class JudgeDispersion(TypedDict):
    """How the 3-sample majority split, attached to evidence for audit."""

    passes: int
    total: int


class ConsensusResult(TypedDict):
    """The chosen pipeline result plus, on the multi-sample path, its dispersion.

    `dispersion` is None on the ensemble path (the ensemble already aggregates its
    own samples, so there is no per-call majority to record). `samples_asked` is how many
    judge samples were dispatched: one on the ensemble path."""

    result: PipelineResult
    dispersion: JudgeDispersion | None
    dimension_majority: MajorityRecord
    samples_asked: int


async def evaluate_with_consensus(
    pipeline: object,
    response: str,
    criteria: EvaluationCriteria,
    rubric: AnalyticRubric,
    context: str,
    context_vars: dict[str, str] | None = None,
    samples: int = DEFAULT_JUDGE_SAMPLES,
    correct: Callable[[PipelineResult], PipelineResult] | None = None,
    tally_dimensions: bool = False,
) -> ConsensusResult:
    """One judge call for an ensemble; a `samples`-way majority for a single judge.

    The majority vote stabilises a single judge's verdict on the same input run to
    run (the only residual non-determinism on the judge path once temperature is 0).
    An ensemble already aggregates internally, so it takes the single-call path and
    reports no dispersion.

    The samples are independent by construction, so they are issued concurrently:
    run serially they triple the latency of every evidence item on this path for no
    accuracy gain. Determinism is preserved because `gather` returns results in input
    order, so the majority vote and the verdict body selected from it are identical to
    the serial implementation's.

    `correct` is an OPTIONAL per-sample fixup applied to every sample BEFORE the vote, for a
    caller that can establish from its own catalogue that a dimension's verdict is not something
    the judge was entitled to decide (V02's catalogue-scoped auto-pass). It must run before the
    vote: correcting only the chosen body would leave `dispersion` describing verdicts that were
    never published, and would let uncorrected samples decide the majority. Default `None` leaves
    every other caller's behaviour untouched.

    A sample the judge could not deliver (any `extraction_error`) does not vote: counted as a fail,
    a judge outage read as the agent failing. A tie, or no verdict at all, leaves the item unscored.
    ``tally_dimensions`` publishes evidence for dimension-scoped gates; ``dispersion`` and the
    per-dimension tally count the same voting samples as the vote.
    """
    if pipeline.is_ensemble_judge():  # type: ignore[attr-defined]
        result = await pipeline.evaluate(  # type: ignore[attr-defined]
            response=response,
            criteria=criteria,
            rubric=rubric,
            references=None,
            context=context,
            context_vars=context_vars,
        )
        result = correct(result) if correct is not None else result
        tally = new_sample_tally()
        tally_sample(tally, result)
        return ConsensusResult(
            result=result,
            dispersion=None,
            dimension_majority=majority_record(tally, 1),
            samples_asked=1,
        )

    results: list[PipelineResult] = list(
        raise_first_error(
            await asyncio.gather(
                *[
                    pipeline.evaluate(  # type: ignore[attr-defined]
                        response=response,
                        criteria=criteria,
                        rubric=rubric,
                        references=None,
                        context=context,
                        context_vars=context_vars,
                    )
                    for _ in range(samples)
                ],
                return_exceptions=True,
            )
        )
    )
    if correct is not None:
        results = [correct(r) for r in results]
    # Only samples with a verdict vote, here as in `vote_on_samples`.
    voted = [r for r in results if r.extraction_error is None]
    dispersion = JudgeDispersion(passes=sum(1 for r in voted if r.passed), total=len(voted))
    tally = new_sample_tally()
    for result in results:
        tally_sample(tally, result)
    return ConsensusResult(
        result=vote_on_samples(results),
        dispersion=dispersion,
        dimension_majority=majority_record(tally, samples),
        samples_asked=samples,
    )


def vote_on_samples(results: list[PipelineResult]) -> PipelineResult:
    """The majority over the samples that returned a verdict; unscored on a tie or none."""
    voted = [r for r in results if r.extraction_error is None]
    pass_votes = sum(1 for r in voted if r.passed)
    if pass_votes * 2 == len(voted):
        lost = next((r for r in results if r.extraction_error is not None), None)
        return lost or results[0].model_copy(
            update={
                "passed": False,
                "extraction_error": JudgeErrorKind.EXTRACTION,
                "evaluation_result": "extraction_error: no clear verdict on this item",
            }
        )
    majority_passed = pass_votes * 2 > len(voted)
    return next(r for r in voted if r.passed == majority_passed)
