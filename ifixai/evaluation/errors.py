"""Typed exceptions for the evaluation pipeline.

These exceptions exist to distinguish *configuration* failures (the run
cannot produce a score because something was not wired up) from
*runtime* judge failures (the judge ran but could not deliver a
verdict). Configuration failures must surface to operators as ERROR;
runtime judge failures stay INCONCLUSIVE.
"""

import os


class JudgeUnavailableError(BaseException):
    """The judge is unreachable after its retry budget was exhausted.

    Raised (fail-fast) at any point when the provider refuses the judge's key or
    account (a rejected key, a spent quota, no credit), and before the judge ever
    answered on an unknown model (at once) or five other hard failures (a rejected
    request). Such a judge won't recover, so grinding through the remaining
    inspections would only produce a scorecard full of INCONCLUSIVE at real cost.
    A judge that times out or 5xxs mid-run only stops the affected inspections (see
    ``EvaluationPipeline.ungraded_reason``). Subclasses ``BaseException`` on purpose
    so it slips past the inspection runners' broad ``except Exception`` handlers
    and aborts the whole run, rather than being downgraded to a per-probe
    INCONCLUSIVE. Opt out with ``IFIXAI_JUDGE_FAIL_FAST=0`` to keep the old
    drop-and-continue behavior.
    """

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


def judge_fail_fast_enabled() -> bool:
    """Whether a judge that will not answer aborts the run.

    Default ON: a rejected key, spent quota or no credit stops the run at any
    point; before any verdict, an unknown model stops it at once, other hard
    failures after five.
    Set ``IFIXAI_JUDGE_FAIL_FAST=0`` (or false/no/off) to fall back to
    dropping the probe as INCONCLUSIVE and continuing.
    """
    return os.environ.get("IFIXAI_JUDGE_FAIL_FAST", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


class JudgePipelineRequiredError(RuntimeError):
    """A benchmark required the analytic judge pipeline but none was configured.

    Raised pre-flight by judge-dependent runners and by the evaluation
    pipeline when invoked without a judge or rubric. The harness maps
    this to ``TestStatus.ERROR`` (a configuration failure), distinct
    from the ``INCONCLUSIVE`` status used for runs where the judge ran
    but could not produce a score.
    """

    def __init__(self, test_id: str, reason: str) -> None:
        super().__init__(f"{test_id}: judge pipeline required ({reason})")
        self.test_id = test_id
        self.reason = reason
