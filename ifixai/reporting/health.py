"""Run-health signals shared so every surface flags measurement failures and
grader fragility identically. The per-inspection scores hide an unreachable
model, a judge that kept breaking its verdict, or inspections that crashed; this
aggregates those so a confident-looking letter atop dead evidence is flagged the
same way everywhere.

Single source of truth, imported by `ifixai run` (ifixai/cli/run.py).
"""

from __future__ import annotations

from typing import NamedTuple

from ifixai.core.types import (
    EvidenceItem,
    JudgeErrorKind,
    TestResult,
    TestRunResult,
    TestStatus,
)
from ifixai.scoring.engine import _is_advisory, _is_attestation, _is_exploratory

# Judge-side outcomes that say nothing about the agent, so no score counts them,
# even where a spec counts its other unscored items as fails.
UNGRADED = frozenset(
    {
        JudgeErrorKind.BUDGET,
        JudgeErrorKind.JUDGE_UNREACHABLE,
        JudgeErrorKind.JUDGE_TRANSIENT,
        JudgeErrorKind.UNAVAILABLE,
    }
)


def counts_toward_score(item: EvidenceItem, count_extraction_errors_as_fail: bool) -> bool:
    """Whether a score counts `item`: a graded one, or, when the spec opts in, any other
    unscored item as a fail, except a budget skip or a dead, flaky or down judge."""
    if count_extraction_errors_as_fail:
        return item.extraction_error not in UNGRADED
    return item.extraction_error is None


def scored_evidence(
    evidence: list[EvidenceItem], count_extraction_errors_as_fail: bool
) -> list[EvidenceItem]:
    """The items a score counts (see `counts_toward_score`)."""
    return [e for e in evidence if counts_toward_score(e, count_extraction_errors_as_fail)]


def measured_count(
    evidence: list[EvidenceItem], count_extraction_errors_as_fail: bool
) -> int:
    """What the min_evidence_items floor counts: scored items, never diagnostics
    (a runner's compute_score already leaves those out)."""
    measurements = [e for e in evidence if not e.is_diagnostic]
    return len(scored_evidence(measurements, count_extraction_errors_as_fail))


def gating_results(result: TestRunResult) -> list[TestResult]:
    """The selected inspections that can gate a run: report-only ones never do."""
    return [
        r
        for r in result.test_results
        if not (_is_exploratory(r) or _is_advisory(r) or _is_attestation(r))
    ]


class _RunHealth(NamedTuple):
    n_inspections: int  # inspections that ran
    errored: int  # inspections that crashed before producing evidence (status ERROR)
    total: int  # measured evidence items (excludes operator diagnostics)
    scorable: int  # produced a graded result (no extraction_error)
    unreachable: int  # a model call failed to communicate (agent OR judge)
    judge_broke: int  # the judge replied but the verdict was unusable
    budget_skipped: int = 0  # no judge call was made after the run budget ran out
    judge_down: int = 0  # of `unreachable`, the ones where the judge rejected the call

    @property
    def attempted_probes(self) -> int:
        return self.total - self.budget_skipped

    @property
    def invalid(self) -> bool:
        """The run measured (almost) nothing — the letter grade rests on noise
        and must be read as a measurement failure, not a result. True when most
        inspections crashed before producing evidence, when every probe was
        unscorable, or when a model was unreachable on most probes."""
        if self.n_inspections and self.errored / self.n_inspections >= 0.5:
            return True  # most inspections crashed (no evidence to grade)
        if self.total == 0:
            # Inspections ran but emitted no gradeable evidence at all: the run
            # measured nothing. (No inspections at all -> nothing to flag.)
            return self.n_inspections > 0
        if self.scorable == 0:
            return True  # every probe was unscorable
        return self.attempted_probes > 0 and self.unreachable / self.attempted_probes >= 0.5

    @property
    def low_confidence(self) -> bool:
        """Not outright invalid, but under half the probes produced a graded
        reply — the grade is thin."""
        return self.total > 0 and not self.invalid and self.scorable / self.total < 0.5

    @property
    def judge_attempts(self) -> int:
        return self.scorable + self.judge_broke


def run_health(result: TestRunResult) -> _RunHealth:
    """Aggregate the failure signals the per-inspection scores hide. The engine
    already EXCLUDES an unscorable probe from a score (via its ``extraction_error``
    flag) and reports a crashed inspection as ERROR with no evidence, so a run
    where the model was unreachable, the judge kept breaking its verdict, or most
    inspections crashed can still print a confident letter from a handful of
    survivors. This counts those signals so the operator can flag a measurement
    failure (model unreachable / inspections crashed) or a weak grader (judge
    broke the contract) right next to the grade.

    COMMUNICATION is a failed call to the agent under test, UNAVAILABLE one to
    the judge; CONTRACT/EXTRACTION mean the judge answered but the verdict was
    unusable — unambiguously a grader-health problem. BUDGET means no judge
    call was made, so it is neither a transport nor a verdict failure.
    JUDGE_TRANSIENT (the judge timed out on an item) is a flaky grader;
    JUDGE_UNREACHABLE (the judge rejected the call) is a failed call, but on
    the judge's side.

    A scoped run exits on the inspections it selected, so a crashed report-only
    one can't invalidate it; a full run counts every inspection."""
    errored = total = scorable = unreachable = judge_broke = budget_skipped = judge_down = 0
    results = result.test_results
    if result.mandatory_minimums_not_run:
        results = gating_results(result) or results
    for br in results:
        if br.status == TestStatus.ERROR:
            errored += 1
        for ev in br.evidence:
            if ev.is_diagnostic:
                continue
            total += 1
            kind = ev.extraction_error
            if kind is None:
                scorable += 1
            elif kind in (JudgeErrorKind.COMMUNICATION, JudgeErrorKind.UNAVAILABLE):
                unreachable += 1
            elif kind in (JudgeErrorKind.CONTRACT, JudgeErrorKind.EXTRACTION, JudgeErrorKind.JUDGE_TRANSIENT):
                judge_broke += 1
            elif kind == JudgeErrorKind.BUDGET:
                budget_skipped += 1
            elif kind == JudgeErrorKind.JUDGE_UNREACHABLE:
                unreachable += 1
                judge_down += 1
            else:
                unreachable += 1
    return _RunHealth(
        len(results), errored, total, scorable, unreachable, judge_broke, budget_skipped, judge_down
    )


def measurement_failure_banner(health: _RunHealth) -> str | None:
    """A loud, grade-suppressing banner when the run did not actually measure the
    agent. Names the dominant cause so the remediation is right — a dead model
    points at the model id/key/endpoint, a broken grader points at the judge.
    Returns None for a healthy run."""
    if not health.invalid:
        return None
    if health.total == 0 and not health.errored:
        cause = (
            f"all {health.n_inspections} inspection(s) ran but produced no gradeable "
            "evidence. Check the model id / key / endpoint and re-run."
        )
    elif health.budget_skipped == health.total and not health.errored:
        cause = (
            f"all {health.budget_skipped} probes were skipped because the judge "
            "budget was exhausted; no judge call was made for those probes. "
            "Increase the judge budget or select fewer inspections and re-run."
        )
    elif health.scorable == 0 and health.judge_broke and health.unreachable == 0 and not health.errored:
        # Every probe reached a model but the judge produced no usable verdict —
        # a broken grader, not a model problem. Don't send the user to the model.
        cause = (
            f"the judge returned {health.judge_broke} unusable verdict(s) and not "
            "one produced a graded reply — a broken grader, not a model problem. "
            "Switch to a more reliable judge and re-run."
        )
    elif health.errored and health.errored >= health.unreachable:
        cause = (
            f"{health.errored} of {health.n_inspections} inspections crashed before "
            "producing any evidence (the agent under test or its configuration "
            "errored out). Check the model id / key / endpoint and re-run."
        )
    elif health.unreachable and health.judge_down == health.unreachable:
        cause = (
            f"the judge rejected its calls on {health.judge_down} of {health.attempted_probes} "
            "attempted probes, so they went ungraded. Check the judge's model id / key / "
            "endpoint and re-run."
        )
    else:
        cause = (
            f"{health.unreachable} of {health.attempted_probes} attempted probes had model calls fail to complete "
            "(the agent under test or a judge was unreachable). Check the model id / "
            "key / endpoint (the preflight catches a bad model id before billing) "
            "and re-run."
        )
    return (
        "*** RUN INVALID — measurement failure, not a result. ***\n"
        f"  {cause} The grade below is computed from almost no evidence — ignore "
        "the letter."
    )


def judge_health_note(health: _RunHealth) -> str | None:
    """A note when the judge broke its verdict contract often enough to make the
    grade untrustworthy. The broken probes were excluded from scoring, but an
    inspection that leaned on a flaky grader is not a reliable finding. Returns
    None when the judge held its contract."""
    if health.judge_broke == 0:
        return None
    return (
        f"⚠ Judge health: the grader returned {health.judge_broke} unusable "
        f"verdict(s) (broke the scoring contract) of {health.judge_attempts} "
        "grading attempts. Those probes were dropped, so inspections that leaned "
        "on a flaky grader are weak signal — prefer a stronger or independent "
        "judge for a result you can trust."
    )
