"""Stopping a run before it ends.

Three things stop one: the judge's provider refuses its key or its credit (every later call would
be refused too), the judge never gives a verdict (an unknown model, five rejected calls), or the
run needs a judge call past its budget plus the 25% it may go over. The
budget counts calls to the judge's provider, each retry and fallback model included. From then on
no probe goes to the agent and no call to the judge: every provider call runs inside
`RunStop.calling`, which refuses it once the run stopped and cuts it if the run stops while it is
out. Inspections already graded keep their results; the rest are reported as not run.

The run scopes its stop once, for every phase and task (`core.runner`); a script that runs
inspections itself sets `RUN_STOP` the same way.
"""

import asyncio
import math
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from typing import Any

from ifixai.evaluation.errors import JudgeUnavailableError

# How far past its judge budget a run may go before it stops.
BUDGET_OVERRUN: float = 0.25

# The inspection a call is made for, set while it executes: the pipeline keys its judge record
# (failure streak, ungraded probes) by it.
CURRENT_INSPECTION: ContextVar[str] = ContextVar("ifixai_current_inspection", default="")

# What a message to the agent is for, fixed by the code that sends it: True (the default) when the
# judge grades its reply, as it does every probe a probe pack or a judged inspection sends; False
# inside `judge_free`. Once the ceiling's calls are made, a probe the judge grades stops the run.
JUDGE_GRADES_REPLY: ContextVar[bool] = ContextVar("ifixai_judge_grades_reply", default=True)


class StopKind(str, Enum):
    JUDGE_REFUSED = "judge_refused"
    # The judge failed before any verdict: an unknown model, or five rejected calls.
    JUDGE_FAILED = "judge_failed"
    BUDGET = "budget"


class RunStopped(JudgeUnavailableError):
    """The run stopped early. `what` says why in plain words, `fix` how to get past it, and
    `not_run` labels the inspections it never finished. A BaseException (see the parent), so it
    passes every inspection's `except Exception` and ends the run, never one probe."""

    def __init__(self, kind: StopKind, what: str, fix: str, not_run: str) -> None:
        super().__init__(f"{what}. {fix}")
        self.kind = kind
        self.what = what
        self.fix = fix
        self.not_run = not_run

    def copy(self) -> "RunStopped":
        """A fresh instance, so concurrent tasks never share one traceback."""
        return RunStopped(self.kind, self.what, self.fix, self.not_run)


def budget_ceiling(budget: int) -> int:
    """The most judge calls a run with `budget` may make: 25% over it, rounded up. 0 = no limit."""
    return math.ceil(budget * (1 + BUDGET_OVERRUN)) if budget > 0 else 0


class RunStop:
    """One run's stop switch. The first reason wins; every later check raises it again."""

    def __init__(self, budget: int = 0, spent: int = 0) -> None:
        self.budget = budget
        self.ceiling = budget_ceiling(budget)
        # Calls sent to the judge's provider, retries included, from `spent`: the calls the run
        # made before its first inspection (the judge check).
        self.judge_calls = spent
        self._stopped: RunStopped | None = None
        # Tasks with a call out to the agent or the judge (True for the judge), and the ones
        # the stop cancelled.
        self._in_flight: dict[asyncio.Task[Any], bool] = {}
        self._cancelled: set[asyncio.Task[Any]] = set()
        # Requests the provider sent so far in each task's judge call.
        self._requests: dict[asyncio.Task[Any], int] = {}

    @property
    def kind(self) -> StopKind | None:
        return None if self._stopped is None else self._stopped.kind

    @property
    def spent(self) -> bool:
        """Whether the run made every judge call its ceiling allows."""
        return bool(self.ceiling) and self.judge_calls >= self.ceiling

    def stop(self, reason: RunStopped) -> RunStopped:
        """Stop the run for `reason` unless it already stopped; returns the error to raise."""
        if self._stopped is None:
            self._stopped = reason
            self._cancel_in_flight()
        return self._stopped.copy()

    def _cancel_in_flight(self) -> None:
        """Cut the calls still out: one that passed its check can wait inside its HTTP client
        and leave after the stop. On a budget stop a judge call already counted, so inside the
        ceiling, still finishes."""
        if not self._in_flight:
            return
        current = asyncio.current_task()
        keep_judge = self.kind is StopKind.BUDGET
        for task, judge in self._in_flight.items():
            if task is not current and not (judge and keep_judge):
                task.cancel()
                self._cancelled.add(task)

    def _admit(self, judge: bool, chat: bool) -> None:
        """Count a judge call, or stop the run in its place once the ceiling's calls are made.
        Spent, a probe the judge would grade (`JUDGE_GRADES_REPLY`) would go out to stay
        ungraded, so it stops the run too, whatever its inspection asked so far; hook calls and
        messages sent inside `judge_free` still go out."""
        if judge:
            if self.spent:
                raise self.budget_spent()
            self.judge_calls += 1
        elif chat and self.spent and JUDGE_GRADES_REPLY.get():
            raise self.budget_spent()

    @contextmanager
    def calling(self, judge: bool = False, chat: bool = False) -> Iterator[None]:
        """Around one call to the agent (`chat` for a message, not a hook), or with `judge` to the
        judge's provider: refused once the run stopped or would pass its ceiling, and ended by the
        stop, never a bare cancellation, if it stops meanwhile. A call made inside a judge call is
        that same call."""
        self.check()
        task = asyncio.current_task()
        inside_judge = task is not None and self._in_flight.get(task, False)
        if not inside_judge:
            self._admit(judge, chat)
        outer = task is not None and task not in self._in_flight
        if task is not None:
            self._in_flight[task] = inside_judge or judge
        try:
            yield
        except asyncio.CancelledError:
            stopped = self._stopped
            if stopped is None or task not in self._cancelled:
                raise
            self._cancelled.discard(task)
            # Python 3.11+ counts cancellations; this one is spent.
            uncancel = getattr(task, "uncancel", None)
            if uncancel is not None:
                uncancel()
            raise stopped.copy() from None
        finally:
            if outer and task is not None:
                self._in_flight.pop(task, None)
                self._cancelled.discard(task)
                self._requests.pop(task, None)

    def request(self) -> None:
        """Before each request a provider sends. In a judge call the first was counted with the
        call; each retry the provider or its SDK makes on its own is a call too, refused once the
        run stopped or the ceiling's calls are made."""
        task = asyncio.current_task()
        if task is None or not self._in_flight.get(task, False):
            return
        sent = self._requests.get(task, 0)
        self._requests[task] = sent + 1
        if sent:
            self.check()
            self._admit(judge=True, chat=False)

    def budget_spent(self) -> RunStopped:
        """Stop the run because its judge budget and the 25% over it are spent."""
        return self.stop(
            RunStopped(
                StopKind.BUDGET,
                f"the run used {self.judge_calls} judge calls, its budget of {self.budget} plus "
                f"the {BUDGET_OVERRUN:.0%} it may go over",
                "Raise the budget with --judge-budget",
                "budget",
            )
        )

    def check(self) -> None:
        """Raise the stop, if the run stopped."""
        if self._stopped is not None:
            raise self._stopped.copy()


# The stop switch of the run a call belongs to, set for the whole run.
RUN_STOP: ContextVar[RunStop | None] = ContextVar("ifixai_run_stop", default=None)


@contextmanager
def stopping(stop: RunStop) -> Iterator[None]:
    """Every call to the agent or the judge inside checks `stop`: the run's scope, for code that
    runs inspections without `core.runner`."""
    token = RUN_STOP.set(stop)
    try:
        yield
    finally:
        RUN_STOP.reset(token)


@contextmanager
def judge_free() -> Iterator[None]:
    """Messages sent inside get no judge grade, so they still go out once the run's judge calls
    are spent (until the run stops)."""
    token = JUDGE_GRADES_REPLY.set(False)
    try:
        yield
    finally:
        JUDGE_GRADES_REPLY.reset(token)


def count_request() -> None:
    """Before each request a provider sends: a retry inside a judge call counts against the run's
    judge budget (see `RunStop.request`)."""
    stop = RUN_STOP.get()
    if stop is not None:
        stop.request()


def check_run_stop() -> None:
    """Raise the run's stop before a call goes out, if the run stopped."""
    stop = RUN_STOP.get()
    if stop is not None:
        stop.check()


def run_stopped() -> bool:
    """Whether the run the current call belongs to has stopped."""
    stop = RUN_STOP.get()
    return stop is not None and stop.kind is not None


def drop_stopped_task_reports(loop: asyncio.AbstractEventLoop) -> None:
    """Keep asyncio quiet about the tasks a stop ended. A bare `gather`'s sibling tasks finish with
    RunStopped that nothing awaits, and asyncio would print each one, with local paths, under the
    stop line that already said why. Every other report goes to the handler set before."""
    previous = loop.get_exception_handler()
    if getattr(previous, "drops_run_stopped", False):
        return

    def handler(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        if isinstance(context.get("exception"), RunStopped):
            return
        if previous is not None:
            previous(loop, context)
        else:
            loop.default_exception_handler(context)

    handler.drops_run_stopped = True  # type: ignore[attr-defined]
    loop.set_exception_handler(handler)


def raise_if_stopped(outcomes: Iterable[object]) -> None:
    """Re-raise a run stop that `asyncio.gather(..., return_exceptions=True)` collected, so the
    caller never reads it as one failed call."""
    for outcome in outcomes:
        if isinstance(outcome, JudgeUnavailableError):
            raise outcome
