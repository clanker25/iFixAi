"""How many judge calls a run will make, for --dry-run and the run's judge budget. One grade asks
one judge (three times where the inspection takes a majority), or every judge once with --eval-mode
full; each ask is a call to that judge's provider, and the budget counts those calls (see
`run_stop`)."""

import statistics
from collections.abc import Collection

from ifixai.evaluation.run_stop import BUDGET_OVERRUN, budget_ceiling
from ifixai.inspections.judge_consensus import DEFAULT_JUDGE_SAMPLES

# Judge calls each inspection makes on the default fixture: judge_stats.total_calls of
# `ifixai run -p mock --eval-mode self -b <ID>`, one judge reading each grade three times where the
# inspection takes a majority (B15 counted as its 10 trajectories reach the judge, three reads each)
# and once in a probe pack.
# Fixed counts, so other fixtures get the default fixture's numbers; re-measure an inspection when
# its judge traffic changes (a test fails when one is missing).
JUDGE_GRADES_BY_INSPECTION: dict[str, int] = {
    "B01": 16, "B02": 13, "B03": 264, "B04": 13, "B05": 22, "B06": 44, "B07": 11, "B08": 477,
    "B09": 132, "B10": 110, "B11": 1, "B12": 24, "B13": 264, "B14": 792, "B15": 30, "B16": 30,
    "B17": 24, "B18": 30, "B19": 30, "B20": 48, "B21": 40, "B22": 24, "B23": 11, "B24": 40,
    "B25": 18, "B26": 24, "B27": 30, "B28": 20, "B29": 12, "B30": 81, "B31": 20, "B32": 20,
    "C02": 15, "C05": 15, "C11": 12, "M02": 20, "M03": 13, "M06": 0, "M07": 20, "M12": 120,
    "P01": 23, "P08": 12, "P13": 180, "P19": 40, "P22": 180, "P27": 180, "P32": 180, "S02": 180,
    "V01": 120, "V02": 120, "V03": 120, "V04": 120, "V05": 180, "V06": 144, "V07": 144, "V08": 11,
    "V09": 144, "V10": 144, "V11": 120, "X04": 12, "X11": 13,
}
# The inspections whose single judge reads each grade `DEFAULT_JUDGE_SAMPLES` times and takes the
# majority. A panel asks each of its judges once per grade instead.
MAJORITY_INSPECTIONS: frozenset[str] = frozenset({
    "B15", "M12", "P13", "P22", "P27", "P32", "S02", "V01", "V02", "V03", "V04", "V05", "V06",
    "V07", "V09", "V10", "V11",
})
# An inspection the table doesn't list yet counts as the table's median, so a newly merged one
# estimates close instead of crashing every run and dry run.
UNLISTED_INSPECTION_GRADES: int = int(statistics.median(JUDGE_GRADES_BY_INSPECTION.values()))

# Without --judge-budget the cap only guards against a runaway run. The estimate is the default
# fixture's, and a fixture twice its size needs about 2.2x, so the guard sits well above that.
RUNAWAY_FACTOR: int = 10
MIN_JUDGE_CALL_CAP: int = 2000


def estimate_judge_grades(test_ids: Collection[str], ensemble: bool = False) -> int:
    """Judge calls for `test_ids` with one judge; with `ensemble`, the calls each panel judge makes,
    one per grade (rounded up: a call made once either way, like a purpose check, stays one)."""
    return sum(
        -(-JUDGE_GRADES_BY_INSPECTION.get(test_id, UNLISTED_INSPECTION_GRADES)
          // (DEFAULT_JUDGE_SAMPLES if ensemble and test_id in MAJORITY_INSPECTIONS else 1))
        for test_id in test_ids
    )


def judge_call_cap(estimated_calls: int, judge_budget: int) -> int:
    """The run's judge budget in calls: --judge-budget when set, else the runaway guard. A soft
    budget: the run may go 25% over it, then stops (see `run_stop.budget_ceiling`)."""
    if judge_budget > 0:
        return judge_budget
    return max(RUNAWAY_FACTOR * estimated_calls, MIN_JUDGE_CALL_CAP)


def describe_judge_cap(cap: int, judge_budget: int) -> str:
    """The cap for the run header and --dry-run: where it came from, and how far over it the
    run may go before it stops."""
    source = (
        ""
        if judge_budget > 0
        else f" (the default: {RUNAWAY_FACTOR}x the estimate, at least {MIN_JUDGE_CALL_CAP})"
    )
    return (
        f"{cap} calls{source}; the run may go {BUDGET_OVERRUN:.0%} over it, up to "
        f"{budget_ceiling(cap)}, then stops"
    )
