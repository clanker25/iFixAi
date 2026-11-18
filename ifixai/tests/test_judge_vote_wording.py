"""How per-judge votes read in a report: a count like "2 of 3" only when all three returned."""

from __future__ import annotations

import pytest

from ifixai.reporting.judge_votes import (
    describe_votes,
    judges_that_failed,
    tally_phrase,
)

pytestmark = pytest.mark.unit

HARM = "no_harm_facilitation"


def returned(judge: str, failed: tuple[str, ...] = ()) -> dict:
    checks = [{"check": name, "passed": name not in failed, "reason": "r"} for name in (HARM, "no_contempt")]
    return {
        "judge": judge,
        "returned": True,
        "verdict": "fail" if failed else "pass",
        "passed": not failed,
        "checks": checks,
    }


def no_verdict(judge: str, error_class: str = "JudgeCommunicationError") -> dict:
    return {
        "judge": judge,
        "returned": False,
        "verdict": "no verdict",
        "passed": None,
        "error_class": error_class,
        "checks": [],
    }


def details(*votes: dict) -> dict:
    return {"judge_votes": list(votes)}


def test_a_probe_without_votes_reads_as_nothing() -> None:
    assert describe_votes({}) == ""
    assert tally_phrase({}, HARM, "found it") == ""
    assert judges_that_failed({}, HARM) == []
    assert describe_votes({"judge_votes": None}) == ""


def test_all_judges_returned_reads_as_a_full_count() -> None:
    d = details(returned("a"), returned("b", (HARM,)), returned("c", (HARM,)))
    assert describe_votes(d) == "3 of 3 judges returned: a pass; b fail (no_harm_facilitation); c fail (no_harm_facilitation)"
    assert tally_phrase(d, HARM, "found the harm") == "2 of 3 judges found the harm"
    assert judges_that_failed(d, HARM) == ["b", "c"]


def test_fewer_judges_returned_says_how_many_and_never_a_count_out_of_three() -> None:
    d = details(returned("a", (HARM,)), no_verdict("b"), returned("c", (HARM,)))
    assert describe_votes(d) == (
        "graded by 2 judges, 1 gave no verdict: a fail (no_harm_facilitation); "
        "b no verdict (JudgeCommunicationError); c fail (no_harm_facilitation)"
    )
    phrase = tally_phrase(d, HARM, "found the harm")
    assert phrase == "2 of the 2 judges that returned found the harm; 1 gave no verdict"
    assert "of 3" not in phrase + describe_votes(d)
    assert judges_that_failed(d, HARM) == ["a", "c"]


def test_a_judge_with_no_verdict_is_never_counted_as_a_failure_or_a_pass() -> None:
    d = details(returned("a"), no_verdict("b"), returned("c"))
    assert judges_that_failed(d, HARM) == []
    assert tally_phrase(d, HARM, "found the harm") == "0 of the 2 judges that returned found the harm; 1 gave no verdict"
    assert "b pass" not in describe_votes(d)


def test_one_judge_left_reads_in_the_singular() -> None:
    d = details(returned("a"), no_verdict("b"), no_verdict("c", "JudgeExtractionError"))
    assert describe_votes(d).startswith("graded by 1 judge, 2 gave no verdict: ")


def test_a_fallback_model_that_graded_is_named_as_a_stand_in() -> None:
    stood_in = {**returned("backup"), "fell_back_from": "b"}
    d = details(returned("a"), stood_in, returned("c"))
    assert "backup (stood in for b) pass" in describe_votes(d)
    assert "stood in" not in describe_votes(details(returned("a"), returned("b")))


def test_a_member_the_panel_dropped_reads_as_no_verdict() -> None:
    d = details(returned("a", (HARM,)), no_verdict("b", "JudgeDropped"), returned("c"))
    assert describe_votes(d) == (
        "graded by 2 judges, 1 gave no verdict: a fail (no_harm_facilitation); "
        "b no verdict (JudgeDropped); c pass"
    )
    assert tally_phrase(d, HARM, "found the harm") == (
        "1 of the 2 judges that returned found the harm; 1 gave no verdict"
    )
