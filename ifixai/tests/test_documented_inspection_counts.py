"""The inspection counts printed in the docs must match the registry.

Three docs quote how many inspections ship: ``README.md`` and
``docs/methodology.md`` in digits, ``docs/what-we-test.md`` in words. Those
numbers were maintained by hand and drifted — ``what-we-test.md`` still said
forty-nine after the extended series grew to twenty-nine. For an audit tool the
count is a load-bearing claim, so it is asserted here against
:data:`ifixai.harness.registry.ALL_SPECS` rather than trusted to review.

The assertions quote whole sentences on purpose. An earlier version of this
file only checked that the right number-word appeared *somewhere* in the
document, which passes vacuously: changing the headline "Sixty-one
inspections." to "Forty-nine inspections." left the word "sixty-one" present
further down, in "Twenty-four of the sixty-one inspections", so the test stayed
green against a document that now stated the wrong figure. Matching the claim
in place is the only version of this test that is worth having.

Adding an inspection is expected to fail this test. Update the prose in the
same commit; that is the point.
"""

import re
from pathlib import Path

import pytest

from ifixai.harness.registry import ALL_SPECS

REPO_ROOT = Path(__file__).resolve().parents[2]

# Only the values these docs actually quote. Kept small on purpose: a general
# number-to-words converter would let a drifted count hide behind a passing
# test, which is the failure mode this file exists to prevent.
NUMBER_WORDS = {
    29: "twenty-nine",
    32: "thirty-two",
    61: "sixty-one",
}

# Figures these docs stated before the V-series grew the extended set. Asserted
# absent so the prose cannot quietly regress to them.
STALE_COUNT_WORDS = ("forty-nine", "seventeen")


def _counts() -> tuple[int, int, int]:
    """Total, core (``B`` series) and extended inspection counts."""
    core = sum(1 for spec in ALL_SPECS if spec.test_id.startswith("B"))
    return len(ALL_SPECS), core, len(ALL_SPECS) - core


def _read(relative: str) -> str:
    path = REPO_ROOT / relative
    if not path.is_file():
        pytest.skip(f"{relative} is not present in this checkout")
    return path.read_text(encoding="utf-8")


def _flowed(relative: str) -> str:
    """The document with newlines collapsed, so a wrapped sentence matches."""
    return re.sub(r"\s+", " ", _read(relative))


def _words() -> tuple[str, str, str]:
    total, core, extended = _counts()
    missing = [n for n in (total, core, extended) if n not in NUMBER_WORDS]
    assert not missing, (
        f"No spelled-out form on record for {missing}. Add it to NUMBER_WORDS "
        "and update docs/what-we-test.md to match the registry."
    )
    return NUMBER_WORDS[total], NUMBER_WORDS[core], NUMBER_WORDS[extended]


def test_registry_splits_into_core_and_extended() -> None:
    total, core, extended = _counts()
    assert total == core + extended
    assert core > 0 and extended > 0


def test_methodology_quotes_the_registry_counts() -> None:
    total, core, extended = _counts()
    expected = f"{total} inspections ({core} core, {extended} extended)"
    assert expected in _read("docs/methodology.md"), (
        f"docs/methodology.md must state {expected!r}. "
        "The registry changed; update the prose in this commit."
    )


def test_readme_quotes_the_total() -> None:
    total, _, _ = _counts()
    assert f"{total} inspections" in _read("README.md"), (
        f"README.md must state '{total} inspections'. "
        "The registry changed; update the prose in this commit."
    )


def test_what_we_test_headline_sentence_states_every_count() -> None:
    """The one sentence that carries all three figures, matched in place."""
    total_word, core_word, extended_word = _words()
    text = _flowed("docs/what-we-test.md")

    pattern = (
        rf"{total_word} inspections\. {core_word} sit in the graded pillars"
        rf".*?{extended_word} are extended"
    )
    assert re.search(pattern, text, re.IGNORECASE), (
        "docs/what-we-test.md's headline sentence must read "
        f"'{total_word.capitalize()} inspections. {core_word.capitalize()} sit "
        f"in the graded pillars … {extended_word} are extended'. "
        "The registry changed; update the prose in this commit."
    )


def test_what_we_test_extended_section_states_the_extended_count() -> None:
    _, _, extended_word = _words()
    text = _flowed("docs/what-we-test.md")
    assert re.search(rf"{extended_word} further inspections", text, re.IGNORECASE), (
        f"docs/what-we-test.md's extended set must open '{extended_word} "
        "further inspections'. The registry changed; update the prose."
    )


def test_what_we_test_faq_denominator_is_the_total() -> None:
    total_word, _, _ = _words()
    text = _flowed("docs/what-we-test.md")
    denominators = re.findall(r"of the ([a-z-]+) inspections", text, re.IGNORECASE)
    assert denominators, "expected at least one 'of the N inspections' phrase"
    wrong = [d for d in denominators if d.lower() != total_word]
    assert not wrong, (
        f"docs/what-we-test.md uses {wrong} as a denominator where the total "
        f"is {total_word!r}. The registry changed; update the prose."
    )


def test_what_we_test_does_not_regress_to_a_stale_count() -> None:
    text = _flowed("docs/what-we-test.md").lower()
    present = [
        word
        for word in STALE_COUNT_WORDS
        if re.search(rf"\b{word}\b(?= (?:further )?inspections)", text)
    ]
    assert not present, (
        f"docs/what-we-test.md counts inspections as {present}, which are "
        "figures from before the extended set grew. Update the prose to the "
        "registry's counts."
    )
