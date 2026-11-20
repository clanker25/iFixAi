"""A suite's description may not quote a size its suite does not have.

`ifixai list suites` prints a count beside a description. The count was always
derived from the resolved suite; the description was hand-written, and
`extended` drifted to "The 28 frontier inspections" while resolving to 29 — the
row printed both numbers.

The descriptions that quote their own size now take `{count}` and are filled
from the same resolution, so drift is no longer possible for them. This test
guards the remaining way to reintroduce it: hard-coding a fresh number into a
description.
"""

import re

import pytest

from ifixai.harness.suites import suite_catalog

#: Numbers a description may state that are not its total. `essential`
#: describes its make-up rather than its size; the parts are a separate claim
#: this test cannot derive, so they are listed explicitly and reviewed by hand.
ALLOWED_NON_COUNT_NUMBERS: dict[str, frozenset[int]] = {
    "essential": frozenset({6, 4}),
    # "P01" names the mandatory minimum the suite leaves out.
    "core": frozenset({1}),
}


def _rows() -> list[dict[str, object]]:
    rows = suite_catalog()
    assert rows, "suite_catalog() returned no rows"
    return rows


def test_every_row_has_a_name_a_count_and_a_description() -> None:
    for row in _rows():
        assert row["name"], row
        assert isinstance(row["count"], int) and row["count"] > 0, row
        assert row["description"], f"{row['name']} has no description"


def test_no_description_quotes_a_number_that_is_not_its_count() -> None:
    offenders: list[str] = []
    for row in _rows():
        name = str(row["name"])
        count = int(row["count"])  # type: ignore[arg-type]
        allowed = ALLOWED_NON_COUNT_NUMBERS.get(name, frozenset())
        for found in (int(m) for m in re.findall(r"\d+", str(row["description"]))):
            if found != count and found not in allowed:
                offenders.append(
                    f"{name}: description says {found}, suite resolves to {count}"
                )
    assert not offenders, (
        "A suite description quotes a size its suite does not have:\n  "
        + "\n  ".join(offenders)
        + "\nUse '{count}' in the description so it is filled from the resolved "
        "suite, or add the number to ALLOWED_NON_COUNT_NUMBERS if it genuinely "
        "is not the total."
    )


def test_no_unfilled_placeholder_reaches_a_reader() -> None:
    """No braces survive into a printed description.

    An unknown field name such as ``{cnt}`` raises at render, so that mistake
    announces itself. This covers the quiet one: a doubled ``{{count}}``, which
    formats to the literal text ``{count}`` and would ship to a reader.
    """
    for row in _rows():
        description = str(row["description"])
        assert "{" not in description and "}" not in description, (
            f"{row['name']} description still holds a placeholder: {description!r}"
        )


@pytest.mark.parametrize("tier", ["core", "extended", "smoke"])
def test_the_size_quoting_tiers_state_their_real_count(tier: str) -> None:
    """The three descriptions that name a size agree with the registry."""
    row = next((r for r in _rows() if r["name"] == tier), None)
    assert row is not None, f"{tier} is not in the catalogue"
    assert str(row["count"]) in str(row["description"]), (
        f"{tier} quotes a size but not its own: {row['description']!r} "
        f"vs count {row['count']}"
    )
