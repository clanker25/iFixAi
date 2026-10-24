"""How a pass rate reads in every report: one format, and "not scored" where nothing was scored.

One place, so the terminal, the markdown, the HTML scorecard and a runner's own prose agree.
"""

from collections.abc import Mapping

NOT_SCORED: str = "not scored"


def percent(rate: float | None) -> str:
    """A pass rate as a whole percentage, or "not scored" when no probe was scored for it."""
    return NOT_SCORED if rate is None else f"{rate * 100:.0f}%"


def per_class_line(rates: Mapping[str, float | None]) -> str:
    """Per-class pass rates as a reader sees them: class names in words, in sorted order."""
    return " · ".join(
        f"{name.replace('_', ' ')} {percent(rate)}" for name, rate in sorted(rates.items())
    )
