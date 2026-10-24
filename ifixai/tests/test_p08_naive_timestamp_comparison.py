"""Regression test: _parse_timestamp must always return timezone-aware datetimes
to prevent TypeError when comparing naive and aware timestamps in _temporal_item.

The bug: _parse_timestamp returned naive datetimes for timestamps without timezone
info (e.g., '2026-10-18T12:00:00'), but _temporal_item compared them with
timezone-aware timestamps from the governance mixin, causing
TypeError: can't compare offset-naive and offset-aware datetimes.
"""
from datetime import timezone

from ifixai.inspections.p08_self_audit_trail_integrity.runner import _parse_timestamp


def test_parse_timestamp_returns_tz_aware_for_naive_input():
    """Naive timestamps (no timezone) must be interpreted as UTC and return tz-aware."""
    naive = _parse_timestamp("2026-10-18T12:00:00")
    assert naive is not None
    assert naive.tzinfo is not None
    assert naive.utcoffset() == timezone.utc.utcoffset(None)


def test_parse_timestamp_preserves_explicit_timezone():
    """Explicit timezone offsets must be preserved."""
    with_offset = _parse_timestamp("2026-10-18T12:00:00-05:00")
    assert with_offset is not None
    assert with_offset.tzinfo is not None
    # -05:00 offset
    assert with_offset.utcoffset().total_seconds() == -5 * 3600


def test_parse_timestamp_handles_z_suffix():
    """Z suffix must be converted to UTC."""
    z_suffix = _parse_timestamp("2026-10-18T12:00:00Z")
    assert z_suffix is not None
    assert z_suffix.tzinfo is not None
    assert z_suffix.utcoffset() == timezone.utc.utcoffset(None)


def test_parse_timestamp_comparison_safe():
    """Comparing naive-parsed and aware-parsed timestamps must not raise TypeError."""
    # This simulates the comparison in _temporal_item:
    # a naive timestamp from a fixture vs a Z-suffix timestamp from governance mixin
    naive_parsed = _parse_timestamp("2026-10-18T12:00:00")  # was naive, now UTC
    aware_parsed = _parse_timestamp("2026-10-18T11:00:00Z")  # was UTC, stays UTC

    # Both are now timezone-aware, comparison must work
    result = naive_parsed < aware_parsed
    assert result is False  # 12:00 is not < 11:00

    # Reverse comparison also works
    result = aware_parsed < naive_parsed
    assert result is True


def test_parse_timestamp_empty_and_invalid():
    """Empty and invalid inputs must return None."""
    assert _parse_timestamp("") is None
    assert _parse_timestamp("invalid") is None
    assert _parse_timestamp("   ") is None