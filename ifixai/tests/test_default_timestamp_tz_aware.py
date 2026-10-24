"""Regression test: default timestamps on TestRunResult and RunManifest must be
timezone-aware UTC rather than naive — ``datetime.utcnow`` is deprecated since
Python 3.12 and raises DeprecationWarning (removed in 3.14).
"""
from datetime import datetime, timezone
from unittest import mock

from ifixai.core.types import RunMode, TestRunResult
from ifixai.evaluation.manifest import build_manifest
from ifixai.evaluation.types import ModelDescriptor


def test_test_run_result_default_timestamp_is_tz_aware():
    result = TestRunResult()
    assert result.timestamp.tzinfo is not None
    assert result.evaluation_date.tzinfo is not None
    assert result.timestamp.utcoffset() == timezone.utc.utcoffset(None)
    assert result.evaluation_date.utcoffset() == timezone.utc.utcoffset(None)


def test_build_manifest_default_timestamp_is_utc_string():
    with mock.patch("ifixai.evaluation.manifest.datetime") as mock_dt:
        fixed = datetime(2026, 10, 5, 12, 30, 45, tzinfo=timezone.utc)
        mock_dt.now.return_value = fixed
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        manifest = build_manifest(
            mode=RunMode.STANDARD,
            model_under_test=ModelDescriptor(
                provider="mock", model_id="test", version="1"
            ),
            judge_models=[],
            normalizer_version="1",
            test_versions={},
            rubric_hashes={},
            fixture_digest="a" * 64,
            run_nonce="b" * 16,
        )
    assert manifest.timestamp == "2026-10-05T12:30:45Z"
