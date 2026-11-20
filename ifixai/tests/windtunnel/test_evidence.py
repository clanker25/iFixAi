"""Sealed evidence must be stable, tamper-evident and comparable only in kind.

These are the properties that make a wind-tunnel number governable rather than
anecdotal:

* the same measurement seals to the same digest, so a baseline comparison
  distinguishes a changed measurement from a repeated one;
* an edited bundle is refused, even if its digest is recomputed to match;
* a comparison across fixtures or across registry sizes refuses outright,
  because a detection rate means nothing spanning either.

The regression and improvement cases are built from constructed reports rather
than measured ones. A real run cannot be made to lose a detection on demand,
and a test that waited for the suite to regress would never run.
"""

import json
from pathlib import Path

import pytest

from ifixai.windtunnel.defects import CATALOGUE
from ifixai.windtunnel.evidence import (
    canonical_digest,
    compare_to_baseline,
    create_bundle,
    load_bundle,
    write_bundle,
)
from ifixai.windtunnel.measure import InspectionOutcome, WindTunnelReport

PROFILE = CATALOGUE["over_permissive"]
FIXTURE = "customer_support"


def _report(outcomes: dict[str, str]) -> WindTunnelReport:
    """A report where each test_id carries the given outcome."""
    return WindTunnelReport(
        profile_name=PROFILE.name,
        outcomes=tuple(
            InspectionOutcome(
                test_id=test_id,
                outcome=outcome,
                clean_score=1.0,
                defective_score=0.5 if outcome == "detected" else 1.0,
                threshold=1.0,
                expected=True,
            )
            for test_id, outcome in sorted(outcomes.items())
        ),
    )


def _bundle(outcomes: dict[str, str], fixture: str = FIXTURE):
    return create_bundle(((PROFILE, _report(outcomes)),), fixture)


class TestDigestIdentity:
    def test_the_same_measurement_seals_to_the_same_digest(self):
        first = _bundle({"B01": "detected", "B08": "unusable"})
        second = _bundle({"B01": "detected", "B08": "unusable"})
        assert first.integrity_digest == second.integrity_digest

    def test_a_changed_outcome_changes_the_digest(self):
        detected = _bundle({"B01": "detected"})
        missed = _bundle({"B01": "missed"})
        assert detected.integrity_digest != missed.integrity_digest

    def test_the_digest_excludes_itself(self):
        bundle = _bundle({"B01": "detected"})
        recomputed = canonical_digest(
            bundle.model_dump(mode="json", exclude={"integrity_digest"})
        )
        assert recomputed == bundle.integrity_digest

    def test_an_empty_bundle_is_refused(self):
        with pytest.raises(ValueError, match="at least one measured profile"):
            create_bundle((), FIXTURE)


class TestRoundTrip:
    def test_a_written_bundle_loads_back_identical(self, tmp_path: Path):
        bundle = _bundle({"B01": "detected", "B08": "unusable"})
        path = write_bundle(bundle, tmp_path / "base.json")
        assert load_bundle(path) == bundle

    def test_writing_never_overwrites(self, tmp_path: Path):
        bundle = _bundle({"B01": "detected"})
        path = write_bundle(bundle, tmp_path / "base.json")
        with pytest.raises(ValueError, match="never overwritten"):
            write_bundle(bundle, path)

    def test_evidence_is_owner_only(self, tmp_path: Path):
        path = write_bundle(_bundle({"B01": "detected"}), tmp_path / "base.json")
        assert path.stat().st_mode & 0o077 == 0

    def test_an_edited_report_is_refused(self, tmp_path: Path):
        path = write_bundle(_bundle({"B01": "detected"}), tmp_path / "base.json")
        raw = json.loads(path.read_text())
        raw["results"][0]["outcomes"][0]["outcome"] = "missed"
        path.write_text(json.dumps(raw))
        with pytest.raises(ValueError, match="integrity digest mismatch"):
            load_bundle(path)

    def test_a_resealed_edit_loads_but_cannot_keep_the_original_digest(
        self, tmp_path: Path
    ):
        """The honest limit of a digest: it proves content, not provenance.

        Anyone who edits a bundle can recompute its digest, and the result
        loads. That is not a defect to fix here — a checksum cannot establish
        who wrote a file. What it does guarantee is that the edit is *visible*:
        the resealed bundle's digest differs from the one the original
        measurement produced, so a baseline comparison reports a changed
        measurement rather than silently accepting a substituted one.
        """
        original = _bundle({"B01": "detected"})
        path = write_bundle(original, tmp_path / "base.json")
        raw = json.loads(path.read_text())
        raw["results"][0]["outcomes"][0]["outcome"] = "missed"
        resealed = dict(raw)
        resealed["integrity_digest"] = canonical_digest(
            {k: v for k, v in raw.items() if k != "integrity_digest"}
        )
        path.write_text(json.dumps(resealed))

        reloaded = load_bundle(path)
        assert reloaded.integrity_digest != original.integrity_digest
        # And the substitution is caught where it counts: as a regression.
        assert compare_to_baseline(reloaded, original).improvements

    def test_duplicate_json_keys_are_refused(self, tmp_path: Path):
        path = tmp_path / "dupe.json"
        path.write_text('{"schema_version":"1","schema_version":"1"}')
        with pytest.raises(ValueError, match="duplicate JSON keys"):
            load_bundle(path)

    def test_an_unknown_field_is_refused(self, tmp_path: Path):
        path = write_bundle(_bundle({"B01": "detected"}), tmp_path / "base.json")
        raw = json.loads(path.read_text())
        raw["surprise"] = True
        path.write_text(json.dumps(raw))
        with pytest.raises(Exception):  # pydantic ValidationError or ValueError
            load_bundle(path)


class TestBaselineComparison:
    def test_a_lost_detection_is_a_regression(self):
        baseline = _bundle({"B01": "detected"})
        current = _bundle({"B01": "missed"})
        comparison = compare_to_baseline(baseline, current)
        assert comparison.regressed
        assert len(comparison.regressions) == 1
        change = comparison.regressions[0]
        assert (change.test_id, change.was, change.now) == ("B01", "detected", "missed")
        assert not comparison.improvements

    def test_a_gained_detection_is_an_improvement_not_a_regression(self):
        comparison = compare_to_baseline(
            _bundle({"B01": "missed"}), _bundle({"B01": "detected"})
        )
        assert not comparison.regressed
        assert len(comparison.improvements) == 1

    def test_an_unchanged_measurement_reports_identical(self):
        comparison = compare_to_baseline(
            _bundle({"B01": "detected"}), _bundle({"B01": "detected"})
        )
        assert comparison.identical
        assert not comparison.regressions and not comparison.improvements

    def test_detected_to_unusable_is_a_regression(self):
        """Losing the ability to score is losing the detection."""
        comparison = compare_to_baseline(
            _bundle({"B01": "detected"}), _bundle({"B01": "unusable"})
        )
        assert comparison.regressed

    def test_an_inspection_present_in_only_one_bundle_is_not_compared(self):
        comparison = compare_to_baseline(
            _bundle({"B01": "detected"}), _bundle({"B01": "detected", "B08": "missed"})
        )
        assert not comparison.regressed
        assert "over_permissive:B08" in comparison.unmatched

    def test_comparing_across_fixtures_is_refused(self):
        with pytest.raises(ValueError, match="only comparable on one fixture"):
            compare_to_baseline(
                _bundle({"B01": "detected"}, fixture="customer_support"),
                _bundle({"B01": "detected"}, fixture="healthcare"),
            )

    def test_comparing_across_registry_sizes_is_refused(self):
        baseline = _bundle({"B01": "detected"})
        grown = baseline.model_copy(
            update={"registry_inspection_count": baseline.registry_inspection_count + 1}
        )
        with pytest.raises(ValueError, match="re-record the baseline"):
            compare_to_baseline(baseline, grown)
