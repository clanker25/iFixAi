"""The CLI's exit codes are the contract CI will depend on.

A wind-tunnel run is consumed by a machine before a person: the useful question
is "did this build lose detection capability", and that is answered by the exit
code. So the three outcomes are pinned here, including the ordering rule that an
inconclusive control outranks a blind spot — an unusable control does not make a
blind spot confirmed, it makes it unmeasurable.
"""

from click.testing import CliRunner

from ifixai.cli.main import ifixai_cli
from ifixai.windtunnel.measure import InspectionOutcome, WindTunnelReport

CLEAN_FIXTURE = "customer_support"
#: The shipped default: its reference provider fails inspections before any
#: defect is injected, so it cannot support a measurement.
UNCONTROLLED_FIXTURE = "default"


def _outcome(test_id: str, outcome: str, expected: bool) -> InspectionOutcome:
    return InspectionOutcome(
        test_id=test_id,
        outcome=outcome,
        clean_score=1.0,
        defective_score=1.0,
        threshold=1.0,
        expected=expected,
    )


def _invoke(*args: str):
    return CliRunner().invoke(ifixai_cli, ["windtunnel", *args])


class TestVerdictOrdering:
    def test_a_false_positive_outranks_a_blind_spot(self):
        from ifixai.cli.windtunnel import _verdict

        report = WindTunnelReport(
            profile_name="mixed",
            outcomes=(
                _outcome("B01", "false_positive", expected=False),
                _outcome("B03", "missed", expected=True),
            ),
        )
        assert _verdict((report,)) == 2

    def test_a_blind_spot_alone_is_one(self):
        from ifixai.cli.windtunnel import _verdict

        report = WindTunnelReport(
            profile_name="gap",
            outcomes=(_outcome("B03", "missed", expected=True),),
        )
        assert _verdict((report,)) == 1

    def test_all_detected_is_zero(self):
        from ifixai.cli.windtunnel import _verdict

        report = WindTunnelReport(
            profile_name="good",
            outcomes=(_outcome("B01", "detected", expected=True),),
        )
        assert _verdict((report,)) == 0

    def test_an_unexpected_miss_is_not_a_blind_spot(self):
        """Only an inspection that claims the defect can have a blind spot."""
        from ifixai.cli.windtunnel import _verdict

        report = WindTunnelReport(
            profile_name="unclaimed",
            outcomes=(_outcome("C05", "missed", expected=False),),
        )
        assert _verdict((report,)) == 0


class TestBaselineAffectsTheVerdict:
    """A regression fails a build that would otherwise have been green.

    This is asserted here rather than through the CLI because the suite cannot
    be made to lose a detection on demand: a run that detects everything is the
    only run available. The verdict function is what the CLI calls, so pinning
    it pins the behaviour the docs promise.
    """

    @staticmethod
    def _clean_report() -> WindTunnelReport:
        return WindTunnelReport(
            profile_name="good",
            outcomes=(_outcome("B01", "detected", expected=True),),
        )

    def _comparison(self, regressions: int, improvements: int = 0):
        from ifixai.windtunnel.evidence import BaselineComparison, DetectionChange

        return BaselineComparison(
            baseline_digest="a" * 64,
            current_digest="b" * 64,
            regressions=tuple(
                DetectionChange(
                    profile="good", test_id=f"B{i:02d}", was="detected", now="missed"
                )
                for i in range(regressions)
            ),
            improvements=tuple(
                DetectionChange(
                    profile="good", test_id=f"C{i:02d}", was="missed", now="detected"
                )
                for i in range(improvements)
            ),
            unmatched=(),
        )

    def test_a_regression_fails_an_otherwise_clean_run(self):
        from ifixai.cli.windtunnel import _verdict

        reports = (self._clean_report(),)
        assert _verdict(reports) == 0, "precondition: this run is clean on its own"
        assert _verdict(reports, self._comparison(regressions=1)) == 1

    def test_an_improvement_alone_does_not_fail(self):
        from ifixai.cli.windtunnel import _verdict

        comparison = self._comparison(regressions=0, improvements=2)
        assert _verdict((self._clean_report(),), comparison) == 0

    def test_an_inconclusive_control_outranks_a_regression(self):
        """A comparison drawn against a broken control is not trustworthy."""
        from ifixai.cli.windtunnel import _verdict

        report = WindTunnelReport(
            profile_name="bad-control",
            outcomes=(_outcome("B01", "false_positive", expected=True),),
        )
        assert _verdict((report,), self._comparison(regressions=1)) == 2


class TestRunCommand:
    def test_a_fully_detected_profile_exits_zero(self, monkeypatch):
        # Which defects the live suite catches is measure's claim; this pins the exit code of a
        # run where every expected detector moved.
        async def all_detected(profiles, fixture):
            return tuple(
                WindTunnelReport(
                    profile_name=profile.name,
                    outcomes=tuple(
                        _outcome(test_id, "detected", expected=True)
                        for test_id in sorted(profile.expected_detectors)
                    ),
                )
                for profile in profiles
            )

        monkeypatch.setattr("ifixai.cli.windtunnel.measure_all", all_detected)
        result = _invoke("run", "over_refusing", "--fixture", CLEAN_FIXTURE)
        assert result.exit_code == 0, result.output
        assert "detected" in result.output

    def test_a_profile_with_a_blind_spot_exits_one(self):
        # over_permissive expects B01 and B08; B08 has no surface on the
        # reference provider, so it reports as a blind spot.
        result = _invoke("run", "over_permissive", "--fixture", CLEAN_FIXTURE)
        assert result.exit_code == 1, result.output
        assert "Blind spots" in result.output

    def test_an_uncontrolled_fixture_exits_inconclusive(self):
        result = _invoke("run", "over_permissive", "--fixture", UNCONTROLLED_FIXTURE)
        assert result.exit_code == 2, result.output
        assert "Inconclusive" in result.output
        # The reason has to be stated, not just the code: a reader who sees
        # only "2" cannot tell a broken control from a broken agent.
        assert "clean control failed" in result.output

    def test_an_unknown_profile_is_rejected_by_the_parser(self):
        result = _invoke("run", "no_such_profile", "--fixture", CLEAN_FIXTURE)
        assert result.exit_code == 2
        assert "no_such_profile" in result.output


class TestRegistration:
    def test_the_group_is_registered_on_the_root_cli(self):
        result = CliRunner().invoke(ifixai_cli, ["--help"])
        assert result.exit_code == 0
        assert "windtunnel" in result.output

    def test_every_subcommand_is_listed(self):
        result = _invoke("--help")
        assert result.exit_code == 0
        for command in ("run", "sweep", "controls", "sensitivity"):
            assert command in result.output


class TestNotExercisedIsNotAPass:
    """An unreached expected detector is inconclusive, never green.

    This distinction was not free. An earlier version classified an unreached
    inspection as `missed`, which reported two blind spots (B03 and P08 under
    the audit-entry defect) that did not exist: both gathered byte-identical
    evidence in each run, so the defect never reached them.
    """

    @staticmethod
    def _report(outcome: str, expected: bool = True) -> WindTunnelReport:
        return WindTunnelReport(
            profile_name="p",
            outcomes=(_outcome("B03", outcome, expected=expected),),
        )

    def test_an_unreached_expected_detector_is_inconclusive(self):
        from ifixai.cli.windtunnel import _verdict

        assert _verdict((self._report("not_exercised"),)) == 2

    def test_an_unreached_unclaimed_inspection_is_ignored(self):
        from ifixai.cli.windtunnel import _verdict

        assert _verdict((self._report("not_exercised", expected=False),)) == 0

    def test_unreached_is_not_counted_as_a_blind_spot(self):
        report = self._report("not_exercised")
        assert report.unreached and not report.blind_spots

    def test_detection_rate_excludes_unreached_detectors(self):
        """A rate over claims that were never tested would be invented."""
        report = WindTunnelReport(
            profile_name="p",
            outcomes=(
                _outcome("B03", "not_exercised", expected=True),
                _outcome("P08", "not_exercised", expected=True),
            ),
        )
        assert report.detection_rate is None

    def test_a_rate_is_still_reported_when_some_detectors_were_reached(self):
        report = WindTunnelReport(
            profile_name="p",
            outcomes=(
                _outcome("B01", "detected", expected=True),
                _outcome("B03", "not_exercised", expected=True),
            ),
        )
        assert report.detection_rate == 1.0
