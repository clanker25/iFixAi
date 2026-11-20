"""Registered diagnostics are summaries, never successful measurements."""

import pytest

from ifixai.core.fixture_loader import load_fixture
from ifixai.core.types import (
    EvaluationPipelineConfig,
    EvidenceItem,
    JudgeErrorKind,
    ProviderConfig,
    TestResult,
    TestRunResult,
    TestStatus,
)
from ifixai.evaluation.pipeline import EvaluationPipeline
from ifixai.harness.registry import ALL_SPECS, create_inspection
from ifixai.providers.base import ChatProvider
from ifixai.reporting.health import (
    judge_health_note,
    measurement_failure_banner,
    run_health,
)
from ifixai.reporting.scorecard import scoped_gate_decides
from ifixai.shared.evidence import flag_diagnostics


class VerboseFixtureProvider(ChatProvider):
    """Owned canned responses exercise M12's real transcript-length guard."""

    def __init__(self):
        self.calls = 0

    async def send_message(self, messages, config):
        self.calls += 1
        return "Owned synthetic transcript content. " * 180


@pytest.mark.asyncio
async def test_registered_m12_diagnostics_cannot_hide_zero_measurements():
    fixture = load_fixture("healthcare")
    provider = VerboseFixtureProvider()
    config = EvaluationPipelineConfig()
    pipeline = EvaluationPipeline(config)
    inspection = create_inspection("M12")
    result = await inspection.execute(
        provider,
        ProviderConfig(provider="mock"),
        fixture,
        pipeline_config=config,
        pipeline=pipeline,
    )

    diagnostics = [item for item in result.evidence if item.is_diagnostic]
    measurements = [item for item in result.evidence if not item.is_diagnostic]
    assert len(diagnostics) == 3
    assert len(measurements) == 40
    assert all(item.extraction_error == JudgeErrorKind.CONTRACT for item in measurements)
    assert provider.calls == 280
    assert pipeline.judge_calls_used == 0

    health = run_health(TestRunResult(test_results=[result]))
    assert health.total == 40
    assert health.scorable == 0
    assert health.invalid
    assert measurement_failure_banner(health) is not None


def _health(evidence):
    inspection = create_inspection("B01")
    return run_health(
        TestRunResult(
            test_results=[
                TestResult(
                    test_id=inspection.spec.test_id,
                    spec=inspection.spec,
                    evidence=evidence,
                )
            ]
        )
    )


def test_diagnostics_do_not_inflate_judge_attempts():
    health = _health(
        [
            EvidenceItem(test_case_id="measured", passed=True),
            EvidenceItem(
                test_case_id="broken", extraction_error=JudgeErrorKind.CONTRACT
            ),
            EvidenceItem(test_case_id="summary", is_diagnostic=True, passed=True),
        ]
    )
    assert health.total == 2
    assert health.scorable == 1
    assert health.judge_broke == 1
    assert health.judge_attempts == 2
    assert "of 2 grading attempts" in judge_health_note(health)


def test_unflagged_evidence_keeps_existing_measurement_accounting():
    health = _health(
        [
            EvidenceItem(test_case_id="one", passed=True),
            EvidenceItem(test_case_id="two", passed=False),
        ]
    )
    assert health.total == health.scorable == 2
    assert not health.invalid
    assert measurement_failure_banner(health) is None


def test_error_under_declared_diagnostic_prefix_remains_a_measurement():
    evidence = flag_diagnostics(
        [
            EvidenceItem(
                test_case_id="summary-contract",
                extraction_error=JudgeErrorKind.CONTRACT,
            )
        ],
        ("summary-",),
    )
    assert not evidence[0].is_diagnostic
    health = _health(evidence)
    assert health.total == health.judge_broke == 1
    assert health.invalid


def _result(test_id, status, evidence=()):
    spec = create_inspection(test_id).spec
    return TestResult(test_id=test_id, spec=spec, status=status, evidence=list(evidence))


def test_a_crashed_report_only_inspection_does_not_invalidate_a_scoped_run():
    # -b B16 -b P19: B16 passes, P19 (exploratory) crashes.
    run = TestRunResult(
        test_results=[
            _result("B16", TestStatus.PASS, [EvidenceItem(test_case_id=f"b16-{n}", passed=True) for n in range(10)]),
            _result("P19", TestStatus.ERROR),
        ],
        mandatory_minimums_not_run=["B01", "B08", "P01"],
    )
    health = run_health(run)
    assert (health.n_inspections, health.errored, health.invalid) == (1, 0, False)
    assert scoped_gate_decides(run)


def test_a_full_run_counts_report_only_inspections():
    # All 61 inspections, 18 of the 35 scored ones crash: 18/61 is under half, so the run
    # stays valid, and a V-series judge timeout still shows in the judge health note.
    scored = [s.test_id for s in ALL_SPECS if not (s.is_exploratory or s.is_advisory or s.is_attestation)]
    crashed = set(scored[:18])
    run = TestRunResult(
        test_results=[
            _result(s.test_id, TestStatus.ERROR)
            if s.test_id in crashed
            else _result(s.test_id, TestStatus.PASS, [EvidenceItem(test_case_id=s.test_id, passed=True)])
            for s in ALL_SPECS
        ]
    )
    run.test_results[-1].evidence.append(
        EvidenceItem(test_case_id="v11-timeout", extraction_error=JudgeErrorKind.JUDGE_TRANSIENT)
    )
    health = run_health(run)
    assert (len(scored), run.test_results[-1].test_id) == (35, "V11")
    assert (health.n_inspections, health.errored, health.judge_broke, health.invalid) == (61, 18, 1, False)
    assert measurement_failure_banner(health) is None


def test_a_run_of_only_report_only_inspections_still_counts_them():
    health = run_health(TestRunResult(test_results=[_result("P19", TestStatus.ERROR)]))
    assert (health.n_inspections, health.errored, health.invalid) == (1, 1, True)


def test_a_flaky_judge_counts_in_the_judge_health_note():
    health = _health(
        [
            EvidenceItem(test_case_id="graded", passed=True),
            EvidenceItem(test_case_id="timed-out", extraction_error=JudgeErrorKind.JUDGE_TRANSIENT),
        ]
    )
    assert (health.judge_broke, health.unreachable, health.judge_attempts) == (1, 0, 2)
    assert not health.invalid
    assert "returned 1 unusable verdict(s)" in judge_health_note(health)


@pytest.mark.parametrize(
    ("kind", "says"),
    [
        (JudgeErrorKind.JUDGE_TRANSIENT, "a broken grader, not a model problem"),
        (JudgeErrorKind.JUDGE_UNREACHABLE, "Check the judge's model id / key / endpoint"),
    ],
)
def test_a_run_invalid_because_of_the_judge_blames_the_judge(kind, says):
    health = _health([EvidenceItem(test_case_id=f"p{n}", extraction_error=kind) for n in range(3)])
    banner = measurement_failure_banner(health)
    assert says in banner
    assert "agent under test" not in banner
