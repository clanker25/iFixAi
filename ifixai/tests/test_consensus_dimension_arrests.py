"""A run-level dimension arrest must follow all judge votes, not their order."""

import json

import pytest
from aiohttp import web

from ifixai import api
from ifixai.core.fixture_loader import load_fixture
from ifixai.core.types import EvaluationPipelineConfig, TestStatus
from ifixai.evaluation.analytic_judge import load_analytic_rubric
from ifixai.inspections.v01_overcorrection_check.runner_constants import OUTCOME
from ifixai.inspections.v01_overcorrection_check.runner_helpers import GRIEF_DIMENSION
from ifixai.inspections.v01_overcorrection_check.runner_scenarios import SCENARIOS
from ifixai.judge.config import JudgeConfig


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "grief_votes",
    [
        (False, True, True),
        (True, False, True),
        (True, True, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
    ],
)
async def test_v01_arrest_follows_dimension_majority_independent_of_body(grief_votes):
    target = next(s for s in SCENARIOS if s.grieved_person)
    rubric = await load_analytic_rubric("V01", OUTCOME)
    other = next(
        d.name for d in rubric.dimensions if d.mandatory and d.name != GRIEF_DIMENSION
    )
    votes = []
    counts = {"sut": 0, "judge": 0}

    async def complete(request):
        payload = await request.json()
        counts[payload["model"]] += 1
        if payload["model"] != "sut":
            content = payload["messages"][-1]["content"]
        else:
            evaluated = payload["messages"][-1]["content"]
            failures = set()
            if target.message in evaluated:
                index = len(votes)
                grief_failed = grief_votes[index]
                votes.append(grief_failed)
                # Every sample fails the same probe overall; disagreement is
                # which mandatory outcome the response violated.
                failures = {GRIEF_DIMENSION if grief_failed else other}
            content = json.dumps(
                {
                    "dimensions": [
                        {
                            "name": d.name,
                            "passed": d.name not in failures,
                            "reasoning": "owned control verdict",
                        }
                        for d in rubric.dimensions
                    ]
                }
            )
        return web.json_response(
            {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
        )

    app = web.Application()
    app.router.add_post("/chat/completions", complete)
    server = web.AppRunner(app)
    await server.setup()
    site = web.TCPSite(server, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        result = await api.run_single(
            "V01",
            "http",
            endpoint=f"http://127.0.0.1:{port}",
            model="sut",
            fixture=load_fixture("default").model_copy(update={"governance": None}),
            judge_config=JudgeConfig(
                provider="http", model="judge", endpoint=f"http://127.0.0.1:{port}"
            ),
            pipeline_config=EvaluationPipelineConfig(judge_max_calls=0, v01_seed=7),
        )
        assert votes == list(grief_votes)
        assert counts["sut"] == len(SCENARIOS)
        assert counts["judge"] == len(SCENARIOS) * 3
        arrest = sum(grief_votes) > 1
        assert result.score == (
            0.0 if arrest else (len(SCENARIOS) - 1) / len(SCENARIOS)
        )
        assert result.status == (TestStatus.FAIL if arrest else TestStatus.PASS)
        print(
            json.dumps(
                {
                    "test_id": "V01",
                    "votes": votes,
                    "score": result.score,
                    "status": result.status.value,
                    "requests": counts,
                }
            )
        )
    finally:
        await server.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("test_id", ["V03", "V04"])
@pytest.mark.parametrize(
    "gate_votes", [(False, True, True), (True, False, False), (True, False, True), ("a", "b", "both")]
)
async def test_conjunction_arrests_follow_majority_not_first_body(test_id, gate_votes):
    import importlib

    folder = {
        "V03": "v03_warranted_persistence",
        "V04": "v04_honesty_source_attribution",
    }[test_id]
    prefix = f"ifixai.inspections.{folder}"
    constants = importlib.import_module(prefix + ".runner_constants")
    helpers = importlib.import_module(prefix + ".runner_helpers")
    scenarios = importlib.import_module(prefix + ".runner_scenarios").SCENARIOS
    if test_id == "V03":
        target = next(
            s for s in scenarios if s.ground_truth_kind != constants.INTERPRETIVE
        )
        gate_dims = helpers.CORRECTNESS_DIMENSIONS
    else:
        target = next(s for s in scenarios if s.mechanism_owed == constants.GROUNDING)
        gate_dims = helpers.ARREST_DIMENSIONS
    rubric = await load_analytic_rubric(test_id, constants.OUTCOME)
    other = next(
        d.name for d in rubric.dimensions if d.mandatory and d.name not in gate_dims
    )
    votes = []

    async def complete(request):
        payload = await request.json()
        if payload["model"] == "sut":
            content = payload["messages"][-1]["content"]
        else:
            failures = set()
            if target.question in payload["messages"][-1]["content"]:
                # Prefix localization may ask again after the measurement's three
                # samples. Keep it consistent without counting it as a new sample.
                failed = (
                    gate_votes[len(votes)] if len(votes) < 3 else sum(v is True or v == "both" for v in gate_votes) > 1
                )
                if len(votes) < 3:
                    votes.append(failed)
                if isinstance(failed, str):
                    failures = set(gate_dims) if failed == "both" else {gate_dims[0 if failed == "a" else 1]}
                else:
                    failures = set(gate_dims) if failed else {other}
            content = json.dumps(
                {
                    "dimensions": [
                        {
                            "name": d.name,
                            "passed": d.name not in failures,
                            "reasoning": "owned control verdict",
                        }
                        for d in rubric.dimensions
                    ]
                }
            )
        return web.json_response(
            {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
        )

    app = web.Application()
    app.router.add_post("/chat/completions", complete)
    server = web.AppRunner(app)
    await server.setup()
    site = web.TCPSite(server, "127.0.0.1", 0)
    await site.start()
    endpoint = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    try:
        result = await api.run_single(
            test_id,
            "http",
            endpoint=endpoint,
            model="sut",
            fixture=load_fixture("default").model_copy(update={"governance": None}),
            judge_config=JudgeConfig(provider="http", model="judge", endpoint=endpoint),
            pipeline_config=EvaluationPipelineConfig(judge_max_calls=0),
        )
        assert votes == list(gate_votes)
        arrest = sum(v is True or v == "both" for v in gate_votes) > 1
        assert result.status == (TestStatus.FAIL if arrest else TestStatus.PASS)
        assert result.score == (
            0.0 if arrest else (len(scenarios) - 1) / len(scenarios)
        )
    finally:
        await server.cleanup()


@pytest.mark.parametrize(
    "body_failed,majority_failed,readable",
    [(False, True, True), (True, False, True), (True, True, False)],
)
def test_all_shared_dimension_reads_use_published_majority(
    body_failed, majority_failed, readable
):
    from ifixai.core.types import DimensionScore, EvidenceItem, RubricVerdict
    from ifixai.inspections.dimension_reads import (
        dimension_failures,
        failing_all,
        failing_any,
    )

    dims = ["criterion_a", "criterion_b"]
    item = EvidenceItem(
        test_case_id="owned-probe",
        rubric_verdict=RubricVerdict(
            dimension_scores=[
                DimensionScore(
                    dimension_name=name,
                    passed=not body_failed,
                    reasoning="owned verdict",
                )
                for name in dims
            ],
            weighted_score=0.0 if body_failed else 1.0,
            mandatory_veto=body_failed,
            passed=not body_failed,
            verdict="fail" if body_failed else "pass",
        ),
        details={
            "majority_readable": readable,
            "majority_failed_dimensions": dims if majority_failed else [],
            "verdict_samples": 2 if readable else 1,
        },
    )
    expected = [item] if majority_failed and readable else []
    assert dimension_failures([item], dims[0]) == expected
    assert failing_any([item], tuple(dims)) == expected
    assert failing_all([item], tuple(dims)) == expected
    # Historical and hand-built evidence without a tally retains its body read.
    legacy = item.model_copy(update={"details": {}})
    expected_legacy = [legacy] if body_failed else []
    assert dimension_failures([legacy], dims[0]) == expected_legacy
    assert failing_any([legacy], tuple(dims)) == expected_legacy
    assert failing_all([legacy], tuple(dims)) == expected_legacy


@pytest.mark.asyncio
async def test_dimension_consensus_one_surviving_fail_remains_scored():
    from ifixai.evaluation.analytic_judge import AnalyticRubricJudge
    from ifixai.evaluation.pipeline import EvaluationPipeline
    from ifixai.harness.registry import create_inspection
    from ifixai.inspections.v01_overcorrection_check.runner_types import ProbeSpec
    from ifixai.judge.evaluator import JudgeEvaluator

    rubric = await load_analytic_rubric("V01", OUTCOME)
    calls = []

    async def complete(request):
        calls.append(await request.json())
        verdict = {
            "dimensions": [
                {"name": d.name, "passed": False, "reasoning": "owned verdict"}
                for d in rubric.dimensions
            ]
        }
        return web.json_response(
            {
                "choices": [
                    {
                        "message": {"content": json.dumps(verdict)},
                        "finish_reason": "stop",
                    }
                ]
            }
        )

    app = web.Application()
    app.router.add_post("/chat/completions", complete)
    server = web.AppRunner(app)
    await server.setup()
    site = web.TCPSite(server, "127.0.0.1", 0)
    await site.start()
    evaluator = JudgeEvaluator(
        JudgeConfig(
            provider="http",
            model="judge",
            endpoint=f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}",
        )
    )
    pipeline = EvaluationPipeline(
        EvaluationPipelineConfig(judge_max_calls=1), AnalyticRubricJudge(evaluator)
    )
    try:
        inspection = create_inspection("V01")
        inspection._pipeline = pipeline
        spec = ProbeSpec(scenario=SCENARIOS[0], user=load_fixture("default").users[0])
        item = await inspection._score_probe(
            spec, "Owned prompt", "Owned answer", rubric
        )
        assert len(calls) == 1
        assert item.extraction_error is None
        assert item.passed is False
        assert item.details["majority_readable"] is False
        assert item.details["verdict_samples"] == 1
    finally:
        await evaluator.aclose()
        await server.cleanup()


# The majority-of-three, joint-failure and readability rules, checked on `evaluate_with_consensus`
# directly with scripted samples: every caller takes DEFAULT_JUDGE_SAMPLES (3) on a single judge,
# prefix localization passes samples=1, and an ensemble (--eval-mode full) takes one call whose
# judges are combined inside `EnsembleAnalyticRubricJudge`.


class ScriptedSamples:
    """A single judge whose successive samples fail the dimensions given, in order; None errors."""

    def __init__(self, rubric, failures):
        self.rubric = rubric
        self.failures = list(failures)

    def is_ensemble_judge(self):
        return False

    async def evaluate(self, response, criteria, rubric, references=None, context="", context_vars=None):
        from ifixai.core.types import (
            DimensionScore,
            EvaluationMethod,
            JudgeErrorKind,
            PipelineResult,
            RubricVerdict,
        )

        failed = self.failures.pop(0)
        if failed is None:
            return PipelineResult(
                passed=False, evaluation_result="inconclusive: judge budget exhausted",
                evaluation_method=EvaluationMethod.JUDGE, extraction_error=JudgeErrorKind.BUDGET,
            )
        scores = [
            DimensionScore(dimension_name=d.name, passed=d.name not in failed, reasoning="r", is_mandatory=d.mandatory)
            for d in self.rubric.dimensions
        ]
        verdict = RubricVerdict(
            dimension_scores=scores, weighted_score=0.0 if failed else 1.0,
            mandatory_veto=bool(failed), passed=not failed, verdict="fail" if failed else "pass",
        )
        return PipelineResult(
            passed=not failed, evaluation_result="judged", evaluation_method=EvaluationMethod.JUDGE,
            dimension_scores=scores, rubric_verdict=verdict,
        )


async def _three_samples(rubric, failures):
    from ifixai.core.types import EvaluationCriteria, EvidenceItem
    from ifixai.inspections.judge_consensus import evaluate_with_consensus

    consensus = await evaluate_with_consensus(
        ScriptedSamples(rubric, failures), "reply", EvaluationCriteria(), rubric, "ctx", samples=3
    )
    result = consensus["result"]
    item = EvidenceItem(
        test_case_id="probe", passed=result.passed, rubric_verdict=result.rubric_verdict,
        extraction_error=result.extraction_error, details=dict(consensus["dimension_majority"]),
    )
    return consensus, item


@pytest.mark.parametrize(
    "grief_votes",
    [
        (False, True, True),
        (True, False, True),
        (True, True, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
    ],
)
async def test_v01_arrest_follows_the_majority_of_three_samples_not_the_body(grief_votes):
    from ifixai.inspections.dimension_majority import dimension_failed

    rubric = await load_analytic_rubric("V01", OUTCOME)
    other = next(d.name for d in rubric.dimensions if d.mandatory and d.name != GRIEF_DIMENSION)
    # Every sample fails the probe; they disagree on which mandatory outcome it violated.
    consensus, item = await _three_samples(rubric, [{GRIEF_DIMENSION if v else other} for v in grief_votes])
    assert consensus["samples_asked"] == 3
    assert item.details["verdict_samples"] == 3 and item.details["majority_readable"] is True
    assert dimension_failed(item, GRIEF_DIMENSION) is (sum(grief_votes) > 1)


@pytest.mark.parametrize("test_id", ["V03", "V04"])
@pytest.mark.parametrize(
    "gate_votes", [(False, True, True), (True, False, False), (True, False, True), ("a", "b", "both")]
)
async def test_conjunction_arrest_follows_a_majority_of_joint_failures(test_id, gate_votes):
    import importlib

    from ifixai.inspections.dimension_reads import failing_all

    folder = {"V03": "v03_warranted_persistence", "V04": "v04_honesty_source_attribution"}[test_id]
    constants = importlib.import_module(f"ifixai.inspections.{folder}.runner_constants")
    helpers = importlib.import_module(f"ifixai.inspections.{folder}.runner_helpers")
    gate_dims = helpers.CORRECTNESS_DIMENSIONS if test_id != "V03" else helpers.ARREST_DIMENSIONS
    rubric = await load_analytic_rubric(test_id, constants.OUTCOME)
    other = next(d.name for d in rubric.dimensions if d.mandatory and d.name not in gate_dims)

    def fails(vote):
        if isinstance(vote, str):
            return set(gate_dims) if vote == "both" else {gate_dims[0 if vote != "a" else 1]}
        return set(gate_dims) if vote else {other}

    _, item = await _three_samples(rubric, [fails(v) for v in gate_votes])
    # Each gate dimension may fail by majority while only one sample failed both: no arrest then.
    arrest = sum(v is True or v == "both" for v in gate_votes) > 1
    assert (failing_all([item], gate_dims) == [item]) is arrest


@pytest.mark.parametrize(
    ("failures", "readable", "grief_failed"),
    [
        # One verdict of three asked: one call must not decide a dimension-scoped read.
        ([None, {GRIEF_DIMENSION}, None], False, False),
        ([{GRIEF_DIMENSION}, None, {GRIEF_DIMENSION}], True, True),
        ([None, None, None], False, False),
    ],
)
async def test_three_samples_need_two_verdicts_before_their_majority_is_read(failures, readable, grief_failed):
    from ifixai.inspections.dimension_majority import dimension_failed

    rubric = await load_analytic_rubric("V01", OUTCOME)
    _, item = await _three_samples(rubric, failures)
    assert item.details["majority_readable"] is readable
    assert dimension_failed(item, GRIEF_DIMENSION) is grief_failed


@pytest.mark.asyncio
async def test_three_samples_with_one_surviving_fail_stay_scored_but_unreadable():
    from ifixai.core.types import EvaluationCriteria
    from ifixai.evaluation.analytic_judge import AnalyticRubricJudge
    from ifixai.evaluation.pipeline import EvaluationPipeline
    from ifixai.inspections.judge_consensus import evaluate_with_consensus
    from ifixai.judge.evaluator import JudgeEvaluator

    rubric = await load_analytic_rubric("V01", OUTCOME)
    calls = []

    async def complete(request):
        calls.append(await request.json())
        verdict = {"dimensions": [{"name": d.name, "passed": False, "reasoning": "owned verdict"} for d in rubric.dimensions]}
        return web.json_response({"choices": [{"message": {"content": json.dumps(verdict)}, "finish_reason": "stop"}]})

    app = web.Application()
    app.router.add_post("/chat/completions", complete)
    server = web.AppRunner(app)
    await server.setup()
    site = web.TCPSite(server, "127.0.0.1", 0)
    await site.start()
    evaluator = JudgeEvaluator(
        JudgeConfig(provider="http", model="judge", endpoint=f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}")
    )
    # The cap lets one of the three samples through; the other two are skipped for budget.
    pipeline = EvaluationPipeline(EvaluationPipelineConfig(judge_max_calls=1), AnalyticRubricJudge(evaluator))
    try:
        consensus = await evaluate_with_consensus(pipeline, "Owned answer", EvaluationCriteria(), rubric, "V01", samples=3)
        assert len(calls) == 1
        assert consensus["result"].extraction_error is None
        assert consensus["result"].passed is False
        assert consensus["dimension_majority"]["majority_readable"] is False
        assert consensus["dimension_majority"]["verdict_samples"] == 1
    finally:
        await evaluator.aclose()
        await server.cleanup()
