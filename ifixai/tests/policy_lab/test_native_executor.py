import pytest
from pydantic import ValidationError

from ifixai.policy_lab import create_exploration_bundle
from ifixai.policy_lab.executor import execute_witness


def setup(native_example):
    bundle = create_exploration_bundle(native_example.demo_model(), native_example.demo_policy())
    return bundle, bundle.report.safety_witnesses[0], native_example.LocalRefundStore()


def test_zero_step_initial_harm_is_observed_without_action_dispatch(native_example):
    model = native_example.demo_model()
    model = model.model_copy(update={"initial_facts": {**model.initial_facts, "paid": True}})
    bundle = create_exploration_bundle(model, native_example.demo_policy())
    witness = bundle.report.safety_witnesses[0]
    facts = {}
    def reset():
        facts.update(model.initial_facts)
    report = execute_witness(bundle, witness, {}, reset, lambda: dict(facts))
    assert not witness.steps and report.decision == "MATCHED"
    assert report.attempted_actions == report.completed_actions == 0
    assert report.initial_observed == model.initial_facts and not report.steps


def test_three_legal_actions_have_real_observed_harm_and_immutable_context(native_example):
    bundle, witness, store = setup(native_example)
    contexts = []
    bindings = store.bindings()
    original = bindings["01_request"]
    def observed(context):
        contexts.append(context)
        with pytest.raises(ValidationError):
            context.actor_id = "different-actor"
        original(context)
        return "return value is not state authority"
    bindings["01_request"] = observed
    report = execute_witness(bundle, witness, bindings, store.reset, store.observe)
    assert report.decision == "MATCHED"
    assert report.attempted_actions == report.completed_actions == 3
    assert store.calls == ["01_request", "02_self_approve", "04_pay"]
    assert store.synthetic_paid_units == 60 and store.facts["independent"] is False
    assert contexts[0].cost_units == 1
    assert len(report.steps) == 3
    assert all(step.outcome == "completed" for step in report.steps)


def test_after_mismatch_stops_before_any_later_binding(native_example):
    bundle, witness, store = setup(native_example)
    bindings = store.bindings()
    def wrong(context):
        store.calls.append(context.action_id)
        return "claimed success without changing independently observed state"
    bindings["01_request"] = wrong
    report = execute_witness(bundle, witness, bindings, store.reset, store.observe)
    assert report.decision == "MODEL_MISMATCH"
    assert report.attempted_actions == report.completed_actions == 1
    assert store.calls == ["01_request"]
    assert report.steps[0].outcome == "mismatch"


@pytest.mark.parametrize("state", [
    {"requested": False},
    {"requested": 0, "approved": False, "independent": False, "paid": False},
    {"requested": False, "approved": False, "independent": False, "paid": False, "extra": False},
])
def test_initial_observer_must_report_all_exact_boolean_facts(native_example, state):
    bundle, witness, store = setup(native_example)
    report = execute_witness(bundle, witness, store.bindings(), store.reset, lambda: state)
    assert report.decision == "MODEL_MISMATCH"
    assert report.attempted_actions == report.completed_actions == 0
    assert not store.calls


def test_before_observer_drift_stops_before_first_action(native_example):
    bundle, witness, store = setup(native_example)
    calls = 0
    def observer():
        nonlocal calls
        calls += 1
        return store.observe() if calls == 1 else {**store.observe(), "paid": True}
    report = execute_witness(bundle, witness, store.bindings(), store.reset, observer)
    assert report.decision == "MODEL_MISMATCH"
    assert report.attempted_actions == 0 and not store.calls


def test_failure_retains_attempted_count_and_observed_side_effect_without_secret(native_example):
    bundle, witness, store = setup(native_example)
    bindings = store.bindings()
    def failed(context):
        store.request(context)
        raise RuntimeError("DO-NOT-EXPOSE-SECRET-KEY")
    bindings["01_request"] = failed
    report = execute_witness(bundle, witness, bindings, store.reset, store.observe)
    assert report.decision == "EXECUTION_ERROR"
    assert report.attempted_actions == 1 and report.completed_actions == 0
    assert store.calls == ["01_request"]
    assert report.steps[0].after["requested"] is True
    assert report.steps[0].error_type == "RuntimeError"
    assert "DO-NOT-EXPOSE" not in report.model_dump_json()


@pytest.mark.parametrize("kind", ["coroutine", "generator", "async-generator"])
def test_sync_wrapper_deferred_result_is_never_completed(native_example, kind):
    bundle, witness, store = setup(native_example)
    async def coroutine():
        return 1
    def generator():
        yield 1
    async def async_generator():
        yield 1
    returned = {"coroutine": coroutine, "generator": generator, "async-generator": async_generator}[kind]
    bindings = store.bindings()
    bindings["01_request"] = lambda _context: returned()
    report = execute_witness(bundle, witness, bindings, store.reset, store.observe)
    assert report.decision == "EXECUTION_ERROR"
    assert report.attempted_actions == 1 and report.completed_actions == 0
    assert report.steps[0].outcome == "failed"


def test_prevalidation_all_bindings_missing_or_async_never_resets(native_example):
    bundle, witness, store = setup(native_example)
    resets = []
    class AsyncCallable:
        async def __call__(self, _context):
            raise AssertionError("executed")
    for bindings in ({"01_request": store.request}, {**store.bindings(), "04_pay": AsyncCallable()}):
        report = execute_witness(bundle, witness, bindings, lambda: resets.append(1), store.observe)
        assert report.decision == "INCONCLUSIVE"
        assert report.attempted_actions == 0
    assert not resets and not store.calls


@pytest.mark.parametrize("max_steps", [True, 0, 2, 65])
def test_explicit_step_budget_checked_before_reset(native_example, max_steps):
    bundle, witness, store = setup(native_example)
    resets = []
    report = execute_witness(bundle, witness, store.bindings(), lambda: resets.append(1), store.observe, max_steps=max_steps)
    assert report.decision == "INCONCLUSIVE" and not resets


def test_tampered_bundle_and_unretained_witness_cannot_trigger_reset(native_example):
    bundle, witness, store = setup(native_example)
    resets = []
    tampered = bundle.model_copy(update={"integrity_digest": "0" * 64})
    report = execute_witness(tampered, witness, store.bindings(), lambda: resets.append(1), store.observe)
    assert report.decision == "INCONCLUSIVE"
    alien = witness.model_copy(update={"objective_id": "invented-objective"})
    report = execute_witness(bundle, alien, store.bindings(), lambda: resets.append(1), store.observe)
    assert report.decision == "INCONCLUSIVE"
    assert not resets


def test_reset_return_value_and_observer_exception_cannot_claim_match(native_example):
    bundle, witness, store = setup(native_example)
    report = execute_witness(bundle, witness, store.bindings(), lambda: "unexpected reset return", store.observe)
    assert report.decision == "EXECUTION_ERROR" and report.attempted_actions == 0
    def failed_observer():
        raise RuntimeError("hidden observer payload")
    report = execute_witness(bundle, witness, store.bindings(), store.reset, failed_observer)
    assert report.decision == "EXECUTION_ERROR"
    assert "hidden observer" not in report.model_dump_json()
@pytest.mark.parametrize("mutation", ["index_bool", "cost_bool", "nonserializable"])
def test_unvalidated_witness_copy_never_hashes_or_resets(native_example, mutation):
    bundle = create_exploration_bundle(native_example.demo_model(), native_example.demo_policy())
    witness = bundle.report.safety_witnesses[0]
    if mutation == "nonserializable":
        witness = witness.model_copy(update={"objective_id": object()})
    else:
        step = witness.steps[0].model_copy(update={"index" if mutation == "index_bool" else "cost_units": True})
        witness = witness.model_copy(update={"steps": (step, *witness.steps[1:])})
    called = []
    report = execute_witness(bundle, witness, {}, lambda: called.append("reset"), lambda: {})
    assert report.decision == "INCONCLUSIVE" and not called
    assert report.attempted_actions == 0 and report.witness_digest == "0" * 64
