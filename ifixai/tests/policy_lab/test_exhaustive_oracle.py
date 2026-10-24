"""Independent primitive-state oracle; never calls implementation helpers."""

from collections import deque
from itertools import product

import pytest

from ifixai.policy_lab import explore_policy
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


def primitive_oracle(allowed, costs, limit):
    initial = (False, False, 0)
    queue = deque([(initial, 0)])
    seen = {initial}
    unsafe = goal = None
    enabled = 0
    # independent transition declarations (guard/effect lambdas), no engine APIs
    operations = (
        (lambda x, _y: not x, lambda _x, y: (True, y)),
        (lambda x, _y: x, lambda _x, y: (False, y)),
        (lambda x, y: x and not y, lambda x, _y: (x, True)),
    )
    while queue:
        (x, y, spent), depth = queue.popleft()
        if x and y and unsafe is None:
            unsafe = depth
        if y and goal is None:
            goal = depth
        for i, (guard, effect) in enumerate(operations):
            if not allowed[i] or not guard(x, y) or (limit is not None and spent + costs[i] > limit):
                continue
            enabled += 1
            next_x, next_y = effect(x, y)
            candidate = (next_x, next_y, spent + costs[i] if limit is not None else 0)
            if candidate not in seen:
                seen.add(candidate)
                queue.append((candidate, depth + 1))
    return len(seen), enabled, unsafe, goal


def tiny_contract(allowed, costs, limit):
    model = BusinessModel(name="Tiny exhaustive oracle", tenant_id="tenant", initial_facts={"x": False, "y": False}, actors={"actor": Actor(roles=("worker",))}, actions=(
        ActionContract(action_id="a", actor_id="actor", tool="set_x", resource="object:one", requires={"x": False}, effects={"x": True}, cost_units=costs[0]),
        ActionContract(action_id="b", actor_id="actor", tool="clear_x", resource="object:one", requires={"x": True}, effects={"x": False}, cost_units=costs[1]),
        ActionContract(action_id="c", actor_id="actor", tool="set_y", resource="object:one", requires={"x": True, "y": False}, effects={"y": True}, cost_units=costs[2]),
    ), properties=(SafetyProperty(property_id="unsafe", forbidden={"x": True, "y": True}),), goals=(BusinessGoal(goal_id="goal", requires={"y": True}),))
    tools = ("set_x", "clear_x", "set_y")
    policy = ExplorationPolicy(name="Tiny policy", tenant_id="tenant", permissions=tuple(PermissionRule(role="worker", tool=tool) for enabled, tool in zip(allowed, tools, strict=True) if enabled), budgets=() if limit is None else (BudgetRule(scope="trace", limit_units=limit),))
    return model, policy


def test_bfs_agrees_with_independent_exhaustive_oracle_across_120_policies():
    compared = 0
    for allowed, costs, limit in product(product((False, True), repeat=3), ((0, 0, 0), (1, 1, 1), (2, 1, 1)), (None, 0, 1, 2, 3)):
        expected_states, expected_edges, unsafe_depth, goal_depth = primitive_oracle(allowed, costs, limit)
        model, policy = tiny_contract(allowed, costs, limit)
        report = explore_policy(model, policy, SearchConfig(max_depth=64))
        assert report.frontier_exhausted
        assert report.states_discovered == expected_states
        assert report.transitions_enabled == expected_edges
        assert report.property_status["unsafe"] == ("COUNTEREXAMPLE" if unsafe_depth is not None else "VERIFIED")
        assert report.goal_status["goal"] == ("REACHABLE" if goal_depth is not None else "UNREACHABLE")
        assert [len(w.steps) for w in report.safety_witnesses] == ([] if unsafe_depth is None else [unsafe_depth])
        assert [len(w.steps) for w in report.goal_witnesses] == ([] if goal_depth is None else [goal_depth])
        if expected_edges == 0:
            assert report.decision != "VERIFIED"
        compared += 1
    assert compared == 120


def test_budget_counters_are_part_of_visited_identity_even_when_facts_cycle():
    model, policy = tiny_contract((True, True, False), (1, 1, 0), 4)
    report = explore_policy(model, policy)
    assert report.states_discovered == 5
    assert report.transitions_enabled == 4
    assert report.frontier_exhausted


@pytest.mark.parametrize("config", [SearchConfig(max_depth=1), SearchConfig(max_states=1), SearchConfig(max_transitions=1)])
def test_finite_limits_do_not_claim_complete_frontier(config):
    model, policy = tiny_contract((True, True, True), (0, 0, 0), None)
    report = explore_policy(model, policy, config)
    assert not report.frontier_exhausted
    assert report.cutoff_reasons
    assert report.decision == "INCONCLUSIVE"
    assert report.property_status["unsafe"] == "INCONCLUSIVE"


def test_depth_boundary_existing_cycle_is_a_complete_fixed_point():
    model, policy = tiny_contract((True, True, False), (0, 0, 0), None)
    report = explore_policy(model, policy, SearchConfig(max_depth=1))
    assert report.frontier_exhausted
    assert report.states_discovered == 2
    assert not report.cutoff_reasons
    assert report.goal_status["goal"] == "UNREACHABLE"
    assert report.decision == "GOAL_BLOCKED"


def test_initial_unsafe_has_zero_step_witness(native_example):
    model = native_example.demo_model()
    model = model.model_copy(update={"initial_facts": {**model.initial_facts, "paid": True}})
    report = explore_policy(model, native_example.demo_policy())
    assert report.decision == "COUNTEREXAMPLE"
    assert len(report.safety_witnesses[0].steps) == 0


def test_simultaneous_effects_preserve_unchanged_facts(native_example):
    report = explore_policy(native_example.demo_model(), native_example.demo_policy())
    witness = report.goal_witnesses[0]
    approved = witness.steps[1]
    assert approved.action_id == "03_independent_approve"
    assert approved.before.facts["requested"] is True
    assert approved.after.facts == {"requested": True, "approved": True, "independent": True, "paid": False}
