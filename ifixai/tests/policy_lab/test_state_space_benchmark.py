"""Exact synthetic binary state-space benchmark, independent integer oracle.

Monotone bit-set transitions produce every integer in [0, size). Explicit
Boolean prefix guards encode the upper bound; maximum path depth is bit width.
These local search measurements do not measure production agent latency.
"""

import json
import os
import time
import tracemalloc

import pytest

from ifixai.policy_lab import create_exploration_bundle
from ifixai.policy_lab.models import (
    ActionContract,
    Actor,
    BusinessGoal,
    BusinessModel,
    ExplorationPolicy,
    PermissionRule,
    SafetyProperty,
    SearchConfig,
)


def binary_state_space(size):
    width = (size - 1).bit_length()
    facts = {f"b{i}": False for i in range(width)}
    facts["unsafe"] = False
    actions = []
    for bit in range(width):
        threshold = size - (1 << bit)
        # x < threshold is a disjoint union of equal-higher-prefix/zero-bit
        # clauses at each one-bit of threshold. Require target bit currently0.
        for pivot in range(width):
            if not threshold & (1 << pivot):
                continue
            guard = {f"b{i}": bool(threshold & (1 << i)) for i in range(pivot + 1, width)}
            guard[f"b{pivot}"] = False
            if guard.get(f"b{bit}") is True:
                continue
            guard[f"b{bit}"] = False
            actions.append(ActionContract(action_id=f"set_{bit:02}_{pivot:02}", actor_id="worker", tool="set_bit", resource="counter:synthetic", requires=guard, effects={f"b{bit}": True}))
    model = BusinessModel(name=f"Synthetic {size} state counter", tenant_id="benchmark", initial_facts=facts, actors={"worker": Actor(roles=("worker",))}, actions=tuple(actions), properties=(SafetyProperty(property_id="safe", forbidden={"unsafe": True}),), goals=(BusinessGoal(goal_id="last", requires={f"b{i}": bool((size - 1) & (1 << i)) for i in range(width)}),))
    policy = ExplorationPolicy(name="Declared bit operations", tenant_id="benchmark", permissions=(PermissionRule(role="worker", tool="set_bit", resource_scope="counter:synthetic"),))
    return model, policy, width


@pytest.mark.parametrize("size", [1000, 10000])
def test_exhaustive_binary_state_count_and_edge_oracle(size):
    model, policy, width = binary_state_space(size)
    expected_edges = sum(1 for value in range(size) for bit in range(width) if not value & (1 << bit) and (value | (1 << bit)) < size)
    config = SearchConfig(max_depth=width, max_states=size, max_transitions=2000000)
    tracemalloc.start()
    started = time.perf_counter()
    bundle = create_exploration_bundle(model, policy, config)
    elapsed = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    report = bundle.report
    assert report.decision == "VERIFIED" and report.frontier_exhausted
    assert report.states_discovered == report.states_expanded == size
    assert report.transitions_enabled == expected_edges
    assert report.transitions_examined == size * len(model.actions)
    assert len(report.goal_witnesses[0].steps) == (size - 1).bit_count()
    encoded_bytes = len(bundle.model_dump_json().encode())
    assert encoded_bytes < 100000
    assert peak_bytes < 128 * 1024 * 1024
    destination = os.environ.get("IFIXAI_POLICY_LAB_BENCHMARK_DIR")
    if destination:
        from pathlib import Path
        path = Path(destination) / f"benchmark-{size}.json"
        path.write_text(json.dumps({"synthetic": True, "production_latency": False, "distinct_states": size, "enabled_edges": expected_edges, "action_contracts": len(model.actions), "transition_examinations": report.transitions_examined, "elapsed_seconds_with_tracemalloc": elapsed, "peak_traced_bytes": peak_bytes, "bundle_bytes": encoded_bytes, "frontier_exhausted": report.frontier_exhausted, "decision": report.decision}, indent=2))
