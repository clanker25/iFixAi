"""Bounded declarative trajectory facts and policies; no executable specifications."""
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

Identifier = Annotated[str, StringConstraints(min_length=1, max_length=200, pattern=r"^\S(?:[\s\S]*\S)?$")]
Resource = Annotated[str, StringConstraints(min_length=1, max_length=1000)]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Timestamp = Annotated[int, Field(ge=0, le=2**63 - 1, strict=True)]
Amount = Annotated[int, Field(ge=0, le=2**63 - 1, strict=True)]
TraceId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
Message = Annotated[str, StringConstraints(max_length=3000)]
Counter = Annotated[int, Field(ge=0, le=3200000, strict=True)]
Total = Annotated[int, Field(ge=0, le=(2**63 - 1) * 100000, strict=True)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def _exact_resource(value: str) -> str:
    if "*" in value:
        raise ValueError("resource must be exact; wildcards belong only in resource_scope")
    return value


def _scope(value: str) -> str:
    if value.count("*") > 1 or ("*" in value and not value.endswith("*")):
        raise ValueError("resource_scope supports at most one terminal wildcard")
    return value


def _trace(value: str) -> str:
    return value.lower() if isinstance(value, str) else value


class StartPayload(StrictModel):
    kind: Literal["start"] = "start"
    label: str = Field(default="", max_length=1000)


class ApprovalPayload(StrictModel):
    kind: Literal["approval"] = "approval"
    actor_id: Identifier
    subject_id: Identifier
    action: Identifier
    resource: Resource
    arguments_digest: Digest
    expires_ns: Timestamp
    uses: int = Field(default=1, ge=1, le=100, strict=True)
    human: bool = Field(default=True, strict=True)
    exact_resource = field_validator("resource")(_exact_resource)


class RevocationPayload(StrictModel):
    kind: Literal["revocation"] = "revocation"
    actor_id: Identifier
    grant_id: Identifier


class DelegationPayload(StrictModel):
    kind: Literal["delegation"] = "delegation"
    actor_id: Identifier
    subject_id: Identifier
    action: Identifier
    resource_scope: Resource
    expires_ns: Timestamp
    parent_grant_id: Identifier | None = None
    valid_scope = field_validator("resource_scope")(_scope)


class ActionPayload(StrictModel):
    kind: Literal["action"] = "action"
    actor_id: Identifier
    action: Identifier
    resource: Resource
    arguments_digest: Digest
    approval_id: Identifier | None = None
    delegation_id: Identifier | None = None
    cost_units: Amount = 0
    outcome: Literal["completed", "failed", "unknown"] = "completed"
    exact_resource = field_validator("resource")(_exact_resource)


class SourcePayload(StrictModel):
    kind: Literal["source"] = "source"
    actor_id: Identifier
    resource: Resource
    content_digest: Digest
    exact_resource = field_validator("resource")(_exact_resource)


class Citation(StrictModel):
    source_event_id: Identifier
    content_digest: Digest


class OutputPayload(StrictModel):
    kind: Literal["output"] = "output"
    actor_id: Identifier
    citations: tuple[Citation, ...] = Field(default=(), max_length=1000)
    label: str = Field(default="", max_length=1000)


class EndPayload(StrictModel):
    kind: Literal["end"] = "end"
    complete: bool = Field(strict=True)
    declared_events: int = Field(ge=2, le=100000, strict=True)


Payload = Annotated[
    StartPayload | ApprovalPayload | RevocationPayload | DelegationPayload | ActionPayload | SourcePayload | OutputPayload | EndPayload,
    Field(discriminator="kind"),
]


class TraceEvent(StrictModel):
    event_id: Identifier
    trace_id: TraceId
    tenant_id: Identifier
    sequence: int = Field(ge=1, le=100000, strict=True)
    time_ns: Timestamp
    parents: tuple[Identifier, ...] = Field(default=(), max_length=32)
    payload: Payload
    normalize_trace = field_validator("trace_id", mode="before")(_trace)


class TraceCapture(StrictModel):
    schema_version: Literal["1"] = "1"
    trace_id: TraceId
    tenant_id: Identifier
    collector_id: Identifier
    events: tuple[TraceEvent, ...] = Field(max_length=100000)
    capture_complete: bool = Field(strict=True)
    import_warnings: tuple[Message, ...] = Field(default=(), max_length=10000)
    source_format: Literal["jsonl", "otlp", "native"] = "native"
    normalize_trace = field_validator("trace_id", mode="before")(_trace)


class ActorPolicy(StrictModel):
    roles: tuple[Identifier, ...] = Field(default=(), max_length=1000)
    human: bool = Field(default=False, strict=True)


class PermissionRule(StrictModel):
    role: Identifier
    action: Identifier
    resource_scope: Resource
    valid_scope = field_validator("resource_scope")(_scope)


class ApprovalRule(StrictModel):
    action: Identifier
    resource_scope: Resource
    approver_roles: tuple[Identifier, ...] = Field(min_length=1, max_length=1000)
    require_human: bool = Field(default=True, strict=True)
    valid_scope = field_validator("resource_scope")(_scope)


class BudgetRule(StrictModel):
    scope: Literal["trace", "actor"]
    limit_units: Amount
    actor_id: Identifier | None = None

    @model_validator(mode="after")
    def valid_scope(self) -> BudgetRule:
        if (self.scope == "actor") != (self.actor_id is not None):
            raise ValueError("actor budgets require actor_id; trace budgets prohibit actor_id")
        return self


class AuditPolicy(StrictModel):
    schema_version: Literal["1"] = "1"
    name: Identifier
    tenant_id: Identifier
    trusted_collectors: tuple[Identifier, ...] = Field(min_length=1, max_length=10000)
    actors: dict[Identifier, ActorPolicy] = Field(min_length=1, max_length=10000)
    permissions: tuple[PermissionRule, ...] = Field(default=(), max_length=10000)
    approval_rules: tuple[ApprovalRule, ...] = Field(default=(), max_length=10000)
    budgets: tuple[BudgetRule, ...] = Field(default=(), max_length=1000)
    require_citations: bool = Field(default=True, strict=True)


class Violation(StrictModel):
    rule: Literal["permission", "approval", "delegation", "budget", "citation"]
    code: Identifier
    event_id: Identifier
    message: str = Field(max_length=3000)
    witness_ids: tuple[Identifier, ...] = Field(default=(), max_length=100000)
    actor_id: Identifier | None = None


class CoverageIssue(StrictModel):
    code: Identifier
    event_id: Identifier | None = None
    message: str = Field(max_length=3000)
    witness_ids: tuple[Identifier, ...] = Field(default=(), max_length=100000)


FamilyStatus = Literal["EVALUATED", "NOT_EVALUATED"]


class AuditReport(StrictModel):
    decision: Literal["CLEAR", "BLOCK", "INCONCLUSIVE"]
    violations: tuple[Violation, ...]
    coverage_issues: tuple[CoverageIssue, ...]
    checked_actions: int = Field(ge=0, le=100000, strict=True)
    completed_actions: int = Field(ge=0, le=100000, strict=True)
    failed_actions: int = Field(ge=0, le=100000, strict=True)
    family_counts: dict[str, Counter]
    family_status: dict[str, FamilyStatus]
    graph_edges: tuple[tuple[str, str], ...]
    authority_paths: dict[str, tuple[str, ...]]
    budget_totals: dict[str, Total]
    policy_digest: Digest
    capture_digest: Digest
    limitations: tuple[Message, ...]


class AuditBundle(StrictModel):
    schema_version: Literal["1"] = "1"
    policy: AuditPolicy
    capture: TraceCapture
    report: AuditReport
    integrity_digest: Digest
