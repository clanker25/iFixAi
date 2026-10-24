import pytest

from ifixai.trajectory import audit_trajectory, scope_matches, scope_subset
from ifixai.trajectory.integrity import canonical_digest
from ifixai.trajectory.models import (
    ActorPolicy,
    ApprovalPayload,
    ApprovalRule,
    AuditPolicy,
    BudgetRule,
    DelegationPayload,
    PermissionRule,
    RevocationPayload,
)
from ifixai.trajectory.recorder import TraceRecorder


def chain(*, child_scope="orders:one", revoke=False, forged_revoker=False, ancestor_expired=False):
    recorder = TraceRecorder("tenant", "collector", clock_ns=lambda: 100)
    first = recorder.emit(DelegationPayload(actor_id="owner", subject_id="manager", action="write", resource_scope="orders:*", expires_ns=100 if ancestor_expired else 1000))
    child = recorder.emit(DelegationPayload(actor_id="manager", subject_id="worker", action="write", resource_scope=child_scope, expires_ns=1000, parent_grant_id=first.event_id))
    if revoke:
        recorder.emit(RevocationPayload(actor_id="worker" if forged_revoker else "owner", grant_id=first.event_id))
    recorder.invoke(lambda: "real local write", actor_id="worker", action="write", resource="orders:one", arguments={}, delegation_id=child.event_id, cost_units=7)
    policy = AuditPolicy(name="Chain test", tenant_id="tenant", trusted_collectors=("collector",), actors={actor: ActorPolicy(roles=("owner",) if actor == "owner" else ()) for actor in ("owner", "manager", "worker")}, permissions=(PermissionRule(role="owner", action="write", resource_scope="orders:*"),), budgets=(BudgetRule(scope="trace", limit_units=10),), require_citations=False)
    return recorder.seal(), policy, first.event_id, child.event_id


def test_chain_attenuation_positive_and_both_grants_in_witness_path():
    capture, policy, first, child = chain()
    report = audit_trajectory(capture, policy)
    assert report.decision == "CLEAR"
    action = next(e for e in capture.events if e.payload.kind == "action")
    assert first in report.authority_paths[action.event_id]
    assert child in report.authority_paths[action.event_id]


@pytest.mark.parametrize("options", [
    {"child_scope": "*"}, {"child_scope": "outside:*"},
    {"revoke": True}, {"ancestor_expired": True},
    {"revoke": True, "forged_revoker": True},
])
def test_escalation_revocation_and_ancestor_expiration_block(options):
    capture, policy, _, _ = chain(**options)
    report = audit_trajectory(capture, policy)
    assert report.decision == "BLOCK"
    assert any(v.rule == "delegation" for v in report.violations)
    assert report.budget_totals["trace"] == 7


def test_multiple_matching_approval_rules_all_apply(native_example):
    capture, _, _ = native_example.record_scenario()
    policy = native_example.demo_policy()
    strict = policy.model_copy(update={"approval_rules": (*policy.approval_rules, ApprovalRule(action="refund", resource_scope="accounts:*", approver_roles=("finance",)))})
    report = audit_trajectory(capture, strict)
    assert report.decision == "BLOCK"
    assert any(v.rule == "approval" for v in report.violations)


def test_approval_use_counter_shared_across_actions():
    for permitted, expected in ((1, "BLOCK"), (2, "CLEAR")):
        recorder = TraceRecorder("tenant", "collector", clock_ns=lambda: 100)
        grant = recorder.emit(ApprovalPayload(actor_id="owner", subject_id="owner", action="write", resource="orders:one", arguments_digest=canonical_digest({}), expires_ns=1000, uses=permitted))
        for _ in range(2):
            recorder.invoke(lambda: 1, actor_id="owner", action="write", resource="orders:one", arguments={}, approval_id=grant.event_id, cost_units=5)
        policy = AuditPolicy(name="Use count", tenant_id="tenant", trusted_collectors=("collector",), actors={"owner": ActorPolicy(roles=("owner",), human=True)}, permissions=(PermissionRule(role="owner", action="write", resource_scope="orders:*"),), approval_rules=(ApprovalRule(action="write", resource_scope="orders:*", approver_roles=("owner",)),), budgets=(BudgetRule(scope="trace", limit_units=10),), require_citations=False)
        report = audit_trajectory(recorder.seal(), policy)
        assert report.decision == expected
        assert report.budget_totals["trace"] == 10


def test_exact_integer_shared_and_actor_ledger_counts_unauthorized_actions():
    recorder = TraceRecorder("tenant", "collector", clock_ns=lambda: 100)
    amounts = (2**62 + 17, 2**62 + 19)
    for amount in amounts:
        recorder.invoke(lambda: 1, actor_id="writer", action="write", resource="orders:one", arguments={}, cost_units=amount)
    policy = AuditPolicy(name="Integer oracle", tenant_id="tenant", trusted_collectors=("collector",), actors={"writer": ActorPolicy(roles=())}, permissions=(PermissionRule(role="different-role", action="write", resource_scope="orders:*"),), budgets=(BudgetRule(scope="trace", limit_units=2**63 - 1), BudgetRule(scope="actor", actor_id="writer", limit_units=2**63 - 1)), require_citations=False)
    report = audit_trajectory(recorder.seal(), policy)
    assert report.decision == "BLOCK"
    assert report.budget_totals == {"trace": sum(amounts), "actor:writer": sum(amounts)}
    assert any(v.rule == "permission" for v in report.violations)
    assert any(v.rule == "budget" for v in report.violations)


def test_failed_callable_never_treated_as_zero_effect_clean_capture():
    recorder = TraceRecorder("tenant", "collector", clock_ns=lambda: 100)
    def failed():
        raise RuntimeError("unknown side effects")
    with pytest.raises(RuntimeError):
        recorder.invoke(failed, actor_id="owner", action="write", resource="orders:one", arguments={}, cost_units=5)
    policy = AuditPolicy(name="Failure coverage", tenant_id="tenant", trusted_collectors=("collector",), actors={"owner": ActorPolicy(roles=("owner",))}, permissions=(PermissionRule(role="owner", action="write", resource_scope="orders:*"),), budgets=(BudgetRule(scope="trace", limit_units=10),), require_citations=False)
    report = audit_trajectory(recorder.seal(), policy)
    assert report.decision == "INCONCLUSIVE"
    assert report.failed_actions == 1
    assert report.coverage_issues


def test_enabled_budget_with_no_applicable_action_cannot_clear():
    recorder = TraceRecorder("tenant", "collector", clock_ns=lambda: 100)
    recorder.invoke(lambda: 1, actor_id="owner", action="write", resource="orders:one", arguments={})
    policy = AuditPolicy(name="Unexercised actor pool", tenant_id="tenant", trusted_collectors=("collector",), actors={"owner": ActorPolicy(roles=("owner",)), "other": ActorPolicy(roles=())}, permissions=(PermissionRule(role="owner", action="write", resource_scope="orders:*"),), budgets=(BudgetRule(scope="actor", actor_id="other", limit_units=10),), require_citations=False)
    report = audit_trajectory(recorder.seal(), policy)
    assert report.decision == "INCONCLUSIVE"
    assert report.family_status["budget"] == "NOT_EVALUATED"
    assert report.coverage_issues


@pytest.mark.parametrize("child,parent,expected", [
    ("orders:one", "orders:*", True), ("orders:one*", "orders:*", True),
    ("orders:*", "orders:one*", False), ("orders:*", "orders:one", False),
    ("orders:one", "orders:one", True), ("orders:one", "orders:two", False),
    ("other:*", "orders:*", False), ("*", "orders:*", False),
    ("orders:*", "*", True), ("orders:one-extra", "orders:one", False),
])
def test_scope_attenuation_against_independent_exact_prefix_table(child, parent, expected):
    assert scope_subset(child, parent) is expected


@pytest.mark.parametrize("scope,resource,expected", [
    ("orders:*", "orders:one", True), ("orders:*", "outside:one", False),
    ("orders:one", "orders:one", True), ("orders:one", "orders:one-extra", False),
    ("*", "outside:one", True), ("orders:one*", "orders:one-extra", True),
])
def test_resource_matching_against_independent_exact_prefix_table(scope, resource, expected):
    assert scope_matches(scope, resource) is expected


@pytest.mark.parametrize("depth,decision", [(128, "CLEAR"), (129, "INCONCLUSIVE")])
def test_deep_authority_chain_work_bound_is_explicit_inconclusive(depth, decision):
    recorder = TraceRecorder("tenant", "collector", clock_ns=lambda: 100)
    previous = None
    for index in range(depth):
        grant = recorder.emit(DelegationPayload(actor_id=f"actor-{index}", subject_id=f"actor-{index + 1}", action="write", resource_scope="orders:one", expires_ns=1000, parent_grant_id=previous))
        previous = grant.event_id
    recorder.invoke(lambda: 1, actor_id=f"actor-{depth}", action="write", resource="orders:one", arguments={}, delegation_id=previous, cost_units=5)
    policy = AuditPolicy(name="Bounded deep chain oracle", tenant_id="tenant", trusted_collectors=("collector",), actors={f"actor-{i}": ActorPolicy(roles=("owner",) if i == 0 else ()) for i in range(depth + 1)}, permissions=(PermissionRule(role="owner", action="write", resource_scope="orders:*"),), budgets=(BudgetRule(scope="trace", limit_units=5),), require_citations=False)
    report = audit_trajectory(recorder.seal(), policy)
    assert report.decision == decision
    assert report.budget_totals["trace"] == 5
    assert not report.violations
    if decision == "INCONCLUSIVE":
        assert report.coverage_issues


def test_identity_normalization_cannot_collapse_denied_and_privileged_actor():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        AuditPolicy(name="Identity collision", tenant_id="tenant", trusted_collectors=("collector",), actors={"worker": ActorPolicy(roles=()), " worker ": ActorPolicy(roles=("admin",))}, permissions=(PermissionRule(role="admin", action="write", resource_scope="orders:*"),), require_citations=False)


def test_many_approval_reuses_keep_bounded_witnesses_without_losing_actions():
    recorder = TraceRecorder("tenant", "collector", clock_ns=lambda: 100)
    grant = recorder.emit(ApprovalPayload(actor_id="owner", subject_id="owner", action="write", resource="orders:one", arguments_digest=canonical_digest({}), expires_ns=1000, uses=1))
    for _ in range(150):
        recorder.invoke(lambda: 1, actor_id="owner", action="write", resource="orders:one", arguments={}, approval_id=grant.event_id)
    policy = AuditPolicy(name="Bounded witness oracle", tenant_id="tenant", trusted_collectors=("collector",), actors={"owner": ActorPolicy(roles=("owner",), human=True)}, permissions=(PermissionRule(role="owner", action="write", resource_scope="orders:*"),), approval_rules=(ApprovalRule(action="write", resource_scope="orders:*", approver_roles=("owner",)),), require_citations=False)
    capture = recorder.seal()
    report = audit_trajectory(capture, policy)
    reuses = [violation for violation in report.violations if violation.code == "APPROVAL_REUSE"]
    assert report.decision == "BLOCK"
    assert report.completed_actions == 150
    assert len(reuses) == 149
    assert all(len(violation.witness_ids) <= 103 for violation in reuses)
    assert all(violation.event_id in violation.witness_ids for violation in reuses)
    assert len(capture.events) == 153


def test_duplicate_budget_rules_are_equivalent_to_strictest_pool_once(native_example):
    capture, _, _ = native_example.record_scenario()
    base = native_example.demo_policy()
    duplicate = base.model_copy(update={"budgets": (BudgetRule(scope="trace", limit_units=100), BudgetRule(scope="trace", limit_units=59), BudgetRule(scope="trace", limit_units=70))})
    strictest = base.model_copy(update={"budgets": (BudgetRule(scope="trace", limit_units=59),)})
    left = audit_trajectory(capture, duplicate)
    right = audit_trajectory(capture, strictest)
    assert left.decision == right.decision == "BLOCK"
    assert left.budget_totals == right.budget_totals == {"trace": 60}
    assert left.family_counts["budget"] == right.family_counts["budget"] == 1
    assert [(v.rule, v.code, v.event_id, v.witness_ids) for v in left.violations] == [(v.rule, v.code, v.event_id, v.witness_ids) for v in right.violations]


def test_ancestry_work_limit_is_inconclusive_and_preserves_observed_integer_cost(native_example, monkeypatch):
    capture, store, _ = native_example.record_scenario()
    assert audit_trajectory(capture, native_example.demo_policy()).decision == "CLEAR"
    monkeypatch.setattr("ifixai.trajectory.audit.MAX_ANCESTRY_WALK_STEPS", 1)
    report = audit_trajectory(capture, native_example.demo_policy())
    assert report.decision == "INCONCLUSIVE"
    assert any(issue.code == "ANCESTRY_WORK_LIMIT" for issue in report.coverage_issues)
    assert not report.violations
    assert report.budget_totals["trace"] == store.refunded_units == 60
    assert report.completed_actions == store.calls == 1
    assert report.checked_actions == 0
    assert report.family_status["permission"] == "NOT_EVALUATED"
