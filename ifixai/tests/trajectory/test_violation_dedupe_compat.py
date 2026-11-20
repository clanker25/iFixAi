"""Only exact repeat findings collapse, and bundles saved with repeats still replay."""

import json

import pytest

from ifixai.trajectory import (
    audit_trajectory,
    canonical_digest,
    create_audit_bundle,
    load_audit_bundle,
)
from ifixai.trajectory.models import (
    ActorPolicy,
    AuditPolicy,
    BudgetRule,
    DelegationPayload,
    PermissionRule,
)
from ifixai.trajectory.recorder import TraceRecorder


def test_two_budget_rules_broken_at_one_action_are_two_findings(native_example):
    capture, _, _ = native_example.record_scenario("budget-overrun")
    policy = native_example.demo_policy().model_copy(update={"budgets": (
        BudgetRule(scope="trace", limit_units=100),
        BudgetRule(scope="actor", actor_id="agent", limit_units=100),
    )})
    report = audit_trajectory(capture, policy)

    budget = sorted(v.message for v in report.violations if v.code == "BUDGET_EXCEEDED")
    assert budget == [
        "Configured actor:agent budget rule 1 exceeded its integer limit.",
        "Configured trace budget rule 0 exceeded its integer limit.",
    ]


def _saved_with_repeats():
    """A bundle for one bad grant used twice, stored the way 4.2.0 wrote it: the
    grant's finding once at the grant and again before each action's own finding."""
    recorder = TraceRecorder("tenant", "collector", clock_ns=lambda: 100)
    grant = recorder.emit(DelegationPayload(
        actor_id="manager", subject_id="worker", action="write",
        resource_scope="orders:*", expires_ns=1000,
    ))
    for index in range(2):
        recorder.invoke(
            lambda: None, actor_id="worker", action="write", resource=f"orders:{index}",
            arguments={}, delegation_id=grant.event_id, parents=(grant.event_id,),
        )
    policy = AuditPolicy(
        name="owned", tenant_id="tenant", trusted_collectors=("collector",),
        actors={actor: ActorPolicy(roles=("owner",) if actor == "owner" else ())
                for actor in ("owner", "manager", "worker")},
        permissions=(PermissionRule(role="owner", action="write", resource_scope="orders:*"),),
        require_citations=False,
    )
    raw = create_audit_bundle(recorder.seal(), policy).model_dump(mode="json")
    issuer, *denied = raw["report"]["violations"]
    assert issuer["code"] == "DELEGATION_ISSUER_PERMISSION"
    assert [v["code"] for v in denied] == ["PERMISSION_DENIED", "PERMISSION_DENIED"]
    raw["report"]["violations"] = [issuer, issuer, denied[0], issuer, denied[1]]
    return raw


def _seal(raw, path):
    raw["integrity_digest"] = canonical_digest({k: v for k, v in raw.items() if k != "integrity_digest"})
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def test_bundle_saved_with_repeated_findings_still_replays(tmp_path):
    bundle = load_audit_bundle(_seal(_saved_with_repeats(), tmp_path / "saved.json"))

    assert bundle.report.decision == "BLOCK"
    # The retained evidence is returned as saved, so its checksum still holds.
    assert len(bundle.report.violations) == 5


@pytest.mark.parametrize("edit", ["reword", "rename", "drop"])
def test_changed_finding_in_saved_bundle_is_still_rejected(tmp_path, edit):
    raw = _saved_with_repeats()
    target = raw["report"]["violations"][2]
    if edit == "reword":
        target["message"] += " (edited)"
    elif edit == "rename":
        target["code"] = "EDITED_CODE"
    else:
        del raw["report"]["violations"][2]

    with pytest.raises(ValueError, match="audit report does not match"):
        load_audit_bundle(_seal(raw, tmp_path / "edited.json"))
