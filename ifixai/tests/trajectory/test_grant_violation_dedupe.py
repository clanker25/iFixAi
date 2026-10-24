"""A defective grant is one finding however many actions cite it."""

from collections import Counter

from ifixai.trajectory import audit_trajectory
from ifixai.trajectory.models import (
    ActorPolicy,
    AuditPolicy,
    DelegationPayload,
    PermissionRule,
)
from ifixai.trajectory.recorder import TraceRecorder


def test_invalid_grant_is_reported_once_per_defect():
    recorder = TraceRecorder("tenant", "collector", clock_ns=lambda: 100)
    # The issuer has no direct permission, so the root grant itself is invalid.
    grant = recorder.emit(DelegationPayload(
        actor_id="manager", subject_id="worker", action="write",
        resource_scope="orders:*", expires_ns=1000,
    ))
    for index in range(3):
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
    report = audit_trajectory(recorder.seal(), policy)

    codes = Counter(violation.code for violation in report.violations)
    assert codes["DELEGATION_ISSUER_PERMISSION"] == 1
    # Each of the three actions is still its own permission finding.
    assert codes["PERMISSION_DENIED"] == 3
