"""Reproducible local observer/auditor benchmark; no production latency claim.

Run: python -m ifixai.tests.trajectory.benchmark_trajectory --output NEW.json
"""

import argparse
import json
import time
import tracemalloc
from pathlib import Path

from ifixai.trajectory import audit_trajectory
from ifixai.trajectory.models import (
    ActorPolicy,
    AuditPolicy,
    BudgetRule,
    PermissionRule,
)
from ifixai.trajectory.recorder import TraceRecorder


def measure(action_count):
    tracemalloc.start()
    started = time.perf_counter()
    recorder = TraceRecorder("benchmark-tenant", "local-benchmark", clock_ns=lambda: 100)
    count = 0
    def increment():
        nonlocal count
        count += 1
        return count
    for _ in range(action_count):
        recorder.invoke(increment, actor_id="writer", action="increment", resource="counter:local", arguments={"increment": 1}, cost_units=3)
    capture = recorder.seal()
    recorded = time.perf_counter()
    policy = AuditPolicy(name="Benchmark exact-count oracle", tenant_id="benchmark-tenant", trusted_collectors=("local-benchmark",), actors={"writer": ActorPolicy(roles=("writer",))}, permissions=(PermissionRule(role="writer", action="increment", resource_scope="counter:*"),), budgets=(BudgetRule(scope="trace", limit_units=3 * action_count),), require_citations=False)
    report = audit_trajectory(capture, policy)
    audited = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert count == action_count == report.checked_actions == report.completed_actions
    assert report.budget_totals["trace"] == 3 * action_count
    assert report.decision == "CLEAR" and not report.violations and not report.coverage_issues
    return {"actual_callable_count": count, "typed_event_count": len(capture.events), "edge_count": len(report.graph_edges), "cost_units": report.budget_totals["trace"], "decision": report.decision, "record_seconds": recorded - started, "audit_seconds": audited - recorded, "peak_python_tracemalloc_bytes": peak}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = {"measurement_kind": "local Python callables, typed recorder, deterministic audit", "network_or_model_calls": 0, "production_latency_claim": False, "sizes": [measure(1000), measure(10000)]}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
