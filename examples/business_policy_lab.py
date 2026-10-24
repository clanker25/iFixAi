"""Find a harmful legal composition, repair it, and replay real local bindings.

Run: python examples/business_policy_lab.py --output ./new-business-lab-demo
The store is benign local accounting. No real money, services or models run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ifixai.policy_lab import create_exploration_bundle
from ifixai.policy_lab.executor import execute_witness
from ifixai.policy_lab.models import (
    ActionContract,
    Actor,
    BudgetRule,
    BusinessGoal,
    BusinessModel,
    ExplorationPolicy,
    PermissionRule,
    SafetyProperty,
    SearchConfig,
)
from ifixai.policy_lab.repair import search_repairs


def demo_model():
    return BusinessModel(
        name="Local refund composition", tenant_id="demo-tenant",
        initial_facts={"requested": False, "approved": False, "independent": False, "paid": False},
        actors={"agent": Actor(roles=("agent",)), "reviewer": Actor(roles=("reviewer",))},
        actions=(
            ActionContract(action_id="01_request", actor_id="agent", tool="request_refund", resource="account:demo", requires={"requested": False, "paid": False}, effects={"requested": True}, cost_units=1),
            ActionContract(action_id="02_self_approve", actor_id="agent", tool="approve_refund", resource="account:demo", requires={"requested": True, "approved": False}, effects={"approved": True, "independent": False}, cost_units=1),
            ActionContract(action_id="03_independent_approve", actor_id="reviewer", tool="approve_refund", resource="account:demo", requires={"requested": True, "approved": False}, effects={"approved": True, "independent": True}, cost_units=1),
            ActionContract(action_id="04_pay", actor_id="agent", tool="pay_refund", resource="account:demo", requires={"approved": True, "paid": False}, effects={"paid": True}, cost_units=1),
        ),
        properties=(SafetyProperty(property_id="no_self_approved_payment", label="Payment requires independent approval", forbidden={"paid": True, "independent": False}),),
        goals=(BusinessGoal(goal_id="legitimate_refund", label="Independent approval still reaches payment", requires={"paid": True, "independent": True}),),
    )


def demo_policy():
    return ExplorationPolicy(name="Permitted local operations", tenant_id="demo-tenant", permissions=(
        PermissionRule(role="agent", tool="request_refund", resource_scope="account:*"),
        PermissionRule(role="agent", tool="approve_refund", resource_scope="account:*"),
        PermissionRule(role="agent", tool="pay_refund", resource_scope="account:*"),
        PermissionRule(role="reviewer", tool="approve_refund", resource_scope="account:*"),
    ), budgets=(BudgetRule(scope="trace", limit_units=3),))


class LocalRefundStore:
    """Observed state comes from this local object, not action return values."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.facts = {"requested": False, "approved": False, "independent": False, "paid": False}
        self.synthetic_paid_units = 0
        self.calls = []

    def observe(self):
        return dict(self.facts)

    def request(self, context):
        self.calls.append(context.action_id)
        self.facts["requested"] = True

    def self_approve(self, context):
        self.calls.append(context.action_id)
        self.facts.update(approved=True, independent=False)

    def independent_approve(self, context):
        self.calls.append(context.action_id)
        self.facts.update(approved=True, independent=True)

    def pay(self, context):
        self.calls.append(context.action_id)
        self.synthetic_paid_units += 60
        self.facts["paid"] = True

    def bindings(self):
        return {"01_request": self.request, "02_self_approve": self.self_approve, "03_independent_approve": self.independent_approve, "04_pay": self.pay}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.absolute()
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    model, policy, config = demo_model(), demo_policy(), SearchConfig()
    original = create_exploration_bundle(model, policy, config)
    assert original.report.decision == "COUNTEREXAMPLE"
    unsafe = original.report.safety_witnesses[0]
    assert len(unsafe.steps) == 3
    store = LocalRefundStore()
    native_unsafe = execute_witness(original, unsafe, store.bindings(), store.reset, store.observe)
    assert native_unsafe.decision == "MATCHED" and store.synthetic_paid_units == 60
    unsafe_observed = {"synthetic": True, "already_paid_units": store.synthetic_paid_units, "actual_calls": list(store.calls), "no_real_money": True}
    repair = search_repairs(original, max_removals=1, max_candidates=64)
    assert repair.status == "FOUND" and repair.selected_bundle is not None
    selected = repair.selected_bundle
    assert selected.report.decision == "VERIFIED"
    assert selected.report.goal_status["legitimate_refund"] == "REACHABLE"
    native_goal = execute_witness(selected, selected.report.goal_witnesses[0], store.bindings(), store.reset, store.observe)
    assert native_goal.decision == "MATCHED" and store.synthetic_paid_units == 60
    values = {
        "model.json": model.model_dump(mode="json"), "policy.json": policy.model_dump(mode="json"),
        "original-bundle.json": original.model_dump(mode="json"), "repair.json": repair.model_dump(mode="json"),
        "selected-policy.json": selected.policy.model_dump(mode="json"), "selected-bundle.json": selected.model_dump(mode="json"),
        "native-unsafe.json": native_unsafe.model_dump(mode="json"), "native-goal.json": native_goal.model_dump(mode="json"),
        "observed-local-store.json": {"unsafe": unsafe_observed, "goal": {"synthetic": True, "paid_units": store.synthetic_paid_units, "actual_calls": list(store.calls)}},
    }
    for name, value in values.items():
        path = destination / name
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")
        path.chmod(0o600)
    root = Path(__file__).resolve().parents[1]
    for command, expected in (
        (["check", str(destination / "model.json"), str(destination / "policy.json")], 1),
        (["replay", str(destination / "original-bundle.json")], 1),
        (["repair", str(destination / "original-bundle.json"), "--max-removals", "1", "--max-candidates", "64"], 0),
        (["compare", str(destination / "original-bundle.json"), str(destination / "selected-policy.json")], 0),
    ):
        output = destination / ("cli-" + command[0])
        completed = subprocess.run([sys.executable, "-m", "ifixai.cli.main", "policy-lab", *command, "--output", str(output)], cwd=root, check=False)
        if completed.returncode != expected:
            raise RuntimeError(f"Unexpected {command[0]} exit={completed.returncode}")
    print("Three individually permitted operations produced an observed synthetic self-approved payment.")
    print("Permission removal eliminated that model path while preserving the observed independent-approval goal.")
    print(f"Explicit-model and local-observer evidence: {destination}")


if __name__ == "__main__":
    main()
