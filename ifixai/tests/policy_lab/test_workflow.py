"""Real CLI, shipped fixture projection and private report boundary checks."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from ifixai.cli.policy_lab import policy_lab
from ifixai.core.fixture_loader import load_fixture
from ifixai.policy_lab import create_exploration_bundle, load_exploration_bundle
from ifixai.policy_lab.fixture import policy_from_fixture
from ifixai.policy_lab.models import ActionContract, Actor
from ifixai.reporting.policy_lab import render_policy_lab_html, write_policy_lab_reports


def test_check_replay_repair_compare_cli_real_files(native_example, tmp_path):
    runner = CliRunner()
    model, policy = native_example.demo_model(), native_example.demo_policy()
    model_file, policy_file = tmp_path / "model.json", tmp_path / "policy.json"
    model_file.write_text(model.model_dump_json())
    policy_file.write_text(policy.model_dump_json())
    original = tmp_path / "check"
    checked = runner.invoke(policy_lab, ["check", str(model_file), str(policy_file), "--output", str(original)])
    assert checked.exit_code == 1, checked.output
    bundle = load_exploration_bundle(original / "bundle.json")
    assert bundle.report.decision == "COUNTEREXAMPLE"
    replayed = runner.invoke(policy_lab, ["replay", str(original / "bundle.json"), "--output", str(tmp_path / "replay")])
    assert replayed.exit_code == 1, replayed.output
    assert load_exploration_bundle(tmp_path / "replay" / "bundle.json") == bundle
    repaired = runner.invoke(policy_lab, ["repair", str(original / "bundle.json"), "--output", str(tmp_path / "repair")])
    assert repaired.exit_code == 0, repaired.output
    selected = load_exploration_bundle(tmp_path / "repair" / "selected-bundle.json")
    assert selected.report.decision == "VERIFIED"
    candidate_file = tmp_path / "candidate.json"
    candidate_file.write_text(selected.policy.model_dump_json())
    compared = runner.invoke(policy_lab, ["compare", str(original / "bundle.json"), str(candidate_file), "--output", str(tmp_path / "compare")])
    assert compared.exit_code == 0, compared.output
    comparison = json.loads((tmp_path / "compare" / "comparison.json").read_text())
    assert comparison["original_decision"] == "COUNTEREXAMPLE" and comparison["candidate_decision"] == "VERIFIED"
    assert comparison["goals"]["legitimate_refund"] == {"before": "REACHABLE", "after": "REACHABLE"}
    assert load_exploration_bundle(original / "bundle.json") == bundle
    for output in (original, tmp_path / "repair", tmp_path / "compare"):
        assert output.stat().st_mode & 0o777 == 0o700
        assert all(path.stat().st_mode & 0o777 == 0o600 for path in output.iterdir())
    duplicate = runner.invoke(policy_lab, ["replay", str(original / "bundle.json"), "--output", str(original)])
    assert duplicate.exit_code == 1 and "FileExistsError" in duplicate.output


def fixture_model(native_example):
    fixture_path = Path(__file__).resolve().parents[2] / "fixtures" / "examples" / "customer_support.yaml"
    fixture = load_fixture(fixture_path)
    original = native_example.demo_model()
    actions = tuple(ActionContract.model_validate({**action.model_dump(mode="python"),
        "actor_id": "agent_jordan" if action.actor_id == "agent" else "sup_taylor",
        "tool": {"request_refund": "lookup_account", "approve_refund": "modify_contact", "pay_refund": "issue_refund"}[action.tool]}) for action in original.actions)
    model = original.model_copy(update={"actors": {user.user_id: Actor(roles=tuple(user.roles)) for user in fixture.users}, "actions": actions})
    return fixture_path, fixture, model


def test_actual_customer_support_bridge_and_cli(native_example, tmp_path):
    fixture_path, fixture, model = fixture_model(native_example)
    policy = policy_from_fixture(fixture, model)
    assert policy.tenant_id == model.tenant_id
    assert all(rule.resource_scope == "*" for rule in policy.permissions)
    assert all(rule.tool in {action.tool for action in model.actions} for rule in policy.permissions)
    assert not any(rule.role == "agent" and rule.tool == "issue_refund" for rule in policy.permissions)
    assert create_exploration_bundle(model, policy).report.decision == "COUNTEREXAMPLE"
    model_file = tmp_path / "model.json"
    model_file.write_text(model.model_dump_json())
    output = tmp_path / "fixture"
    result = CliRunner().invoke(policy_lab, ["from-fixture", str(model_file), str(fixture_path), "--output", str(output)])
    assert result.exit_code == 1, result.output
    assert load_exploration_bundle(output / "bundle.json").policy == policy
    metadata = json.loads((output / "metadata.json").read_text())
    assert "resource scope is wildcard" in metadata["boundary"]


@pytest.mark.parametrize("mutation", ["actor", "role", "tool"])
def test_fixture_bridge_rejects_unknown_or_invented_identity(native_example, mutation):
    _, fixture, model = fixture_model(native_example)
    if mutation == "actor":
        model = model.model_copy(update={"actors": {**model.actors, "fictional": Actor(roles=("supervisor",))}})
    elif mutation == "role":
        model = model.model_copy(update={"actors": {**model.actors, "agent_jordan": Actor(roles=("supervisor",))}})
    else:
        action = model.actions[0].model_copy(update={"tool": "python:os.system"})
        model = model.model_copy(update={"actions": (action, *model.actions[1:])})
    with pytest.raises(ValueError):
        policy_from_fixture(fixture, model)


def test_html_escapes_untrusted_label_and_existing_destination_refused(native_example, tmp_path):
    model = native_example.demo_model()
    prop = model.properties[0].model_copy(update={"label": '<script>alert("model")</script>'})
    bundle = create_exploration_bundle(model.model_copy(update={"properties": (prop,)}), native_example.demo_policy())
    rendered = render_policy_lab_html(bundle)
    assert '<script>alert("model")</script>' not in rendered
    assert "&lt;script&gt;" in rendered
    destination = tmp_path / "private"
    write_policy_lab_reports(bundle, destination)
    existing = (destination / "bundle.json").read_bytes()
    with pytest.raises(FileExistsError):
        write_policy_lab_reports(bundle, destination)
    assert (destination / "bundle.json").read_bytes() == existing


@pytest.mark.parametrize("raw", ['{"name":"x","name":"x"}', '{"amount":NaN}'])
def test_cli_strict_json_cannot_import_code(tmp_path, raw):
    input_file = tmp_path / "model.json"
    input_file.write_text(raw)
    result = CliRunner().invoke(policy_lab, ["check", str(input_file), str(input_file), "--output", str(tmp_path / "out")])
    assert result.exit_code == 1 and not (tmp_path / "out").exists()


@pytest.mark.parametrize("mutation", ["duplicate_permission", "unknown_role", "unknown_tool", "governance_conflict"])
def test_fixture_authorization_contradictions_rejected(native_example, mutation):
    _, fixture, model = fixture_model(native_example)
    if mutation == "governance_conflict":
        fixture = fixture.model_copy(update={"governance": SimpleNamespace(policies=SimpleNamespace(authorization=[]))})
    else:
        rule = fixture.permissions[0]
        if mutation == "duplicate_permission":
            permissions = [*fixture.permissions, rule]
        elif mutation == "unknown_role":
            permissions = [rule.model_copy(update={"role": "invented"}), *fixture.permissions[1:]]
        else:
            permissions = [rule.model_copy(update={"tools": ["invented"]}), *fixture.permissions[1:]]
        fixture = fixture.model_copy(update={"permissions": permissions})
    with pytest.raises(ValueError):
        policy_from_fixture(fixture, model)


def test_fixture_projection_revalidates_strict_copied_effects(native_example):
    _, fixture, model = fixture_model(native_example)
    bad_action = model.actions[0].model_copy(update={"cost_units": True})
    model = model.model_copy(update={"actions": (bad_action, *model.actions[1:])})
    with pytest.raises(ValueError):
        policy_from_fixture(fixture, model)
