"""`ifixai author` turns an agent profile into a fixture, and refuses a bad profile by name."""

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from ifixai.cli.author import REFERENCE_PROFILE
from ifixai.cli.main import ifixai_cli

REFERENCE_FIXTURE = REFERENCE_PROFILE.with_name("fixture.yaml")


def _author(tmp_path: Path, profile: object) -> tuple[int, str, Path]:
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    out = tmp_path / "fixture.yaml"
    result = CliRunner().invoke(ifixai_cli, ["author", str(path), "-o", str(out)])
    return result.exit_code, result.output, out


def _profile() -> dict:
    return json.loads(REFERENCE_PROFILE.read_text(encoding="utf-8"))


@pytest.mark.parametrize("source", ["file", "reference"])
def test_reference_profile_authors_the_reference_fixture(tmp_path, source):
    out = tmp_path / "fixture.yaml"
    arg = str(REFERENCE_PROFILE) if source == "file" else "reference"
    result = CliRunner().invoke(ifixai_cli, ["author", arg, "-o", str(out)])

    assert result.exit_code == 0, result.output
    assert out.read_text(encoding="utf-8") == REFERENCE_FIXTURE.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("change", "line"),
    [
        ({"favourite_colour": "blue"}, "favourite_colour: isn't a profile field, remove it"),
        ({"roles": "lots of them"}, "roles: Input should be a valid list"),
        ({"domain": "Fintech Support"}, "domain: String should match pattern '^[a-z0-9_]*$'"),
        ({"regulations": [{"name": "GDPR", "required_tests": ["B99"]}]}, "regulation 1, required_tests: B99 isn't a B inspection id (like B05)"),
        ({"safety_rule": ["Never share a card number."]}, "safety_rule: isn't a profile field; did you mean safety_rules?"),
    ],
)
def test_bad_profile_is_refused_naming_the_field(tmp_path, change, line):
    code, output, out = _author(tmp_path, {**_profile(), **change})

    assert code == 2, output
    assert line in output
    assert "Traceback" not in output
    assert not out.exists()


def _with_third_tool(**changes: str) -> dict:
    profile = _profile()
    profile["tools"][2] = {**profile["tools"][2], **changes}
    return profile


@pytest.mark.parametrize(
    ("profile", "line"),
    [
        (_with_third_tool(risk_level="severe"), "tool 3, risk_level: Input should be 'low', 'medium', 'high' or 'critical'"),
        (_with_third_tool(risklevel="high"), "tool 3, risklevel: isn't a tool field; did you mean risk_level?"),
        (_with_third_tool(colour="blue"), "tool 3, colour: isn't a tool field, remove it"),
    ],
    ids=["bad-value", "close-field", "unknown-field"],
)
def test_a_field_in_a_list_is_named_by_its_item_counted_from_one(tmp_path, profile, line):
    code, output, out = _author(tmp_path, profile)

    assert code == 2, output
    assert f"  {line}" in output.splitlines()
    assert "tools.2" not in output
    assert not out.exists()


def test_example_writes_the_reference_profile_to_edit_and_never_over_a_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reference = REFERENCE_PROFILE.read_text(encoding="utf-8")

    wrote = CliRunner().invoke(ifixai_cli, ["author", "--example"])
    (tmp_path / "profile.json").write_text("mine\n", encoding="utf-8")
    kept = CliRunner().invoke(ifixai_cli, ["author", "--example"])
    kept_text = (tmp_path / "profile.json").read_text(encoding="utf-8")
    forced = CliRunner().invoke(ifixai_cli, ["author", "--example", "--force"])
    authored = CliRunner().invoke(ifixai_cli, ["author", "profile.json"])

    assert wrote.exit_code == 0, wrote.output
    assert "Wrote profile.json, an example to edit. Next: ifixai author profile.json -o fixture.yaml" in wrote.output
    assert kept.exit_code == 1, kept.output
    assert "Error: profile.json already exists. Pass --force to replace it." in kept.output
    assert kept_text == "mine\n"
    assert forced.exit_code == 0, forced.output
    assert (tmp_path / "profile.json").read_text(encoding="utf-8") == reference
    assert authored.exit_code == 0, authored.output
    assert (tmp_path / "fixture.yaml").read_text(encoding="utf-8") == REFERENCE_FIXTURE.read_text(encoding="utf-8")


def test_example_writes_where_o_says_and_never_over_a_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reference = REFERENCE_PROFILE.read_text(encoding="utf-8")

    wrote = CliRunner().invoke(ifixai_cli, ["author", "--example", "-o", "mine.json"])
    (tmp_path / "mine.json").write_text("mine\n", encoding="utf-8")
    kept = CliRunner().invoke(ifixai_cli, ["author", "--example", "-o", "mine.json"])
    kept_text = (tmp_path / "mine.json").read_text(encoding="utf-8")
    forced = CliRunner().invoke(ifixai_cli, ["author", "--example", "-o", "mine.json", "--force"])

    assert wrote.exit_code == 0, wrote.output
    assert "Wrote mine.json, an example to edit. Next: ifixai author mine.json -o fixture.yaml" in wrote.output
    assert kept.exit_code == 1, kept.output
    assert "Error: mine.json already exists. Pass --force to replace it." in kept.output
    assert kept_text == "mine\n"
    assert forced.exit_code == 0, forced.output
    assert (tmp_path / "mine.json").read_text(encoding="utf-8") == reference
    assert sorted(path.name for path in tmp_path.iterdir()) == ["mine.json"]



@pytest.mark.parametrize(
    ("target", "fixture"),
    [("fixture.yaml", "my-fixture.yaml"), ("./fixture.yaml", "my-fixture.yaml"), ("mine.json", "fixture.yaml")],
)
def test_the_next_command_after_example_never_writes_the_fixture_over_the_profile(tmp_path, monkeypatch, target, fixture):
    monkeypatch.chdir(tmp_path)

    wrote = CliRunner().invoke(ifixai_cli, ["author", "--example", "-o", target])
    line = next(text for text in wrote.output.splitlines() if "Next: " in text)
    nxt = CliRunner().invoke(ifixai_cli, line.split("Next: ifixai ", 1)[1].split())

    assert wrote.exit_code == 0, wrote.output
    assert line.endswith(f"Next: ifixai author {target} -o {fixture}")
    assert nxt.exit_code == 0, nxt.output
    assert (tmp_path / target).read_text(encoding="utf-8") == REFERENCE_PROFILE.read_text(encoding="utf-8")
    assert (tmp_path / fixture).read_text(encoding="utf-8") == REFERENCE_FIXTURE.read_text(encoding="utf-8")


def test_author_with_no_profile_points_at_example(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(ifixai_cli, ["author"])

    assert result.exit_code == 2, result.output
    assert "Missing argument 'PROFILE'. Pass --example to get a profile to edit." in result.output
    assert list(tmp_path.iterdir()) == []


def test_grant_of_a_tool_the_profile_lacks_is_refused(tmp_path):
    profile = _profile()
    profile["roles"][1]["allowed_tools"].append("wire_money")

    code, output, out = _author(tmp_path, profile)

    assert code == 2, output
    assert "profile: role customer is allowed wire_money, which isn't in tools" in output
    assert not out.exists()


@pytest.mark.parametrize("broken", [{"nme": "auditor"}, {"name": 7}, "auditor"])
def test_an_unreadable_role_never_hides_another_roles_bad_grant(tmp_path, broken):
    profile = _profile()
    profile["roles"][1]["allowed_tools"].append("wire_money")
    profile["roles"].append(broken)

    code, output, out = _author(tmp_path, profile)

    # One refusal names both: the broken role and the readable role's grant.
    assert code == 2, output
    assert "role 4" in output
    assert "profile: role customer is allowed wire_money, which isn't in tools" in output
    assert not out.exists()


def test_missing_profile_file_is_refused(tmp_path):
    result = CliRunner().invoke(ifixai_cli, ["author", str(tmp_path / "nope.json"), "-o", str(tmp_path / "f.yaml")])

    assert result.exit_code == 2
    assert "nope.json: no such file" in result.output


@pytest.mark.parametrize(
    ("name", "content", "line"),
    [
        ("adir", None, "adir: can't read it (Is a directory)"),
        ("list.json", "[1, 2]", "list.json: should be one JSON object with the profile's fields"),
        ("bad.json", "{name: 1}", "bad.json: not valid JSON (Expecting property name enclosed in double quotes"),
    ],
)
def test_unusable_profile_file_is_refused_in_plain_words(tmp_path, monkeypatch, name, content, line):
    monkeypatch.chdir(tmp_path)
    if content is None:
        (tmp_path / name).mkdir()
    else:
        (tmp_path / name).write_text(content, encoding="utf-8")

    result = CliRunner().invoke(ifixai_cli, ["author", name, "-o", "f.yaml"])

    assert result.exit_code == 2, result.output
    assert line in result.output
    assert "Traceback" not in result.output
    assert not (tmp_path / "f.yaml").exists()


def test_unwritable_output_is_a_plain_error(tmp_path):
    out = tmp_path / "missing-dir" / "f.yaml"

    result = CliRunner().invoke(ifixai_cli, ["author", str(REFERENCE_PROFILE), "-o", str(out)])

    assert result.exit_code == 1, result.output
    assert f"Error: can't write {out} (No such file or directory)" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_no_roles_gives_one_role_with_every_tool_and_ten_users(tmp_path):
    sources = [{**source, "accessible_by_roles": ["user"]} for source in _profile()["data_sources"]]
    profile = {**_profile(), "roles": [], "data_sources": sources}

    code, output, out = _author(tmp_path, profile)

    fixture = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert code == 0, output
    assert fixture["roles"] == [{"name": "user", "description": "The people who use the agent."}]
    assert fixture["permissions"] == [{"role": "user", "tools": sorted(t["tool_id"] for t in profile["tools"])}]
    assert [u["user_id"] for u in fixture["users"]] == [f"u{n:03d}" for n in range(1, 11)]
    assert fixture["policies"]["override_allowed_roles"] == ["user"]


def test_users_spread_over_roles_with_the_approver_first(tmp_path):
    code, output, out = _author(tmp_path, _profile())

    fixture = yaml.safe_load(out.read_text(encoding="utf-8"))
    roles = [u["roles"][0] for u in fixture["users"]]
    assert code == 0, output
    # support_lead holds the most tools, so it comes first; the rest by name.
    assert [r["name"] for r in fixture["roles"]] == ["support_lead", "customer", "support_agent"]
    assert roles == ["support_lead"] * 4 + ["customer"] * 3 + ["support_agent"] * 3
    assert "governance" not in fixture


def _source(profile: dict, source_id: str) -> dict:
    return next(source for source in profile["data_sources"] if source["source_id"] == source_id)


def _with_reader(role: str) -> dict:
    profile = _profile()
    _source(profile, "fraud_signals")["accessible_by_roles"] = ["support_lead", role]
    return profile


def _with_repeat(field: str) -> dict:
    profile = _profile()
    profile[field].append(dict(profile[field][0]))
    return profile


@pytest.mark.parametrize(
    ("profile", "line"),
    [
        (
            _with_reader("auditor"),
            "profile: data source fraud_signals: accessible_by_roles names auditor, which isn't in roles",
        ),
        (_with_repeat("tools"), "tools: tool_id block_card is listed more than once"),
        (_with_repeat("roles"), "roles: role support_lead is listed more than once"),
    ],
    ids=["unknown-reader", "repeated-tool", "repeated-role"],
)
def test_names_that_do_not_add_up_are_refused_naming_the_field(tmp_path, profile, line):
    code, output, out = _author(tmp_path, profile)

    assert code == 2, output
    assert line in output
    assert "Traceback" not in output
    assert not out.exists()


def test_an_existing_output_is_kept_unless_forced(tmp_path):
    out = tmp_path / "fixture.yaml"
    out.write_text("mine\n", encoding="utf-8")
    args = ["author", str(REFERENCE_PROFILE), "-o", str(out)]

    kept = CliRunner().invoke(ifixai_cli, args)
    forced = CliRunner().invoke(ifixai_cli, [*args, "--force"])

    assert kept.exit_code == 1, kept.output
    assert f"Error: {out} already exists. Pass --force to replace it." in kept.output
    assert forced.exit_code == 0, forced.output
    assert out.read_text(encoding="utf-8") == REFERENCE_FIXTURE.read_text(encoding="utf-8")


def _every_tool_but(missing: int) -> dict:
    """The reference profile with every role allowed every tool, except customer, which lacks
    its first `missing` tools."""
    profile = _profile()
    tools = [tool["tool_id"] for tool in profile["tools"]]
    for role in profile["roles"]:
        role["allowed_tools"] = tools[missing:] if role["name"] == "customer" else tools
    return profile


def _validate(fixture: Path) -> tuple[int, list[str]]:
    # An authored fixture must clear every inspection's minimums, not only a bare run's.
    result = CliRunner().invoke(ifixai_cli, ["validate", str(fixture), "--full-run"])
    return result.exit_code, [line.strip() for line in result.output.splitlines()]


def test_a_restricted_role_gets_enough_users_for_b08(tmp_path):
    # Customer lacks one tool and gets 3 of the 10 users: 3 requests for a tool the role may not
    # use, and B08 needs 4, so author gives customer a 4th user.
    code, output, out = _author(tmp_path, _every_tool_but(1))

    fixture = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert code == 0, output
    assert [user["roles"][0] for user in fixture["users"]] == ["support_agent"] * 4 + ["customer"] * 4 + ["support_lead"] * 3
    assert [user["user_id"] for user in fixture["users"]] == [f"u{n:03d}" for n in range(1, 12)]
    # Customer lacks only block_card, no tool that destroys, so no role is refused one and P01 has
    # nothing to ask.
    assert _validate(out) == (0, [
        "Valid",
        "Note: P01 won't run: every role in the fixture may use each tool that can delete or "
        "overwrite, so no request for one crosses a role's limits.",
    ])


def test_a_role_already_lacking_enough_tools_keeps_ten_users(tmp_path):
    code, output, out = _author(tmp_path, _every_tool_but(2))

    fixture = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert code == 0, output
    assert [user["roles"][0] for user in fixture["users"]] == ["support_agent"] * 4 + ["customer"] * 3 + ["support_lead"] * 3


def test_a_profile_where_every_role_has_every_tool_is_told_to_give_a_role_fewer(tmp_path):
    code, output, out = _author(tmp_path, _every_tool_but(0))

    checked, lines = _validate(out)
    assert code == 0, output
    assert len(yaml.safe_load(out.read_text(encoding="utf-8"))["users"]) == 10
    assert checked == 1, lines
    assert (
        "Add a role that can't use at least one tool, or give a role fewer tools: "
        "B08 needs 4 requests for a tool the role may not use, and finds 0."
    ) in lines


def test_every_role_gets_a_test_user_past_ten_roles(tmp_path):
    profile = _profile()
    tools = [tool["tool_id"] for tool in profile["tools"]]
    profile["roles"] = [{"name": f"role_{n:02d}", "allowed_tools": tools[: n + 1]} for n in range(12)]

    code, output, out = _author(tmp_path, {**profile, "data_sources": []})

    fixture = yaml.safe_load(out.read_text(encoding="utf-8"))
    names = [user["name"] for user in fixture["users"]]
    assert code == 0, output
    assert len(fixture["users"]) == 12
    assert {user["roles"][0] for user in fixture["users"]} == {f"role_{n:02d}" for n in range(12)}
    assert names[:10] == [
        "Sam Patel", "Alex Morgan", "Priya Shah", "Jordan Lee", "Maria Costa",
        "Tom Becker", "Aisha Khan", "Daniel Novak", "Grace Okafor", "Leo Martin",
    ]
    assert names[10:] == ["Sam Morgan", "Alex Shah"]


# The guides tell the profile writer to say what the agent does in `abilities`: author carries it
# into the fixture, so the guided flow can set a check aside, and leaves the fixture alone without it.
def test_abilities_in_the_profile_reach_the_fixture(tmp_path):
    code, output, out = _author(tmp_path, {**_profile(), "abilities": {"trains_models": False, "serves_consumers": True}})

    assert code == 0, output
    assert yaml.safe_load(out.read_text(encoding="utf-8"))["abilities"] == {"trains_models": False, "serves_consumers": True}


def test_a_profile_without_abilities_writes_none(tmp_path):
    code, output, out = _author(tmp_path, _profile())

    assert code == 0, output
    assert "abilities" not in yaml.safe_load(out.read_text(encoding="utf-8"))


def test_a_misspelt_ability_is_refused_naming_it(tmp_path):
    code, output, out = _author(tmp_path, {**_profile(), "abilities": {"trains_model": False}})

    assert code == 2, output
    assert "trains_model" in output and "Traceback" not in output
    assert not out.exists()
