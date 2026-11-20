"""Fixes from the real-user pass on the easy flow: the mock run's judge, the author hint, the
progress list, the dry run's cost and a profile's problems all in one refusal."""

import io
import json
import logging
import re
import sys
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from ifixai.cli.author import REFERENCE_PROFILE
from ifixai.cli.main import ifixai_cli
from ifixai.cli.orchestrator import BenchmarkProgressDisplay
from ifixai.cli.run import PROVIDER_CHOICES, describe_cost
from ifixai.cli.validate import profile_behind
from ifixai.providers.resolver import credential_env_vars

MOCK_NEEDS_A_JUDGE = (
    "Error: the mock run has no judge. Add --eval-mode self for a free run on the stand-in agent "
    "(it judges itself), or set a judge key such as OPENAI_API_KEY or ANTHROPIC_API_KEY for a "
    "cross-vendor judge."
)
OFFER = "No judge key found. Run the mock free, with the stand-in agent judging itself (--eval-mode self)?"
QUIET = ["--no-promo", "--no-telemetry"]


@pytest.fixture
def no_keys(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("IFIXAI_TELEMETRY", "0")
    for provider in PROVIDER_CHOICES:
        for key in credential_env_vars(provider):
            monkeypatch.delenv(key, raising=False)


def _lines(output: str) -> list[str]:
    return [line.strip() for line in output.splitlines()]


# 1. A mock run with no key and no --eval-mode says the fix that works, never "set OPENAI_API_KEY".
def test_a_mock_run_with_no_judge_says_add_eval_mode_self(no_keys, tmp_path):
    result = CliRunner().invoke(ifixai_cli, ["run", "--provider", "mock", "--fixture", "default", *QUIET])

    assert result.exit_code == 2, result.output
    assert MOCK_NEEDS_A_JUDGE in _lines(result.output)
    assert "Set at least one of" not in result.output
    assert "Testing connection" not in result.output
    assert not (tmp_path / "ifixai-results").exists()


def test_a_mock_run_with_a_judge_key_pairs_it_and_never_self_judges(no_keys, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "owned-synthetic-anthropic-key")

    result = CliRunner().invoke(ifixai_cli, ["run", "--provider", "mock", "--fixture", "default", "--dry-run", *QUIET])

    assert result.exit_code == 0, result.output
    assert "auto-paired judge provider 'anthropic' (SUT='mock')" in result.output
    assert "the mock run has no judge" not in result.output


# 2. Interactive setup on mock: no API key prompt, and with no judge key an offer of the free
# self-judged run, which only an explicit yes takes.
SETUP_MOCK = "n\nmock\nn\nn\n"


def test_interactive_mock_never_asks_a_key_and_a_yes_runs_it_self_judged(no_keys, tmp_path):
    args = ["run", "-b", "B01", *QUIET, "-o", str(tmp_path / "out"), "--reliability-out", str(tmp_path / "runs")]

    result = CliRunner().invoke(ifixai_cli, args, input=f"{SETUP_MOCK}y\n")

    # It ran: B01 fails on the demo agent's planted session elevation.
    assert result.exit_code == 2, result.output
    assert "API key" not in result.output
    assert OFFER in result.output
    assert "Evaluation mode: self (system-under-test 'mock' acts as its own judge" in result.output
    assert "second distinct-provider credential" not in result.output


def test_interactive_mock_no_to_the_offer_says_the_fix(no_keys):
    result = CliRunner().invoke(ifixai_cli, ["run", "--dry-run", *QUIET], input=f"{SETUP_MOCK}n\n")

    assert result.exit_code == 2, result.output
    assert MOCK_NEEDS_A_JUDGE in _lines(result.output)
    assert "second distinct-provider credential" not in result.output
    assert "Dry run" not in result.output


def test_interactive_mock_takes_no_default_on_the_offer(no_keys):
    # Enter alone is no answer: self-judging stays an explicit yes.
    result = CliRunner().invoke(ifixai_cli, ["run", "--dry-run", *QUIET], input=f"{SETUP_MOCK}\nn\n")

    assert result.exit_code == 2, result.output
    assert result.output.count(OFFER) == 2


# 3. The author hint is a command to copy: the profile that wrote the fixture when it is found
# next to it, else a placeholder that says what goes there.
def _short_profile(tmp_path: Path) -> tuple[Path, Path]:
    profile = json.loads(REFERENCE_PROFILE.read_text(encoding="utf-8"))
    profile["escalation_triggers"] = profile["escalation_triggers"][:2]
    profile_path = tmp_path / "pantry.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    fixture = tmp_path / "pantry.yaml"
    authored = CliRunner().invoke(ifixai_cli, ["author", str(profile_path), "-o", str(fixture)])
    assert authored.exit_code == 0, authored.output
    return profile_path, fixture


def test_validate_names_the_profile_that_wrote_the_fixture(tmp_path):
    profile, fixture = _short_profile(tmp_path)

    result = CliRunner().invoke(ifixai_cli, ["validate", str(fixture), "--full-run"])

    assert result.exit_code == 1, result.output
    assert _lines(result.output)[-2:] == [
        f"Fix {profile} and run ifixai author {profile} -o {fixture} --force, or edit {fixture}.",
        f"Then run ifixai validate {fixture} --full-run",
    ]


def test_run_names_the_profile_that_wrote_the_fixture(tmp_path):
    profile, fixture = _short_profile(tmp_path)
    out = ["-o", str(tmp_path / "out"), "--reliability-out", str(tmp_path / "runs")]

    result = CliRunner().invoke(
        ifixai_cli, ["run", "-p", "mock", "--eval-mode", "self", "-f", str(fixture), "--full-run", *QUIET, *out]
    )

    assert result.exit_code == 1, result.output
    assert f"Fix {profile} and run ifixai author {profile} -o {fixture} --force, or edit {fixture}." in _lines(result.output)
    assert not (tmp_path / "out").exists()


def test_a_profile_that_no_longer_writes_the_fixture_is_not_named(tmp_path):
    profile, fixture = _short_profile(tmp_path)
    edited = json.loads(profile.read_text(encoding="utf-8"))
    edited["name"] = "Someone Else"
    profile.write_text(json.dumps(edited), encoding="utf-8")

    result = CliRunner().invoke(ifixai_cli, ["validate", str(fixture), "--full-run"])

    assert result.exit_code == 1, result.output
    assert (
        f"Fix the agent profile and run ifixai author <your profile> -o {fixture} --force, or edit {fixture}."
        in _lines(result.output)
    )
    assert f"author {profile}" not in result.output


def test_the_profile_is_found_whatever_order_the_folder_lists_its_files(tmp_path, monkeypatch):
    profile, fixture = _short_profile(tmp_path)
    # More JSON files than are tried, all listed before the profile in name order.
    for n in range(60):
        (tmp_path / f"a{n:02d}.json").write_text("{}", encoding="utf-8")
    listed = sorted(tmp_path.glob("*.json"))
    assert listed[-1] == profile
    glob = Path.glob
    for order in (listed, listed[::-1]):
        monkeypatch.setattr(
            Path, "glob", lambda self, pattern, order=order: iter(order) if self == tmp_path else glob(self, pattern)
        )
        assert profile_behind(str(fixture)) == str(profile)


# 4. A log line mid-run (the purpose check's) never strands a progress row above the list.
def _screen(raw: str) -> list[str]:
    """What a terminal shows for `raw`: cursor-up, carriage return + clear line, newline."""
    rows, row = [""], 0
    for token in re.split(r"(\x1b\[\d+A|\r\x1b\[K|\n)", raw):
        if move := re.fullmatch(r"\x1b\[(\d+)A", token):
            row = max(0, row - int(move[1]))
        elif token == "\r\x1b[K":
            rows[row] = ""
        elif token == "\n":
            row += 1
            rows.extend([""] * (row + 1 - len(rows)))
        else:
            rows[row] += token
    return [re.sub(r"\x1b\[[0-9;]*m", "", text) for text in rows]


def test_a_log_line_mid_run_never_strands_a_progress_row(monkeypatch):
    terminal = io.StringIO()
    monkeypatch.setattr(sys, "stdout", terminal)
    monkeypatch.setattr(sys, "stderr", terminal)
    root = logging.getLogger()
    # The console handler the CLI installs at startup, bound to the stream it had then.
    handler = logging.StreamHandler(sys.stderr)
    root.addHandler(handler)
    level = root.level
    root.setLevel(logging.WARNING)
    tests = [("B01", "Tool Invocation Governance"), ("B02", "Non-LLM Governance Layer"), ("B03", "Auditability")]
    try:
        display = BenchmarkProgressDisplay(tests, [])
        display.start()
        for _ in range(2):
            logging.getLogger("ifixai.evaluation.pipeline").warning("Purpose check answered off-format")
        display.stop()
    finally:
        root.removeHandler(handler)
        root.setLevel(level)

    screen = _screen(terminal.getvalue())
    for test_id, _ in tests:
        assert sum(test_id in row for row in screen) == 1, screen
    assert screen.count("Purpose check answered off-format") == 2, screen


# 5. The dry run says what the run costs: nothing for the offline mock, and who pays otherwise,
# never a judge-call count passed off as the cost.
def _cost(output: str) -> str:
    [line] = [text for text in _lines(output) if text.startswith("Cost:")]
    return " ".join(line.split())


def test_dry_run_of_the_self_judged_mock_says_it_costs_nothing(no_keys):
    result = CliRunner().invoke(ifixai_cli, ["run", "-p", "mock", "--eval-mode", "self", "-f", "default", "--dry-run", *QUIET])

    assert result.exit_code == 0, result.output
    assert _cost(result.output) == "Cost: nothing: the mock agent and its judge run offline, so no call is billed."
    assert "real cost depends" not in result.output


@pytest.mark.parametrize(
    ("agent", "cost"),
    [
        (
            ["-p", "mock"],
            "Cost: not priced here (iFixAi has no price list). The judge calls bill the openai key's account; "
            "the mock agent is free.",
        ),
        (
            ["-p", "http", "-e", "http://127.0.0.1:9/v1", "--auth-method", "none"],
            "Cost: not priced here (iFixAi has no price list). The judge calls bill the openai key's account; "
            "the probes bill your agent's own infrastructure.",
        ),
        (
            ["-p", "anthropic"],
            "Cost: not priced here (iFixAi has no price list). The judge calls bill the openai key's account; "
            "the probes bill the anthropic key's account.",
        ),
    ],
    ids=["mock", "http", "model"],
)
def test_dry_run_with_a_paid_judge_names_who_pays(no_keys, monkeypatch, agent, cost):
    monkeypatch.setenv("OPENAI_API_KEY", "owned-synthetic-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "owned-synthetic-anthropic-key")
    args = ["run", *agent, "--judge-provider", "openai", "--eval-mode", "single", "-f", "default", "--dry-run", *QUIET]

    result = CliRunner().invoke(ifixai_cli, args)

    assert result.exit_code == 0, result.output
    assert _cost(result.output) == cost
    assert "Estimated judge calls:" in result.output


@pytest.mark.parametrize(
    ("provider", "eval_mode", "judges", "cost"),
    [
        ("http", "self", (), "The judge calls and the probes bill your agent's own infrastructure."),
        ("openai", "deterministic", (), "The probes bill the openai key's account."),
        ("mock", "full", ("anthropic", "openai"), "The judge calls bill the anthropic and openai keys' accounts; the mock agent is free."),
    ],
)
def test_the_cost_names_each_payer(provider, eval_mode, judges, cost):
    assert describe_cost(provider, eval_mode, judges) == f"not priced here (iFixAi has no price list). {cost}"


# 6. A bad profile gets every problem in its first refusal, the cross-checks included.
def _author(tmp_path: Path, profile: dict) -> tuple[int, list[str], Path]:
    path = tmp_path / "pantry.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    out = tmp_path / "pantry.yaml"
    result = CliRunner().invoke(ifixai_cli, ["author", str(path), "-o", str(out)])
    return result.exit_code, result.output.splitlines(), out


def _profile() -> dict:
    return json.loads(REFERENCE_PROFILE.read_text(encoding="utf-8"))


def test_every_problem_comes_in_the_first_refusal(tmp_path):
    profile = _profile()
    profile["tools"][2]["risk_level"] = "hgh"
    profile["saftey_rules"] = profile.pop("safety_rules")
    profile["roles"][1]["allowed_tools"].append("refnd_item")
    profile["data_sources"][0]["accessible_by_roles"] = ["auditor"]

    code, lines, out = _author(tmp_path, profile)

    assert code == 2, lines
    assert lines == [
        "Profile refused. Fix these, then run ifixai author again:",
        "  tool 3, risk_level: Input should be 'low', 'medium', 'high' or 'critical'",
        "  saftey_rules: isn't a profile field; did you mean safety_rules?",
        "  profile: role customer is allowed refnd_item, which isn't in tools",
        "  profile: data source card_processor: accessible_by_roles names auditor, which isn't in roles",
    ]
    assert not out.exists()


def test_every_role_with_an_unknown_tool_is_named_once(tmp_path):
    profile = _profile()
    profile["roles"][0]["allowed_tools"].append("refnd_item")
    profile["roles"][2]["allowed_tools"].append("wipe_ledger")

    code, lines, _ = _author(tmp_path, profile)

    assert code == 2, lines
    assert lines[1:] == [
        "  profile: role support_lead is allowed refnd_item, which isn't in tools",
        "  profile: role support_agent is allowed wipe_ledger, which isn't in tools",
    ]


def test_a_repeated_tool_is_named_beside_a_bad_tool(tmp_path):
    profile = _profile()
    profile["tools"].append(dict(profile["tools"][0]))
    profile["tools"][1]["category"] = "reed"

    code, lines, _ = _author(tmp_path, profile)

    assert code == 2, lines
    assert "  tools: tool_id block_card is listed more than once" in lines
    assert sum("tool 2, category" in line for line in lines) == 1


@pytest.mark.parametrize("broken", ["tools", "roles"])
def test_a_list_that_is_not_a_list_adds_no_made_up_cross_check(tmp_path, broken):
    # With no tools (or roles) to check against, every grant would look unknown: say nothing.
    code, lines, _ = _author(tmp_path, {**_profile(), broken: "lots of them"})

    assert code == 2, lines
    assert lines[1:] == [f"  {broken}: Input should be a valid list"]


def test_a_good_profile_still_authors(tmp_path):
    code, lines, out = _author(tmp_path, _profile())

    assert code == 0, lines
    assert yaml.safe_load(out.read_text(encoding="utf-8"))["metadata"]["name"] == _profile()["name"]
