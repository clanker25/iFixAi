"""Any agent but the mock runs without a bundled fixture's simulated governance, and the run says
so in one line; the retired `--governance` never reaches the run; library callers keep the
fixture's own block; a real agent (http, langchain) is warned off fixture grounding."""

import http.server
import json
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from ifixai.api import run_selected, run_single
from ifixai.cli.main import ifixai_cli
from ifixai.cli.run import BUNDLED_GOVERNANCE_SKIPPED, EMPTY_GOVERNANCE_SKIPPED
from ifixai.core.fixture_loader import load_fixture
from ifixai.core.types import EvaluationMode, EvaluationPipelineConfig, TestResult
from ifixai.judge.config import JudgeConfig
from ifixai.providers.governance_fixture import GovernanceFixture
from ifixai.shared.evidence import DECLARED_KEY

DEFAULT_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "default" / "fixture.yaml"
EXAMPLES = DEFAULT_FIXTURE.parents[1] / "examples"
# Checks whose hooks a composed `governance:` block answers (as notes the report marks declared).
STRUCTURAL = ["P08", "M06", "M07"]
JUDGED = {"judge_config": JudgeConfig(provider="mock"), "pipeline_config": EvaluationPipelineConfig(mode=EvaluationMode.SINGLE)}


class _Agent(http.server.BaseHTTPRequestHandler):
    """A plain chat agent: no tools, no audit trail. Also answers as a LangServe app (POST .../invoke)."""

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers["Content-Length"]))
        reply = "I can't do that."
        answer = {"output": reply} if self.path.endswith("/invoke") else {
            "choices": [{"finish_reason": "stop", "message": {"content": reply}}]}
        body = json.dumps(answer).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        pass


@pytest.fixture
def agent() -> Iterator[str]:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Agent)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(2)


class _Run:
    def __init__(self, output: str, results: dict[str, dict], governance_source: str) -> None:
        self.output = output
        self.results = results
        self.governance_source = governance_source

    def declared(self) -> int:
        """Evidence items the fixture's `governance:` block answered."""
        return sum(
            1 for t in self.results.values() for e in t["evidence"] if (e.get("details") or {}).get(DECLARED_KEY)
        )


def _run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, endpoint: str, *extra: str, provider: str = "http"
) -> _Run:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("IFIXAI_TELEMETRY", "0")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    out = tmp_path / "reports"
    args = [
        "run", "--provider", provider, "--endpoint", endpoint, "--auth-method", "none",
        "--eval-mode", "single", "--judge-provider", "mock", "--no-telemetry", "--no-parallel",
        "--format", "json", "--output", str(out), "--reliability-out", str(tmp_path / "reliability"),
        *[a for tid in STRUCTURAL for a in ("--test", tid)], *extra,
    ]
    result = CliRunner().invoke(ifixai_cli, args)
    reports = list(out.glob("*.json"))
    assert len(reports) == 1, result.output
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    [manifest] = (tmp_path / "reliability").glob("*/manifest.json")
    return _Run(
        result.output,
        {t["test_id"]: t for t in report["test_results"]},
        json.loads(manifest.read_text(encoding="utf-8"))["governance_source"],
    )


def _declared(result: TestResult) -> int:
    return sum(1 for e in result.evidence if e.details.get(DECLARED_KEY))


@pytest.mark.parametrize(("provider", "extra"), [
    ("http", ()), ("http", ("-f", "default")), ("langchain", ("--api-key", "k")),
    ("openai", ("--api-key", "k", "--model", "agent")),
    ("http", ("-f", "customer_support")), ("http", ("-f", str(EXAMPLES / "customer_support.yaml"))),
])
def test_real_endpoint_on_a_bundled_fixture_gets_no_simulated_governance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent: str, provider: str, extra: tuple[str, ...]
) -> None:
    run = _run(tmp_path, monkeypatch, agent, *extra, provider=provider)
    assert run.results, run.output
    assert run.declared() == 0, run.output
    assert run.governance_source == "runtime"
    assert run.output.splitlines().count(BUNDLED_GOVERNANCE_SKIPPED) == 1, run.output
    assert "composed from fixture" not in run.output


def test_real_endpoint_on_own_fixture_keeps_its_governance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent: str
) -> None:
    # The same block, copied out of the package, is the user's own declared policy.
    own = tmp_path / "own.yaml"
    own.write_text((EXAMPLES / "customer_support.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    run = _run(tmp_path, monkeypatch, agent, "-f", str(own))
    assert run.declared() > 0, run.output
    assert run.governance_source == "explicit_fixture"
    assert "Governance: composed from fixture (source=explicit" in run.output
    assert BUNDLED_GOVERNANCE_SKIPPED not in run.output


def test_the_retired_governance_flag_never_puts_rules_on_a_real_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent: str
) -> None:
    governance = yaml.safe_load(DEFAULT_FIXTURE.read_text(encoding="utf-8"))["governance"]
    governance["version"] = "flag-policy-v7"
    policy = tmp_path / "policy.yaml"
    policy.write_text(yaml.safe_dump(governance), encoding="utf-8")
    run = _run(tmp_path, monkeypatch, agent, "--governance", str(policy))
    assert run.declared() == 0, run.output
    assert run.governance_source == "runtime"
    assert "flag-policy-v7" not in run.output
    assert run.output.count(f"Ignoring the governance file {policy}.") == 1
    assert run.output.splitlines().count(BUNDLED_GOVERNANCE_SKIPPED) == 1, run.output


async def test_library_caller_keeps_the_fixture_block(agent: str) -> None:
    run = await run_selected(
        set(STRUCTURAL), provider="http", endpoint=agent, auth_method="none", fixture="default", **JUDGED
    )
    assert run.test_results
    assert all(_declared(t) > 0 for t in run.test_results)


@pytest.mark.parametrize(("decision", "composed"), [({}, True), ({"governance": None}, False)])
async def test_run_single_takes_the_governance_decision(agent: str, decision: dict, composed: bool) -> None:
    result = await run_single(
        "P08", provider="http", endpoint=agent, auth_method="none", fixture="default", **JUDGED, **decision
    )
    assert (_declared(result) > 0) is composed


@pytest.mark.parametrize(
    "content",
    [
        "hello: world\n",
        "",
        "version: [unclosed\n  - : :\n",
        "seed_audit_records: oops\n",
        "tools:\n  - name: lookup\n",
    ],
)
def test_an_unusable_governance_file_is_ignored_like_any_other(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent: str, content: str
) -> None:
    # The flag is retired: the file is never read, so a broken one neither stops the run nor
    # reaches it.
    policy = tmp_path / "policy.yaml"
    policy.write_text(content, encoding="utf-8")
    run = _run(tmp_path, monkeypatch, agent, "--governance", str(policy))
    assert not [line for line in run.output.splitlines() if line.startswith("Error:")], run.output
    assert run.output.count(f"Ignoring the governance file {policy}.") == 1
    assert run.declared() == 0


@pytest.fixture
def empty_block_fixture(tmp_path: Path) -> Path:
    """The default fixture with a `governance:` block that declares nothing to simulate."""
    data = yaml.safe_load(DEFAULT_FIXTURE.read_text(encoding="utf-8"))
    data["governance"] = {"version": "1.0.0"}
    path = tmp_path / "empty-governance.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.mark.parametrize(("provider", "extra"), [("http", ()), ("langchain", ("--api-key", "k"))])
def test_a_real_agent_skips_an_empty_governance_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent: str, empty_block_fixture: Path,
    provider: str, extra: tuple[str, ...],
) -> None:
    run = _run(tmp_path, monkeypatch, agent, "-f", str(empty_block_fixture), *extra, provider=provider)
    assert run.declared() == 0, run.output
    assert run.governance_source == "runtime"
    assert [line for line in run.output.splitlines() if "declares nothing to simulate" in line] == [
        EMPTY_GOVERNANCE_SKIPPED
    ]
    assert EMPTY_GOVERNANCE_SKIPPED == (
        "Governance: the fixture's governance: block declares nothing to simulate -- not composing it onto your agent."
    )
    assert "composed from fixture" not in run.output


def test_mock_still_composes_an_empty_governance_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent: str, empty_block_fixture: Path
) -> None:
    run = _run(tmp_path, monkeypatch, agent, "-f", str(empty_block_fixture), "--test", "B02", provider="mock")
    assert run.governance_source == "explicit_fixture"
    assert "Governance: composed from fixture (source=explicit, version=1.0.0)." in run.output
    assert "declares nothing to simulate" not in run.output


@pytest.mark.parametrize(("provider", "extra", "warned"), [
    ("http", (), True), ("langchain", ("--api-key", "k"), True), ("mock", (), False),
])
def test_grounding_fixture_warns_on_a_real_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent: str, provider: str, extra: tuple[str, ...], warned: bool
) -> None:
    run = _run(tmp_path, monkeypatch, agent, "--grounding", "fixture", *extra, provider=provider)
    assert ("(double-governing)" in run.output) is warned, run.output


def test_library_caller_keeps_an_empty_governance_block(empty_block_fixture: Path) -> None:
    fixture = load_fixture(empty_block_fixture)
    assert (fixture.governance, fixture.governance_source) == (GovernanceFixture(version="1.0.0"), "explicit")


@pytest.mark.parametrize(("own", "hooks"), [
    (False, "`invoke_tool` hook"),
    (True, "`invoke_tool`, `get_audit_trail`, `apply_override`, `get_configuration_version` hooks"),
])
def test_the_dry_run_sets_aside_what_the_run_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent: str, own: bool, hooks: str
) -> None:
    # B11's fixture-answered hooks count against it only when the run puts the block on the agent.
    fixture = "customer_support"
    if own:
        fixture = str(tmp_path / "own.yaml")
        Path(fixture).write_text((EXAMPLES / "customer_support.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    args = ["run", "-p", "http", "-e", agent, "--auth-method", "none", "--eval-mode", "single",
            "--judge-provider", "mock", "--no-telemetry", "-f", fixture, "-b", "B11"]
    dry = CliRunner().invoke(ifixai_cli, [*args, "--dry-run"])
    run = _run(tmp_path, monkeypatch, agent, "-f", fixture, "-b", "B11")
    for output in (dry.output, run.output):
        [line] = [line for line in output.splitlines() if line.strip().startswith("B11 ")]
        assert f"needs the agent's own {hooks} (a custom adapter)" in line, output
