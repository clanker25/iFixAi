import json
import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from ifixai.cli.config_file import RunConfig, write_config
from ifixai.cli.main import ifixai_cli

GOVERNANCE_FILE = str(Path(__file__).parents[1] / "fixtures" / "governance" / "mock.yaml")
# Both paths: `ifixai author --force` rewrites a whole fixture, so an authored one's rules go in
# the profile; only a hand-written fixture holds them in its governance: block.
NOTICE = (
    f"Ignoring the governance file {GOVERNANCE_FILE}. Fixture from ifixai author: add these rules to "
    "the profile's safety_rules and run ifixai author again with --force. Fixture written by hand: "
    "put them in its governance: block."
)
RUN = [
    "run", "--provider", "mock", "--api-key", "x", "--fixture", "default", "--eval-mode", "self",
    "-b", "B01", "-b", "B02", "--run-nonce", "0123456789abcdef",
    # Every pinnable seed pinned. The run id also covers a fresh inspection seed, so the runs are
    # compared on the manifest's governance fields instead.
    *[arg for name in ("b12", "b14", "b28", "b29", "b30", "b32", "holdout") for arg in (f"--{name}-seed", "7")],
    "--output", "out", "--format", "json", "--no-telemetry", "--no-promo",
]


def mock_run(
    workdir: Path, monkeypatch: pytest.MonkeyPatch, extra: list[str], config: RunConfig | None = None
) -> dict:
    workdir.mkdir()
    if config is not None:
        write_config(config, workdir)
    monkeypatch.chdir(workdir)
    result = CliRunner().invoke(ifixai_cli, [*RUN, *extra])
    # A two-check run has no headline score, so it exits on B01 and B02, which fail here on the
    # demo agent's planted replies.
    assert result.exit_code == 2, result.output
    [report_file] = [f for f in (workdir / "out").glob("*.json") if "summary" not in f.name]
    report = json.loads(report_file.read_text(encoding="utf-8"))
    [manifest_file] = (workdir / "runs").glob("*/manifest.json")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    return {
        "output": result.output,
        "governance": (manifest["governance_source"], manifest.get("governance_fixture_digest")),
        "results": [(r["test_id"], r["status"], r["score"]) for r in report["test_results"]],
        "not_applicable": report["not_applicable"],
        "warnings": report["warnings"],
    }


# J1: an old script or ifixai.yaml still naming a governance file runs as before and is told,
# once, where the rules go now. No line claims a wrap the run didn't do.
@pytest.mark.parametrize(
    ("extra", "config"),
    [
        (["--governance", GOVERNANCE_FILE], None),
        ([], RunConfig(governance=GOVERNANCE_FILE)),
    ],
    ids=["flag", "ifixai.yaml"],
)
def test_a_governance_file_changes_nothing_and_says_where_rules_go(tmp_path, monkeypatch, extra, config):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    plain = mock_run(tmp_path / "plain", monkeypatch, [])
    governed = mock_run(tmp_path / "governed", monkeypatch, extra, config)
    output = governed.pop("output")

    assert governed == {key: value for key, value in plain.items() if key != "output"}
    assert output.count(NOTICE) == 1
    assert NOTICE not in plain["output"]
    assert not re.search(r"wrapped|composing", output, re.IGNORECASE), output


def test_governance_is_hidden_from_help():
    result = CliRunner().invoke(ifixai_cli, ["run", "--help"])
    assert result.exit_code == 0, result.output
    assert "--governance" not in result.output
