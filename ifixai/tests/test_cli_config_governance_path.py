"""A `governance:` path from ifixai.yaml bypasses click.Path(exists=True), so an
unreadable path must still produce the CLI's clean governance error."""

import pytest
from click.testing import CliRunner

from ifixai.cli.main import ifixai_cli


@pytest.mark.parametrize("target", ["missing.yaml", "policy_dir"])
def test_unreadable_config_governance_path_reports_cleanly(tmp_path, monkeypatch, target):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IFIXAI_TELEMETRY", "0")
    (tmp_path / "policy_dir").mkdir()
    (tmp_path / "ifixai.yaml").write_text(f"governance: {target}\n", encoding="utf-8")
    result = CliRunner().invoke(ifixai_cli, [
        "run", "--provider", "http", "--endpoint", "http://127.0.0.1:9",
        "--auth-method", "none", "--eval-mode", "self", "--no-telemetry",
        "--reliability-out", str(tmp_path / "runs"), "--output", str(tmp_path / "out"),
    ])
    assert isinstance(result.exception, SystemExit), result.exception
    assert result.exit_code == 1
    # A run reads only the fixture's rules, so the file is named and never opened.
    assert f"Ignoring the governance file {target}." in result.output
    assert "Traceback" not in result.output
