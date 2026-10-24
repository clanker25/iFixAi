"""`ifixai run` rejects --governance with --provider mock, so the setup wizard
must not save a governance path for a mock agent."""

import yaml
from click.testing import CliRunner

from ifixai.cli import setup_cmd
from ifixai.cli.main import ifixai_cli


def test_mock_setup_does_not_save_governance(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IFIXAI_TELEMETRY", "0")
    asked = []

    def select(message, choices, default=None, **kwargs):
        return "mock" if message.startswith("What is the system under test") else default

    def text(message, default=""):
        asked.append(message)
        return str(tmp_path / "policy.yaml") if "governance" in message.lower() else default

    confirms = iter([True, False])  # save the config; don't run now
    monkeypatch.setattr(setup_cmd.ui, "is_interactive", lambda: True)
    monkeypatch.setattr(setup_cmd.ui, "select", select)
    monkeypatch.setattr(setup_cmd.ui, "text", text)
    monkeypatch.setattr(setup_cmd.ui, "confirm", lambda message, default=True: next(confirms))

    result = CliRunner().invoke(ifixai_cli, ["setup"])
    assert result.exit_code == 0, result.output
    saved = yaml.safe_load((tmp_path / "ifixai.yaml").read_text(encoding="utf-8"))
    assert saved["provider"] == "mock"
    assert "governance" not in saved
    assert not any("governance" in message.lower() for message in asked)
