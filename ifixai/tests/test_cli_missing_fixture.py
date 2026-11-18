"""A --fixture path that doesn't exist stops the run before any agent or judge call."""

import pytest
from click.testing import CliRunner

from ifixai.cli.main import ifixai_cli


@pytest.mark.parametrize("dry_run", [True, False])
def test_missing_fixture_fails_before_agent_call(tmp_path, monkeypatch, dry_run):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("IFIXAI_TELEMETRY", "0")
    missing = tmp_path / "missing.yaml"
    args = [
        "run", "--provider", "mock", "--eval-mode", "self", "--fixture", str(missing),
        "--test", "B01", "--no-promo", "--no-telemetry",
        "--output", str(tmp_path / "reports"), "--reliability-out", str(tmp_path / "runs"),
    ]
    result = CliRunner().invoke(ifixai_cli, [*args, "--dry-run"] if dry_run else args)

    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit), result.exception
    assert f"Error: Fixture not found: '{missing}'." in result.output
    assert "Dry run" not in result.output
    assert "Testing connection" not in result.output
    assert not (tmp_path / "runs").exists()


# A typo in the fixture file reads as one plain line, as `ifixai validate` says it, not a traceback.
@pytest.mark.parametrize(
    ("name", "text", "says"),
    [("bad.yaml", "metadata:\n  name: x\ntools: [\n", "invalid YAML"),
     ("bad.json", '{"metadata": {"name": "x"},\n', "invalid JSON")],
)
def test_a_fixture_that_does_not_parse_fails_with_one_line(tmp_path, monkeypatch, name, text, says):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("IFIXAI_TELEMETRY", "0")
    (tmp_path / name).write_text(text, encoding="utf-8")
    result = CliRunner().invoke(ifixai_cli, [
        "run", "--provider", "mock", "--eval-mode", "self", "--fixture", name, "--test", "B01",
        "--no-promo", "--no-telemetry", "--output", "reports", "--reliability-out", "runs",
    ])

    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit), result.exception
    assert f"Error: Fixture error: {says}: " in result.output
    assert "Testing connection" not in result.output
