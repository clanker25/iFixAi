from click.testing import CliRunner

from ifixai.cli import setup_cmd
from ifixai.cli.config_file import load_config
from ifixai.cli.setup_cmd import setup


def _accept_default(message, choices, default=None, **kwargs):
    if "mock" in choices:
        return "mock"
    return default if default in choices else choices[0]


def test_setup_defaults_to_the_graded_suite(monkeypatch):
    # `core` alone never ends in a grade (P01 isn't in it); pressing Enter must give one:
    # the essential suite holds every mandatory minimum.
    monkeypatch.setattr(setup_cmd.ui, "is_interactive", lambda: True)
    monkeypatch.setattr(setup_cmd.ui, "select", _accept_default)
    monkeypatch.setattr(setup_cmd.ui, "text", lambda *args, **kwargs: "")
    monkeypatch.setattr(setup_cmd.ui, "confirm", lambda message, **kwargs: message.startswith("Save"))
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(setup)
        assert result.exit_code == 0, result.output
        assert load_config().suite == "essential"
