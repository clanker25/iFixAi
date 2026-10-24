"""A full run resumes with a bare --resume even when ifixai.yaml saves a suite."""

import json

from click.testing import CliRunner

from ifixai.cli import run as run_module
from ifixai.cli.main import ifixai_cli


def test_full_run_resumes_despite_saved_config_suite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IFIXAI_TELEMETRY", "0")
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    (tmp_path / "ifixai.yaml").write_text("suite: smoke\n", encoding="utf-8")
    selections = []

    async def interrupted(**kwargs):
        selections.append((kwargs["strategic"], tuple(kwargs["test_ids"] or ())))
        raise KeyboardInterrupt

    monkeypatch.setattr(run_module, "execute_tests", interrupted)
    common = [
        "run", "--provider", "mock", "--eval-mode", "self", "--no-telemetry", "--no-promo",
        "--reliability-out", str(tmp_path / "runs"), "--output", str(tmp_path / "out"),
    ]
    CliRunner().invoke(ifixai_cli, [*common, "--full-run"])
    manifests = list((tmp_path / "runs").glob("*/manifest.json"))
    assert len(manifests) == 1
    run_id = json.loads(manifests[0].read_text(encoding="utf-8"))["run_id"]

    resumed = CliRunner().invoke(ifixai_cli, [*common, "--resume", run_id])
    assert "cannot resume" not in resumed.output, resumed.output
    assert len(selections) == 2
    assert selections[1] == selections[0]
