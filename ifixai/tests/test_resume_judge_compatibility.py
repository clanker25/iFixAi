import importlib
import json

import pytest
from click.testing import CliRunner

from ifixai.cli.main import ifixai_cli


def flags(tmp_path):
    return ["run", "--provider", "mock", "--model", "owned-sut", "--test", "B07",
            "--eval-mode", "single", "--judge-provider", "mock", "--judge-model", "owned-judge-a",
            "--judge-budget", "10000", "--output", str(tmp_path / "reports"),
            "--reliability-out", str(tmp_path / "runs"), "--no-telemetry", "--no-parallel"]


def owned_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IFIXAI_TELEMETRY", "0")
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    return importlib.import_module("ifixai.cli.run")

@pytest.mark.parametrize("change", ["model", "budget", "legacy"])
def test_native_resume_allows_dead_judge_swap_and_legacy_identity(tmp_path, monkeypatch, change):
    run = owned_environment(tmp_path, monkeypatch)
    original = run.build_manifest
    if change == "legacy":
        from ifixai.evaluation.manifest import compute_run_id
        def legacy(**kwargs):
            kwargs["judge_models"] = []
            kwargs["judge_identity"] = None
            manifest = original(**kwargs)
            payload = manifest.model_dump(mode="json")
            payload.pop("evaluation_mode", None)
            payload.pop("judge_budget", None)
            payload["schema_version"] = 4
            manifest.schema_version = 4
            manifest.run_id = compute_run_id(payload)
            return manifest
        monkeypatch.setattr(run, "build_manifest", legacy)
    first = CliRunner().invoke(ifixai_cli, flags(tmp_path))
    paths = list((tmp_path / "runs").glob("*/manifest.json"))
    assert len(paths) == 1, first.output
    manifest = json.loads(paths[0].read_text())
    monkeypatch.setattr(run, "build_manifest", original)
    args = flags(tmp_path)
    if change == "model":
        args[args.index("owned-judge-a")] = "owned-judge-b"
    elif change == "budget":
        args[args.index("10000")] = "1"
    resumed = CliRunner().invoke(ifixai_cli, [*args, "--resume", manifest["run_id"]])
    assert "[reused] B07" in resumed.output, resumed.output
    assert resumed.exit_code == first.exit_code


def test_native_full_ensemble_with_same_sut_model_does_not_crash(tmp_path, monkeypatch):
    owned_environment(tmp_path, monkeypatch)
    args = flags(tmp_path)
    args[args.index("single")] = "full"
    args[args.index("owned-judge-a")] = "owned-sut"
    result = CliRunner().invoke(ifixai_cli, [*args, "--judge-provider", "mock", "--judge-model", "owned-judge-b"])
    # A scoped run exits on its selection (#317): B07 passes on mock.
    assert result.exit_code == 0, (result.output, result.exception)
    assert len(list((tmp_path / "runs").glob("*/manifest.json"))) == 1
