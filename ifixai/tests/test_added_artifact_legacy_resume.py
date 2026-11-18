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

@pytest.mark.parametrize("mutate_existing", [False, True])
def test_added_keys_preserve_legacy_native_resume_but_existing_edits_reject(tmp_path, monkeypatch, mutate_existing):
    run = owned_environment(tmp_path, monkeypatch)
    current = run.compute_rubric_digests_for_tests_layout
    def legacy(*args, **kwargs):
        return {key: value for key, value in current(*args, **kwargs).items()
                if ":on_topic_prompts" not in key and key != "mutation_bank:synonyms"}
    monkeypatch.setattr(run, "compute_rubric_digests_for_tests_layout", legacy)
    first = CliRunner().invoke(ifixai_cli, flags(tmp_path))
    paths = list((tmp_path / "runs").glob("*/manifest.json"))
    assert len(paths) == 1, first.output
    manifest = json.loads(paths[0].read_text())
    def upgraded(*args, **kwargs):
        result = current(*args, **kwargs)
        if mutate_existing:
            result[next(iter(manifest["rubric_hashes"]))] = "f" * 64
        return result
    monkeypatch.setattr(run, "compute_rubric_digests_for_tests_layout", upgraded)
    resumed = CliRunner().invoke(ifixai_cli, [*flags(tmp_path), "--resume", manifest["run_id"]])
    if mutate_existing:
        assert "differs in: rubric_hashes" in resumed.output, resumed.output
        assert resumed.exit_code == 1
    else:
        assert "[reused] B07" in resumed.output, resumed.output
        assert resumed.exit_code == first.exit_code
