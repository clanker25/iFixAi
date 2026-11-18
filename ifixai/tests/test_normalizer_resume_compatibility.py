"""Changing unused normalization must not reject a completed native CLI resume."""
import importlib
import json

from click.testing import CliRunner

from ifixai.cli.main import ifixai_cli


def test_completed_run_with_existing_normalizer_version_still_resumes(tmp_path, monkeypatch):
    run = importlib.import_module('ifixai.cli.run')
    current_version = run.NORMALIZER_VERSION
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('IFIXAI_TELEMETRY', '0')
    common = ['run', '--provider', 'mock', '--model', 'owned-model',
              '--test', 'B07', '--eval-mode', 'single', '--judge-provider', 'mock',
              '--output', str(tmp_path / 'reports'), '--reliability-out', str(tmp_path / 'runs'),
              '--no-telemetry', '--no-parallel']
    monkeypatch.setattr(run, 'NORMALIZER_VERSION', '1.0.0')
    first = CliRunner().invoke(ifixai_cli, common)
    manifests = list((tmp_path / 'runs').glob('*/manifest.json'))
    assert len(manifests) == 1, first.output
    manifest = json.loads(manifests[0].read_text())
    monkeypatch.setattr(run, 'NORMALIZER_VERSION', current_version)
    resumed = CliRunner().invoke(ifixai_cli, [*common, '--resume', manifest['run_id']])
    assert '[reused] B07' in resumed.output, resumed.output
    assert 'run configuration changed' not in resumed.output
    assert resumed.exit_code == first.exit_code
