"""`python -m ifixai.cli.main` is the fallback the README gives when the `ifixai` script isn't on PATH."""

import os
import subprocess
import sys
from pathlib import Path


def test_python_dash_m_runs_the_cli():
    result = subprocess.run(
        [sys.executable, "-m", "ifixai.cli.main", "--version"], capture_output=True, text=True, check=False, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("ifixai, version ")


# Piped output on Windows, a coding agent or CI can be a code page without the scorecard's bars and
# marks. The run prints `?` for those and still writes every report, instead of crashing at the
# scorecard after every inspection ran.
def test_a_run_on_a_non_utf8_console_finishes_and_writes_its_reports(tmp_path: Path) -> None:
    env = {**os.environ, "PYTHONIOENCODING": "cp1252", "IFIXAI_TELEMETRY": "0", "IFIXAI_NO_PROMO": "1"}
    result = subprocess.run(
        [sys.executable, "-m", "ifixai.cli.main", "run", "--provider", "mock", "--eval-mode", "self",
         "-b", "B16", "--no-telemetry", "--no-promo", "--output", "out", "--reliability-out", "runs"],
        capture_output=True, check=False, timeout=300, cwd=tmp_path, env=env,
    )
    # It finished: B16 fails on the demo agent's planted password reply, so it exits 2.
    assert result.returncode == 2, result.stderr.decode("cp1252", "replace")
    assert b"Traceback" not in result.stderr
    assert len(list((tmp_path / "out").glob("*.json"))) == 1
    assert b"Grade:" in result.stdout

