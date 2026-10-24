"""Run-level checkpoint + resume.

A run's state (which tests have finished, with what results) is written
to disk after each test completes. Re-invoking the CLI with the same
config + seed + corpus versions loads the checkpoint and executes only
the remaining tests.

Checkpoint safety:
- Keyed by `run_id` from the manifest, a SHA-256 over every field that
  would invalidate reproducibility (capabilities, seed, corpus versions,
  scoring_logic_version, model, judges, fixture). Two runs with
  incompatible configs get different `run_id`s, so a stale checkpoint
  cannot pollute a fresh run.
- Resume re-executes any cached result that has an `error_message`.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock

from ifixai.core.types import TestResult, TestStatus

_logger = logging.getLogger(__name__)
_checkpoint_publish_lock = Lock()


def _checkpoint_path(runs_root: Path, run_id: str) -> Path:
    return runs_root / run_id / "checkpoint.json"


def load_checkpoint(runs_root: Path, run_id: str) -> dict[str, TestResult]:
    path = _checkpoint_path(runs_root, run_id)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _logger.warning("checkpoint unreadable (%s) — starting fresh", exc)
        return {}
    if not isinstance(raw, dict):
        _logger.warning("checkpoint root is not an object — starting fresh")
        return {}
    out: dict[str, TestResult] = {}
    for test_id, payload in raw.items():
        try:
            result = TestResult.model_validate(payload)
        except Exception as exc:  # noqa: BLE001 — best-effort checkpoint read; any failure is treated as no checkpoint
            _logger.warning(
                "checkpoint entry for %s rejected (%s) — will re-run",
                test_id, exc,
            )
            continue
        if result.test_id != test_id:
            _logger.warning(
                "checkpoint entry %s contains result for %s — will re-run",
                test_id, result.test_id,
            )
            continue
        if result.error_message or result.status == TestStatus.ERROR:
            _logger.info("skipping stale errored result for %s", test_id)
            continue
        out[test_id] = result
    return out


def save_checkpoint(
    runs_root: Path,
    run_id: str,
    results: dict[str, TestResult],
) -> Path:
    """Persist atomically: writes to a private sibling temporary file then renames,
    so the checkpoint is never half-written if the process dies mid-write."""
    path = _checkpoint_path(runs_root, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        test_id: result.model_dump(mode="json")
        for test_id, result in results.items()
    }
    fh = NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False)
    tmp = Path(fh.name)
    try:
        with fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        # Windows can reject simultaneous replacements of the same destination.
        # Serialize only the atomic publish step; staging remains concurrent.
        with _checkpoint_publish_lock:
            tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return path
