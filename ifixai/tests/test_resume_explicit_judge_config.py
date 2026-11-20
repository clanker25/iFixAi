"""Historical grading manifests retain their original hash verification."""
import json
from pathlib import Path

import pytest

from ifixai.evaluation.manifest import load_manifest, verify_run_id


@pytest.mark.parametrize("version", [1, 3, 4])
def test_frozen_pre_grading_context_manifests_keep_original_identity(version):
    path = Path(__file__).parent / "fixtures" / f"manifest_before_judge_config_v{version}.json"
    original = json.loads(path.read_text())
    loaded = load_manifest(path)
    assert loaded.run_id == original["run_id"]
    assert verify_run_id(loaded)
