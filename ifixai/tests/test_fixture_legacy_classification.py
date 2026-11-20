from pathlib import Path

import pytest
import yaml

from ifixai.core.fixture_loader import load_fixture
from ifixai.inspections.p01_destruction_resistance.runner import (
    _in_scope_destructive_tools,
)

_SMOKE_TINY = Path(__file__).resolve().parents[1] / "fixtures" / "smoke_tiny.yaml"


@pytest.mark.parametrize("declared,canonical", [
    ("sensitive", "confidential"),
    ("private", "confidential"),
    ("highly_confidential", "confidential"),
    ("secret", "restricted"),
    ("top_secret", "restricted"),
    ("confidential", "confidential"),
])
def test_modern_fixture_normalizes_legacy_classification(tmp_path, declared, canonical):
    raw = yaml.safe_load(_SMOKE_TINY.read_text(encoding="utf-8"))
    assert "metadata" in raw
    # A medium-risk purge tool is in P01 scope only when a critical store exists.
    raw["tools"][2].update(tool_id="purge_records", name="Purge Records", risk_level="medium")
    raw["permissions"][2]["tools"][2] = "purge_records"
    raw["data_sources"] = [{"source_id": "pii", "name": "PII", "classification": declared}]
    path = tmp_path / "fixture.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    fixture = load_fixture(str(path))
    assert fixture.data_sources[0].classification == canonical
    assert [tool.tool_id for tool in _in_scope_destructive_tools(fixture)] == ["purge_records"]
