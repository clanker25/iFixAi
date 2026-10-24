"""The authoring schemas accept the contracts actually shipped by the package."""

import json
import re
from pathlib import Path

import jsonschema
import pytest
import yaml

from ifixai.rules import loader as rules_loader
from ifixai.rules.loader import RuleLoader, RuleLoadError, load_inspection_definition

_ROOT = Path(__file__).parents[1]
_ARTIFACTS = [
    (kind, path)
    for kind in ("rubric", "references")
    for path in sorted((_ROOT / "inspections").glob(f"*/{kind}*.yaml"))
]


@pytest.mark.parametrize("kind,path", _ARTIFACTS, ids=[p.parent.name + "/" + p.name for _, p in _ARTIFACTS])
def test_shipped_judge_artifacts_match_their_authoring_schema(kind, path):
    schema = json.loads((_ROOT / "schemas" / f"{kind}.schema.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(yaml.safe_load(path.read_text()))


@pytest.mark.parametrize("test_id", ["P27", "P32", "S02"])
def test_frontier_definitions_load_through_the_public_schema_checked_reader(test_id):
    plan = load_inspection_definition(test_id)
    assert plan is not None
    assert plan.test_id == test_id
    assert plan.steps


@pytest.mark.parametrize("kind", ["rubric", "references"])
def test_noncanonical_identifiers_remain_rejected(kind):
    schema = json.loads((_ROOT / "schemas" / f"{kind}.schema.json").read_text())
    sample = next(p for k, p in _ARTIFACTS if k == kind)
    document = yaml.safe_load(sample.read_text())
    document["test_id"] = "P1"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(document)


_STEPLESS = ["B01", "B02", "B04", "B05", "B07", "B12", "B19", "B20", "B22", "B23", "B28", "B29",
             "B30", "B32", "C02", "C05", "C11", "M02", "M03", "M06", "M07", "X04", "X11"]


@pytest.mark.parametrize("test_id", _STEPLESS)
def test_stepless_definitions_pass_the_run_time_schema_check(test_id):
    assert load_inspection_definition(test_id) is None


@pytest.mark.parametrize("steps", [None, [{"prompt_template": "hi"}]])
def test_run_time_schema_check_covers_stepless_definitions(tmp_path, monkeypatch, steps):
    folder = tmp_path / "b01_broken"
    folder.mkdir()
    raw = {"test_id": "B01", "fixture_requirements": "users"}
    if steps:
        raw["steps"] = steps
    (folder / "definition.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.setattr(rules_loader, "_DEFAULT_LOADER", RuleLoader(tmp_path))
    message = "Schema validation failed for B01: 'users' is not of type 'array'"
    with pytest.raises(RuleLoadError, match=re.escape(message)):
        load_inspection_definition("B01")
