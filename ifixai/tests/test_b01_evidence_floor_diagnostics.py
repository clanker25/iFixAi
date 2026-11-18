"""B01's coverage summary is a diagnostic: it must not count toward the evidence floor."""

import asyncio
from pathlib import Path

import pytest

from ifixai.core.fixture_loader import load_fixture
from ifixai.core.types import (
    ProviderCapabilities,
    ProviderConfig,
    TestStatus,
    ToolInvocationResult,
)
from ifixai.inspections.b01_tool_governance.runner import B01ToolGovernance
from ifixai.scoring.mandatory_minimums import check_mandatory_minimums

_SMOKE_TINY = Path(__file__).resolve().parents[1] / "fixtures" / "smoke_tiny.yaml"


def _governed_run(tool_count: int):
    fixture = load_fixture(str(_SMOKE_TINY))
    fixture = fixture.model_copy(update={"tools": fixture.tools[:tool_count]})
    allowed: dict[str, set[str]] = {}
    for permission in fixture.permissions:
        allowed.setdefault(permission.role, set()).update(permission.tools)

    class Governor:
        async def authorize_tool(self, tool_id, user_role, config):
            granted = tool_id in allowed.get(user_role, set())
            return ToolInvocationResult(authorized=granted, executed=False,
                                        policy_rule=None if granted else "RBAC-DENY")

    return asyncio.run(B01ToolGovernance().execute(
        Governor(), ProviderConfig(provider="mock", model="owned"), fixture,
        capabilities=ProviderCapabilities(has_authorization=True),
    ))


def test_floor_counts_only_scored_decisions():
    # 3 users x 2 tools + 3 holdouts = 9 decisions, one short of B01's floor of 10.
    result = _governed_run(tool_count=2)
    scored = [e for e in result.evidence if not e.is_diagnostic]
    assert len(scored) < result.spec.min_evidence_items
    assert result.insufficient_evidence is True
    assert result.status is TestStatus.INCONCLUSIVE
    gate = check_mandatory_minimums([result], selected_ids={"B01"})["minimum_status"]["B01"]
    assert gate is not TestStatus.PASS


def test_floor_is_met_when_enough_decisions_are_scored():
    result = _governed_run(tool_count=3)
    assert sum(not e.is_diagnostic for e in result.evidence) >= result.spec.min_evidence_items
    assert result.status is TestStatus.PASS
    assert result.score == pytest.approx(1.0)
