"""Inspections that score a subset must publish their CI over that subset.

The runner's own leg computes it. A run sets these inspections aside without a judge (their chat
probes need one), so each runner is executed directly.
"""

import importlib
from copy import copy

import pytest

from ifixai.core.fixture_loader import load_fixture
from ifixai.core.types import EvaluationMode, EvaluationPipelineConfig, ProviderConfig
from ifixai.harness.registry import INSPECTION_REGISTRY
from ifixai.providers.base import detect_capabilities
from ifixai.providers.mock_governance import MockGovernanceProvider

_RUNNERS = {
    "C02": "c02_low_confidence_abstention",
    "C05": "c05_human_fallback_routing",
    "C11": "c11_operational_outcome_conformance",
    "X04": "x04_detection_performance_gate",
    "X11": "x11_pre_action_confirmation_gate",
}


@pytest.mark.asyncio
@pytest.mark.parametrize("test_id", sorted(_RUNNERS))
async def test_ci_is_computed_over_the_scored_population(test_id):
    runner = importlib.import_module(f"ifixai.inspections.{_RUNNERS[test_id]}.runner")
    fixture = load_fixture("default")
    provider = MockGovernanceProvider(fixture.governance)
    config = ProviderConfig(provider="mock")
    result = await copy(INSPECTION_REGISTRY[test_id]).execute(
        provider,
        config,
        fixture,
        capabilities=await detect_capabilities(provider, config),
        pipeline_config=EvaluationPipelineConfig(mode=EvaluationMode.FULL),
    )
    scored = runner._scored_items(result.evidence)
    ci = result.confidence_interval
    assert 0 < len(scored) < len(result.evidence)
    assert ci.sample_size == len(scored)
    assert ci.lower <= result.score <= ci.upper
