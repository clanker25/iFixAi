"""Evaluator wind tunnel: measure what the inspections actually detect.

The suite makes a claim about an agent. This package measures whether that
claim holds, by running the inspections against a reference provider that is
correct everywhere except one named, seeded defect, and reporting per
inspection whether the defect was caught.

    from ifixai.providers.governance_fixture import GovernanceFixture
    from ifixai.windtunnel import CATALOGUE, measure

    governance = GovernanceFixture.load("ifixai/fixtures/governance/mock.yaml")
    report = await measure(CATALOGUE["over_permissive"], governance)
    print(report.summary())
    for gap in report.blind_spots:
        print("blind spot:", gap.test_id)

What it does not do: a detection rate measured here is against *these* seeded
defects on *this* fixture. It is a lower bound on what the suite catches, not a
statement about a real agent, and not a substitute for an adversary that gets
to choose its own attack.
"""

from ifixai.windtunnel.defects import (
    CATALOGUE,
    CLEAN,
    LEAK_RATES,
    Defect,
    DefectKind,
    DefectProfile,
    leak_profiles,
)
from ifixai.windtunnel.measure import (
    STRUCTURAL_INSPECTIONS,
    InspectionOutcome,
    Outcome,
    WindTunnelReport,
    measure,
    measure_all,
)
from ifixai.windtunnel.provider import DefectInjectingProvider

__all__ = [
    "CATALOGUE",
    "CLEAN",
    "LEAK_RATES",
    "STRUCTURAL_INSPECTIONS",
    "Defect",
    "DefectInjectingProvider",
    "DefectKind",
    "DefectProfile",
    "InspectionOutcome",
    "Outcome",
    "WindTunnelReport",
    "leak_profiles",
    "measure",
    "measure_all",
]

from ifixai.windtunnel.evidence import (
    SCHEMA_VERSION,
    BaselineComparison,
    DetectionChange,
    WindTunnelBundle,
    canonical_digest,
    compare_to_baseline,
    create_bundle,
    load_bundle,
    write_bundle,
)

__all__ += [
    "SCHEMA_VERSION",
    "BaselineComparison",
    "DetectionChange",
    "WindTunnelBundle",
    "canonical_digest",
    "compare_to_baseline",
    "create_bundle",
    "load_bundle",
    "write_bundle",
]
