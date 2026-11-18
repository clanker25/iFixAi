import pytest

from ifixai.core.discovery import generate_test_cases
from ifixai.core.types import Permission, Role, Tool


def test_wildcard_permissions_generate_real_allow_cases():
    cases = generate_test_cases(
        [Role(name="admin")],
        [Permission(role="admin", tools=["*"])],
        [Tool(tool_id="read", name="Read"), Tool(tool_id="delete", name="Delete")],
    )
    assert {(case.tool_id, case.expected_result) for case in cases} == {
        ("read", "allow"),
        ("delete", "allow"),
    }
    assert all(case.test == "B01" for case in cases)


@pytest.mark.parametrize("test_id", ["B01", "B08"])
async def test_b01_and_b08_probe_actual_tools_granted_by_wildcard(test_id):
    from ifixai.core.types import (
        Fixture,
        ProviderCapabilities,
        ProviderConfig,
        ToolInvocationResult,
        User,
    )
    from ifixai.inspections.b01_tool_governance.runner import B01ToolGovernance
    from ifixai.inspections.b08_privilege_escalation.runner import (
        B08PrivilegeEscalation,
    )
    from ifixai.providers.base import ChatProvider

    class OwnedWildcardProvider(ChatProvider):
        async def send_message(self, messages, config):
            return "owned reply"
        async def authorize_tool(self, tool_id, user_role, config):
            return ToolInvocationResult(authorized=user_role == "ci_pipeline", executed=False, policy_rule="owned wildcard policy")
        async def invoke_tool(self, tool_id, user_role, params, config):
            return ToolInvocationResult(authorized=user_role == "ci_pipeline", executed=user_role == "ci_pipeline", policy_rule="owned wildcard policy")

    fixture = Fixture.model_construct(users=[User(user_id="ci", name="ci", roles=["ci_pipeline"])],
        permissions=[Permission(role="ci_pipeline", tools=["*"])],
        tools=[Tool(tool_id=f"tool_{i}", name=f"Tool {i}") for i in range(24)])
    for inspection in [B01ToolGovernance() if test_id == "B01" else B08PrivilegeEscalation()]:
        inspection.capabilities = ProviderCapabilities(has_authorization=True, has_tool_calling=True)
        evidence = await inspection.run(OwnedWildcardProvider(), ProviderConfig(provider="owned"), fixture)
        probes = [e for e in evidence if e.test_case_id.startswith(("B01-auth-ci-", "B08-struct-ci-"))]
        assert len(probes) == (24 if test_id == "B01" else 0)
        assert all(e.passed for e in probes)
        # B08 only probes attempts to escalate into tools outside the grant.
        # A wildcard user must have no ordinary unauthorized-tool probes.
