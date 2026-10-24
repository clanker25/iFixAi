import pytest

from ifixai.core.types import Fixture, Permission, Role, Tool
from ifixai.shared.authorized_scope import authorized_tool_name


@pytest.mark.parametrize("tools", [[], ["missing_tool"]])
def test_explicit_permission_without_usable_tool_avoids_forbidden_objective(tools):
    fixture = Fixture.model_construct(
        permissions=[Permission(role="reader", tools=tools)],
        tools=[Tool(tool_id="delete", name="Delete records")],
    )
    assert authorized_tool_name(fixture, "reader", 0) == "general query"


def test_absent_permission_retains_unrestricted_fallback():
    fixture = Fixture.model_construct(
        permissions=[], tools=[Tool(tool_id="read", name="Read records")]
    )
    assert authorized_tool_name(fixture, "reader", 0) == "Read records"


@pytest.mark.parametrize("requested", ["Guest", "role_guest"])
def test_permission_role_id_and_name_resolve_to_same_excluded_scope(requested):
    fixture = Fixture.model_construct(roles=[Role(name="Guest", role_id="role_guest")],
        permissions=[Permission(role="role_guest", tools=[])],
        tools=[Tool(tool_id="refund", name="Issue Refund")])
    assert authorized_tool_name(fixture, requested, 0) == "general query"
