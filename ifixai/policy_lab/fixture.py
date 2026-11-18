"""Import fixture permissions only; business effects always remain explicit."""
from __future__ import annotations

from ifixai.core.types import Fixture

from .models import BusinessModel, ExplorationPolicy, PermissionRule


def policy_from_fixture(fixture: Fixture, model: BusinessModel) -> ExplorationPolicy:
    model = BusinessModel.model_validate(model.model_dump(mode="python", warnings=False))
    role_ids = [role.role_id or role.name for role in fixture.roles]
    tool_ids = [tool.tool_id for tool in fixture.tools]
    user_ids = [user.user_id for user in fixture.users]
    for identifiers in (role_ids, tool_ids, user_ids):
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("fixture role, tool and user identifiers must be unique")
    roles, tools = set(role_ids), set(tool_ids)
    users = {user.user_id: user for user in fixture.users}
    for user in fixture.users:
        if len(set(user.roles)) != len(user.roles) or not set(user.roles) <= roles:
            raise ValueError("fixture user role declarations are ambiguous or unknown")
    for actor_id, actor in model.actors.items():
        if actor_id not in users or set(actor.roles) != set(users[actor_id].roles):
            raise ValueError("model actors must exactly match fixture users and their roles")
    if any(action.tool not in tools for action in model.actions):
        raise ValueError("business actions must use explicitly declared fixture tools")
    matrix: dict[str, set[str]] = {}
    for permission in fixture.permissions:
        if permission.role not in roles or permission.role in matrix:
            raise ValueError("fixture permission roles are unknown or duplicated")
        if not set(permission.tools) <= tools | {"*"}:
            raise ValueError("fixture permissions refer to undeclared tools")
        matrix[permission.role] = tools.copy() if "*" in permission.tools else set(permission.tools)
    governance = fixture.governance
    if governance is not None:
        # Conflicting declarative matrices must not be silently promoted to one policy.
        authorization = getattr(getattr(governance, "policies", None), "authorization", None)
        if authorization is None:
            raise ValueError("fixture governance authorization cannot be verified")
        governed: dict[str, set[str]] = {}
        for rule in authorization:
            if rule.role not in roles or rule.role in governed or not set(rule.tools) <= tools | {"*"}:
                raise ValueError("fixture governance authorization is ambiguous")
            governed[rule.role] = tools.copy() if "*" in rule.tools else set(rule.tools)
        if any(matrix.get(role, set()) != governed.get(role, set()) for role in roles):
            raise ValueError("fixture permission and governance matrices disagree")
    referenced_tools = {action.tool for action in model.actions}
    permissions = tuple(PermissionRule(role=role, tool=tool, resource_scope="*")
                        for role in sorted(matrix) for tool in sorted(matrix[role] & referenced_tools))
    return ExplorationPolicy(name="fixture permission projection", tenant_id=model.tenant_id, permissions=permissions)
