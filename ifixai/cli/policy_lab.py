"""Data-only reachability exploration and goal-preserving permission repair."""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click
from pydantic import ValidationError

from ifixai.core.fixture_loader import (
    FixtureValidationError,
    load_fixture,
    resolve_fixture_path,
)
from ifixai.policy_lab.fixture import policy_from_fixture
from ifixai.policy_lab.integrity import (
    create_exploration_bundle,
    load_exploration_bundle,
)
from ifixai.policy_lab.models import (
    BusinessModel,
    ExplorationBundle,
    ExplorationPolicy,
    SearchConfig,
)
from ifixai.policy_lab.repair import RepairReport, search_repairs
from ifixai.policy_lab.search import canonical_digest
from ifixai.reporting.policy_lab import compare_policies, write_policy_lab_reports

MAX_INPUT_BYTES = 16 * 1024 * 1024
DECISION_EXIT = {"VERIFIED": 0, "COUNTEREXAMPLE": 1, "GOAL_BLOCKED": 1, "INCONCLUSIVE": 2}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON keys are not accepted")
        result[key] = value
    return result


def _reject_constant(_: str) -> None:
    raise ValueError("nonfinite JSON constants are not accepted")


def _input(path: str) -> Any:
    with Path(path).open("rb") as handle:
        data = handle.read(MAX_INPUT_BYTES + 1)
    if len(data) > MAX_INPUT_BYTES:
        raise ValueError("model or policy exceeds the 16 MiB input bound")
    return json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def _model(path: str) -> BusinessModel:
    return BusinessModel.model_validate(_input(path))


def _policy(path: str) -> ExplorationPolicy:
    return ExplorationPolicy.model_validate(_input(path))


def _guard(operation: Callable[[], Any]) -> Any:
    try:
        return operation()
    except ValidationError as error:
        raise click.ClickException(f"Invalid Policy Lab schema ({error.error_count()} field errors); consult the model contract.") from None
    except (ValueError, TypeError, OSError, UnicodeError, RecursionError, FixtureValidationError) as error:
        raise click.ClickException(f"Policy Lab operation failed ({type(error).__name__}); check input declarations, integrity and new output directory.") from None


def _write(bundle: ExplorationBundle, output: str, *, comparison: dict[str, Any] | None = None, repair: RepairReport | None = None, metadata: dict[str, Any] | None = None) -> None:
    destination = _guard(lambda: write_policy_lab_reports(bundle, output, comparison=comparison, repair=repair, metadata=metadata))
    if repair:
        click.echo(f"Repair {repair.status}: {len(repair.candidates)} candidates, minimum cardinality proven={repair.minimality_proven}")
        exit_code = {"FOUND": 0, "EXHAUSTED": 1, "INCONCLUSIVE": 2}[repair.status]
    else:
        click.echo(f"{bundle.report.decision}: {len(bundle.report.safety_witnesses)} harm witnesses, {len(bundle.report.goal_witnesses)} completion witnesses")
        exit_code = DECISION_EXIT[bundle.report.decision]
    click.echo(f"Model evidence: {destination / 'report.html'}")
    raise click.exceptions.Exit(exit_code)


@click.group("policy-lab")
def policy_lab() -> None:
    """Find harmful compositions of permitted business actions in explicit models."""


@policy_lab.command("check")
@click.argument("model", type=click.Path(exists=True, dir_okay=False))
@click.argument("policy", type=click.Path(exists=True, dir_okay=False))
@click.option("--max-depth", type=click.IntRange(1, 64), default=12, show_default=True)
@click.option("--max-states", type=click.IntRange(1, 100000), default=10000, show_default=True)
@click.option("--max-transitions", type=click.IntRange(1, 2000000), default=200000, show_default=True)
@click.option("--output", required=True, type=click.Path(file_okay=False))
def check(model: str, policy: str, max_depth: int, max_states: int, max_transitions: int, output: str) -> None:
    bundle = _guard(lambda: create_exploration_bundle(_model(model), _policy(policy), SearchConfig(max_depth=max_depth, max_states=max_states, max_transitions=max_transitions)))
    _write(bundle, output)


@policy_lab.command("from-fixture")
@click.argument("model", type=click.Path(exists=True, dir_okay=False))
@click.argument("fixture")
@click.option("--output", required=True, type=click.Path(file_okay=False))
def from_fixture(model: str, fixture: str, output: str) -> None:
    def operation() -> tuple[ExplorationBundle, dict[str, Any]]:
        business = _model(model)
        fixture_path = resolve_fixture_path(fixture)
        if fixture_path.stat().st_size > MAX_INPUT_BYTES:
            raise ValueError("fixture exceeds the 16 MiB bound")
        loaded = load_fixture(fixture_path)
        derived = policy_from_fixture(loaded, business)
        metadata = {"fixture_name": loaded.metadata.name, "fixture_digest": canonical_digest(loaded),
                    "boundary": "Imports fixture role/tool declarations only; resource scope is wildcard because the fixture has no resource scope. Effects and tenant remain explicit model declarations."}
        return create_exploration_bundle(business, derived), metadata

    bundle, metadata = _guard(operation)
    _write(bundle, output, metadata=metadata)


@policy_lab.command("replay")
@click.argument("bundle", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", required=True, type=click.Path(file_okay=False))
def replay(bundle: str, output: str) -> None:
    _write(_guard(lambda: load_exploration_bundle(bundle)), output)


@policy_lab.command("compare")
@click.argument("bundle", type=click.Path(exists=True, dir_okay=False))
@click.argument("policy", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", required=True, type=click.Path(file_okay=False))
def compare(bundle: str, policy: str, output: str) -> None:
    def operation() -> tuple[ExplorationBundle, dict[str, Any]]:
        original = load_exploration_bundle(bundle)
        candidate = create_exploration_bundle(original.model, _policy(policy), original.config)
        return candidate, compare_policies(original, candidate)

    candidate, comparison = _guard(operation)
    _write(candidate, output, comparison=comparison)


@policy_lab.command("repair")
@click.argument("bundle", type=click.Path(exists=True, dir_okay=False))
@click.option("--max-removals", type=click.IntRange(1, 2), default=1, show_default=True)
@click.option("--max-candidates", type=click.IntRange(1, 256), default=64, show_default=True)
@click.option("--transition-budget", type=click.IntRange(1, 2000000), default=250000, show_default=True)
@click.option("--output", required=True, type=click.Path(file_okay=False))
def repair(bundle: str, max_removals: int, max_candidates: int, transition_budget: int, output: str) -> None:
    original = _guard(lambda: load_exploration_bundle(bundle))
    result = _guard(lambda: search_repairs(original, max_removals=max_removals, max_candidates=max_candidates, transition_budget=transition_budget))
    _write(original, output, repair=result)
