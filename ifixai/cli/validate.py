
import json
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path

import click
import yaml
from pydantic import ValidationError

from ifixai.cli.config_file import CONFIG_FILENAME, load_config
from ifixai.core.fixture_loader import (
    FixtureValidationError,
    ProfileNotFixtureError,
    is_builtin_fixture,
    read_valid_fixture,
    resolve_fixture_path,
)
from ifixai.core.profile import AgentProfile, expand_profile
from ifixai.harness.completeness import check_fixture
from ifixai.harness.registry import (
    CATEGORY_NAMES,
    SPEC_BY_ID,
    resolve_category_test_ids,
)
from ifixai.harness.suites import DEFAULT_SUITE, SUITE_NAMES, resolve_suite
from ifixai.harness.validator import LayoutValidationError, validate_layout
from ifixai.scoring.category_weights import STRATEGIC_TEST_IDS

# Where the advice for a short built-in fixture says to put the user's own copy.
_OWN_COPY = "my-fixture.yaml"
# The JSON files per folder tried as the profile a fixture was authored from, and the largest read.
_PROFILE_CANDIDATES = 50
_PROFILE_MAX_BYTES = 1_000_000


def profile_behind(fixture: str) -> str | None:
    """The profile `ifixai author` wrote `fixture` from: a JSON file beside it, or in the working
    directory, that still expands to exactly this fixture. None when none does, so the hint never
    names a profile whose `--force` would write a different fixture over this one."""
    try:
        written = yaml.safe_load(Path(fixture).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return None
    for folder in dict.fromkeys([Path(fixture).parent, Path()]):
        for candidate in _likely_profiles(folder, Path(fixture).stem):
            if _writes(candidate, written):
                return str(candidate)
    return None


def _likely_profiles(folder: Path, fixture_stem: str) -> list[Path]:
    """The JSON files in `folder` small enough to be a profile, the likeliest first (named like the
    fixture or `*profile*`), then by name, at most `_PROFILE_CANDIDATES`: the same pick whatever
    order the folder lists them in."""
    try:
        found = [path for path in folder.glob("*.json") if _small_file(path)]
    except OSError:
        return []

    def rank(path: Path) -> tuple[bool, str]:
        stem = path.stem.lower()
        return (not (stem == fixture_stem.lower() or "profile" in stem), path.name)

    return sorted(found, key=rank)[:_PROFILE_CANDIDATES]


def _small_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size <= _PROFILE_MAX_BYTES
    except OSError:
        return False


def _writes(profile: Path, written: object) -> bool:
    """Whether `ifixai author` on `profile` writes exactly the fixture `written`."""
    try:
        parsed = AgentProfile.model_validate(json.loads(profile.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError):
        return False
    return expand_profile(parsed) == written


def echo_misses(misses: list[str], fixture: str, selection: Sequence[str]) -> None:
    """The evidence minimums a fixture falls short of, one line each saying what to add, then how
    to fix it and the `ifixai validate` that checks the same selection again."""
    click.echo(click.style("Not ready: some inspections couldn't reach a verdict on this fixture.", fg="red"))
    for line in misses:
        click.echo(f"  {line}")
    if is_builtin_fixture(fixture):
        # A shipped fixture is never edited in place: the fix goes in a copy of the user's own.
        source = shlex.quote(str(resolve_fixture_path(fixture)))
        click.echo(
            f"{fixture} ships with iFixAi, so leave it as it is: copy it to a new file "
            f"(cp {source} {_OWN_COPY}) and fix that, or write a profile and run "
            f"ifixai author <your profile> -o {_OWN_COPY}."
        )
        fixture = _OWN_COPY
    else:
        path = shlex.quote(fixture)
        profile = profile_behind(fixture)
        if profile is None:
            click.echo(
                f"Fix the agent profile and run ifixai author <your profile> -o {path} --force, or edit {path}."
            )
        else:
            source = shlex.quote(profile)
            click.echo(f"Fix {source} and run ifixai author {source} -o {path} --force, or edit {path}.")
    click.echo(f"Then run {shlex.join(['ifixai', 'validate', fixture, *selection])}")


def refuse_full_run_with_a_pick(
    full_run: bool, strategic: bool, test: Sequence[str], categories: Sequence[str], suite: str | None
) -> None:
    """`--full-run` already picks every inspection: with another pick too, exit 1."""
    if full_run and (strategic or test or categories or suite):
        click.echo(
            click.style(
                "Error: --full-run runs every inspection and cannot be combined "
                "with --suite, --test, --category or --strategic.",
                fg="red",
            ),
            err=True,
        )
        sys.exit(1)


def resolve_pick(
    suite: str | None,
    *,
    full_run: bool,
    strategic: bool,
    test: Sequence[str],
    categories: Sequence[str],
    config_suite: str | None,
) -> tuple[str | None, bool]:
    """The suite a pick runs, and whether it is the default. With no pick of its own, a run takes
    ifixai.yaml's suite, else the essential one. `ifixai run` and `ifixai validate` both pick
    here, so a bare validate checks exactly what a bare run runs."""
    picked = full_run or strategic or bool(test) or bool(categories)
    if not picked and suite is None and config_suite:
        return config_suite, False
    if picked or suite:
        return suite, False
    return DEFAULT_SUITE, True


def expand_selection(test: Sequence[str], categories: Sequence[str], suite: str | None) -> tuple[str, ...]:
    """The `-b` ids plus every id `--suite` and `-c` name, in order, once each. An unknown suite,
    category or id exits 1, naming the choices."""
    if suite:
        suite_resolution = resolve_suite(suite)
        if suite_resolution["unknown"]:
            click.echo(
                click.style(
                    f"Error: unknown suite '{suite}'. "
                    f"Available: {', '.join(SUITE_NAMES)}",
                    fg="red",
                )
            )
            sys.exit(1)
        test = tuple(
            dict.fromkeys(
                [*(tid.upper() for tid in test), *suite_resolution["test_ids"]]
            )
        )

    if categories:
        resolution = resolve_category_test_ids(categories)
        if resolution["unknown"]:
            click.echo(
                click.style(
                    f"Error: unknown category(ies): "
                    f"{', '.join(resolution['unknown'])}. "
                    f"Available: {', '.join(CATEGORY_NAMES)}",
                    fg="red",
                )
            )
            sys.exit(1)
        # Merge category-expanded IDs with any explicit -b IDs (dedup, order
        # preserved) so the run flows through the existing selected-subset path.
        test = tuple(
            dict.fromkeys([*(tid.upper() for tid in test), *resolution["test_ids"]])
        )

    if test:
        unknown_ids = [tid for tid in test if tid.upper() not in SPEC_BY_ID]
        if unknown_ids:
            click.echo(
                click.style(
                    f"Error: unknown test ID(s): {', '.join(unknown_ids)}. "
                    f"Available: {', '.join(sorted(SPEC_BY_ID))}",
                    fg="red",
                )
            )
            sys.exit(1)
    return tuple(test)


def select_test_ids(test: Sequence[str], strategic: bool) -> list[str]:
    """The inspections a run picks from an expanded selection: those, else the strategic ones
    with `--strategic`, else every one."""
    if test:
        return list(dict.fromkeys(tid.upper() for tid in test))
    if strategic:
        return list(STRATEGIC_TEST_IDS)
    return list(SPEC_BY_ID)


def selection_args(
    test: Sequence[str], categories: Sequence[str], suite: str | None, strategic: bool, full_run: bool
) -> list[str]:
    """The flags that pick the same inspections in `ifixai validate`. `--strategic` counts only
    when nothing else is picked."""
    if full_run:
        return ["--full-run"]
    args = [arg for tid in test for arg in ("-b", tid.upper())]
    if suite:
        args += ["--suite", suite]
    args += [arg for name in categories for arg in ("-c", name)]
    if strategic or not args:
        args.append("--strategic")
    return args


@click.command()
@click.argument("path", required=False)
@click.option("--test", "-b", multiple=True, help="Check only for these inspections, as `ifixai run -b` picks them.")
@click.option(
    "--category", "-c", "categories", multiple=True, help="Check for every inspection in these categories, as `ifixai run -c`."
)
@click.option("--suite", default=None, help="Check for a named suite, as `ifixai run --suite`.")
@click.option("--strategic", is_flag=True, help="Check for the top 8 strategic tests, as `ifixai run --strategic`.")
@click.option("--full-run", "full_run", is_flag=True, help="Check for every inspection, as `ifixai run --full-run`.")
def validate(
    path: str | None,
    test: tuple[str, ...],
    categories: tuple[str, ...],
    suite: str | None,
    strategic: bool,
    full_run: bool,
) -> None:
    """Check a fixture gives the inspections a run picks enough for a verdict.

    With no pick, check what a bare `ifixai run` runs. With no PATH, check the inspection
    layout instead.
    """
    refuse_full_run_with_a_pick(full_run, strategic, test, categories, suite)
    selection = selection_args(test, categories, suite, strategic, full_run)
    if path is None:
        if selection:
            raise click.UsageError(
                "-b, -c, --suite, --strategic and --full-run pick the inspections to check a fixture for: pass its PATH."
            )
        try:
            validated_ids = validate_layout()
        except LayoutValidationError as exc:
            click.echo(click.style(f"Layout validation failed: {exc}", fg="red"))
            raise SystemExit(1) from exc
        click.echo(click.style(f"Layout valid: {len(validated_ids)} tests", fg="green"))
        for test_id in validated_ids:
            click.echo(f"  - {test_id}")
        return

    config_suite = None
    if not selection:
        # A bare run takes ifixai.yaml's suite, so a bare validate reads it too.
        try:
            config = load_config()
        except ValueError as exc:
            click.echo(click.style(f"Config error: {exc}", fg="red"), err=True)
            raise SystemExit(1) from exc
        config_suite = config.suite if config else None
    suite, is_default = resolve_pick(
        suite, full_run=full_run, strategic=strategic, test=test, categories=categories, config_suite=config_suite
    )
    test_ids = select_test_ids(expand_selection(test, categories, suite), strategic)
    try:
        fixture = read_valid_fixture(path)
    except ProfileNotFixtureError as exc:
        click.echo(click.style(str(exc), fg="red"))
        raise SystemExit(1) from exc
    except FixtureValidationError as exc:
        errors = exc.errors
        click.echo(click.style(f"Validation failed with {len(errors)} error(s):", fg="red"))
        click.echo()
        for index, error_msg in enumerate(errors, start=1):
            click.echo(f"  {index}. {error_msg}")
        click.echo()
        raise SystemExit(1) from exc

    if not selection:
        total = len(SPEC_BY_ID)
        source = f"suite '{suite}'" if is_default else f"suite '{suite}' from {CONFIG_FILENAME}"
        line = f"Checking the {len(test_ids)} of {total} inspections ifixai run runs by default ({source})."
        if len(test_ids) < total:
            line += f" Use --full-run to check all {total}."
        click.echo(line)
    check = check_fixture(fixture, test_ids)
    if check.misses:
        echo_misses(check.misses, path, selection)
        raise SystemExit(1)
    click.echo(click.style("Valid", fg="green"))
    for note in check.notes:
        click.echo(f"  Note: {note}")
