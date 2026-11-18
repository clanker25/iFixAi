import json
import shlex
from pathlib import Path
from typing import NoReturn

import click
import yaml
from pydantic import ValidationError

from ifixai.core.profile import AgentProfile, expand_profile, profile_errors

REFERENCE_PROFILE = Path(__file__).resolve().parent.parent / "fixtures" / "reference" / "profile.json"
# Where `--example` writes the reference profile for someone to edit, and where the fixture goes,
# when `-o` doesn't say.
EXAMPLE_OUTPUT = "profile.json"
FIXTURE_OUTPUT = "fixture.yaml"
# The fixture's path in the suggested next command when the profile went to FIXTURE_OUTPUT.
FIXTURE_ELSEWHERE = "my-fixture.yaml"


def _refuse(lines: list[str]) -> NoReturn:
    click.echo(click.style("Profile refused. Fix these, then run ifixai author again:", fg="red"), err=True)
    for line in lines:
        click.echo(f"  {line}", err=True)
    raise SystemExit(2)


def _write_new(path: str, text: str, force: bool) -> None:
    """Write `text` to `path`, never over an existing file unless `force`."""
    try:
        with Path(path).open("w" if force else "x", encoding="utf-8") as handle:
            handle.write(text)
    except FileExistsError as err:
        raise click.ClickException(f"{path} already exists. Pass --force to replace it.") from err
    except OSError as err:
        raise click.ClickException(f"can't write {path} ({err.strerror})") from err


@click.command()
@click.argument("profile", required=False)
@click.option(
    "-o",
    "--output",
    default=None,
    help=f"Where to write the fixture (default {FIXTURE_OUTPUT}), or with --example the profile (default {EXAMPLE_OUTPUT}).",
)
@click.option("--force", is_flag=True, help="Replace the output file if it already exists.")
@click.option(
    "--example",
    is_flag=True,
    help=f"Write the example profile, a support agent at a fictional fintech, to edit (to {EXAMPLE_OUTPUT} unless -o says).",
)
def author(profile: str | None, output: str | None, force: bool, example: bool) -> None:
    """Turn an agent profile (JSON) into a fixture. `reference` is the built-in example profile."""
    if example:
        if profile is not None:
            raise click.UsageError("--example writes the example profile and takes no PROFILE.")
        target = output or EXAMPLE_OUTPUT
        _write_new(target, REFERENCE_PROFILE.read_text(encoding="utf-8"), force)
        # The next command must not write the fixture over the profile it reads.
        fixture = FIXTURE_OUTPUT
        if Path(fixture).exists() and Path(fixture).samefile(target):
            fixture = FIXTURE_ELSEWHERE
        click.echo(f"Wrote {target}, an example to edit. Next: ifixai author {shlex.quote(target)} -o {fixture}")
        return
    if profile is None:
        raise click.UsageError("Missing argument 'PROFILE'. Pass --example to get a profile to edit.")
    path = REFERENCE_PROFILE if profile == "reference" and not Path(profile).exists() else Path(profile)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _refuse([f"{profile}: no such file"])
    except OSError as err:
        _refuse([f"{profile}: can't read it ({err.strerror})"])
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        _refuse([f"{profile}: not valid JSON ({err})"])
    if not isinstance(raw, dict):
        _refuse([f"{profile}: should be one JSON object with the profile's fields"])
    try:
        parsed = AgentProfile.model_validate(raw)
    except ValidationError as err:
        _refuse(profile_errors(err, raw))
    target = output or FIXTURE_OUTPUT
    _write_new(target, yaml.safe_dump(expand_profile(parsed), sort_keys=False, allow_unicode=True), force)
    click.echo(f"Wrote {target}. Next: ifixai validate {target}")
