"""Offline trajectory auditing; importing never resolves providers or tools."""
from __future__ import annotations

from collections.abc import Callable

import click
from pydantic import ValidationError

from ifixai.reporting.trajectory import policy_comparison, write_trajectory_reports
from ifixai.trajectory.importers import (
    load_jsonl_capture,
    load_otlp_capture,
    read_input,
    strict_json,
)
from ifixai.trajectory.integrity import create_audit_bundle, load_audit_bundle
from ifixai.trajectory.models import AuditBundle, AuditPolicy


def _policy(path: str) -> AuditPolicy:
    return AuditPolicy.model_validate(strict_json(read_input(path)))


def _execute(operation: Callable[[], AuditBundle], output: str, original: AuditBundle | None = None) -> None:
    try:
        bundle = operation()
        comparison = policy_comparison(original, bundle) if original else None
        destination = write_trajectory_reports(bundle, output, comparison=comparison)
    except ValidationError as error:
        # Do not print Pydantic's input_value, imported JSON or model response text.
        count = error.error_count()
        raise click.ClickException(f"Invalid trajectory schema ({count} field errors); inspect the schema documentation.") from None
    except (ValueError, TypeError, OSError, UnicodeError, RecursionError) as error:
        raise click.ClickException(f"Trajectory operation failed ({type(error).__name__}); check input format, integrity and new output directory.") from None
    click.echo(f"{bundle.report.decision}: {len(bundle.report.violations)} violations, {len(bundle.report.coverage_issues)} coverage issues")
    click.echo(f"Evidence: {destination / 'report.html'}")
    raise click.exceptions.Exit({"CLEAR": 0, "BLOCK": 1, "INCONCLUSIVE": 2}[bundle.report.decision])


@click.group()
def trajectory() -> None:
    """Audit and replay recorded agent actions offline."""


@trajectory.command("audit")
@click.argument("policy", type=click.Path(exists=True, dir_okay=False))
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--format", "input_format", type=click.Choice(["jsonl", "otlp"]), default="jsonl", show_default=True)
@click.option("--trace-id", help="Select one trace from a multi-trace OTLP export.")
@click.option("--output", required=True, type=click.Path(file_okay=False), help="New owner-only evidence directory; never overwrite.")
def audit(policy: str, input_path: str, input_format: str, trace_id: str | None, output: str) -> None:
    def operation() -> AuditBundle:
        if input_format == "jsonl" and trace_id is not None:
            raise ValueError("trace selection applies only to OTLP")
        capture = load_jsonl_capture(input_path) if input_format == "jsonl" else load_otlp_capture(input_path, trace_id=trace_id)
        return create_audit_bundle(capture, _policy(policy))

    _execute(operation, output)


@trajectory.command("replay")
@click.argument("bundle", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", required=True, type=click.Path(file_okay=False))
def replay(bundle: str, output: str) -> None:
    _execute(lambda: load_audit_bundle(bundle), output)


@trajectory.command("what-if")
@click.argument("bundle", type=click.Path(exists=True, dir_okay=False))
@click.argument("policy", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", required=True, type=click.Path(file_okay=False))
def what_if(bundle: str, policy: str, output: str) -> None:
    """Re-evaluate the same facts under another policy without executing actions."""
    try:
        original = load_audit_bundle(bundle)
        scenario = create_audit_bundle(original.capture, _policy(policy))
    except (ValidationError, ValueError, TypeError, OSError, UnicodeError, RecursionError):
        raise click.ClickException("Cannot replay the retained bundle or validate the alternate policy.") from None
    _execute(lambda: scenario, output, original)
