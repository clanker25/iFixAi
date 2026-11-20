import copy
import json

import pytest
from click.testing import CliRunner

from ifixai.cli.main import ifixai_cli
from ifixai.trajectory import (
    audit_trajectory,
    canonical_digest,
    create_audit_bundle,
    load_audit_bundle,
)
from ifixai.trajectory.importers import load_jsonl_capture, load_otlp_capture
from ifixai.trajectory.recorder import write_capture_jsonl


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def test_jsonl_and_explicit_otlp_export_preserve_semantic_audit(native_example, tmp_path):
    capture, store, _ = native_example.record_scenario("approval-replay")
    native = audit_trajectory(capture, native_example.demo_policy())
    write_capture_jsonl(capture, tmp_path / "trace.jsonl")
    raw = native_example.otlp_export(capture)
    # Export order is unconstrained; semantic sequence restores it explicitly.
    raw["resourceSpans"][0]["scopeSpans"][0]["spans"].reverse()
    write_json(tmp_path / "trace.otlp.json", raw)
    for imported in (load_jsonl_capture(tmp_path / "trace.jsonl"), load_otlp_capture(tmp_path / "trace.otlp.json")):
        report = audit_trajectory(imported, native_example.demo_policy())
        assert report.decision == native.decision == "BLOCK"
        assert report.violations == native.violations
        assert report.graph_edges == native.graph_edges
        assert report.budget_totals == native.budget_totals
        assert not imported.import_warnings
    assert store.calls == 2 and store.refunded_units == 120


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\x85"])
def test_jsonl_export_round_trips_unicode_line_separators(separator, tmp_path):
    from ifixai.trajectory.recorder import TraceRecorder

    trace = TraceRecorder("tenant", "collector", label=f"line one{separator}line two", clock_ns=lambda: 10)
    capture = trace.write_jsonl(tmp_path / "trace.jsonl")
    imported = load_jsonl_capture(tmp_path / "trace.jsonl")
    assert imported.events == capture.events
    assert imported.events[0].payload.label == f"line one{separator}line two"


@pytest.mark.parametrize("change", [
    lambda d: d["resourceSpans"][0]["scopeSpans"][0]["spans"][1].update(droppedEventsCount=1),
    lambda d: d["resourceSpans"][0]["scopeSpans"][0]["spans"][1].update(droppedLinksCount=1),
    lambda d: d["resourceSpans"][0]["resource"].update(droppedAttributesCount=1),
    lambda d: d["resourceSpans"][0]["scopeSpans"][0]["spans"][1].update(parentSpanId=""),
    lambda d: d["resourceSpans"][0]["scopeSpans"][0]["spans"][1].update(parentSpanId="a" * 16),
    lambda d: d["resourceSpans"][0]["scopeSpans"][0]["spans"][1].update(attributes=[]),
    lambda d: d["resourceSpans"][0]["resource"].update(attributes=[]),
])
def test_otlp_missing_or_dropped_instrumentation_never_clears(native_example, tmp_path, change):
    capture, _, _ = native_example.record_scenario()
    raw = native_example.otlp_export(capture)
    change(raw)
    write_json(tmp_path / "input.json", raw)
    imported = load_otlp_capture(tmp_path / "input.json")
    assert imported.import_warnings
    assert audit_trajectory(imported, native_example.demo_policy()).decision == "INCONCLUSIVE"


@pytest.mark.parametrize("change", [
    lambda s: s.update(spanId="f" * 16),
    lambda s: s.update(endTimeUnixNano="999"),
    lambda s: s.update(endTimeUnixNano=1.25),
    lambda s: s.update(endTimeUnixNano=True),
    lambda s: s.update(startTimeUnixNano="999999"),
    lambda s: s.update(traceId="g" * 32),
])
def test_otlp_binding_and_exact_integer_timestamp_required(native_example, tmp_path, change):
    capture, _, _ = native_example.record_scenario()
    raw = native_example.otlp_export(capture)
    change(raw["resourceSpans"][0]["scopeSpans"][0]["spans"][1])
    write_json(tmp_path / "bad.json", raw)
    with pytest.raises(ValueError):
        load_otlp_capture(tmp_path / "bad.json")


def test_multi_trace_requires_explicit_selection(native_example, tmp_path):
    first, _, _ = native_example.record_scenario()
    second, _, _ = native_example.record_scenario()
    raw = native_example.otlp_export(first)
    raw["resourceSpans"].extend(native_example.otlp_export(second)["resourceSpans"])
    write_json(tmp_path / "multi.json", raw)
    with pytest.raises(ValueError):
        load_otlp_capture(tmp_path / "multi.json")
    capture = load_otlp_capture(tmp_path / "multi.json", trace_id=second.trace_id)
    assert capture.trace_id == second.trace_id
    assert capture.events == second.events
    assert not capture.import_warnings


@pytest.mark.parametrize("value", ["1", True, -1, 6])
def test_otlp_kind_requires_numeric_valid_enum(native_example, tmp_path, value):
    capture, _, _ = native_example.record_scenario()
    raw = native_example.otlp_export(capture)
    raw["resourceSpans"][0]["scopeSpans"][0]["spans"][1]["kind"] = value
    write_json(tmp_path / "bad-kind.json", raw)
    with pytest.raises(ValueError):
        load_otlp_capture(tmp_path / "bad-kind.json")


@pytest.mark.parametrize("missing", ["startTimeUnixNano", "endTimeUnixNano"])
def test_otlp_one_timestamp_only_is_visible_incomplete_coverage(native_example, tmp_path, missing):
    capture, _, _ = native_example.record_scenario()
    raw = native_example.otlp_export(capture)
    raw["resourceSpans"][0]["scopeSpans"][0]["spans"][1].pop(missing)
    write_json(tmp_path / "partial-time.json", raw)
    imported = load_otlp_capture(tmp_path / "partial-time.json")
    assert imported.import_warnings
    assert audit_trajectory(imported, native_example.demo_policy()).decision == "INCONCLUSIVE"


@pytest.mark.parametrize("resign", [False, True])
def test_native_bundle_tamper_detected_even_with_new_checksum(native_example, tmp_path, resign):
    capture, _, _ = native_example.record_scenario()
    bundle = create_audit_bundle(capture, native_example.demo_policy())
    raw = bundle.model_dump(mode="json")
    next(event for event in raw["capture"]["events"] if event["payload"]["kind"] == "action")["payload"]["cost_units"] = 999
    if resign:
        raw["integrity_digest"] = canonical_digest({k: v for k, v in raw.items() if k != "integrity_digest"})
    write_json(tmp_path / "bad.json", raw)
    with pytest.raises(ValueError):
        load_audit_bundle(tmp_path / "bad.json")


def test_cli_audit_replay_whatif_does_not_reexecute_local_store(native_example, tmp_path, monkeypatch):
    capture, store, _ = native_example.record_scenario("budget-overrun")
    write_capture_jsonl(capture, tmp_path / "trace.jsonl")
    policy = native_example.demo_policy()
    write_json(tmp_path / "policy.json", policy.model_dump(mode="json"))
    runner = CliRunner()
    def forbidden(*_args, **_kwargs):
        raise AssertionError("audit executed imported action")
    monkeypatch.setattr(native_example.LocalRefundStore, "refund", forbidden)
    result = runner.invoke(ifixai_cli, ["trajectory", "audit", str(tmp_path / "policy.json"), str(tmp_path / "trace.jsonl"), "--output", str(tmp_path / "audit")])
    assert result.exit_code == 1, (result.output, result.exception)
    original_path = tmp_path / "audit" / "bundle.json"
    original = original_path.read_bytes()
    replay = runner.invoke(ifixai_cli, ["trajectory", "replay", str(original_path), "--output", str(tmp_path / "replay")])
    assert replay.exit_code == 1, (replay.output, replay.exception)
    raw = policy.model_dump(mode="json")
    raw["budgets"][0]["limit_units"] = 200
    write_json(tmp_path / "alternate.json", raw)
    scenario = runner.invoke(ifixai_cli, ["trajectory", "what-if", str(original_path), str(tmp_path / "alternate.json"), "--output", str(tmp_path / "scenario")])
    assert scenario.exit_code == 0, (scenario.output, scenario.exception)
    assert original_path.read_bytes() == original
    assert store.calls == 2 and store.refunded_units == 120
    assert (tmp_path / "scenario" / "report.html").is_file()
    assert original_path.stat().st_mode & 0o077 == 0
    # Existing evidence never overwritten, even for successful replay.
    refused = runner.invoke(ifixai_cli, ["trajectory", "replay", str(original_path), "--output", str(tmp_path / "replay")])
    assert refused.exit_code != 0
    assert original_path.read_bytes() == original


def test_strict_import_duplicate_keys_nonfinite_and_unknown_fields(tmp_path, native_example):
    capture, _, _ = native_example.record_scenario()
    path = tmp_path / "input.jsonl"
    write_capture_jsonl(capture, path)
    lines = path.read_text().splitlines()
    for invalid in ('{"schema_version":"1","schema_version":"1"}', '{"unexpected":NaN}'):
        path.write_text(invalid, encoding="utf-8")
        with pytest.raises(ValueError):
            load_jsonl_capture(path)
    raw = copy.deepcopy(json.loads(lines[1]))
    raw["payload"]["unrecognized"] = "no implicit executable extension"
    path.write_text(lines[0] + "\n" + json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_jsonl_capture(path)
