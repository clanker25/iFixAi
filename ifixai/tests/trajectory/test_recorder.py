import json
import threading

import pytest
from pydantic import ValidationError

from ifixai.trajectory.integrity import canonical_digest
from ifixai.trajectory.models import ActionPayload, SourcePayload
from ifixai.trajectory.recorder import TraceRecorder, write_capture_jsonl


def recorder():
    return TraceRecorder("tenant", "collector", clock_ns=lambda: 10)


def test_real_callable_executes_once_returns_actual_value_without_raw_args():
    observed = []
    trace = recorder()
    result = trace.invoke(lambda x: observed.append(x) or x + 1, 41, actor_id="actor", action="increment", resource="counter:one", arguments={"value": 41}, cost_units=7)
    capture = trace.seal()
    action = capture.events[1].payload
    assert observed == [41] and result == 42
    assert isinstance(action, ActionPayload)
    assert action.arguments_digest == canonical_digest({"value": 41})
    assert action.outcome == "completed" and action.cost_units == 7
    assert "value" not in capture.model_dump_json()
    assert capture.events[-1].payload.declared_events == 3
    assert capture.capture_complete
    assert all(len(event.event_id) == 16 for event in capture.events)
    assert len(capture.trace_id) == 32


def test_original_exception_and_side_effect_preserved_capture_is_incomplete():
    state = []
    original = LookupError("application error")
    def function():
        state.append("real side effect")
        raise original
    trace = recorder()
    with pytest.raises(LookupError) as raised:
        trace.invoke(function, actor_id="actor", action="write", resource="local:store", arguments={})
    assert raised.value is original
    assert state == ["real side effect"]
    capture = trace.capture(complete=True)
    assert not capture.capture_complete
    assert not capture.events[-1].payload.complete
    assert capture.events[1].payload.outcome == "failed"


def test_pending_call_cannot_be_silently_sealed_complete():
    entered, release = threading.Event(), threading.Event()
    trace = recorder()
    def function():
        entered.set()
        assert release.wait(5)
        return "finished"
    thread = threading.Thread(target=lambda: trace.invoke(function, actor_id="actor", action="wait", resource="local:one", arguments={}))
    thread.start()
    assert entered.wait(5)
    with pytest.raises(RuntimeError):
        trace.seal()
    release.set()
    thread.join(5)
    assert not thread.is_alive()
    assert trace.seal().capture_complete


def test_threaded_observation_has_unique_contiguous_events():
    trace = recorder()
    threads = [threading.Thread(target=lambda: trace.invoke(lambda: 1, actor_id="actor", action="read", resource="local:one", arguments={})) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()
    capture = trace.seal()
    assert [event.sequence for event in capture.events] == list(range(1, 23))
    assert len({event.event_id for event in capture.events}) == 22
    assert sum(event.payload.kind == "action" for event in capture.events) == 20


@pytest.mark.parametrize("cost", [True, -1, 1.2])
def test_invalid_observed_cost_rejected_before_callable(cost):
    calls = []
    with pytest.raises(ValidationError):
        recorder().invoke(lambda: calls.append(1), actor_id="actor", action="write", resource="local:one", arguments={}, cost_units=cost)
    assert not calls


@pytest.mark.parametrize("arguments", [{"nested": {True: "ambiguous key"}}, {"nested": {1: "coerced JSON key"}}, {"number": float("nan")}])
def test_noncanonical_argument_declarations_rejected_before_callable(arguments):
    calls = []
    with pytest.raises((TypeError, ValueError)):
        recorder().invoke(lambda: calls.append(1), actor_id="actor", action="write", resource="local:one", arguments=arguments)
    assert not calls


def test_sealed_recorder_refuses_future_calls_and_overwriting(tmp_path):
    trace = recorder()
    path = tmp_path / "trace.jsonl"
    capture = trace.write_jsonl(path)
    original = path.read_bytes()
    assert path.stat().st_mode & 0o077 == 0
    rows = [json.loads(line) for line in original.splitlines()]
    assert rows[0]["source_format"] == "jsonl"
    assert len(rows) == len(capture.events) + 1
    with pytest.raises(FileExistsError):
        write_capture_jsonl(capture, path)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(RuntimeError):
        trace.seal()
    calls = []
    with pytest.raises(RuntimeError):
        trace.invoke(lambda: calls.append(1), actor_id="actor", action="write", resource="local:one", arguments={})
    assert not calls


@pytest.mark.parametrize("changes", [
    {"source_format": "otlp", "import_warnings": ("dropped observed events",)},
    {"source_format": "jsonl"}, {"import_warnings": ("instrumentation missing",)},
])
def test_recorder_export_never_launders_import_coverage_warnings(tmp_path, changes):
    capture = recorder().seal().model_copy(update=changes)
    path = tmp_path / "laundered.jsonl"
    with pytest.raises(ValueError):
        write_capture_jsonl(capture, path)
    assert not path.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_async_callable_rejected_without_creating_unawaited_coroutine():
    async def function():
        raise AssertionError("async callable executed")
    with pytest.raises(TypeError):
        recorder().invoke(function, actor_id="actor", action="read", resource="local:one", arguments={})


def test_async_callable_object_and_generator_rejected_before_execution():
    class AsyncCallable:
        async def __call__(self):
            raise AssertionError("async object executed")
    def generator():
        yield "not executed until iteration"
    for function in (AsyncCallable(), generator):
        with pytest.raises(TypeError):
            recorder().invoke(function, actor_id="actor", action="read", resource="local:one", arguments={})


def test_sync_function_returning_coroutine_cannot_be_recorded_completed():
    async def work():
        return "not executed"
    trace = recorder()
    with pytest.raises(TypeError):
        trace.invoke(lambda: work(), actor_id="actor", action="read", resource="local:one", arguments={})
    capture = trace.seal()
    assert not capture.capture_complete
    assert capture.events[1].payload.outcome == "failed"


def test_sync_wrapper_returning_generator_or_async_generator_never_completed():
    def deferred():
        yield "pending work"
    async def async_deferred():
        yield "pending asynchronous work"
    for function in (deferred, async_deferred):
        trace = recorder()
        with pytest.raises(TypeError):
            trace.invoke(lambda: function(), actor_id="actor", action="read", resource="local:one", arguments={})
        capture = trace.seal()
        assert not capture.capture_complete
        assert capture.events[1].payload.outcome == "failed"


def test_clock_regression_is_clamped_and_invalid_clock_fails():
    clock = iter([10, 9, 8])
    trace = TraceRecorder("tenant", "collector", clock_ns=lambda: next(clock))
    trace.emit(SourcePayload(actor_id="a", resource="source:one", content_digest="a" * 64))
    assert [event.time_ns for event in trace.seal().events] == [10, 10, 10]
    with pytest.raises(ValueError):
        TraceRecorder("tenant", "collector", clock_ns=lambda: True)


def test_event_bound_rejected_before_callable_executes():
    trace = recorder()
    source = SourcePayload(actor_id="a", resource="source:one", content_digest="a" * 64)
    while len(trace._events) < 99999:
        trace.emit(source)
    calls = []
    with pytest.raises(ValueError, match="event bound"):
        trace.invoke(lambda: calls.append("side effect"), actor_id="actor", action="write", resource="local:one", arguments={})
    assert calls == []
    assert len(trace._events) == 99999
