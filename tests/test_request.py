import pytest

from inferscale.engine.request import (
    EngineError,
    GenerationRequest,
    InvalidTransitionError,
    RequestState,
    SamplingParams,
    StreamEvent,
)


def make() -> GenerationRequest:
    return GenerationRequest("r1", "hi", [1, 2], SamplingParams())


def test_happy_path():
    r = make()
    assert r.state is RequestState.WAITING
    r.mark_running()
    assert r.started_at is not None
    r.finish("stop")
    assert r.state is RequestState.FINISHED and r.finished_at is not None


def test_cancel_waiting_and_running():
    r = make()
    assert r.cancel() and r.state is RequestState.CANCELLED
    assert not r.cancel()  # idempotent
    r2 = make()
    r2.mark_running()
    assert r2.cancel()


@pytest.mark.parametrize("terminal", ["finish", "cancel", "fail"])
def test_terminal_states_are_final(terminal):
    r = make()
    r.mark_running()
    {"finish": lambda: r.finish("stop"), "cancel": r.cancel, "fail": lambda: r.fail(Exception())}[
        terminal
    ]()
    with pytest.raises(InvalidTransitionError):
        r.mark_running()
    assert not r.cancel() and not r.fail(Exception("late"))


def test_cannot_finish_waiting_request():
    with pytest.raises(InvalidTransitionError):
        make().finish("stop")


async def test_stream_yields_events_then_stops():
    r = make()
    r.mark_running()
    r.emit(StreamEvent("a"))
    r.finish("length", "b")
    events = [e async for e in r.stream()]
    assert [e.text for e in events] == ["a", "b"]
    assert events[-1].finish_reason == "length"
    assert r.ttft is not None


async def test_stream_raises_engine_error_on_failure():
    r = make()
    r.mark_running()
    r.fail(ValueError("bad"))
    with pytest.raises(EngineError, match="bad"):
        await r.result()
