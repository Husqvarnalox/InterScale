import asyncio

import pytest

from conftest import EOS, ScriptedRunner, make_engine
from inferscale.engine.engine import EngineClosedError, PromptTooLongError
from inferscale.engine.request import EngineError, RequestState, SamplingParams
from inferscale.engine.scheduler import QueueFullError

GREEDY = SamplingParams(temperature=0, max_tokens=50)


def submit(engine, first: int, params=GREEDY):
    return engine.submit("p", [first], params)


@pytest.fixture
async def scripted():
    runner = ScriptedRunner(scripts={1: [104, 105, EOS], 2: list(b"abcdefgh"), 3: [ord("q")] * 100})
    engine = make_engine(runner)
    await engine.start()
    yield engine, runner
    await engine.stop()


async def test_generation_stops_at_eos(scripted):
    engine, _ = scripted
    result = await submit(engine, 1).result()
    assert (result.text, result.finish_reason) == ("hi", "stop")
    assert result.completion_tokens == 2 and result.prompt_tokens == 1


async def test_max_tokens_limit(scripted):
    engine, _ = scripted
    result = await submit(engine, 3, SamplingParams(temperature=0, max_tokens=5)).result()
    assert (result.text, result.finish_reason) == ("qqqqq", "length")


async def test_max_tokens_clamped_to_model_len(scripted):
    engine, runner = scripted
    runner.max_model_len = 4
    result = await engine.submit(
        "p", [3, 3, 3], SamplingParams(temperature=0, max_tokens=99)
    ).result()
    assert result.completion_tokens == 1 and result.finish_reason == "length"


async def test_prompt_too_long(scripted):
    engine, runner = scripted
    with pytest.raises(PromptTooLongError):
        engine.submit("p", [1] * runner.max_model_len, GREEDY)


async def test_streaming_yields_incremental_deltas(scripted):
    engine, _ = scripted
    deltas = [e.text async for e in submit(engine, 2).stream() if e.text]
    assert deltas[:3] == ["a", "b", "c"]  # new text per event, never accumulated


async def test_concurrent_requests_are_batched_and_isolated(scripted):
    engine, runner = scripted
    a, b = submit(engine, 1), submit(engine, 2)
    ra, rb = await asyncio.gather(a.result(), b.result())
    assert ra.text == "hi"
    assert rb.text.startswith("abcdefgh")
    assert runner.batch_sizes[0] == 2


async def test_batch_limit_forms_multiple_batches():
    runner = ScriptedRunner(scripts={1: [ord("a"), EOS]})
    engine = make_engine(runner, batch=2, queue=10)
    await engine.start()
    reqs = [submit(engine, 1) for _ in range(5)]
    await asyncio.gather(*(r.result() for r in reqs))
    assert runner.batch_sizes == [2, 2, 1]
    await engine.stop()


async def test_queue_full_backpressure():
    runner = ScriptedRunner(scripts={3: [ord("q")] * 100}, delay=0.05)
    engine = make_engine(runner, batch=1, queue=1)
    await engine.start()
    first = submit(engine, 3)
    await asyncio.sleep(0.02)  # first is now running; queue is empty
    second = submit(engine, 3)
    with pytest.raises(QueueFullError):
        submit(engine, 3)
    assert 'status="rejected"' in engine.metrics.render().decode()
    engine.cancel(first.request_id)
    engine.cancel(second.request_id)
    await engine.stop()


async def test_cancel_running_request_frees_batch_row():
    runner = ScriptedRunner(scripts={3: [ord("q")] * 1000, 1: [ord("a"), EOS]}, delay=0.01)
    engine = make_engine(runner)
    await engine.start()
    long, short = (
        submit(engine, 3, SamplingParams(temperature=0, max_tokens=1000)),
        submit(engine, 1),
    )
    await asyncio.sleep(0.05)
    assert engine.cancel(long.request_id)
    assert long.state is RequestState.CANCELLED
    assert (await short.result()).text == "a"
    await asyncio.sleep(0.1)  # loop notices and stops decoding
    assert engine.metrics.requests_active._value.get() == 0
    assert len(long.generated_token_ids) < 1000
    assert not engine.cancel(long.request_id)
    await engine.stop()


async def test_cancel_waiting_request():
    runner = ScriptedRunner(scripts={3: [ord("q")] * 100}, delay=0.02)
    engine = make_engine(runner, batch=1)
    await engine.start()
    running, waiting = submit(engine, 3), submit(engine, 3)
    await asyncio.sleep(0.03)
    assert engine.cancel(waiting.request_id)
    assert engine.scheduler.queue_depth == 0
    engine.cancel(running.request_id)
    await engine.stop()


async def test_model_failure_fails_batch_but_engine_survives():
    runner = ScriptedRunner(scripts={1: [ord("a")] * 10}, fail_on_decode=True)
    engine = make_engine(runner)
    await engine.start()
    with pytest.raises(EngineError, match="inference failed") as info:
        await submit(engine, 1).result()
    assert "boom" not in str(info.value)  # internal detail stays in the logs
    runner.fail_on_decode = False
    runner.scripts[1] = [ord("o"), EOS]
    assert (await submit(engine, 1).result()).text == "o"
    await engine.stop()


async def test_submit_after_stop_rejected_and_inflight_failed():
    runner = ScriptedRunner(scripts={3: [ord("q")] * 100}, delay=0.02)
    engine = make_engine(runner)
    await engine.start()
    req = submit(engine, 3)
    await asyncio.sleep(0.03)
    await engine.stop()
    assert req.state is RequestState.FAILED
    with pytest.raises(EngineClosedError):
        submit(engine, 3)


async def test_requests_are_forgotten_after_completion(scripted):
    engine, _ = scripted
    await submit(engine, 1).result()
    assert engine._requests == {}


async def test_stop_string_end_to_end(scripted):
    engine, _ = scripted
    result = await submit(
        engine, 2, SamplingParams(temperature=0, max_tokens=50, stop=("de",))
    ).result()
    assert (result.text, result.finish_reason) == ("abc", "stop")


async def test_real_tiny_model_generates(tiny_engine):
    ids = tiny_engine.tokenizer.encode("hello")
    params = SamplingParams(temperature=0.8, seed=1, max_tokens=12)
    one = await tiny_engine.submit("hello", ids, params).result()
    two = await tiny_engine.submit("hello", ids, params).result()
    assert one.completion_tokens > 0 and one.text == two.text  # seeded => reproducible
    m = tiny_engine.metrics.render().decode()
    assert "inferscale_generated_tokens_total" in m and 'phase="decode"' in m
