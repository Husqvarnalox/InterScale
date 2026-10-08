import asyncio
import json

import httpx
import pytest

from conftest import EOS, ScriptedRunner
from inferscale.api.app import create_app
from inferscale.config import ServerConfig


async def make_client(runner, **cfg):
    app = create_app(ServerConfig(**cfg), runner)
    ctx = app.router.lifespan_context(app)
    await ctx.__aenter__()
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")
    return app, client, ctx


@pytest.fixture
async def api():
    # Prompt "hi" ends in 'i'; chat prompts end in '\n' (fallback template). Scripts are
    # keyed by the last prompt token.
    script = [*list(b"Hello!"), EOS]
    runner = ScriptedRunner(scripts={ord("i"): script, ord("\n"): script})
    app, client, ctx = await make_client(runner)
    yield client, app, runner
    await client.aclose()
    await ctx.__aexit__(None, None, None)


async def test_health_and_models(api):
    client, _, _ = api
    assert (await client.get("/health")).json() == {"status": "ok"}
    models = (await client.get("/v1/models")).json()
    assert models["data"][0]["id"] == "scripted"


async def test_completion(api):
    client, _, _ = api
    r = await client.post(
        "/v1/completions", json={"prompt": "hi", "temperature": 0, "max_tokens": 20}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["text"] == "Hello!"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"] == {"prompt_tokens": 3, "completion_tokens": 6, "total_tokens": 9}
    assert r.headers["x-request-id"]


async def test_completion_max_tokens_and_stop(api):
    client, _, _ = api
    r = await client.post(
        "/v1/completions", json={"prompt": "hi", "temperature": 0, "max_tokens": 2}
    )
    assert r.json()["choices"][0]["text"] == "He"
    assert r.json()["choices"][0]["finish_reason"] == "length"
    r = await client.post(
        "/v1/completions", json={"prompt": "hi", "temperature": 0, "stop": "ll", "max_tokens": 20}
    )
    assert r.json()["choices"][0]["text"] == "He"


async def test_chat_completion(api):
    client, _, _ = api
    r = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "temperature": 0},
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"] == {"role": "assistant", "content": "Hello!"}
    assert r.json()["object"] == "chat.completion"


async def sse(client, url, payload):
    chunks, done = [], False
    async with client.stream("POST", url, json=payload) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        async for line in r.aiter_lines():
            if not line:
                continue
            assert line.startswith("data: ")
            data = line[6:]
            if data == "[DONE]":
                done = True
            else:
                chunks.append(json.loads(data))
    return chunks, done


async def test_streaming_completion(api):
    client, _, _ = api
    chunks, done = await sse(
        client,
        "/v1/completions",
        {
            "prompt": "hi",
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    assert done
    text = [c["choices"][0]["text"] for c in chunks if c["choices"]]
    assert "".join(text) == "Hello!" and max(map(len, text)) == 1  # deltas, not accumulations
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["usage"]["completion_tokens"] == 6


async def test_streaming_chat(api):
    client, _, _ = api
    chunks, done = await sse(
        client,
        "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "hi"}], "temperature": 0, "stream": True},
    )
    assert done
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert content == "Hello!"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.parametrize(
    ("url", "payload"),
    [
        ("/v1/completions", {}),
        ("/v1/completions", {"prompt": ""}),
        ("/v1/completions", {"prompt": "x", "temperature": -1}),
        ("/v1/completions", {"prompt": "x", "top_p": 0}),
        ("/v1/completions", {"prompt": "x", "max_tokens": 0}),
        ("/v1/completions", {"prompt": "x", "n": 2}),
        ("/v1/completions", {"prompt": ["a", "b"]}),
        ("/v1/completions", {"prompt": "x", "stop": list("abcde")}),
        ("/v1/chat/completions", {"messages": []}),
        ("/v1/chat/completions", {"messages": [{"role": "robot", "content": "x"}]}),
    ],
)
async def test_validation_errors(api, url, payload):
    client, _, _ = api
    r = await client.post(url, json=payload)
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"


async def test_prompt_too_long_is_400(api):
    client, _, runner = api
    runner.max_model_len = 8
    r = await client.post("/v1/completions", json={"prompt": "x" * 50})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "context_length_exceeded"


async def test_queue_overflow_returns_429():
    runner = ScriptedRunner(scripts={ord("i"): [ord("q")] * 500}, delay=0.05)
    app, client, ctx = await make_client(runner, max_batch_size=1, max_queue_size=1)
    body = {"prompt": "hi", "temperature": 0, "max_tokens": 400}
    try:

        def post() -> asyncio.Task:
            return asyncio.create_task(client.post("/v1/completions", json=body))

        tasks = [post()]
        await asyncio.sleep(0.15)  # first request is now running, queue empty
        tasks += [post(), post()]  # one fills the queue, one is rejected
        await asyncio.sleep(0.3)
        done = [t for t in tasks if t.done()]
        assert len(done) == 1 and done[0].result().status_code == 429
        assert done[0].result().json()["error"]["code"] == "queue_full"
        for t in tasks:
            t.cancel()
    finally:
        app.state.engine.scheduler.drain()
        await ctx.__aexit__(None, None, None)
        await client.aclose()


async def test_engine_failure_is_reported_as_500():
    runner = ScriptedRunner(scripts={ord("i"): [ord("q")] * 5}, fail_on_decode=True)
    _, client, ctx = await make_client(runner)
    r = await client.post("/v1/completions", json={"prompt": "hi", "temperature": 0})
    assert r.status_code == 500 and r.json()["error"]["code"] == "engine_error"
    await client.aclose()
    await ctx.__aexit__(None, None, None)


async def test_metrics_endpoint(api):
    client, _, _ = api
    await client.post("/v1/completions", json={"prompt": "hi", "temperature": 0})
    text = (await client.get("/metrics")).text
    for name in (
        "inferscale_requests_total",
        "inferscale_requests_active",
        "inferscale_queue_depth",
        "inferscale_request_latency_seconds",
        "inferscale_time_to_first_token_seconds",
        "inferscale_time_per_output_token_seconds",
        "inferscale_prompt_tokens_total",
        "inferscale_generated_tokens_total",
        "inferscale_batch_size",
        "inferscale_model_forward_seconds",
    ):
        assert name in text
    assert 'inferscale_requests_total{status="finished"} 1.0' in text


async def test_stream_generator_cancels_request_on_close(api):
    """Closing the SSE generator (what Starlette does on client disconnect) cancels."""
    from inferscale.api.routes import _stream

    _, app, runner = api
    runner.scripts[ord("i")] = [ord("q")] * 500
    runner.delay = 0.01
    engine = app.state.engine
    from inferscale.engine.request import SamplingParams

    gen = engine.submit("hi", [258, 104, 105], SamplingParams(temperature=0, max_tokens=400))
    agen = _stream(engine, gen, lambda ev, first: {"t": ev.text}, None)
    await agen.__anext__()
    await agen.aclose()
    assert gen.state.value == "cancelled"
