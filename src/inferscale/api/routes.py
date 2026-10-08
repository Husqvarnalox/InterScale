"""HTTP routes. Handlers tokenize, submit to the engine, and format output.

They never call the model: all inference happens in the engine's background loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST

from inferscale.api.schemas import (
    ChatCompletionRequest,
    CompletionRequest,
    error_body,
    make_usage,
    new_id,
    now,
)
from inferscale.engine.engine import EngineClosedError, InferenceEngine, PromptTooLongError
from inferscale.engine.request import EngineError, GenerationRequest, SamplingParams, StreamEvent
from inferscale.engine.scheduler import QueueFullError
from inferscale.model.chat import ChatMessage

DEFAULT_COMPLETION_MAX_TOKENS = 16
DEFAULT_CHAT_MAX_TOKENS = 256

router = APIRouter()


def _engine(request: Request) -> InferenceEngine:
    engine: InferenceEngine = request.app.state.engine
    return engine


def _error(status: int, message: str, type_: str, code: str | None = None) -> JSONResponse:
    return JSONResponse(error_body(message, type_, code), status_code=status)


def _submit(
    engine: InferenceEngine, prompt: str, add_special_tokens: bool, params: SamplingParams
) -> GenerationRequest | JSONResponse:
    ids = engine.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
    try:
        return engine.submit(prompt, ids, params)
    except QueueFullError as exc:
        return _error(429, str(exc), "rate_limit_error", "queue_full")
    except PromptTooLongError as exc:
        return _error(400, str(exc), "invalid_request_error", "context_length_exceeded")
    except ValueError as exc:
        return _error(400, str(exc), "invalid_request_error")
    except EngineClosedError as exc:
        return _error(503, str(exc), "server_error", "engine_unavailable")


async def _await_or_cancel(http: Request, engine: InferenceEngine, gen: GenerationRequest) -> Any:
    """Await the result; cancel the request if the client disconnects meanwhile."""
    task = asyncio.ensure_future(gen.result())
    try:
        while not task.done():
            await asyncio.wait({task}, timeout=0.5)
            if not task.done() and await http.is_disconnected():
                engine.cancel(gen.request_id)
                task.cancel()
                return None
        return task.result()
    except asyncio.CancelledError:
        engine.cancel(gen.request_id)
        task.cancel()
        raise
    finally:
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError, EngineError):
            await task


def _sse(payload: dict[str, Any] | str) -> bytes:
    body = payload if isinstance(payload, str) else json.dumps(payload, separators=(",", ":"))
    return f"data: {body}\n\n".encode()


async def _stream(
    engine: InferenceEngine,
    gen: GenerationRequest,
    chunk: Callable[[StreamEvent, bool], dict[str, Any]],
    usage_chunk: Callable[[dict[str, int]], dict[str, Any]] | None,
) -> AsyncIterator[bytes]:
    """SSE generator. ``finally`` fires on client disconnect and cancels the request."""
    first = True
    try:
        async for event in gen.stream():
            yield _sse(chunk(event, first))
            first = False
        if usage_chunk is not None:
            usage = make_usage(len(gen.prompt_token_ids), len(gen.generated_token_ids))
            yield _sse(usage_chunk(usage))
    except EngineError as exc:
        yield _sse(error_body(str(exc), "server_error", "engine_error"))
    finally:
        engine.cancel(gen.request_id)  # no-op if already finished
    yield _sse("[DONE]")


def _sse_response(
    engine: InferenceEngine,
    gen: GenerationRequest,
    chunk: Callable[[StreamEvent, bool], dict[str, Any]],
    usage_chunk: Callable[[dict[str, int]], dict[str, Any]] | None,
) -> StreamingResponse:
    headers = {
        "X-Request-Id": gen.request_id,
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        _stream(engine, gen, chunk, usage_chunk), media_type="text/event-stream", headers=headers
    )


@router.get("/health")
async def health(request: Request) -> Response:
    if not _engine(request).is_running:
        return JSONResponse({"status": "unavailable"}, status_code=503)
    return JSONResponse({"status": "ok"})


@router.get("/metrics")
async def metrics(request: Request) -> Response:
    return Response(_engine(request).metrics.render(), media_type=CONTENT_TYPE_LATEST)


@router.get("/v1/models")
async def list_models(request: Request) -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": _engine(request).model_name,
                "object": "model",
                "created": request.app.state.started_at,
                "owned_by": "inferscale",
            }
        ],
    }


@router.post("/v1/completions", response_model=None)
async def completions(body: CompletionRequest, request: Request) -> Response:
    engine = _engine(request)
    params = body.to_sampling_params(DEFAULT_COMPLETION_MAX_TOKENS)
    submitted = _submit(engine, str(body.prompt), True, params)
    if isinstance(submitted, JSONResponse):
        return submitted
    gen = submitted
    cid, created, model = new_id("cmpl"), now(), engine.model_name

    def base(choices: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "id": cid,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": choices,
        }

    if body.stream:
        include_usage = bool(body.stream_options and body.stream_options.include_usage)

        def chunk(ev: StreamEvent, _first: bool) -> dict[str, Any]:
            return base([{"index": 0, "text": ev.text, "finish_reason": ev.finish_reason}])

        def usage_chunk(usage: dict[str, int]) -> dict[str, Any]:
            return {**base([]), "usage": usage}

        return _sse_response(engine, gen, chunk, usage_chunk if include_usage else None)

    try:
        result = await _await_or_cancel(request, engine, gen)
    except EngineError as exc:
        return _error(500, str(exc), "server_error", "engine_error")
    if result is None:
        return _error(499, "client closed request", "invalid_request_error", "cancelled")
    payload = base([{"index": 0, "text": result.text, "finish_reason": result.finish_reason}])
    payload["usage"] = make_usage(result.prompt_tokens, result.completion_tokens)
    return JSONResponse(payload, headers={"X-Request-Id": gen.request_id})


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(body: ChatCompletionRequest, request: Request) -> Response:
    engine = _engine(request)
    params = body.to_sampling_params(DEFAULT_CHAT_MAX_TOKENS)
    messages = [ChatMessage(role=m.role, content=m.content) for m in body.messages]
    try:
        prompt = engine.tokenizer.format_chat(messages)
    except Exception as exc:
        return _error(400, f"could not apply chat template: {exc}", "invalid_request_error")
    # Chat templates already contain any special tokens (e.g. BOS).
    submitted = _submit(engine, prompt, False, params)
    if isinstance(submitted, JSONResponse):
        return submitted
    gen = submitted
    cid, created, model = new_id("chatcmpl"), now(), engine.model_name

    def base(choices: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": choices,
        }

    if body.stream:
        include_usage = bool(body.stream_options and body.stream_options.include_usage)

        def chunk(ev: StreamEvent, first: bool) -> dict[str, Any]:
            delta: dict[str, Any] = {"content": ev.text} if ev.text else {}
            if first:
                delta = {"role": "assistant", **delta}
            return base([{"index": 0, "delta": delta, "finish_reason": ev.finish_reason}])

        def usage_chunk(usage: dict[str, int]) -> dict[str, Any]:
            return {**base([]), "usage": usage}

        return _sse_response(engine, gen, chunk, usage_chunk if include_usage else None)

    try:
        result = await _await_or_cancel(request, engine, gen)
    except EngineError as exc:
        return _error(500, str(exc), "server_error", "engine_error")
    if result is None:
        return _error(499, "client closed request", "invalid_request_error", "cancelled")
    return JSONResponse(
        {
            "id": cid,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.text},
                    "finish_reason": result.finish_reason,
                }
            ],
            "usage": make_usage(result.prompt_tokens, result.completion_tokens),
        },
        headers={"X-Request-Id": gen.request_id},
    )
