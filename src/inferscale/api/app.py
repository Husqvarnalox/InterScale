"""FastAPI application factory and engine lifecycle."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from inferscale import __version__
from inferscale.api.routes import router
from inferscale.api.schemas import error_body
from inferscale.config import ServerConfig
from inferscale.engine.engine import InferenceEngine
from inferscale.engine.scheduler import Scheduler
from inferscale.model.runner import Runner


def create_app(config: ServerConfig, runner: Runner | None = None) -> FastAPI:
    """Build the app. If ``runner`` is None the model is loaded during startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        nonlocal runner
        if runner is None:
            from inferscale.model.loader import load_runner

            runner = await asyncio.to_thread(load_runner, config)
        engine = InferenceEngine(runner, Scheduler(config.max_batch_size, config.max_queue_size))
        await engine.start()
        app.state.engine = engine
        try:
            yield
        finally:
            await engine.stop()

    app = FastAPI(title="InferScale", version=__version__, lifespan=lifespan)
    app.state.started_at = int(time.time())
    app.include_router(router)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        problems: list[Any] = [
            f"{'.'.join(str(p) for p in e['loc'] if p != 'body')}: {e['msg']}" for e in exc.errors()
        ]
        return JSONResponse(
            error_body("; ".join(problems), "invalid_request_error", "invalid_request"),
            status_code=400,
        )

    return app
