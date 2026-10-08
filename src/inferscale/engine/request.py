"""Request objects: the public lifecycle of one generation call."""

from __future__ import annotations

import asyncio
import enum
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field


class RequestState(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in (RequestState.FINISHED, RequestState.CANCELLED, RequestState.FAILED)


_ALLOWED: dict[RequestState, frozenset[RequestState]] = {
    RequestState.WAITING: frozenset(
        {RequestState.RUNNING, RequestState.CANCELLED, RequestState.FAILED}
    ),
    RequestState.RUNNING: frozenset(
        {RequestState.FINISHED, RequestState.CANCELLED, RequestState.FAILED}
    ),
    RequestState.FINISHED: frozenset(),
    RequestState.CANCELLED: frozenset(),
    RequestState.FAILED: frozenset(),
}


class InvalidTransitionError(RuntimeError):
    pass


class EngineError(RuntimeError):
    """A request failed inside the engine (model error, shutdown, ...)."""


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0  # 0 disables
    max_tokens: int = 16
    stop: tuple[str, ...] = ()
    seed: int | None = None


@dataclass(frozen=True)
class StreamEvent:
    """Incremental output. ``finish_reason`` is set only on the last event."""

    text: str = ""
    finish_reason: str | None = None


@dataclass(frozen=True)
class GenerationResult:
    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int


@dataclass
class _Failure:
    error: Exception


@dataclass
class GenerationRequest:
    request_id: str
    prompt: str
    prompt_token_ids: list[int]
    params: SamplingParams
    created_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    first_token_at: float | None = None
    finished_at: float | None = None
    state: RequestState = RequestState.WAITING
    finish_reason: str | None = None
    error: Exception | None = None
    generated_token_ids: list[int] = field(default_factory=list)
    _events: asyncio.Queue[StreamEvent | _Failure] = field(default_factory=asyncio.Queue)

    # ---- lifecycle -------------------------------------------------------
    def _transition(self, new: RequestState) -> None:
        if new not in _ALLOWED[self.state]:
            raise InvalidTransitionError(f"{self.state.value} -> {new.value}")
        self.state = new

    @property
    def cancel_requested(self) -> bool:
        return self.state is RequestState.CANCELLED

    def mark_running(self) -> None:
        self._transition(RequestState.RUNNING)
        self.started_at = time.monotonic()

    def emit(self, event: StreamEvent) -> None:
        if self.first_token_at is None and event.text:
            self.first_token_at = time.monotonic()
        self._events.put_nowait(event)

    def finish(self, reason: str, text: str = "") -> None:
        self._transition(RequestState.FINISHED)
        self.finish_reason = reason
        self.finished_at = time.monotonic()
        self.emit(StreamEvent(text=text, finish_reason=reason))

    def cancel(self) -> bool:
        """Cancel if not yet terminal. Returns True if the state changed."""
        if self.state.is_terminal:
            return False
        self._transition(RequestState.CANCELLED)
        self.finish_reason = "cancelled"
        self.finished_at = time.monotonic()
        self._events.put_nowait(StreamEvent(finish_reason="cancelled"))
        return True

    def fail(self, error: Exception) -> bool:
        if self.state.is_terminal:
            return False
        self._transition(RequestState.FAILED)
        self.error = error
        self.finish_reason = "error"
        self.finished_at = time.monotonic()
        self._events.put_nowait(_Failure(error))
        return True

    # ---- derived metrics -------------------------------------------------
    @property
    def latency(self) -> float | None:
        return None if self.finished_at is None else self.finished_at - self.created_at

    @property
    def ttft(self) -> float | None:
        return None if self.first_token_at is None else self.first_token_at - self.created_at

    # ---- consumption -----------------------------------------------------
    async def stream(self) -> AsyncIterator[StreamEvent]:
        """Yield events until the terminal one; re-raise engine failures."""
        while True:
            item = await self._events.get()
            if isinstance(item, _Failure):
                raise EngineError(str(item.error)) from item.error
            yield item
            if item.finish_reason is not None:
                return

    async def result(self) -> GenerationResult:
        parts: list[str] = []
        reason = "stop"
        async for event in self.stream():
            parts.append(event.text)
            if event.finish_reason is not None:
                reason = event.finish_reason
        return GenerationResult(
            text="".join(parts),
            finish_reason=reason,
            prompt_tokens=len(self.prompt_token_ids),
            completion_tokens=len(self.generated_token_ids),
        )
