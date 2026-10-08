"""Scheduler: admission, FIFO ordering, batch formation, cancellation.

Pure synchronous bookkeeping, only ever touched from the event-loop thread, so
it needs no locks. It never touches the model.
"""

from __future__ import annotations

from collections import deque

from inferscale.engine.request import GenerationRequest


class QueueFullError(RuntimeError):
    """Admission rejected: the waiting queue is at ``max_queue_size``."""


class Scheduler:
    def __init__(self, max_batch_size: int, max_queue_size: int) -> None:
        if max_batch_size < 1 or max_queue_size < 1:
            raise ValueError("max_batch_size and max_queue_size must be >= 1")
        self.max_batch_size = max_batch_size
        self.max_queue_size = max_queue_size
        self._waiting: deque[GenerationRequest] = deque()

    @property
    def queue_depth(self) -> int:
        return len(self._waiting)

    def add(self, request: GenerationRequest) -> None:
        if len(self._waiting) >= self.max_queue_size:
            raise QueueFullError(f"queue is full ({self.max_queue_size} waiting requests)")
        self._waiting.append(request)

    def remove(self, request_id: str) -> GenerationRequest | None:
        """Remove a waiting request (used for cancellation before it starts)."""
        for req in self._waiting:
            if req.request_id == request_id:
                self._waiting.remove(req)
                return req
        return None

    def next_batch(self) -> list[GenerationRequest]:
        """Pop up to ``max_batch_size`` live requests in arrival order."""
        batch: list[GenerationRequest] = []
        while self._waiting and len(batch) < self.max_batch_size:
            req = self._waiting.popleft()
            if req.state.is_terminal:  # cancelled while queued
                continue
            batch.append(req)
        return batch

    def drain(self) -> list[GenerationRequest]:
        """Remove and return everything still waiting (shutdown)."""
        drained = list(self._waiting)
        self._waiting.clear()
        return drained
