"""InferenceEngine: owns the background decode loop, independent of HTTP handlers.

Threading model
---------------
* All request/scheduler state is mutated on the asyncio event-loop thread only.
* Model forward + sampling run on a single dedicated worker thread, one call at
  a time, so the loop stays responsive (streaming, new admissions, cancellation)
  and the model/KV cache are never touched concurrently.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch

from inferscale.engine.request import (
    EngineError,
    GenerationRequest,
    RequestState,
    SamplingParams,
    StreamEvent,
)
from inferscale.engine.sampler import make_generator, sample_batch
from inferscale.engine.scheduler import QueueFullError, Scheduler
from inferscale.engine.sequence import Sequence
from inferscale.logging import log_event
from inferscale.metrics.prometheus import EngineMetrics
from inferscale.model.runner import BatchState, Runner
from inferscale.model.tokenizer import TokenizerLike

logger = logging.getLogger("inferscale.engine")


class PromptTooLongError(ValueError):
    pass


class EngineClosedError(RuntimeError):
    pass


class InferenceEngine:
    def __init__(
        self,
        runner: Runner,
        scheduler: Scheduler,
        metrics: EngineMetrics | None = None,
    ) -> None:
        self.runner = runner
        self.scheduler = scheduler
        self.metrics = metrics or EngineMetrics()
        self._requests: dict[str, GenerationRequest] = {}
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inferscale-model")

    # ---- public API ------------------------------------------------------
    @property
    def tokenizer(self) -> TokenizerLike:
        return self.runner.tokenizer

    @property
    def model_name(self) -> str:
        return self.runner.name

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="inferscale-engine-loop")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        for req in list(self._requests.values()):
            if req.fail(EngineError("engine shut down")):
                self._on_terminal(req, "failed")
        self.scheduler.drain()
        self._sync_gauges(0)
        self._executor.shutdown(wait=False, cancel_futures=True)

    def submit(
        self,
        prompt: str,
        prompt_token_ids: list[int],
        params: SamplingParams,
        request_id: str | None = None,
    ) -> GenerationRequest:
        """Admit a request or raise ``QueueFullError`` / ``PromptTooLongError``."""
        if not self.is_running:
            raise EngineClosedError("engine is not running")
        limit = self.runner.max_model_len
        if not prompt_token_ids:
            raise ValueError("prompt must contain at least one token")
        if len(prompt_token_ids) >= limit:
            raise PromptTooLongError(
                f"prompt has {len(prompt_token_ids)} tokens; max_model_len is {limit}"
            )
        clamped = min(params.max_tokens, limit - len(prompt_token_ids))
        request = GenerationRequest(
            request_id=request_id or uuid.uuid4().hex,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            params=dataclasses.replace(params, max_tokens=clamped),
        )
        try:
            self.scheduler.add(request)
        except QueueFullError:
            self.metrics.requests_total.labels(status="rejected").inc()
            raise
        self._requests[request.request_id] = request
        self.metrics.queue_depth.set(self.scheduler.queue_depth)
        log_event(
            logger,
            logging.INFO,
            "request accepted",
            request_id=request.request_id,
            prompt_tokens=len(prompt_token_ids),
            max_tokens=clamped,
            queue_depth=self.scheduler.queue_depth,
        )
        self._wake.set()
        return request

    def cancel(self, request_id: str) -> bool:
        """Cancel a waiting or running request. Idempotent; False if unknown/finished."""
        request = self._requests.get(request_id)
        if request is None:
            return False
        self.scheduler.remove(request_id)
        self.metrics.queue_depth.set(self.scheduler.queue_depth)
        if request.cancel():
            self._on_terminal(request, "cancelled")
            return True
        return False

    # ---- internals -------------------------------------------------------
    def _sync_gauges(self, active: int) -> None:
        self.metrics.requests_active.set(active)
        self.metrics.queue_depth.set(self.scheduler.queue_depth)

    def _on_terminal(self, request: GenerationRequest, status: str) -> None:
        """Single place where a request leaves the engine: metrics + logs + cleanup."""
        self._requests.pop(request.request_id, None)
        m = self.metrics
        m.requests_total.labels(status=status).inc()
        generated = len(request.generated_token_ids)
        latency = request.latency
        ttft = request.ttft
        if status == "finished" and latency is not None:
            m.request_latency.observe(latency)
            if ttft is not None:
                m.ttft.observe(ttft)
                if generated > 1 and request.finished_at and request.first_token_at:
                    m.tpot.observe((request.finished_at - request.first_token_at) / (generated - 1))
        fields: dict[str, Any] = {
            "request_id": request.request_id,
            "prompt_tokens": len(request.prompt_token_ids),
            "generated_tokens": generated,
            "latency": latency,
            "ttft": ttft,
        }
        if status == "failed":
            fields["error"] = repr(request.error)
            log_event(logger, logging.ERROR, "request failed", **fields)
        else:
            log_event(logger, logging.INFO, f"request {status}", **fields)

    async def _in_thread(self, fn: Any, *args: Any) -> Any:
        return await asyncio.get_running_loop().run_in_executor(self._executor, fn, *args)

    def _sample(self, seqs: list[Sequence], logits: torch.Tensor) -> list[int]:
        gens: list[torch.Generator | None] = []
        for seq in seqs:
            seed = seq.request.params.seed
            if seed is not None and seq.generator is None:
                seq.generator = make_generator(seed, logits.device)
            gens.append(seq.generator)
        return sample_batch(logits, [s.request.params for s in seqs], gens)

    def _prefill_and_sample(self, seqs: list[Sequence]) -> tuple[BatchState, list[int]]:
        state, logits = self.runner.prefill([s.request.prompt_token_ids for s in seqs])
        return state, self._sample(seqs, logits)

    def _decode_and_sample(
        self, state: BatchState, seqs: list[Sequence], tokens: list[int]
    ) -> list[int]:
        return self._sample(seqs, self.runner.decode(state, tokens))

    async def _loop(self) -> None:
        while True:
            batch = self.scheduler.next_batch()
            self.metrics.queue_depth.set(self.scheduler.queue_depth)
            if not batch:
                await self._wake.wait()
                self._wake.clear()
                continue
            try:
                await self._run_batch(batch)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("batch failed")
                for req in batch:
                    if req.fail(exc):
                        self._on_terminal(req, "failed")
            finally:
                self._sync_gauges(0)

    async def _run_batch(self, batch: list[GenerationRequest]) -> None:
        """Prefill the batch, then decode token-by-token until every row is done.

        Rows are evicted from the KV cache as they finish or are cancelled. No new
        requests join a running batch (v0.1: no iteration-level admission).
        """
        for req in batch:
            req.mark_running()
            log_event(
                logger,
                logging.INFO,
                "request started",
                request_id=req.request_id,
                batch_size=len(batch),
            )
        self.metrics.batch_size.observe(len(batch))
        self.metrics.prompt_tokens.inc(sum(len(r.prompt_token_ids) for r in batch))

        active = [Sequence(r, self.tokenizer) for r in batch]
        self._sync_gauges(len(active))
        state, tokens = await self._in_thread(self._prefill_and_sample, active)
        phase = "prefill"

        while True:
            self.metrics.model_forward.labels(phase=phase).observe(self.runner.last_forward_seconds)
            for seq, token in zip(active, tokens, strict=True):
                req = seq.request
                if req.state is not RequestState.RUNNING:
                    continue  # cancelled while the forward pass was in flight
                before = len(req.generated_token_ids)
                out = seq.append(token)
                self.metrics.generated_tokens.inc(len(req.generated_token_ids) - before)
                if out.finish_reason is not None:
                    req.finish(out.finish_reason, out.text)
                    self._on_terminal(req, "finished")
                elif out.text:
                    req.emit(StreamEvent(out.text))

            keep = [i for i, s in enumerate(active) if s.request.state is RequestState.RUNNING]
            self._sync_gauges(len(keep))
            if not keep:
                return
            if len(keep) < len(active):
                state.keep(keep)
                active = [active[i] for i in keep]

            last = [s.last_token for s in active]
            assert all(t is not None for t in last)
            tokens = await self._in_thread(self._decode_and_sample, state, active, last)
            phase = "decode"
