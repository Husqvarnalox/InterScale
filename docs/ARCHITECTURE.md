# Architecture

```
api/routes.py ──► engine/engine.py ──► engine/scheduler.py
                        │                  (queue, batches)
                        ├──► engine/sequence.py, sampler.py
                        └──► model/runner.py ──► PyTorch / Transformers
```

Dependency direction: `api → engine → model`. `metrics`, `config` and `logging` are leaf modules.

## API layer (`api/`)

`app.py` builds the FastAPI app (`create_app(config, runner=None)`); the lifespan creates the
`InferenceEngine`, starts its loop and stops it on shutdown. There is no module-level singleton:
the engine lives in `app.state`. `schemas.py` holds Pydantic request models and converts them to
`SamplingParams`. `routes.py` tokenizes, calls `engine.submit`, and formats JSON or SSE. Handlers
never call the model. Validation errors are mapped to OpenAI-shaped 400s.

## Engine (`engine/engine.py`)

`InferenceEngine` owns the background loop task. `submit()` validates length, clamps
`max_tokens` to `max_model_len - prompt_len`, admits the request to the scheduler (or raises
`QueueFullError`) and wakes the loop. `cancel()` is idempotent and works for waiting and running
requests. `_on_terminal()` is the one place where a request leaves the engine: it updates
metrics, logs and drops the request from the registry (so nothing leaks).

The loop: `next_batch()` → `_run_batch()`: prefill, then repeat {process sampled tokens, evict
finished/cancelled rows via `state.keep(rows)`, decode one step} until no row is running. A
failure inside a batch fails only that batch's requests; the loop keeps serving.

## Scheduler (`engine/scheduler.py`)

Synchronous bookkeeping: a FIFO `deque`, `max_queue_size` admission check, `next_batch()` that
pops up to `max_batch_size` live requests (skipping ones cancelled while queued), `remove()` for
cancelling waiting requests, `drain()` for shutdown. It never touches the model. v0.1 does not
add requests to a batch that is already running.

## Model runner (`model/runner.py`)

The only code calling `model(...)`. `HFModelRunner.prefill(prompts)` left-pads, builds the
attention mask and position ids, runs one forward under `torch.inference_mode()` and returns
`(HFBatchState, last-position logits)`. `decode(state, tokens)` appends a mask column and runs a
single-token forward with the cache. It requests `logits_to_keep=1` when the model supports it so
prefill does not compute logits for every prompt position. The engine depends only on the `Runner`
and `BatchState` protocols, so tests inject a scripted runner and alternative cache designs can be
added behind the same interface. `model/loader.py` loads HF models (or the offline `builtin:tiny`)
and wraps failures in `ModelLoadError`; `model/device.py` picks device and dtype.

## Sampler (`engine/sampler.py`)

Pure functions over logits: greedy for `temperature == 0`, otherwise temperature → top-k →
top-p → multinomial, optionally with a per-request seeded `torch.Generator`. `sample_batch`
samples every row on-device and transfers the batch's token ids to the host once.

## Sequence (`engine/sequence.py`)

Per-running-request state: an incremental detokenizer (decodes only a short trailing window per
token, withholds incomplete UTF-8), EOS detection, `max_tokens`, and stop-string handling that
withholds text that could still become a stop string and truncates at the match.

## Request lifecycle (`engine/request.py`)

`WAITING → RUNNING → FINISHED`, with `CANCELLED` / `FAILED` reachable from non-terminal states.
Transitions are validated (`InvalidTransitionError`). Each request owns an `asyncio.Queue` of
`StreamEvent`s; `stream()` yields until the terminal event and `result()` aggregates it.

## Concurrency

- Request/scheduler state is touched only on the event-loop thread: no locks needed.
- Model forward + sampling run on one dedicated worker thread, one call at a time, via
  `run_in_executor`. The loop remains free to accept requests, stream, and process cancellations
  during a forward pass. The worker thread also exclusively touches per-request generators.
- Cancellation sets state on the loop thread; the batch loop skips cancelled rows when processing
  the in-flight step and evicts them before the next one (worst case one wasted step).
- Streaming responses cancel their request in a `finally`, which Starlette triggers on client
  disconnect. Non-streaming handlers poll `is_disconnected()` while waiting.

## KV cache ownership

A `BatchState` (HF `DynamicCache` + mask + positions) is created by `prefill`, owned by the running
batch inside `_run_batch`, shrunk by `keep()` and released when the batch ends. No cache outlives a
batch; there is no sharing between requests, no prefix reuse and no paging.

## Error propagation

| Failure | Result |
|---|---|
| Queue full | `QueueFullError` → HTTP 429 `queue_full` |
| Prompt too long / empty | → HTTP 400 |
| Model error in a batch | batch's requests `FAILED`; non-stream → 500, stream → an `error` SSE event then `[DONE]` |
| Engine stopped | in-flight requests `FAILED`, new submits → 503 |
| Model load failure | `inferscale serve` prints a clear error and exits 1 before binding the port |
