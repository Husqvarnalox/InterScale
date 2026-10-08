# InferScale

[![CI](https://github.com/Husqvarnalox/InterScale/actions/workflows/ci.yml/badge.svg)](https://github.com/Husqvarnalox/InterScale/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green)

A small LLM inference runtime built from first principles with PyTorch.

InferScale exposes the mechanics normally hidden behind production serving frameworks: request
scheduling, batched autoregressive decoding, KV caching, streaming, cancellation and inference
metrics. Hugging Face supplies weights and tokenization, but the serving loop is its own:
`model.generate()` is not used by the runtime.

It is a tool for understanding and experimenting, not a vLLM or TGI competitor. Designed for
decoder-only Hugging Face causal language models. API and internals may change before 1.0.

<p align="center">
  <img src="docs/assets/streaming.gif" alt="curl streaming tokens from InferScale" width="820">
</p>

<sub>Recording: `curl -N` against InferScale serving Qwen2.5-0.5B-Instruct on Apple MPS. SSE lines
are abbreviated (`id` elided) and playback is slowed ~2.5x; the whole stream took 0.83 s.</sub>

## What is implemented

- **Decode loop**: explicit prefill and decode phases over `model(...)` with the Transformers
  `DynamicCache`; left-padded batches with attention masks and explicit `position_ids`. A test
  checks batched, row-evicted output against `generate()` token for token.
- **Scheduling and batching**: FIFO queue with a bound (HTTP 429 when full), up to
  `max_batch_size` sequences per batch, finished and cancelled rows evicted from the KV cache
  mid-run. New requests do **not** join a running batch (see [Limitations](#limitations)).
- **Streaming, cancellation, observability**: SSE token deltas with stop-string hold-back and
  incremental UTF-8-safe detokenization; client disconnect cancels the request; Prometheus
  metrics (TTFT, TPOT, queue depth, batch size, forward time) and key=value logs.
- **Sampler**: greedy, temperature, top-p, top-k, per-request seed.

Devices: CI and the tests run on CPU only. MPS has been run manually (the recording above); the
CUDA path is implemented but untested. `--device auto` picks CUDA, then MPS, then CPU.

## Architecture

```mermaid
flowchart LR
    C([Client]) -->|HTTP / SSE| API
    subgraph InferScale
        direction LR
        API[FastAPI<br/>routes + schemas] -->|submit / cancel| ENG[InferenceEngine<br/>async loop]
        ENG <-->|admit, next_batch| SCH[Scheduler<br/>FIFO + backpressure]
        ENG -->|logits| SMP[Sampler<br/>greedy / temp / top-p]
        ENG -->|prefill / decode| RUN[ModelRunner<br/>left-pad + KV cache]
        ENG -.->|token events| API
    end
    RUN --> PT[PyTorch + Transformers]
    PT --> DEV[(CPU / MPS / CUDA)]
```

Dependencies point one way: `api → engine → model`. Details in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); inference background in
[docs/INFERENCE.md](docs/INFERENCE.md).

## Quick start

```bash
pip install -e .
inferscale serve --model Qwen/Qwen2.5-0.5B-Instruct   # downloads ~1 GB on first run

curl localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages": [{"role": "user", "content": "Say hi in five words."}],
  "max_tokens": 32, "stream": true }'
```

For an offline smoke test without downloads, use `--model builtin:tiny` (a random-weight
~100k-parameter Llama; its output is gibberish by design). `scripts/demo.sh` exercises a running
server. Docker (CPU): `docker build -t inferscale . && docker run -p 8000:8000 inferscale`.

## API

An OpenAI-compatible subset: `GET /v1/models`, `POST /v1/completions`,
`POST /v1/chat/completions` (JSON or SSE), plus `GET /health` and `GET /metrics`.

Supported: `max_tokens`, `temperature` (0 = greedy), `top_p`, `top_k` (extension), `stop` (up to
4), `seed`, `stream`, `stream_options.include_usage`. `model` is accepted and ignored; `n` must
be 1; logprobs, tools, `echo`, `best_of` and penalties are not supported. Errors use the OpenAI
shape: validation → 400, queue full → 429, engine stopped → 503, inference failure → 500 with
no internal detail. Chat uses the tokenizer's chat template, or a plain `<|role|>` format
if it has none. `usage.completion_tokens` excludes the EOS token.

## Configuration

Precedence: CLI flag > `INFERSCALE_*` environment variable > default (see `.env.example`;
`.env` files are not loaded).

| Flag | Default |
|---|---|
| `--model` | `Qwen/Qwen2.5-0.5B-Instruct` |
| `--device` / `--dtype` | `auto` |
| `--max-batch-size` / `--max-queue-size` | 8 / 64 |
| `--max-model-len` | 2048 (capped by the model's own limit) |
| `--host` / `--port` | `127.0.0.1` / 8000 |
| `--log-level` | `info` |

## Benchmarking

```bash
python benchmarks/benchmark.py --url http://localhost:8000 --requests 50 --concurrency 8 --stream
```

Closed-loop client reporting requests/sec, tokens/sec (from server-reported `usage`), latency
P50/P95/P99, average TTFT and error count. Needs `httpx` (installed with `.[dev]`). No results
are published here: they depend on device, model and batch size, so run it on your hardware.

## Metrics

`GET /metrics` (no per-request labels): `inferscale_requests_total{status}`,
`inferscale_requests_active`, `inferscale_queue_depth`, `inferscale_request_latency_seconds`,
`inferscale_time_to_first_token_seconds`, `inferscale_time_per_output_token_seconds`,
`inferscale_prompt_tokens_total`, `inferscale_generated_tokens_total`, `inferscale_batch_size`,
`inferscale_model_forward_seconds{phase}`.

## Limitations

- **No continuous batching.** A batch runs until every row finishes, so a long generation delays
  queued requests (head-of-line blocking at batch granularity).
- Left-padding columns stay in the KV cache for the batch's lifetime; there is no KV memory
  admission control, so large batches with long contexts can run out of memory.
- No PagedAttention, prefix caching, quantization, parallelism or authentication.
- One model per process, one batch in flight. Prompts are rejected, not truncated, when too long.
- Assumes a decoder-only causal LM with a `DynamicCache` that supports `batch_select_indices`
  and a tokenizer with an EOS token. Not every Hugging Face model will work.
- Not hardened for untrusted networks; see [SECURITY.md](SECURITY.md).

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md).

## Development

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check . && mypy && pytest -q
```

Tests run on CPU without network access. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0.
