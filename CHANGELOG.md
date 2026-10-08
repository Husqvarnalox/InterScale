# Changelog

## 0.1.0

Initial release.

- OpenAI-style HTTP API: `/v1/completions`, `/v1/chat/completions` (JSON and SSE streaming),
  `/v1/models`, `/health`, `/metrics`.
- Manual prefill/decode loop on top of Hugging Face causal LMs, using the standard KV cache.
- Dynamic batching: FIFO scheduler with bounded queue (HTTP 429 on overflow); batches run
  token-by-token with finished/cancelled rows evicted from the cache.
- Custom sampler (greedy, temperature, top-p, top-k, per-request seed), stop strings, EOS.
- Prometheus metrics, structured logs, cancellation on client disconnect.
- `inferscale serve` CLI, CPU Dockerfile, benchmark script, tests, CI.
