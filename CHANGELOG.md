# Changelog

## 0.1.0

- OpenAI-compatible subset: `/v1/completions`, `/v1/chat/completions` (JSON and SSE),
  `/v1/models`, `/health`, `/metrics`.
- Manual prefill/decode loop over Hugging Face causal LMs with the Transformers KV cache.
- FIFO scheduler with a bounded queue and batched decoding; finished and cancelled rows are
  evicted from the cache. No continuous batching yet.
- Sampler (greedy, temperature, top-p, top-k, seed), stop strings, EOS, cancellation on
  client disconnect.
- Prometheus metrics, key=value logging, `inferscale serve` CLI, CPU Dockerfile, benchmark
  script.
