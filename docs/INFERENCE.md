# Inference concepts, mapped to InferScale

## Autoregressive decoding

An LLM predicts one next token given all previous tokens. Generation repeats: run the model,
turn the last position's logits into a token (`engine/sampler.py`), append it, repeat until EOS or
`max_tokens` (`engine/sequence.py`). InferScale writes this loop itself (`InferenceEngine._run_batch`);
it never calls `model.generate()`.

## Prefill

All prompt tokens are known up front, so they are processed in one parallel forward pass
(`HFModelRunner.prefill`). Prompts of different lengths are **left-padded** into one tensor, with an
attention mask and explicit `position_ids` so padded rows behave as if unpadded (tested against
`generate`). Prefill produces the first token's logits and fills the KV cache. It is compute-bound.

## Decode

Each later step feeds a single new token per sequence (`HFModelRunner.decode`). Attention for
earlier tokens comes from the cache, so per-step compute is small but every step must read the
weights and the whole cache: memory-bandwidth-bound. That is why batching decode helps: one weight
read serves many sequences.

## KV cache

Attention needs the keys/values of all previous tokens. Recomputing them each step would cost
O(n²) work overall; caching makes each step O(n) in attention and O(1) in everything else. In
InferScale this is Transformers' `DynamicCache`, wrapped in `HFBatchState`. Cache memory grows by
`2 × layers × kv_heads × head_dim × dtype_bytes` per token per sequence (including padding columns).
When a sequence finishes, `HFBatchState.keep()` drops its row.

## Batching

Static batching runs a fixed set of sequences to completion. InferScale v0.1 does **dynamic batching
with row eviction**: the scheduler picks a batch, it decodes token-by-token, finished rows leave early.
New requests still wait for the next batch. *Continuous batching* additionally admits new sequences
between decode iterations; that is planned for v0.2, not implemented.

## TTFT and TPOT

- **TTFT** (time to first token): request creation → first emitted text. It includes queue wait
  (so batching and head-of-line blocking show up here) plus prefill. Metric:
  `inferscale_time_to_first_token_seconds`.
- **TPOT** (time per output token): mean inter-token time after the first,
  `(finished − first_token) / (tokens − 1)`. Dominated by decode speed and batch size. Metric:
  `inferscale_time_per_output_token_seconds`.
- `inferscale_model_forward_seconds{phase}` isolates raw forward time from queueing and sampling.
