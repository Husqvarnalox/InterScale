# Roadmap

Nothing below is implemented in v0.1. No dates are promised.

## v0.2
- Iteration-level continuous admission: merge new prefills into a running batch.
- Memory-aware KV admission control.
- Cache compaction to reduce padding.
- Broader model compatibility (cache implementations, tokenizers without EOS).

## v0.3
- Prefix caching.
- Paged KV cache exploration.
- Quantization.
- Speculative decoding experiments.

## Later
- Custom CUDA kernels.
- Tensor parallelism and multi-GPU.
