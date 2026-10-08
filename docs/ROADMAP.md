# Roadmap

Nothing here is implemented in v0.1.

## v0.2
- True continuous batching with iteration-level admission (merge new prefills into a running
  batch, right-sized KV cache merging instead of left-padding).
- Prefix caching.
- Improved KV management (memory-aware admission, trimming dead padding).
- CUDA graph experiments for the decode step.

## v0.3
- Paged KV cache with a block allocator.
- Custom CUDA kernels.
- Speculative decoding.
- LoRA.
- Quantization experiments.

## Later
- Tensor parallelism, multi-GPU, distributed serving.
