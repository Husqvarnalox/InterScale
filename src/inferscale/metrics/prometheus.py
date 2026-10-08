"""Prometheus metrics. One registry per engine; labels are low-cardinality only."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)
_TTFT_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)
_TPOT_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1)
_FORWARD_BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 5)


class EngineMetrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        r = self.registry
        self.requests_total = Counter(
            "inferscale_requests_total",
            "Requests by terminal status (finished, cancelled, failed, rejected).",
            ["status"],
            registry=r,
        )
        self.requests_active = Gauge(
            "inferscale_requests_active", "Requests currently being decoded.", registry=r
        )
        self.queue_depth = Gauge(
            "inferscale_queue_depth", "Requests waiting for admission to a batch.", registry=r
        )
        self.request_latency = Histogram(
            "inferscale_request_latency_seconds",
            "End-to-end request latency (queue + prefill + decode).",
            buckets=_LATENCY_BUCKETS,
            registry=r,
        )
        self.ttft = Histogram(
            "inferscale_time_to_first_token_seconds",
            "Time from request creation to the first emitted text.",
            buckets=_TTFT_BUCKETS,
            registry=r,
        )
        self.tpot = Histogram(
            "inferscale_time_per_output_token_seconds",
            "Mean time per output token after the first, per request.",
            buckets=_TPOT_BUCKETS,
            registry=r,
        )
        self.prompt_tokens = Counter(
            "inferscale_prompt_tokens_total", "Prompt tokens processed.", registry=r
        )
        self.generated_tokens = Counter(
            "inferscale_generated_tokens_total", "Tokens generated.", registry=r
        )
        self.batch_size = Histogram(
            "inferscale_batch_size",
            "Number of sequences per scheduled batch.",
            buckets=(1, 2, 4, 8, 16, 32, 64),
            registry=r,
        )
        self.model_forward = Histogram(
            "inferscale_model_forward_seconds",
            "Wall time of one model forward pass.",
            ["phase"],  # prefill | decode
            buckets=_FORWARD_BUCKETS,
            registry=r,
        )

    def render(self) -> bytes:
        return generate_latest(self.registry)
