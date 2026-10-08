#!/usr/bin/env python3
"""Closed-loop load generator for an InferScale (or any OpenAI-compatible) server.

Example:
    python benchmarks/benchmark.py --url http://localhost:8000 \
        --requests 50 --concurrency 8 --max-tokens 64 --stream

``--concurrency`` workers each send requests back-to-back until ``--requests``
have been issued. TTFT is only measurable with ``--stream``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass

import httpx

PROMPTS = [
    "Explain what a KV cache is in one paragraph.",
    "Write a short story about a robot learning to paint.",
    "List five differences between prefill and decode in LLM inference.",
    "What is the capital of France, and why is it historically important?",
    "Describe how batching improves GPU utilisation.",
]


@dataclass
class Result:
    ok: bool
    latency: float
    tokens: int = 0
    ttft: float | None = None
    error: str = ""


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


async def one_request(client: httpx.AsyncClient, args: argparse.Namespace, index: int) -> Result:
    payload = {
        "messages": [{"role": "user", "content": PROMPTS[index % len(PROMPTS)]}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": args.stream,
    }
    if args.model:
        payload["model"] = args.model
    start = time.perf_counter()
    try:
        if not args.stream:
            r = await client.post("/v1/chat/completions", json=payload)
            r.raise_for_status()
            tokens = r.json()["usage"]["completion_tokens"]
            return Result(True, time.perf_counter() - start, tokens)

        payload["stream_options"] = {"include_usage": True}
        ttft: float | None = None
        tokens = 0
        async with client.stream("POST", "/v1/chat/completions", json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                if "error" in chunk:
                    raise RuntimeError(chunk["error"]["message"])
                if chunk.get("usage"):
                    tokens = chunk["usage"]["completion_tokens"]
                elif ttft is None and chunk["choices"][0]["delta"].get("content"):
                    ttft = time.perf_counter() - start
        return Result(True, time.perf_counter() - start, tokens, ttft)
    except (httpx.HTTPError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
        return Result(False, time.perf_counter() - start, error=f"{type(exc).__name__}: {exc}")


async def run(args: argparse.Namespace) -> tuple[list[Result], float]:
    counter = iter(range(args.requests))
    results: list[Result] = []
    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(base_url=args.url, timeout=timeout) as client:

        async def worker() -> None:
            for index in counter:  # shared iterator: each index handed out once
                results.append(await one_request(client, args, index))

        start = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(args.concurrency)))
        return results, time.perf_counter() - start


def report(args: argparse.Namespace, results: list[Result], wall: float) -> None:
    good = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    lat = [r.latency for r in good]
    ttfts = [r.ttft for r in good if r.ttft is not None]
    tokens = sum(r.tokens for r in good)
    mode = "stream" if args.stream else "non-stream"
    print(f"\nInferScale benchmark ({mode}, concurrency={args.concurrency})")
    print("-" * 52)
    print(f"{'total requests':<22}{len(results)}")
    print(f"{'failed requests':<22}{len(failed)}")
    print(f"{'wall time (s)':<22}{wall:.2f}")
    print(f"{'requests/sec':<22}{len(good) / wall:.2f}")
    print(f"{'tokens/sec':<22}{tokens / wall:.2f}")
    for p in (50, 95, 99):
        print(f"{f'P{p} latency (s)':<22}{percentile(lat, p):.3f}")
    ttft = f"{statistics.fmean(ttfts):.3f}" if ttfts else "n/a (use --stream)"
    print(f"{'avg TTFT (s)':<22}{ttft}")
    for msg in sorted({r.error for r in failed})[:5]:
        print(f"  error: {msg}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--requests", type=int, default=50)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--model", default=None)
    p.add_argument("--stream", action="store_true", help="use SSE streaming; enables TTFT")
    p.add_argument("--timeout", type=float, default=300.0)
    args = p.parse_args()
    if args.requests < 1 or args.concurrency < 1:
        p.error("--requests and --concurrency must be >= 1")
    results, wall = asyncio.run(run(args))
    report(args, results, wall)
    return 1 if any(not r.ok for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
