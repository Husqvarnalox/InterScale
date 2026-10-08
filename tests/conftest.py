from __future__ import annotations

import time
from dataclasses import dataclass, field

import pytest
import torch

from inferscale.engine.engine import InferenceEngine
from inferscale.engine.scheduler import Scheduler
from inferscale.model.loader import build_tiny_runner
from inferscale.model.tokenizer import ByteTokenizer

EOS = ByteTokenizer.eos_token_id


@dataclass
class ScriptedState:
    scripts: list[list[int]]
    step: int = 0

    def keep(self, rows: list[int]) -> None:
        self.scripts = [self.scripts[i] for i in rows]


@dataclass
class ScriptedRunner:
    """Deterministic runner: the first prompt token selects which script to emit.

    Implements the ``Runner`` protocol without a model, so engine/scheduler logic is
    tested in isolation. Scripts are token-id lists (greedy decoding only).
    """

    scripts: dict[int, list[int]] = field(default_factory=dict)
    delay: float = 0.0
    fail_on_decode: bool = False
    tokenizer: ByteTokenizer = field(default_factory=ByteTokenizer)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    max_model_len: int = 512
    name: str = "scripted"
    last_forward_seconds: float = 0.0
    batch_sizes: list[int] = field(default_factory=list)

    def _logits(self, state: ScriptedState) -> torch.Tensor:
        logits = torch.zeros((len(state.scripts), ByteTokenizer.vocab_size))
        for row, script in enumerate(state.scripts):
            logits[row, script[min(state.step, len(script) - 1)]] = 10.0
        return logits

    def prefill(self, prompts: list[list[int]]) -> tuple[ScriptedState, torch.Tensor]:
        self.batch_sizes.append(len(prompts))
        time.sleep(self.delay)
        state = ScriptedState([self.scripts.get(p[-1], [ord("x")]) for p in prompts])
        return state, self._logits(state)

    def decode(self, state: ScriptedState, tokens: list[int]) -> torch.Tensor:  # type: ignore[override]
        time.sleep(self.delay)
        if self.fail_on_decode:
            raise RuntimeError("boom")
        state.step += 1
        return self._logits(state)


def make_engine(runner: object, batch: int = 4, queue: int = 8) -> InferenceEngine:
    return InferenceEngine(runner, Scheduler(batch, queue))  # type: ignore[arg-type]


@pytest.fixture
def tiny_runner():
    return build_tiny_runner(max_model_len=256)


@pytest.fixture
async def tiny_engine(tiny_runner):
    engine = make_engine(tiny_runner)
    await engine.start()
    yield engine
    await engine.stop()
