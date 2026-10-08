"""OpenAI-compatible request/response schemas (the subset InferScale supports)."""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from inferscale.engine.request import SamplingParams

MAX_STOP_SEQUENCES = 4


class StreamOptions(BaseModel):
    include_usage: bool = False


class _SamplingFields(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    max_tokens: int | None = Field(None, ge=1)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    top_k: int = Field(0, ge=0)  # InferScale extension
    n: int = Field(1)
    stop: str | list[str] | None = None
    seed: int | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None

    @field_validator("n")
    @classmethod
    def _only_one_choice(cls, v: int) -> int:
        if v != 1:
            raise ValueError("only n=1 is supported")
        return v

    @field_validator("stop")
    @classmethod
    def _check_stop(cls, v: str | list[str] | None) -> list[str] | None:
        if v is None:
            return None
        stops = [v] if isinstance(v, str) else v
        if len(stops) > MAX_STOP_SEQUENCES:
            raise ValueError(f"at most {MAX_STOP_SEQUENCES} stop sequences are supported")
        return stops

    def to_sampling_params(self, default_max_tokens: int) -> SamplingParams:
        return SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens or default_max_tokens,
            stop=tuple(self.stop or ()),
            seed=self.seed,
        )


class CompletionRequest(_SamplingFields):
    prompt: str | list[str]

    @model_validator(mode="after")
    def _single_prompt(self) -> CompletionRequest:
        if isinstance(self.prompt, list):
            if len(self.prompt) != 1:
                raise ValueError("prompt must be a string or a list with exactly one string")
            self.prompt = self.prompt[0]
        if not self.prompt:
            raise ValueError("prompt must not be empty")
        return self


class ChatMessageIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(_SamplingFields):
    messages: list[ChatMessageIn] = Field(min_length=1)


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def make_usage(prompt: int, completion: int) -> dict[str, int]:
    return Usage(
        prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
    ).model_dump()


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


def now() -> int:
    return int(time.time())


def error_body(message: str, type_: str, code: str | None = None) -> dict[str, Any]:
    return {"error": {"message": message, "type": type_, "code": code}}
