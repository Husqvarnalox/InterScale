"""The manual batched prefill/decode loop must match Hugging Face's reference decoding."""

import torch

from inferscale.model.runner import HFModelRunner


def greedy(runner: HFModelRunner, prompts: list[list[int]], steps: int) -> list[list[int]]:
    state, logits = runner.prefill(prompts)
    outs: list[list[int]] = [[] for _ in prompts]
    for _ in range(steps):
        toks = logits.argmax(-1).tolist()
        for o, t in zip(outs, toks, strict=True):
            o.append(t)
        logits = runner.decode(state, toks)
    return outs


def reference(runner: HFModelRunner, prompt: list[int], steps: int) -> list[int]:
    ids = torch.tensor([prompt])
    with torch.inference_mode():
        out = runner.model.generate(
            ids,
            max_new_tokens=steps,
            do_sample=False,
            min_new_tokens=steps,
            pad_token_id=runner.tokenizer.pad_token_id,
        )
    return out[0, len(prompt) :].tolist()


def test_batched_left_padded_decode_matches_reference(tiny_runner):
    prompts = [[258, 10, 20, 30, 40, 50], [258, 7], [258, 99, 98, 97]]
    got = greedy(tiny_runner, prompts, steps=6)
    assert got == [reference(tiny_runner, p, 6) for p in prompts]


def test_keep_evicts_rows_without_corrupting_the_rest(tiny_runner):
    prompts = [[258, 10, 20, 30, 40], [258, 7], [258, 99, 98]]
    state, logits = tiny_runner.prefill(prompts)
    toks = logits.argmax(-1).tolist()
    logits = tiny_runner.decode(state, toks)
    toks2 = logits.argmax(-1).tolist()
    state.keep([0, 2])
    logits = tiny_runner.decode(state, [toks2[0], toks2[2]])
    got0 = [toks[0], toks2[0], int(logits[0].argmax())]
    got2 = [toks[2], toks2[2], int(logits[1].argmax())]
    assert got0 == reference(tiny_runner, prompts[0], 3)
    assert got2 == reference(tiny_runner, prompts[2], 3)
    assert tiny_runner.last_forward_seconds > 0
