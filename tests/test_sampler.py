import torch

from inferscale.engine.request import SamplingParams
from inferscale.engine.sampler import make_generator, sample_batch, sample_token

LOGITS = torch.tensor([1.0, 3.0, 2.0, -1.0])


def test_greedy_picks_argmax():
    assert int(sample_token(LOGITS, SamplingParams(temperature=0))) == 1


def test_seeded_sampling_is_reproducible():
    p = SamplingParams(temperature=1.0)
    a = [int(sample_token(LOGITS, p, make_generator(7, LOGITS.device))) for _ in range(5)]
    b = [int(sample_token(LOGITS, p, make_generator(7, LOGITS.device))) for _ in range(5)]
    assert a == b


def test_temperature_sharpens_distribution():
    g = make_generator(0, LOGITS.device)
    hot = [int(sample_token(LOGITS, SamplingParams(temperature=5.0), g)) for _ in range(400)]
    cold = [int(sample_token(LOGITS, SamplingParams(temperature=0.1), g)) for _ in range(400)]
    assert cold.count(1) > hot.count(1)
    assert cold.count(1) > 390


def test_top_p_restricts_to_nucleus():
    # probs ~ [.09, .67, .24, .01]; top_p=0.5 keeps only the argmax.
    g = make_generator(1, LOGITS.device)
    p = SamplingParams(temperature=1.0, top_p=0.5)
    assert {int(sample_token(LOGITS, p, g)) for _ in range(200)} == {1}


def test_top_p_keeps_tokens_until_mass_reached():
    g = make_generator(2, LOGITS.device)
    p = SamplingParams(temperature=1.0, top_p=0.8)
    seen = {int(sample_token(LOGITS, p, g)) for _ in range(500)}
    assert seen == {1, 2}


def test_top_k():
    g = make_generator(3, LOGITS.device)
    p = SamplingParams(temperature=1.0, top_k=2)
    assert {int(sample_token(LOGITS, p, g)) for _ in range(300)} == {1, 2}


def test_sample_batch_mixed_params():
    logits = torch.stack([LOGITS, -LOGITS])
    out = sample_batch(logits, [SamplingParams(temperature=0)] * 2, [None, None])
    assert out == [1, 3]
