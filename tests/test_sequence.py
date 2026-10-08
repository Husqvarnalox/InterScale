from inferscale.engine.request import GenerationRequest, SamplingParams
from inferscale.engine.sequence import Sequence
from inferscale.model.tokenizer import ByteTokenizer

TOK = ByteTokenizer()


def run(text: str, **params) -> tuple[str, str | None]:
    req = GenerationRequest("r", "", [1], SamplingParams(**params))
    seq = Sequence(req, TOK)
    out = ""
    for t in [*text.encode(), TOK.eos_token_id, ord("Z")]:
        step = seq.append(t)
        out += step.text
        if step.finish_reason:
            return out, step.finish_reason
    raise AssertionError("never finished")


def test_eos_finishes_with_stop():
    assert run("hello", max_tokens=100) == ("hello", "stop")


def test_max_tokens_finishes_with_length():
    assert run("hello", max_tokens=3) == ("hel", "length")


def test_stop_string_truncates_output():
    assert run("hello world", max_tokens=100, stop=("o w",)) == ("hell", "stop")


def test_partial_stop_prefix_is_flushed_if_not_completed():
    assert run("ab<c", max_tokens=100, stop=("<END>",)) == ("ab<c", "stop")


def test_stop_prefix_held_back_while_streaming():
    req = GenerationRequest("r", "", [1], SamplingParams(max_tokens=50, stop=("<END>",)))
    seq = Sequence(req, TOK)
    emitted = [seq.append(t).text for t in b"hi<EN"]
    assert "".join(emitted) == "hi"  # "<EN" withheld


def test_multibyte_character_not_split():
    seq = Sequence(GenerationRequest("r", "", [1], SamplingParams(max_tokens=50)), TOK)
    pieces = [seq.append(b).text for b in "é".encode()]
    assert pieces == ["", "é"]
