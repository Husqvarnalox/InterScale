import pytest

from inferscale.engine.request import GenerationRequest, SamplingParams
from inferscale.engine.scheduler import QueueFullError, Scheduler


def req(i: int) -> GenerationRequest:
    return GenerationRequest(f"r{i}", "p", [1], SamplingParams())


def test_fifo_and_max_batch_size():
    s = Scheduler(max_batch_size=2, max_queue_size=10)
    for i in range(5):
        s.add(req(i))
    assert [r.request_id for r in s.next_batch()] == ["r0", "r1"]
    assert [r.request_id for r in s.next_batch()] == ["r2", "r3"]
    assert [r.request_id for r in s.next_batch()] == ["r4"]
    assert s.next_batch() == []


def test_queue_full_raises():
    s = Scheduler(max_batch_size=2, max_queue_size=2)
    s.add(req(0))
    s.add(req(1))
    with pytest.raises(QueueFullError):
        s.add(req(2))
    s.next_batch()  # frees capacity
    s.add(req(2))


def test_remove_and_cancelled_requests_are_skipped():
    s = Scheduler(max_batch_size=4, max_queue_size=10)
    a, b, c = req(0), req(1), req(2)
    for r in (a, b, c):
        s.add(r)
    assert s.remove("r0") is a
    assert s.remove("nope") is None
    b.cancel()
    assert s.next_batch() == [c]


def test_queue_depth_and_drain():
    s = Scheduler(2, 10)
    s.add(req(0))
    s.add(req(1))
    assert s.queue_depth == 2
    assert len(s.drain()) == 2 and s.queue_depth == 0


def test_invalid_sizes():
    with pytest.raises(ValueError):
        Scheduler(0, 1)
