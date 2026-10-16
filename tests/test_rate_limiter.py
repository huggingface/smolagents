import threading
import time

import pytest

import smolagents.utils as utils
from smolagents.utils import RateLimiter


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_rate_limiter_serializes_concurrent_admissions(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(utils, "time", clock)

    limiter = RateLimiter(requests_per_minute=60)
    start = threading.Barrier(3)
    admitted = []

    def worker():
        start.wait()
        limiter.throttle()
        admitted.append(clock.time())

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()

    assert sorted(admitted) == [0.0, 1.0]


def test_rate_limiters_do_not_share_state(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(utils, "time", clock)

    first = RateLimiter(requests_per_minute=60)
    second = RateLimiter(requests_per_minute=60)

    first.throttle()
    second.throttle()

    assert clock.now == 0.0


def test_disabled_rate_limiter_does_not_touch_clock(monkeypatch):
    class FailingClock:
        def time(self):
            raise AssertionError("disabled limiter must not read the clock")

    monkeypatch.setattr(utils, "time", FailingClock())
    RateLimiter(requests_per_minute=None).throttle()


def test_rate_limiter_releases_lock_when_sleep_raises(monkeypatch):
    class FailingClock:
        def __init__(self):
            self.calls = 0

        def time(self):
            self.calls += 1
            return 0.0

        def sleep(self, seconds):
            raise RuntimeError("interrupted")

    clock = FailingClock()
    monkeypatch.setattr(utils, "time", clock)

    limiter = RateLimiter(requests_per_minute=60)
    limiter._last_call = 0.0

    with pytest.raises(RuntimeError, match="interrupted"):
        limiter.throttle()

    limiter._last_call = -1.0
    limiter.throttle()
