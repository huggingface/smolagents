import threading

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


class ControlledClock(FakeClock):
    def __init__(self):
        super().__init__()
        self.sleep_entered = threading.Event()
        self.release_sleep = threading.Event()
        self.second_time_called = threading.Event()
        self.time_calls = 0

    def time(self):
        self.time_calls += 1
        if self.time_calls >= 2:
            self.second_time_called.set()
        return self.now

    def sleep(self, seconds):
        self.sleep_entered.set()
        self.release_sleep.wait(timeout=5)
        self.now += seconds


def test_rate_limiter_serializes_concurrent_admissions(monkeypatch):
    clock = ControlledClock()
    monkeypatch.setattr(utils, "time", clock)

    limiter = RateLimiter(requests_per_minute=60)
    first = threading.Thread(target=limiter.throttle)
    first.start()
    assert clock.sleep_entered.wait(timeout=2)

    second = threading.Thread(target=limiter.throttle)
    second.start()

    # The first call is deliberately blocked inside sleep(). A second caller
    # must not enter the critical section until the first reservation completes.
    assert not clock.second_time_called.wait(timeout=0.1)

    clock.release_sleep.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert clock.time_calls == 4
    assert limiter._last_call == pytest.approx(2.0)


def test_rate_limiters_do_not_share_state(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(utils, "time", clock)

    first = RateLimiter(requests_per_minute=60)
    second = RateLimiter(requests_per_minute=60)

    first.throttle()
    second.throttle()

    assert clock.now == 2.0


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

    with pytest.raises(RuntimeError, match="interrupted"):
        limiter.throttle()

    limiter._last_call = -1.0
    limiter.throttle()
