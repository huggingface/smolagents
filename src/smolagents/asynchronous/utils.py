import time
import asyncio
import threading
from typing import Optional

class RateLimiter:
    """Min-delay rate limiter with sync and async APIs."""

    def __init__(self, requests_per_minute: Optional[float] = None):
        self._enabled = requests_per_minute is not None
        self._interval = 60.0 / requests_per_minute if self._enabled else 0.0
        self._last_call = 0.0  # monotonic seconds
        # Locks for concurrent usage
        self._sync_lock = threading.Lock()
        self._async_lock = asyncio.Lock()

    def throttle(self) -> None:
        """Synchronous throttle (blocks the thread)."""
        if not self._enabled:
            return
        with self._sync_lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            delay = self._interval - elapsed
            if delay > 0:
                time.sleep(delay)
            self._last_call = time.monotonic()

    async def athrottle(self) -> None:
        """Asynchronous throttle (does not block the event loop)."""
        if not self._enabled:
            return
        async with self._async_lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            delay = self._interval - elapsed
            if delay > 0:
                await asyncio.sleep(delay)
            self._last_call = time.monotonic()

    # (옵션) 다음 호출에 필요한 대기시간을 알고 싶을 때
    def next_delay(self) -> float:
        if not self._enabled:
            return 0.0
        now = time.monotonic()
        elapsed = now - self._last_call
        return max(0.0, self._interval - elapsed)
