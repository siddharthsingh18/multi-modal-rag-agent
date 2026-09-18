"""In-memory request rate limiting for a single API process."""

import time
from collections import defaultdict, deque
from threading import Lock
from typing import Any


class SlidingWindowRateLimiter:
    """Track request timestamps per client within a rolling time window."""

    def __init__(self, limit: int, window_seconds: int = 60):
        self.limit = limit
        self.window_seconds = window_seconds
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str) -> bool:
        """Return whether the client can make another request."""
        now = time.monotonic()
        cutoff = now - self.window_seconds

        with self._lock:
            timestamps = self._requests[key]
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()

            if len(timestamps) >= self.limit:
                return False

            timestamps.append(now)
            return True


class DistributedRateLimiter:
    """Use Redis for shared limits and fall back to an in-memory limiter."""

    def __init__(self, limit: int, window_seconds: int = 60, cache_manager: Any = None):
        self.limit = limit
        self.window_seconds = window_seconds
        self.cache_manager = cache_manager
        self.fallback = SlidingWindowRateLimiter(limit, window_seconds)

    async def allow(self, key: str) -> bool:
        """Return whether a client can make another request across workers."""
        if self.cache_manager is None:
            return self.fallback.allow(key)

        window = int(time.time() // self.window_seconds)
        redis_key = f"rate-limit:{key}:{window}"
        try:
            client = await self.cache_manager.get_client()
            count = await client.incr(redis_key)
            if count == 1:
                await client.expire(redis_key, self.window_seconds)
            return count <= self.limit
        except Exception:
            return self.fallback.allow(key)
