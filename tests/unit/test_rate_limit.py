"""Tests for API rate limiting."""

from src.api.rate_limit import SlidingWindowRateLimiter


class FakeRedis:
    def __init__(self):
        self.counts = {}

    async def incr(self, key):
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    async def expire(self, key, seconds):
        return True


class FakeCacheManager:
    def __init__(self):
        self.redis = FakeRedis()

    async def get_client(self):
        return self.redis


def test_rate_limiter_rejects_requests_over_limit():
    limiter = SlidingWindowRateLimiter(limit=2, window_seconds=60)

    assert limiter.allow("client") is True
    assert limiter.allow("client") is True
    assert limiter.allow("client") is False


def test_rate_limiter_tracks_clients_independently():
    limiter = SlidingWindowRateLimiter(limit=1, window_seconds=60)

    assert limiter.allow("first-client") is True
    assert limiter.allow("second-client") is True


def test_distributed_rate_limiter_uses_redis():
    import asyncio

    from src.api.rate_limit import DistributedRateLimiter

    limiter = DistributedRateLimiter(limit=1, cache_manager=FakeCacheManager())

    assert asyncio.run(limiter.allow("client")) is True
    assert asyncio.run(limiter.allow("client")) is False