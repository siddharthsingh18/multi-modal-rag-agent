"""Tests for API rate limiting."""

from src.api.rate_limit import SlidingWindowRateLimiter


def test_rate_limiter_rejects_requests_over_limit():
    limiter = SlidingWindowRateLimiter(limit=2, window_seconds=60)

    assert limiter.allow("client") is True
    assert limiter.allow("client") is True
    assert limiter.allow("client") is False


def test_rate_limiter_tracks_clients_independently():
    limiter = SlidingWindowRateLimiter(limit=1, window_seconds=60)

    assert limiter.allow("first-client") is True
    assert limiter.allow("second-client") is True