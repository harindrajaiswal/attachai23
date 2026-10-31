"""Sliding-window per-IP and per-tenant rate limiter."""

import time
from collections import deque
from typing import Deque, Dict
from app.core.exceptions import RateLimitExceededError


class SlidingWindowRateLimiter:
    """In-memory sliding window rate limiter scoped by IP and/or tenant ID."""

    def __init__(self, window_seconds: int = 60) -> None:
        self.window_seconds = window_seconds
        self._buckets: Dict[str, Deque[float]] = {}

    def check(self, bucket_key: str, limit: int) -> None:
        """Verify request count for bucket_key within the sliding window."""
        if limit <= 0:
            return
        now = time.monotonic()
        cutoff = now - self.window_seconds
        dq = self._buckets.get(bucket_key)
        if dq is None:
            dq = deque()
            self._buckets[bucket_key] = dq

        while dq and dq[0] <= cutoff:
            dq.popleft()

        if len(dq) >= limit:
            retry_after = max(1, int(self.window_seconds - (now - dq[0])))
            raise RateLimitExceededError(
                message="Too many requests. Please wait before sending more requests.",
                retry_after=retry_after,
            )

        dq.append(now)

    def reset(self) -> None:
        """Clear all rate-limit buckets (used in tests)."""
        self._buckets.clear()
