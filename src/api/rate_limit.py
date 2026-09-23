from __future__ import annotations

import threading
import time


class TokenBucket:
    def __init__(self, rate_per_second: float, capacity: float) -> None:
        self.rate = rate_per_second
        self.capacity = capacity
        self._tokens = capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        with self._lock:
            now = time.monotonic()
            refill = (now - self._last) * self.rate
            self._tokens = min(self.capacity, self._tokens + refill)
            self._last = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False


class TokenBucketLimiter:
    def __init__(self, rate_per_min: int) -> None:
        self.rate_per_min = rate_per_min
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def try_acquire(self, key: str) -> bool:
        if self.rate_per_min <= 0:
            return True
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = TokenBucket(self.rate_per_min / 60.0, float(self.rate_per_min))
                self._buckets[key] = bucket
            return bucket.try_acquire()
