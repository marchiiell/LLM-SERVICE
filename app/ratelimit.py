"""Rate limiter со скользящим окном, отдельный счётчик на каждого клиента."""
import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, max_requests: int, window: int):
        self.max_requests = max_requests
        self.window = window
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, client_id: str) -> tuple[bool, int]:
        """Возвращает (разрешено, сколько секунд ждать до следующей попытки)."""
        now = time.monotonic()
        hits = self._hits[client_id]
        while hits and hits[0] <= now - self.window:
            hits.popleft()
        if len(hits) >= self.max_requests:
            retry_after = int(hits[0] + self.window - now) + 1
            return False, retry_after
        hits.append(now)
        return True, 0
