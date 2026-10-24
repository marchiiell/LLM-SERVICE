"""Простой in-memory кэш с TTL и вытеснением старых записей (LRU)."""
import hashlib
import json
import time
from collections import OrderedDict


def make_key(model: str, system: str, messages: list[dict], max_tokens: int) -> str:
    raw = json.dumps(
        {"m": model, "s": system, "msg": messages, "t": max_tokens},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode()).hexdigest()


class TTLCache:
    def __init__(self, ttl: int, max_items: int):
        self.ttl = ttl
        self.max_items = max_items
        self._data: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> dict | None:
        item = self._data.get(key)
        if item is None:
            self.misses += 1
            return None
        expires, value = item
        if expires < time.monotonic():
            del self._data[key]
            self.misses += 1
            return None
        self._data.move_to_end(key)
        self.hits += 1
        return value

    def set(self, key: str, value: dict) -> None:
        self._data[key] = (time.monotonic() + self.ttl, value)
        self._data.move_to_end(key)
        while len(self._data) > self.max_items:
            self._data.popitem(last=False)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "items": len(self._data),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
        }
