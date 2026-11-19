"""轻量可观测性：请求 ID、计数、延迟分位、缓存命中。

不引入 Prometheus 等外部依赖；指标进程内累计，通过 /api/metrics 暴露。
生产多实例时把这里换成 OpenTelemetry + Prometheus exporter，接口语义保持一致。
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

_LATENCY_WINDOW = 500


@dataclass
class Metrics:
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _requests: int = field(default=0, init=False)
    _errors: int = field(default=0, init=False)
    _latencies: deque = field(default_factory=lambda: deque(maxlen=_LATENCY_WINDOW), init=False, repr=False)
    _cache_hits: int = field(default=0, init=False)
    _cache_misses: int = field(default=0, init=False)
    _inflight: int = field(default=0, init=False)
    _max_inflight: int = field(default=0, init=False)

    def new_request_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def start(self) -> None:
        with self._lock:
            self._requests += 1
            self._inflight += 1
            self._max_inflight = max(self._max_inflight, self._inflight)

    def finish(self, started: float, ok: bool) -> None:
        with self._lock:
            self._inflight -= 1
            if not ok:
                self._errors += 1
            self._latencies.append((time.perf_counter() - started) * 1000)

    def cache_hit(self) -> None:
        with self._lock:
            self._cache_hits += 1

    def cache_miss(self) -> None:
        with self._lock:
            self._cache_misses += 1

    def snapshot(self) -> dict:
        with self._lock:
            lat = sorted(self._latencies)
            def pct(p: float) -> float:
                if not lat:
                    return 0.0
                return round(lat[min(len(lat) - 1, int(len(lat) * p))], 1)
            hits, misses = self._cache_hits, self._cache_misses
            return {
                "requests": self._requests,
                "errors": self._errors,
                "inflight": self._inflight,
                "max_inflight": self._max_inflight,
                "latency_ms": {"p50": pct(0.5), "p95": pct(0.95), "p99": pct(0.99)},
                "cache": {
                    "hits": hits,
                    "misses": misses,
                    "hit_rate": round(hits / (hits + misses), 3) if hits + misses else 0.0,
                },
            }


metrics = Metrics()
