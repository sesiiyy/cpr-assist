"""Thread-safe session harness registry with idle TTL (vision runtime)."""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)

_registry: dict[str, tuple[Any, float]] = {}
_lock = threading.Lock()
_sweeper: threading.Thread | None = None
_stop = threading.Event()


def put(session_id: str, state: Any) -> None:
    now = time.monotonic()
    with _lock:
        _registry[session_id] = (state, now)


def get(session_id: str) -> Any | None:
    with _lock:
        hit = _registry.get(session_id)
        if hit is None:
            return None
        state, _ = hit
        return state


def touch(session_id: str) -> None:
    now = time.monotonic()
    with _lock:
        hit = _registry.get(session_id)
        if hit is None:
            return
        state, _ = hit
        _registry[session_id] = (state, now)


def release(session_id: str) -> None:
    with _lock:
        _registry.pop(session_id, None)


def _sweep_loop() -> None:
    ttl = float(settings.cpr_harness_ttl_seconds)
    interval = min(30.0, max(5.0, ttl / 10.0))
    while not _stop.wait(interval):
        now = time.monotonic()
        evict: list[str] = []
        with _lock:
            for sid, (_st, last) in _registry.items():
                if now - last > ttl:
                    evict.append(sid)
            for sid in evict:
                _registry.pop(sid, None)
        for sid in evict:
            logger.info("harness TTL evicted session_id=%s", sid)


def start_ttl_sweeper() -> None:
    global _sweeper
    with _lock:
        if _sweeper is not None and _sweeper.is_alive():
            return
        _stop.clear()
        _sweeper = threading.Thread(target=_sweep_loop, name="cpr-harness-ttl", daemon=True)
        _sweeper.start()


def stop_ttl_sweeper() -> None:
    _stop.set()
    t = _sweeper
    if t is not None and t.is_alive():
        t.join(timeout=2.0)
