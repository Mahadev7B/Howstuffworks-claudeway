"""Per-device hourly limits kept only in memory, so no IP address is ever stored.

Addresses are turned into a keyed hash with a random key made when the server
starts; the key is never saved, so the hashes can't be traced back after a
restart. Each server process keeps its own counts.
"""
import hashlib
import hmac
import secrets
import threading
import time
from collections import deque

_KEY = secrets.token_bytes(32)
_WINDOW_S = 3600.0
_lock = threading.Lock()
_hits: dict[tuple[str, str], deque] = {}


def _fingerprint(ip: str) -> str:
    return hmac.new(_KEY, ip.encode("utf-8"), hashlib.sha256).hexdigest()


def _recent(bucket: str, ip: str, now: float) -> deque:
    times = _hits.setdefault((bucket, _fingerprint(ip)), deque())
    while times and now - times[0] > _WINDOW_S:
        times.popleft()
    return times


def allow(bucket: str, ip: str | None, limit: int) -> bool:
    """Count one use and say whether this device is still under its hourly limit."""
    if not ip:
        return True
    now = time.time()
    with _lock:
        times = _recent(bucket, ip, now)
        if len(times) >= limit:
            return False
        times.append(now)
        if len(_hits) > 50_000:  # Forget idle devices so memory stays small.
            for key in [k for k, v in _hits.items() if not v or now - v[-1] > _WINDOW_S]:
                del _hits[key]
        return True


def reset() -> None:
    """Test helper."""
    with _lock:
        _hits.clear()
