"""Recent refused questions (spam bots, rude words, too long) for the admin page.

Nothing goes in the database. Entries live in a small temporary file on the server, shared by the
web workers and wiped on every deploy or restart. IP addresses are never written: each entry gets a
short device tag (a keyed hash of the address and the day) so repeat attempts can be grouped.
"""
import hashlib
import hmac
import json
import logging
import os
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from lesson_platform.db import lesson_log_id
from lesson_platform.geo import extract_and_lookup

logger = logging.getLogger(__name__)

_PATH = Path(os.getenv("REFUSED_LOG_PATH") or Path(tempfile.gettempdir()) / "asklilowl-refused.jsonl")
_MAX_BYTES = 512_000  # About 1,500 entries; the older half is kept in one ".old" file.
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="refused-log")


def _submit(fn, *args):
    _executor.submit(fn, *args)


def device_tag(ip: str | None) -> str:
    """Same device on the same day -> same 6-letter tag. The address can't be read back from it."""
    if not ip:
        return "—"
    key = hashlib.sha256((os.getenv("FLASK_SECRET_KEY", "dev-secret-key") + "|device-tag").encode()).digest()
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return hmac.new(key, f"{ip}|{day}".encode(), hashlib.sha256).hexdigest()[:6]


def _write(entry: dict, forwarded_for: str | None, remote_addr: str | None) -> None:
    try:
        _, city, region, country = extract_and_lookup(forwarded_for, remote_addr)
    except Exception:
        city = region = country = None
    entry["place"] = ", ".join(part for part in (city, region, country) if part) or "—"
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    try:
        if _PATH.exists() and _PATH.stat().st_size > _MAX_BYTES:
            os.replace(_PATH, _PATH.with_suffix(".old"))
        with open(_PATH, "a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        logger.exception("Could not note a refused question")
    logger.warning("Refused question (%s) lesson=%s from device %s, %s, %s", entry["reason"],
                   lesson_log_id(entry.get("question") or ""), entry["device"], entry["place"],
                   (entry["user_agent"] or "-")[:120])


def record(*, endpoint: str, question: str, reason: str, ip: str | None, forwarded_for: str | None,
           remote_addr: str | None, user_agent: str | None) -> None:
    """Note one refused question. Returns at once; the location lookup happens in the background."""
    entry = {"at": time.time(), "endpoint": endpoint[:50], "reason": reason, "question": question[:300],
             "user_agent": (user_agent or "")[:300], "device": device_tag(ip)}
    try:
        _submit(_write, entry, forwarded_for, remote_addr)
    except Exception:
        logger.exception("Could not schedule a refused-question note")


def _read() -> list[dict]:
    entries = []
    for path in (_PATH.with_suffix(".old"), _PATH):
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        entries.append(json.loads(line))
                    except ValueError:
                        continue
        except OSError:
            continue
    return entries


def summary(tz_name: str, limit: int = 200) -> dict:
    """Counts, the most active devices, and the latest entries (newest first) for the admin page."""
    entries = sorted(_read(), key=lambda e: e.get("at", 0), reverse=True)
    now = time.time()
    tz = ZoneInfo(tz_name)

    def when(ts):
        return datetime.fromtimestamp(ts, tz).strftime("%Y-%m-%d %H:%M")

    by_device = Counter(e.get("device", "—") for e in entries if e.get("device", "—") != "—")
    devices = []
    for tag, count in by_device.most_common(10):
        latest = next(e for e in entries if e.get("device") == tag)
        devices.append({"device": tag, "count": count, "last": when(latest["at"]), "place": latest.get("place", "—"),
                        "user_agent": latest.get("user_agent") or "—",
                        "reasons": ", ".join(sorted({e["reason"] for e in entries if e.get("device") == tag}))})
    rows = [{"created_at": when(e["at"]), "reason": e.get("reason", ""), "question": e.get("question", ""),
             "place": e.get("place", "—"), "user_agent": e.get("user_agent") or "—", "device": e.get("device", "—")}
            for e in entries[:limit]]
    oldest = when(entries[-1]["at"]) if entries else None
    return {"total": len(entries), "last_1d": sum(now - e.get("at", 0) < 86400 for e in entries),
            "links": sum(e.get("reason") == "link" for e in entries), "since": oldest,
            "devices": devices, "rows": rows}
