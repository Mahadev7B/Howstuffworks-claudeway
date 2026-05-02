"""Postgres storage — feedback, API call telemetry, cached lessons, guardrails."""
import hashlib
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any

try:
    from zoneinfo import ZoneInfo  # Py 3.9+
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore

import psycopg
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)

_pool: ConnectionPool | None = None

# Hard cap on how long lesson-path operations wait for a pool connection.
# Admin uses a longer timeout (see _ADMIN_POOL_TIMEOUT_S) so it queues rather
# than fails, but still uses the same pool to avoid hitting Neon connection limits.
_POOL_TIMEOUT_S = 5.0
_ADMIN_POOL_TIMEOUT_S = 30.0
# Background fire-and-forget writes get a much shorter timeout — they are
# non-critical telemetry. If the pool is busy, drop the write immediately
# rather than blocking a slot for 5 seconds and starving synchronous callers.
_BG_POOL_TIMEOUT_S = 0.5
# libpq-level connect timeout (when the pool has to open a new socket).
_CONNECT_TIMEOUT_S = 5

# Background executor for fire-and-forget writes (record_api_call, cache writes,
# pin updates). Lesson generation must never wait on these.
_bg_writer = ThreadPoolExecutor(max_workers=4, thread_name_prefix="db-bg")


def _bg(fn, *args, **kwargs):
    """Fire-and-forget DB write. Returns immediately; logs on submit failure."""
    try:
        _bg_writer.submit(fn, *args, **kwargs)
    except Exception:
        logger.exception("Failed to schedule background DB write")

SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
  id         SERIAL PRIMARY KEY,
  question   TEXT NOT NULL,
  rating     TEXT NOT NULL CHECK (rating IN ('up', 'down')),
  comment    TEXT,
  ip_address TEXT,
  city       TEXT,
  region     TEXT,
  country    TEXT,
  user_agent TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS api_calls (
  id             SERIAL PRIMARY KEY,
  endpoint       TEXT NOT NULL,
  question       TEXT NOT NULL,
  model          TEXT NOT NULL,
  input_tokens   INTEGER NOT NULL DEFAULT 0,
  output_tokens  INTEGER NOT NULL DEFAULT 0,
  input_chars    INTEGER NOT NULL DEFAULT 0,
  output_chars   INTEGER NOT NULL DEFAULT 0,
  cost_usd       NUMERIC(10, 6) NOT NULL DEFAULT 0,
  duration_ms    INTEGER NOT NULL DEFAULT 0,
  success        BOOLEAN NOT NULL,
  error          TEXT,
  ip_address     TEXT,
  city           TEXT,
  region         TEXT,
  country        TEXT,
  user_agent     TEXT,
  lesson         JSONB,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS api_calls_created_at_idx ON api_calls (created_at DESC);
CREATE INDEX IF NOT EXISTS api_calls_ip_created_idx  ON api_calls (ip_address, created_at DESC);

CREATE TABLE IF NOT EXISTS cached_lessons (
  question_hash TEXT PRIMARY KEY,
  question      TEXT NOT NULL,
  lesson        JSONB NOT NULL,
  hit_count     INTEGER NOT NULL DEFAULT 0,
  last_hit_at   TIMESTAMPTZ,
  pinned        BOOLEAN NOT NULL DEFAULT false,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
ALTER TABLE cached_lessons ADD COLUMN IF NOT EXISTS pinned BOOLEAN NOT NULL DEFAULT false;

-- Idempotent column adds for pre-existing tables
ALTER TABLE feedback  ADD COLUMN IF NOT EXISTS ip_address TEXT;
ALTER TABLE feedback  ADD COLUMN IF NOT EXISTS city       TEXT;
ALTER TABLE feedback  ADD COLUMN IF NOT EXISTS region     TEXT;
ALTER TABLE feedback  ADD COLUMN IF NOT EXISTS country    TEXT;
ALTER TABLE feedback  ADD COLUMN IF NOT EXISTS user_agent TEXT;
ALTER TABLE api_calls ADD COLUMN IF NOT EXISTS ip_address TEXT;
ALTER TABLE api_calls ADD COLUMN IF NOT EXISTS city       TEXT;
ALTER TABLE api_calls ADD COLUMN IF NOT EXISTS region     TEXT;
ALTER TABLE api_calls ADD COLUMN IF NOT EXISTS country    TEXT;
ALTER TABLE api_calls ADD COLUMN IF NOT EXISTS user_agent TEXT;
ALTER TABLE api_calls ADD COLUMN IF NOT EXISTS lesson       JSONB;
ALTER TABLE api_calls ADD COLUMN IF NOT EXISTS input_chars  INTEGER NOT NULL DEFAULT 0;
ALTER TABLE api_calls ADD COLUMN IF NOT EXISTS output_chars INTEGER NOT NULL DEFAULT 0;
"""


def _check_connection(conn) -> None:
    """psycopg_pool health-check: run a trivial query; raises on dead connection."""
    conn.execute("SELECT 1")


def init_db() -> bool:
    """Create the pool and ensure schema exists. Returns True if DB is available."""
    global _pool
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        logger.warning("DATABASE_URL not set — DB storage disabled")
        return False
    try:
        _pool = ConnectionPool(
            dsn,
            min_size=1,
            max_size=8,
            timeout=_POOL_TIMEOUT_S,
            kwargs={"autocommit": True, "connect_timeout": _CONNECT_TIMEOUT_S},
            check=_check_connection,
            reconnect_timeout=10,
        )
        with _pool.connection(timeout=_POOL_TIMEOUT_S) as conn, conn.cursor() as cur:
            cur.execute(SCHEMA)
        logger.info("DB ready (feedback + api_calls)")
        return True
    except Exception:
        logger.exception("DB init failed")
        _pool = None
        return False


def enabled() -> bool:
    return _pool is not None


def _do_save_feedback(params: tuple) -> None:
    if _pool is None:
        return
    sql = """
        INSERT INTO feedback
          (question, rating, comment, ip_address, city, region, country, user_agent)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    try:
        with _pool.connection(timeout=_BG_POOL_TIMEOUT_S) as conn, conn.cursor() as cur:
            cur.execute(sql, params)
    except Exception:
        logger.exception("Failed to save feedback")


def save_feedback(
    question: str,
    rating: str,
    comment: str | None,
    *,
    ip_address: str | None = None,
    city: str | None = None,
    region: str | None = None,
    country: str | None = None,
    user_agent: str | None = None,
) -> None:
    if _pool is None:
        raise RuntimeError("DB not configured")
    if rating not in ("up", "down"):
        raise ValueError("rating must be 'up' or 'down'")
    params = (
        question[:500],
        rating,
        (comment or "")[:2000] or None,
        (ip_address or "")[:64] or None,
        (city or "")[:128] or None,
        (region or "")[:128] or None,
        (country or "")[:128] or None,
        (user_agent or "")[:500] or None,
    )
    _bg(_do_save_feedback, params)


def _do_record_api_call(params: tuple) -> None:
    if _pool is None:
        return
    try:
        with _pool.connection(timeout=_BG_POOL_TIMEOUT_S) as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO api_calls
                  (endpoint, question, model, input_tokens, output_tokens,
                   input_chars, output_chars, cost_usd, duration_ms, success, error,
                   ip_address, city, region, country, user_agent, lesson)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                params,
            )
    except Exception:
        logger.exception("Failed to record api_call")


def record_api_call(
    *,
    endpoint: str,
    question: str,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    input_chars: int = 0,
    output_chars: int = 0,
    cost_usd: float = 0.0,
    duration_ms: int = 0,
    success: bool,
    error: str | None = None,
    ip_address: str | None = None,
    city: str | None = None,
    region: str | None = None,
    country: str | None = None,
    user_agent: str | None = None,
    lesson: dict[str, Any] | None = None,
) -> None:
    """Telemetry write — non-blocking. Submits to the background executor and
    returns immediately so lesson generation is never delayed by DB latency."""
    if _pool is None:
        return
    params = (
        endpoint[:100],
        question[:500],
        model[:100],
        input_tokens,
        output_tokens,
        input_chars,
        output_chars,
        cost_usd,
        duration_ms,
        success,
        (error or "")[:1000] or None,
        (ip_address or "")[:64] or None,
        (city or "")[:128] or None,
        (region or "")[:128] or None,
        (country or "")[:128] or None,
        (user_agent or "")[:500] or None,
        Jsonb(lesson) if lesson is not None else None,
    )
    _bg(_do_record_api_call, params)


def _normalize_question(q: str) -> str:
    q = q.lower().strip()
    q = re.sub(r"[^\w\s]", "", q)
    q = re.sub(r"\s+", " ", q)
    return q


def question_hash(question: str) -> str:
    return hashlib.sha256(_normalize_question(question).encode("utf-8")).hexdigest()


def get_cached_lesson(question: str, ttl_days: int = 30) -> dict[str, Any] | None:
    """Return cached lesson if it exists and hasn't expired.

    Pinned lessons (thumbs-up feedback) ignore TTL and never expire.
    ttl_days=0 disables expiry entirely.
    """
    if _pool is None:
        return None
    qhash = question_hash(question)
    try:
        with _pool.connection(timeout=_POOL_TIMEOUT_S) as conn, conn.cursor() as cur:
            if ttl_days > 0:
                cur.execute(
                    """
                    SELECT lesson FROM cached_lessons
                    WHERE question_hash = %s
                      AND (pinned = true OR created_at > NOW() - INTERVAL '1 day' * %s)
                    """,
                    (qhash, ttl_days),
                )
            else:
                cur.execute(
                    "SELECT lesson FROM cached_lessons WHERE question_hash = %s",
                    (qhash,),
                )
            row = cur.fetchone()
            if row is None:
                return None
            cur.execute(
                """
                UPDATE cached_lessons
                SET hit_count = hit_count + 1, last_hit_at = NOW()
                WHERE question_hash = %s
                """,
                (qhash,),
            )
            return row[0]
    except Exception:
        logger.exception("Cache read failed — skipping")
        return None


def _do_pin_cached_lesson(qhash: str) -> None:
    if _pool is None:
        return
    try:
        with _pool.connection(timeout=_BG_POOL_TIMEOUT_S) as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE cached_lessons SET pinned = true WHERE question_hash = %s",
                (qhash,),
            )
    except Exception:
        logger.exception("Pin cached lesson failed — skipping")


def pin_cached_lesson(question: str) -> None:
    """Mark a cached lesson as pinned (thumbs-up). Fire-and-forget."""
    if _pool is None:
        return
    _bg(_do_pin_cached_lesson, question_hash(question))


def get_lesson_from_calls(question: str) -> dict[str, Any] | None:
    """Return the most recent successfully-generated lesson for this question from api_calls."""
    if _pool is None:
        return None
    try:
        with _pool.connection(timeout=_POOL_TIMEOUT_S) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT lesson FROM api_calls
                WHERE question = %s AND lesson IS NOT NULL AND success = true
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (question[:500],),
            )
            row = cur.fetchone()
            return row[0] if row else None
    except Exception:
        logger.exception("get_lesson_from_calls failed — skipping")
        return None


def _do_delete_cached_lesson(qhash: str, label: str) -> None:
    if _pool is None:
        return
    try:
        with _pool.connection(timeout=_BG_POOL_TIMEOUT_S) as conn, conn.cursor() as cur:
            cur.execute(
                "DELETE FROM cached_lessons WHERE question_hash = %s",
                (qhash,),
            )
        logger.info("Deleted stale cache entry for: %s", label)
    except Exception:
        logger.exception("Cache delete failed — skipping")


def delete_cached_lesson(question: str) -> None:
    """Remove a cached lesson so it will be regenerated on next request. Fire-and-forget."""
    if _pool is None:
        return
    _bg(_do_delete_cached_lesson, question_hash(question), question[:80])


def _do_save_cached_lesson(qhash: str, question: str, lesson: dict[str, Any]) -> None:
    if _pool is None:
        return
    try:
        with _pool.connection(timeout=_BG_POOL_TIMEOUT_S) as conn, conn.cursor() as cur:
            # UPSERT so a re-thumbs-up on a previously cached (text-only) entry
            # replaces it with the newer copy that has image_data_url baked in.
            # Preserves hit_count, last_hit_at, pinned, created_at.
            cur.execute(
                """
                INSERT INTO cached_lessons (question_hash, question, lesson)
                VALUES (%s, %s, %s)
                ON CONFLICT (question_hash) DO UPDATE
                  SET lesson = EXCLUDED.lesson
                """,
                (qhash, question, Jsonb(lesson)),
            )
    except Exception:
        logger.exception("Cache write failed — skipping")


def save_cached_lesson(question: str, lesson: dict[str, Any]) -> None:
    """Insert a lesson into the cache. Fire-and-forget so the request thread is not delayed."""
    if _pool is None:
        return
    _bg(_do_save_cached_lesson, question_hash(question), question[:500], lesson)


# Cache the budget total in-process. Recomputing on every TTS call exhausts
# the pool during lesson playback (4 slides × N users × prefetch).
_BUDGET_CACHE_TTL_S = 300.0
_budget_cache: tuple[float, float] | None = None  # (timestamp, value)


def today_spend_usd() -> float:
    """Sum of cost_usd across all api_calls in the last 24h. Cached for 60s.

    Returns 0.0 on failure or when DB unavailable. The cache is fine because
    daily budget enforcement only needs to be approximately correct.
    """
    global _budget_cache
    if _pool is None:
        return 0.0
    now = time.time()
    if _budget_cache is not None and now - _budget_cache[0] < _BUDGET_CACHE_TTL_S:
        return _budget_cache[1]
    try:
        with _pool.connection(timeout=_POOL_TIMEOUT_S) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT COALESCE(SUM(cost_usd), 0)
                FROM api_calls
                WHERE created_at > NOW() - INTERVAL '1 day'
                """
            )
            row = cur.fetchone()
            value = float(row[0]) if row and row[0] is not None else 0.0
            _budget_cache = (now, value)
            return value
    except Exception:
        logger.exception("Budget query failed — skipping")
        # Serve last known good value if we have one, else 0.0
        return _budget_cache[1] if _budget_cache is not None else 0.0


# Per-IP rate-limit cache: avoids a DB hit on every lesson/feedback request.
# Key: (ip, endpoints_tuple). Value: (timestamp, count).
_IP_RATE_CACHE: dict[tuple, tuple[float, int]] = {}
_IP_RATE_CACHE_TTL_S = 30.0


def ip_calls_last_hour(ip: str | None, endpoints: tuple[str, ...]) -> int:
    """Count of api_calls from this IP hitting given endpoints in the last hour. 0 on failure."""
    if _pool is None or not ip:
        return 0
    cache_key = (ip, endpoints)
    now = time.time()
    cached = _IP_RATE_CACHE.get(cache_key)
    if cached is not None and now - cached[0] < _IP_RATE_CACHE_TTL_S:
        return cached[1]
    try:
        with _pool.connection(timeout=_POOL_TIMEOUT_S) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM api_calls
                WHERE ip_address = %s
                  AND endpoint = ANY(%s)
                  AND created_at > NOW() - INTERVAL '1 hour'
                """,
                (ip, list(endpoints)),
            )
            row = cur.fetchone()
            count = int(row[0]) if row else 0
            _IP_RATE_CACHE[cache_key] = (now, count)
            return count
    except Exception:
        logger.exception("IP rate-limit query failed — skipping")
        return cached[1] if cached is not None else 0


# ---------------------------------------------------------------------------
# Admin dashboard queries — read-only, fail-safe (return empty/zero on error)
# ---------------------------------------------------------------------------
# All admin queries reuse the existing api_calls table. Cost is grouped by
# `endpoint` to attribute spend to Claude / Flux / TTS / cache.
_LESSON_GEN_ENDPOINTS = ("/lesson", "/api/lesson")
_FLUX_ENDPOINT = "/internal/flux"
_TTS_ENDPOINT = "/api/tts"

# All admin date buckets are computed in this timezone. Default America/New_York
# (Eastern Time, handles EST/EDT automatically). Override via ADMIN_TZ env var
# with any IANA timezone name (e.g. "America/New_York", "Asia/Kolkata", "UTC").
ADMIN_TZ = os.getenv("ADMIN_TZ", "America/New_York")
_LOCAL_DATE_SQL = f"DATE(created_at AT TIME ZONE '{ADMIN_TZ}')"
_LOCAL_HOUR_SQL = f"EXTRACT(HOUR FROM created_at AT TIME ZONE '{ADMIN_TZ}')::int"

# Comma-separated IPs or IPv6 prefixes to exclude from all admin analytics.
# Full IPs:    ADMIN_IPS=1.2.3.4,2a02:26f7:1234:5678::1
# Prefixes:    ADMIN_IPS=2a02:26f7:1234:   (trailing colon = prefix match)
_raw_admin_ips = [ip.strip() for ip in os.getenv("ADMIN_IPS", "").split(",") if ip.strip()]
ADMIN_IPS: list[str] = [ip for ip in _raw_admin_ips if not ip.endswith(":")]
ADMIN_IP_PREFIXES: list[str] = [ip for ip in _raw_admin_ips if ip.endswith(":")]

# Comma-separated cities to exclude (e.g. datacenter cities like Ashburn).
# ADMIN_EXCLUDE_CITIES=Ashburn,Frankfurt
ADMIN_EXCLUDE_CITIES: list[str] = [
    c.strip() for c in os.getenv("ADMIN_EXCLUDE_CITIES", "Ashburn,New York,Ohio,Columbus").split(",") if c.strip()
]


def _exclude_admin_ip_sql() -> tuple[str, list]:
    """Return (sql_fragment, params) to append to a WHERE clause.

    Excludes by exact/prefix IP match and by city (for datacenter traffic that
    rotates IPs, e.g. Render/AWS us-east-1 appearing as Ashburn, Virginia).
    Returns ("", []) when nothing is configured.
    """
    clauses: list[str] = []
    params: list = []

    if ADMIN_IPS:
        clauses.append("ip_address != ALL(%s)")
        params.append(ADMIN_IPS)

    for prefix in ADMIN_IP_PREFIXES:
        clauses.append("ip_address NOT LIKE %s")
        params.append(prefix + "%")

    if ADMIN_EXCLUDE_CITIES:
        clauses.append("(city IS NULL OR city != ALL(%s))")
        params.append(ADMIN_EXCLUDE_CITIES)

    if not clauses:
        return "", []

    inner = " AND ".join(clauses)
    return f"AND ({inner})", params


def today_start_local() -> datetime:
    """Return the timezone-aware datetime at midnight ADMIN_TZ today.

    Computed in Python and passed as a SQL parameter so we don't rely on
    Postgres timezone arithmetic for the day boundary. Comparable directly
    to `timestamptz` columns.
    """
    if ZoneInfo is not None:
        try:
            tz = ZoneInfo(ADMIN_TZ)
            now = datetime.now(tz)
            return now.replace(hour=0, minute=0, second=0, microsecond=0)
        except Exception:
            logger.warning("ADMIN_TZ %r not found — falling back to UTC", ADMIN_TZ)
    # Last-resort fallback: UTC midnight, with explicit UTC offset.
    from datetime import timezone as _tz
    return datetime.now(_tz.utc).replace(hour=0, minute=0, second=0, microsecond=0)


def _safe_query(default, fn):
    if _pool is None:
        return default
    try:
        with _pool.connection(timeout=_ADMIN_POOL_TIMEOUT_S) as conn, conn.cursor() as cur:
            return fn(cur)
    except Exception:
        logger.exception("Admin query failed — returning default")
        return default


# A "real lesson" = a successful generation that produced a lesson JSON. This
# excludes cache-hit rows (which are recorded with model='cache' and lesson=NULL).
_REAL_LESSON_FILTER = "success = true AND endpoint = ANY(%s) AND lesson IS NOT NULL"

_ADMIN_DEFAULTS = {
    "overview": {
        "total_lessons": 0, "lessons_today": 0, "unique_ips_today": 0,
        "cost_today": 0.0, "cost_all_time": 0.0, "avg_cost_per_lesson": 0.0,
        "avg_claude_ms": 0, "avg_flux_ms": 0, "avg_gen_ms": 0,
        "errors_today": 0, "errors_today_total": 0, "today_start_iso": "",
    },
    "lessons_by_day": [],
    "lessons_by_hour": [{"hour": h, "count": 0} for h in range(24)],
    "device_split": {"mobile": 0, "desktop": 0},
    "top_questions": [],
    "cost_by_day": [],
    "cost_by_provider": [],
    "recent_lessons": [],
    "perf_by_day": [],
    "slowest_lessons": [],
    "recent_errors": [],
    "lessons_by_country": [],
    "lessons_by_region": [],
}


def admin_load_all(
    *,
    days_lessons: int = 14,
    days_cost: int = 14,
    days_perf: int = 14,
    limit_questions: int = 15,
    limit_recent: int = 50,
    limit_slowest: int = 10,
    limit_errors: int = 30,
) -> dict[str, Any]:
    """Run every admin dashboard query in one pool checkout — single connection."""
    import copy
    defaults = copy.deepcopy(_ADMIN_DEFAULTS)

    if _pool is None:
        return defaults

    endpoints = list(_LESSON_GEN_ENDPOINTS)

    def _run(cur) -> dict[str, Any]:
        today_start = today_start_local()
        logger.info("Admin load_all: today_start=%s (tz=%s)", today_start.isoformat(), ADMIN_TZ)

        xip_sql, xip_params = _exclude_admin_ip_sql()

        # ── Overview ─────────────────────────────────────────────────────────
        cur.execute(f"SELECT COUNT(*) FROM api_calls WHERE {_REAL_LESSON_FILTER} {xip_sql}",
                    (endpoints, *xip_params))
        total_lessons = int(cur.fetchone()[0] or 0)

        cur.execute(f"""
            SELECT COUNT(*) FROM api_calls
            WHERE {_REAL_LESSON_FILTER} AND created_at >= %s {xip_sql}
        """, (endpoints, today_start, *xip_params))
        lessons_today = int(cur.fetchone()[0] or 0)

        cur.execute(f"""
            SELECT COUNT(DISTINCT ip_address) FROM api_calls
            WHERE ip_address IS NOT NULL AND created_at >= %s {xip_sql}
        """, (today_start, *xip_params))
        unique_ips_today = int(cur.fetchone()[0] or 0)

        cur.execute(f"""
            SELECT COALESCE(SUM(cost_usd), 0) FROM api_calls
            WHERE created_at >= %s {xip_sql}
        """, (today_start, *xip_params))
        cost_today = float(cur.fetchone()[0] or 0.0)

        _all_time_where = f"WHERE {xip_sql[4:]}" if xip_sql else ""
        cur.execute(f"SELECT COALESCE(SUM(cost_usd), 0) FROM api_calls {_all_time_where}",
                    (*xip_params,))
        cost_all_time = float(cur.fetchone()[0] or 0.0)

        avg_cost_per_lesson = (cost_all_time / total_lessons) if total_lessons else 0.0

        cur.execute(f"""
            SELECT COALESCE(AVG(duration_ms), 0) FROM api_calls
            WHERE {_REAL_LESSON_FILTER} {xip_sql}
        """, (endpoints, *xip_params))
        avg_claude_ms = float(cur.fetchone()[0] or 0.0)

        cur.execute(f"""
            SELECT COALESCE(AVG(duration_ms), 0) FROM api_calls
            WHERE endpoint = %s AND success = true {xip_sql}
        """, (_FLUX_ENDPOINT, *xip_params))
        avg_flux_ms = float(cur.fetchone()[0] or 0.0)

        cur.execute(f"""
            SELECT COUNT(*) FROM api_calls
            WHERE success = false AND endpoint = ANY(%s) AND created_at >= %s {xip_sql}
        """, (endpoints, today_start, *xip_params))
        lesson_errors_today = int(cur.fetchone()[0] or 0)

        cur.execute(f"""
            SELECT COUNT(*) FROM api_calls
            WHERE success = false AND created_at >= %s {xip_sql}
        """, (today_start, *xip_params))
        all_errors_today = int(cur.fetchone()[0] or 0)

        overview = {
            "total_lessons": total_lessons,
            "lessons_today": lessons_today,
            "unique_ips_today": unique_ips_today,
            "cost_today": round(cost_today, 4),
            "cost_all_time": round(cost_all_time, 4),
            "avg_cost_per_lesson": round(avg_cost_per_lesson, 5),
            "avg_claude_ms": int(avg_claude_ms),
            "avg_flux_ms": int(avg_flux_ms),
            "avg_gen_ms": int(avg_claude_ms + avg_flux_ms),
            "errors_today": lesson_errors_today,
            "errors_today_total": all_errors_today,
            "today_start_iso": today_start.isoformat(),
            "admin_ips_excluded": len(ADMIN_IPS) + len(ADMIN_IP_PREFIXES) + len(ADMIN_EXCLUDE_CITIES),
        }

        # ── Lessons by day ────────────────────────────────────────────────────
        cur.execute(f"""
            SELECT {_LOCAL_DATE_SQL} AS d, COUNT(*) AS n
            FROM api_calls
            WHERE {_REAL_LESSON_FILTER}
              AND created_at > NOW() - INTERVAL '1 day' * %s {xip_sql}
            GROUP BY d ORDER BY d
        """, (endpoints, days_lessons, *xip_params))
        lessons_by_day = [{"date": str(r[0]), "count": int(r[1])} for r in cur.fetchall()]

        # ── Lessons by hour ───────────────────────────────────────────────────
        cur.execute(f"""
            SELECT {_LOCAL_HOUR_SQL} AS h, COUNT(*) AS n
            FROM api_calls
            WHERE {_REAL_LESSON_FILTER}
              AND created_at > NOW() - INTERVAL '7 days' {xip_sql}
            GROUP BY h ORDER BY h
        """, (endpoints, *xip_params))
        hour_rows = {int(r[0]): int(r[1]) for r in cur.fetchall()}
        lessons_by_hour = [{"hour": h, "count": hour_rows.get(h, 0)} for h in range(24)]

        # ── Device split ──────────────────────────────────────────────────────
        cur.execute(f"""
            SELECT user_agent FROM api_calls
            WHERE user_agent IS NOT NULL AND {_REAL_LESSON_FILTER}
              AND created_at > NOW() - INTERVAL '30 days' {xip_sql}
        """, (endpoints, *xip_params))
        mobile = desktop = 0
        for (ua,) in cur.fetchall():
            ua_l = (ua or "").lower()
            if any(k in ua_l for k in ("iphone", "android", "mobile", "ipad")):
                mobile += 1
            else:
                desktop += 1
        device_split = {"mobile": mobile, "desktop": desktop}

        # ── Top questions ─────────────────────────────────────────────────────
        cur.execute(f"""
            SELECT question, COUNT(*) AS n
            FROM api_calls
            WHERE {_REAL_LESSON_FILTER} {xip_sql}
            GROUP BY question ORDER BY n DESC
            LIMIT %s
        """, (endpoints, *xip_params, limit_questions))
        top_questions = [{"question": r[0], "count": int(r[1])} for r in cur.fetchall()]

        # ── Cost by day ───────────────────────────────────────────────────────
        cur.execute(f"""
            SELECT {_LOCAL_DATE_SQL} AS d,
                   COALESCE(SUM(CASE WHEN endpoint = ANY(%s) THEN cost_usd ELSE 0 END), 0),
                   COALESCE(SUM(CASE WHEN endpoint = %s      THEN cost_usd ELSE 0 END), 0),
                   COALESCE(SUM(CASE WHEN endpoint = %s      THEN cost_usd ELSE 0 END), 0)
            FROM api_calls
            WHERE created_at > NOW() - INTERVAL '1 day' * %s {xip_sql}
            GROUP BY d ORDER BY d
        """, (endpoints, _FLUX_ENDPOINT, _TTS_ENDPOINT, days_cost, *xip_params))
        cost_by_day = []
        for r in cur.fetchall():
            c, f, t = float(r[1] or 0), float(r[2] or 0), float(r[3] or 0)
            cost_by_day.append({"date": str(r[0]), "claude": round(c, 5),
                                 "flux": round(f, 5), "tts": round(t, 5),
                                 "total": round(c + f + t, 5)})

        # ── Cost by provider ──────────────────────────────────────────────────
        cur.execute(f"""
            SELECT
              CASE
                WHEN endpoint = ANY(%s) THEN 'Claude'
                WHEN endpoint = %s      THEN 'Flux'
                WHEN endpoint = %s      THEN 'OpenAI TTS'
                ELSE endpoint
              END AS provider,
              COALESCE(SUM(cost_usd), 0) AS total
            FROM api_calls
            {_all_time_where}
            GROUP BY provider ORDER BY total DESC
        """, (endpoints, _FLUX_ENDPOINT, _TTS_ENDPOINT, *xip_params))
        cost_by_provider = [{"provider": r[0], "cost": round(float(r[1] or 0), 5)}
                            for r in cur.fetchall()]

        # ── Recent lessons — single query with correlated subqueries (no N+1) ─
        cur.execute(f"""
            SELECT
              a.id,
              to_char(a.created_at AT TIME ZONE '{ADMIN_TZ}', 'YYYY-MM-DD HH24:MI'),
              a.question,
              a.model,
              a.cost_usd,
              a.duration_ms,
              a.user_agent,
              (SELECT COALESCE(SUM(w.cost_usd), 0) FROM api_calls w
               WHERE w.endpoint = %s AND w.success = true
                 AND w.created_at BETWEEN a.created_at AND a.created_at + INTERVAL '5 minutes'),
              (SELECT COALESCE(SUM(w.cost_usd), 0) FROM api_calls w
               WHERE w.endpoint = %s AND w.success = true
                 AND w.created_at BETWEEN a.created_at AND a.created_at + INTERVAL '5 minutes'),
              (SELECT COALESCE(SUM(w.duration_ms), 0) FROM api_calls w
               WHERE w.endpoint = %s AND w.success = true
                 AND w.created_at BETWEEN a.created_at AND a.created_at + INTERVAL '5 minutes')
            FROM api_calls a
            WHERE {_REAL_LESSON_FILTER} {xip_sql}
            ORDER BY a.created_at DESC
            LIMIT %s
        """, (_FLUX_ENDPOINT, _TTS_ENDPOINT, _FLUX_ENDPOINT, endpoints, *xip_params, limit_recent))
        recent_lessons = []
        for r in cur.fetchall():
            rid, created_local, question, model, claude_cost, dur_ms, ua, flux_c, tts_c, flux_ms = r
            ua_l = (ua or "").lower()
            device = "mobile" if any(k in ua_l for k in ("iphone", "android", "mobile", "ipad")) else "desktop"
            claude_c = float(claude_cost or 0)
            flux_c = float(flux_c or 0)
            tts_c = float(tts_c or 0)
            recent_lessons.append({
                "id": int(rid),
                "created_at": created_local,
                "question": question,
                "claude_model": model,
                "claude_cost": round(claude_c, 5),
                "flux_cost": round(flux_c, 5),
                "tts_cost": round(tts_c, 5),
                "total_cost": round(claude_c + flux_c + tts_c, 5),
                "duration_ms": int(dur_ms or 0) + int(flux_ms or 0),
                "device": device,
            })

        # ── Perf by day ───────────────────────────────────────────────────────
        cur.execute(f"""
            SELECT {_LOCAL_DATE_SQL} AS d,
                   COALESCE(AVG(CASE WHEN endpoint = ANY(%s) AND lesson IS NOT NULL THEN duration_ms END), 0),
                   COALESCE(AVG(CASE WHEN endpoint = %s THEN duration_ms END), 0)
            FROM api_calls
            WHERE success = true AND created_at > NOW() - INTERVAL '1 day' * %s {xip_sql}
            GROUP BY d ORDER BY d
        """, (endpoints, _FLUX_ENDPOINT, days_perf, *xip_params))
        perf_by_day = [{"date": str(r[0]), "claude_ms": int(r[1] or 0), "flux_ms": int(r[2] or 0)}
                       for r in cur.fetchall()]

        # ── Slowest lessons ───────────────────────────────────────────────────
        cur.execute(f"""
            SELECT to_char(created_at AT TIME ZONE '{ADMIN_TZ}', 'YYYY-MM-DD HH24:MI'),
                   question, duration_ms, cost_usd
            FROM api_calls
            WHERE {_REAL_LESSON_FILTER} {xip_sql}
            ORDER BY duration_ms DESC
            LIMIT %s
        """, (endpoints, *xip_params, limit_slowest))
        slowest_lessons = [{"created_at": r[0], "question": r[1],
                            "duration_ms": int(r[2] or 0), "cost": round(float(r[3] or 0), 5)}
                           for r in cur.fetchall()]

        # ── Recent errors ─────────────────────────────────────────────────────
        cur.execute(f"""
            SELECT to_char(created_at AT TIME ZONE '{ADMIN_TZ}', 'YYYY-MM-DD HH24:MI'),
                   endpoint, question, error, ip_address
            FROM api_calls
            WHERE success = false {xip_sql}
            ORDER BY created_at DESC
            LIMIT %s
        """, (*xip_params, limit_errors))
        recent_errors = [{"created_at": r[0], "endpoint": r[1], "question": r[2],
                          "error": r[3], "ip": r[4]}
                         for r in cur.fetchall()]

        # ── Lessons by country (all time) ─────────────────────────────────────
        cur.execute(f"""
            SELECT COALESCE(country, 'Unknown') AS c, COUNT(*) AS n
            FROM api_calls
            WHERE {_REAL_LESSON_FILTER} AND country IS NOT NULL {xip_sql}
            GROUP BY c ORDER BY n DESC
            LIMIT 30
        """, (endpoints, *xip_params))
        lessons_by_country = [{"country": r[0], "count": int(r[1])} for r in cur.fetchall()]

        # ── Lessons by region/state (all time) ────────────────────────────────
        cur.execute(f"""
            SELECT COALESCE(country, 'Unknown') AS c,
                   COALESCE(region, 'Unknown')  AS r,
                   COUNT(*) AS n
            FROM api_calls
            WHERE {_REAL_LESSON_FILTER} AND region IS NOT NULL {xip_sql}
            GROUP BY c, r ORDER BY n DESC
            LIMIT 100
        """, (endpoints, *xip_params))
        lessons_by_region = [{"country": r[0], "region": r[1], "count": int(r[2])}
                             for r in cur.fetchall()]

        return {
            "overview": overview,
            "lessons_by_day": lessons_by_day,
            "lessons_by_hour": lessons_by_hour,
            "device_split": device_split,
            "top_questions": top_questions,
            "cost_by_day": cost_by_day,
            "cost_by_provider": cost_by_provider,
            "recent_lessons": recent_lessons,
            "perf_by_day": perf_by_day,
            "slowest_lessons": slowest_lessons,
            "recent_errors": recent_errors,
            "lessons_by_country": lessons_by_country,
            "lessons_by_region": lessons_by_region,
        }

    return _safe_query(defaults, _run)


def admin_lessons_filtered(
    *,
    country: str | None = None,
    region: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return recent lessons filtered by country and/or region. One pool checkout."""
    endpoints = list(_LESSON_GEN_ENDPOINTS)

    def _run(cur):
        xip_sql, xip_params = _exclude_admin_ip_sql()
        clauses = [_REAL_LESSON_FILTER]
        params: list[Any] = [endpoints]
        if country:
            clauses.append("country = %s")
            params.append(country)
        if region:
            clauses.append("region = %s")
            params.append(region)
        if xip_sql:
            clauses.append(xip_sql[4:])  # strip leading "AND "
            params.extend(xip_params)
        params.append(limit)
        cur.execute(f"""
            SELECT
              to_char(created_at AT TIME ZONE '{ADMIN_TZ}', 'YYYY-MM-DD HH24:MI'),
              question, country, region, city, cost_usd, duration_ms, user_agent
            FROM api_calls
            WHERE {' AND '.join(clauses)}
            ORDER BY created_at DESC
            LIMIT %s
        """, params)
        out = []
        for r in cur.fetchall():
            created_local, question, ctry, rgn, city, cost, dur_ms, ua = r
            ua_l = (ua or "").lower()
            device = "mobile" if any(k in ua_l for k in ("iphone", "android", "mobile", "ipad")) else "desktop"
            out.append({
                "created_at": created_local,
                "question": question,
                "country": ctry or "—",
                "region": rgn or "—",
                "city": city or "—",
                "cost": round(float(cost or 0), 5),
                "duration_ms": int(dur_ms or 0),
                "device": device,
            })
        return out

    return _safe_query([], _run)


# Individual wrappers kept for backwards-compat; all delegate to admin_load_all
# so callers that import them directly still work.

def admin_overview() -> dict[str, Any]:
    return admin_load_all()["overview"]

def admin_lessons_by_day(days: int = 14) -> list[dict[str, Any]]:
    return admin_load_all(days_lessons=days)["lessons_by_day"]

def admin_lessons_by_hour() -> list[dict[str, Any]]:
    return admin_load_all()["lessons_by_hour"]

def admin_device_split() -> dict[str, int]:
    return admin_load_all()["device_split"]

def admin_top_questions(limit: int = 15) -> list[dict[str, Any]]:
    return admin_load_all(limit_questions=limit)["top_questions"]

def admin_cost_by_day(days: int = 14) -> list[dict[str, Any]]:
    return admin_load_all(days_cost=days)["cost_by_day"]

def admin_cost_by_provider() -> list[dict[str, Any]]:
    return admin_load_all()["cost_by_provider"]

def admin_recent_lessons(limit: int = 50) -> list[dict[str, Any]]:
    return admin_load_all(limit_recent=limit)["recent_lessons"]

def admin_perf_by_day(days: int = 14) -> list[dict[str, Any]]:
    return admin_load_all(days_perf=days)["perf_by_day"]

def admin_slowest_lessons(limit: int = 10) -> list[dict[str, Any]]:
    return admin_load_all(limit_slowest=limit)["slowest_lessons"]

def admin_recent_errors(limit: int = 30) -> list[dict[str, Any]]:
    return admin_load_all(limit_errors=limit)["recent_errors"]
