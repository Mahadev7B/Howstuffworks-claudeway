"""Postgres storage — feedback, API call telemetry, cached lessons, guardrails."""
import hashlib
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import psycopg
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)

_pool: ConnectionPool | None = None

# Hard cap on how long ANY db operation may wait for a pool connection.
# If the pool is exhausted we skip the op rather than block lesson generation.
_POOL_TIMEOUT_S = 3.0
# libpq-level connect timeout (when the pool has to open a new socket).
_CONNECT_TIMEOUT_S = 5

# Background executor for fire-and-forget writes (record_api_call, cache writes,
# pin updates). Lesson generation must never wait on these.
_bg_writer = ThreadPoolExecutor(max_workers=2, thread_name_prefix="db-bg")


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
            max_size=4,
            timeout=_POOL_TIMEOUT_S,                      # default checkout timeout
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
    sql = """
        INSERT INTO feedback
          (question, rating, comment, ip_address, city, region, country, user_agent)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    # Retry once on stale-connection errors (SSL reset, server timeout, etc.)
    for attempt in range(2):
        try:
            with _pool.connection(timeout=_POOL_TIMEOUT_S) as conn, conn.cursor() as cur:
                cur.execute(sql, params)
            return
        except psycopg.OperationalError:
            if attempt == 0:
                logger.warning("Feedback insert hit stale connection, retrying…")
                time.sleep(0.1)
            else:
                raise


def _do_record_api_call(params: tuple) -> None:
    if _pool is None:
        return
    try:
        with _pool.connection(timeout=_POOL_TIMEOUT_S) as conn, conn.cursor() as cur:
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
        with _pool.connection(timeout=_POOL_TIMEOUT_S) as conn, conn.cursor() as cur:
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
        with _pool.connection(timeout=_POOL_TIMEOUT_S) as conn, conn.cursor() as cur:
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
        with _pool.connection(timeout=_POOL_TIMEOUT_S) as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO cached_lessons (question_hash, question, lesson)
                VALUES (%s, %s, %s)
                ON CONFLICT (question_hash) DO NOTHING
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


def today_spend_usd() -> float:
    """Sum of cost_usd across all api_calls in the last 24h. 0.0 on failure or DB unavailable."""
    if _pool is None:
        return 0.0
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
            return float(row[0]) if row and row[0] is not None else 0.0
    except Exception:
        logger.exception("Budget query failed — skipping")
        return 0.0


def ip_calls_last_hour(ip: str | None, endpoints: tuple[str, ...]) -> int:
    """Count of api_calls from this IP hitting given endpoints in the last hour. 0 on failure."""
    if _pool is None or not ip:
        return 0
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
            return int(row[0]) if row else 0
    except Exception:
        logger.exception("IP rate-limit query failed — skipping")
        return 0


# ---------------------------------------------------------------------------
# Admin dashboard queries — read-only, fail-safe (return empty/zero on error)
# ---------------------------------------------------------------------------
# All admin queries reuse the existing api_calls table. Cost is grouped by
# `endpoint` to attribute spend to Claude / Flux / TTS / cache.
_LESSON_GEN_ENDPOINTS = ("/lesson", "/api/lesson")
_FLUX_ENDPOINT = "/internal/flux"
_TTS_ENDPOINT = "/api/tts"


def _safe_query(default, fn):
    if _pool is None:
        return default
    try:
        with _pool.connection(timeout=_POOL_TIMEOUT_S) as conn, conn.cursor() as cur:
            return fn(cur)
    except Exception:
        logger.exception("Admin query failed — returning default")
        return default


def admin_overview() -> dict[str, Any]:
    """Return headline numbers for the dashboard cards."""
    def q(cur):
        cur.execute("SELECT COUNT(*) FROM api_calls WHERE success = true AND endpoint = ANY(%s)",
                    (list(_LESSON_GEN_ENDPOINTS),))
        total_lessons = int(cur.fetchone()[0] or 0)

        cur.execute("""
            SELECT COUNT(*) FROM api_calls
            WHERE success = true AND endpoint = ANY(%s)
              AND created_at > NOW() - INTERVAL '1 day'
        """, (list(_LESSON_GEN_ENDPOINTS),))
        lessons_today = int(cur.fetchone()[0] or 0)

        cur.execute("""
            SELECT COUNT(DISTINCT ip_address) FROM api_calls
            WHERE ip_address IS NOT NULL
              AND created_at > NOW() - INTERVAL '1 day'
        """)
        unique_ips_today = int(cur.fetchone()[0] or 0)

        cur.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM api_calls WHERE created_at > NOW() - INTERVAL '1 day'")
        cost_today = float(cur.fetchone()[0] or 0.0)

        cur.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM api_calls")
        cost_all_time = float(cur.fetchone()[0] or 0.0)

        cur.execute("""
            SELECT COALESCE(AVG(cost_usd), 0) FROM api_calls
            WHERE success = true AND endpoint = ANY(%s)
        """, (list(_LESSON_GEN_ENDPOINTS),))
        avg_cost_per_lesson = float(cur.fetchone()[0] or 0.0)

        cur.execute("""
            SELECT COALESCE(AVG(duration_ms), 0) FROM api_calls
            WHERE success = true AND endpoint = ANY(%s)
        """, (list(_LESSON_GEN_ENDPOINTS),))
        avg_gen_ms = float(cur.fetchone()[0] or 0.0)

        cur.execute("""
            SELECT COUNT(*) FROM api_calls
            WHERE success = false AND created_at > NOW() - INTERVAL '1 day'
        """)
        errors_today = int(cur.fetchone()[0] or 0)

        return {
            "total_lessons": total_lessons,
            "lessons_today": lessons_today,
            "unique_ips_today": unique_ips_today,
            "cost_today": round(cost_today, 4),
            "cost_all_time": round(cost_all_time, 4),
            "avg_cost_per_lesson": round(avg_cost_per_lesson, 5),
            "avg_gen_ms": int(avg_gen_ms),
            "errors_today": errors_today,
        }
    return _safe_query({
        "total_lessons": 0, "lessons_today": 0, "unique_ips_today": 0,
        "cost_today": 0.0, "cost_all_time": 0.0, "avg_cost_per_lesson": 0.0,
        "avg_gen_ms": 0, "errors_today": 0,
    }, q)


def admin_lessons_by_day(days: int = 14) -> list[dict[str, Any]]:
    def q(cur):
        cur.execute("""
            SELECT DATE(created_at) AS d, COUNT(*) AS n
            FROM api_calls
            WHERE success = true AND endpoint = ANY(%s)
              AND created_at > NOW() - INTERVAL '1 day' * %s
            GROUP BY d ORDER BY d
        """, (list(_LESSON_GEN_ENDPOINTS), days))
        return [{"date": str(r[0]), "count": int(r[1])} for r in cur.fetchall()]
    return _safe_query([], q)


def admin_lessons_by_hour() -> list[dict[str, Any]]:
    def q(cur):
        cur.execute("""
            SELECT EXTRACT(HOUR FROM created_at)::int AS h, COUNT(*) AS n
            FROM api_calls
            WHERE success = true AND endpoint = ANY(%s)
              AND created_at > NOW() - INTERVAL '7 days'
            GROUP BY h ORDER BY h
        """, (list(_LESSON_GEN_ENDPOINTS),))
        rows = {int(r[0]): int(r[1]) for r in cur.fetchall()}
        return [{"hour": h, "count": rows.get(h, 0)} for h in range(24)]
    return _safe_query([{"hour": h, "count": 0} for h in range(24)], q)


def admin_device_split() -> dict[str, int]:
    def q(cur):
        cur.execute("""
            SELECT user_agent FROM api_calls
            WHERE user_agent IS NOT NULL AND endpoint = ANY(%s)
              AND created_at > NOW() - INTERVAL '30 days'
        """, (list(_LESSON_GEN_ENDPOINTS),))
        mobile = desktop = 0
        for (ua,) in cur.fetchall():
            ua_l = (ua or "").lower()
            if any(k in ua_l for k in ("iphone", "android", "mobile", "ipad")):
                mobile += 1
            else:
                desktop += 1
        return {"mobile": mobile, "desktop": desktop}
    return _safe_query({"mobile": 0, "desktop": 0}, q)


def admin_top_questions(limit: int = 15) -> list[dict[str, Any]]:
    def q(cur):
        cur.execute("""
            SELECT question, COUNT(*) AS n
            FROM api_calls
            WHERE success = true AND endpoint = ANY(%s)
            GROUP BY question
            ORDER BY n DESC
            LIMIT %s
        """, (list(_LESSON_GEN_ENDPOINTS), limit))
        return [{"question": r[0], "count": int(r[1])} for r in cur.fetchall()]
    return _safe_query([], q)


def admin_cost_by_day(days: int = 14) -> list[dict[str, Any]]:
    def q(cur):
        cur.execute("""
            SELECT DATE(created_at) AS d,
                   COALESCE(SUM(CASE WHEN endpoint = ANY(%s) THEN cost_usd ELSE 0 END), 0) AS claude,
                   COALESCE(SUM(CASE WHEN endpoint = %s        THEN cost_usd ELSE 0 END), 0) AS flux,
                   COALESCE(SUM(CASE WHEN endpoint = %s        THEN cost_usd ELSE 0 END), 0) AS tts
            FROM api_calls
            WHERE created_at > NOW() - INTERVAL '1 day' * %s
            GROUP BY d ORDER BY d
        """, (list(_LESSON_GEN_ENDPOINTS), _FLUX_ENDPOINT, _TTS_ENDPOINT, days))
        out = []
        for r in cur.fetchall():
            claude_c, flux_c, tts_c = float(r[1] or 0), float(r[2] or 0), float(r[3] or 0)
            out.append({
                "date": str(r[0]),
                "claude": round(claude_c, 5),
                "flux": round(flux_c, 5),
                "tts": round(tts_c, 5),
                "total": round(claude_c + flux_c + tts_c, 5),
            })
        return out
    return _safe_query([], q)


def admin_cost_by_provider() -> list[dict[str, Any]]:
    def q(cur):
        cur.execute("""
            SELECT
              CASE
                WHEN endpoint = ANY(%s) THEN 'Claude'
                WHEN endpoint = %s        THEN 'Flux'
                WHEN endpoint = %s        THEN 'OpenAI TTS'
                ELSE endpoint
              END AS provider,
              COALESCE(SUM(cost_usd), 0) AS total
            FROM api_calls
            GROUP BY provider
            ORDER BY total DESC
        """, (list(_LESSON_GEN_ENDPOINTS), _FLUX_ENDPOINT, _TTS_ENDPOINT))
        return [{"provider": r[0], "cost": round(float(r[1] or 0), 5)} for r in cur.fetchall()]
    return _safe_query([], q)


def admin_recent_lessons(limit: int = 50) -> list[dict[str, Any]]:
    """Per-lesson cost rows: pair each lesson generation with surrounding
    Flux + TTS calls within a small time window."""
    def q(cur):
        cur.execute("""
            SELECT id, created_at, question, model, cost_usd, duration_ms,
                   ip_address, user_agent
            FROM api_calls
            WHERE success = true AND endpoint = ANY(%s)
            ORDER BY created_at DESC
            LIMIT %s
        """, (list(_LESSON_GEN_ENDPOINTS), limit))
        lessons = cur.fetchall()
        out = []
        for row in lessons:
            lesson_id, created, question, model, claude_cost, duration_ms, ip, ua = row
            # Sum Flux + TTS within the 5 min window after this lesson
            cur.execute("""
                SELECT
                  COALESCE(SUM(CASE WHEN endpoint = %s THEN cost_usd ELSE 0 END), 0) AS flux,
                  COALESCE(SUM(CASE WHEN endpoint = %s THEN cost_usd ELSE 0 END), 0) AS tts,
                  COALESCE(SUM(CASE WHEN endpoint = %s THEN duration_ms ELSE 0 END), 0) AS flux_ms
                FROM api_calls
                WHERE created_at BETWEEN %s AND %s + INTERVAL '5 minutes'
                  AND success = true
            """, (_FLUX_ENDPOINT, _TTS_ENDPOINT, _FLUX_ENDPOINT, created, created))
            flux_cost, tts_cost, flux_ms = cur.fetchone()
            flux_cost, tts_cost, flux_ms = float(flux_cost or 0), float(tts_cost or 0), int(flux_ms or 0)
            ua_l = (ua or "").lower()
            device = "mobile" if any(k in ua_l for k in ("iphone", "android", "mobile", "ipad")) else "desktop"
            out.append({
                "id": int(lesson_id),
                "created_at": created.isoformat() if created else None,
                "question": question,
                "claude_model": model,
                "claude_cost": round(float(claude_cost or 0), 5),
                "flux_cost": round(flux_cost, 5),
                "tts_cost": round(tts_cost, 5),
                "total_cost": round(float(claude_cost or 0) + flux_cost + tts_cost, 5),
                "duration_ms": int(duration_ms or 0) + flux_ms,
                "device": device,
            })
        return out
    return _safe_query([], q)


def admin_perf_by_day(days: int = 14) -> list[dict[str, Any]]:
    def q(cur):
        cur.execute("""
            SELECT DATE(created_at) AS d,
                   COALESCE(AVG(CASE WHEN endpoint = ANY(%s) THEN duration_ms END), 0) AS claude_ms,
                   COALESCE(AVG(CASE WHEN endpoint = %s        THEN duration_ms END), 0) AS flux_ms
            FROM api_calls
            WHERE success = true AND created_at > NOW() - INTERVAL '1 day' * %s
            GROUP BY d ORDER BY d
        """, (list(_LESSON_GEN_ENDPOINTS), _FLUX_ENDPOINT, days))
        return [{"date": str(r[0]), "claude_ms": int(r[1] or 0), "flux_ms": int(r[2] or 0)} for r in cur.fetchall()]
    return _safe_query([], q)


def admin_slowest_lessons(limit: int = 10) -> list[dict[str, Any]]:
    def q(cur):
        cur.execute("""
            SELECT created_at, question, duration_ms, cost_usd
            FROM api_calls
            WHERE success = true AND endpoint = ANY(%s)
            ORDER BY duration_ms DESC
            LIMIT %s
        """, (list(_LESSON_GEN_ENDPOINTS), limit))
        return [{
            "created_at": r[0].isoformat() if r[0] else None,
            "question": r[1],
            "duration_ms": int(r[2] or 0),
            "cost": round(float(r[3] or 0), 5),
        } for r in cur.fetchall()]
    return _safe_query([], q)


def admin_recent_errors(limit: int = 30) -> list[dict[str, Any]]:
    def q(cur):
        cur.execute("""
            SELECT created_at, endpoint, question, error, ip_address
            FROM api_calls
            WHERE success = false
            ORDER BY created_at DESC
            LIMIT %s
        """, (limit,))
        return [{
            "created_at": r[0].isoformat() if r[0] else None,
            "endpoint": r[1],
            "question": r[2],
            "error": r[3],
            "ip": r[4],
        } for r in cur.fetchall()]
    return _safe_query([], q)
