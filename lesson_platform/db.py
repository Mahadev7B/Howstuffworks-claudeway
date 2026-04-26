"""Postgres storage — feedback, API call telemetry, cached lessons, guardrails."""
import hashlib
import logging
import os
import re
import time
from typing import Any

import psycopg
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)

_pool: ConnectionPool | None = None

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
            kwargs={"autocommit": True},
            check=_check_connection,       # validate before handing out
            reconnect_timeout=30,          # keep trying to reconnect for 30s
        )
        with _pool.connection() as conn, conn.cursor() as cur:
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
            with _pool.connection() as conn, conn.cursor() as cur:
                cur.execute(sql, params)
            return
        except psycopg.OperationalError:
            if attempt == 0:
                logger.warning("Feedback insert hit stale connection, retrying…")
                time.sleep(0.1)
            else:
                raise


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
    if _pool is None:
        return
    try:
        with _pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO api_calls
                  (endpoint, question, model, input_tokens, output_tokens,
                   input_chars, output_chars, cost_usd, duration_ms, success, error,
                   ip_address, city, region, country, user_agent, lesson)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
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
                ),
            )
    except Exception:
        logger.exception("Failed to record api_call")


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
        with _pool.connection() as conn, conn.cursor() as cur:
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
        logger.exception("Cache read failed")
        return None


def pin_cached_lesson(question: str) -> None:
    """Mark a cached lesson as pinned so it never expires (called on thumbs-up)."""
    if _pool is None:
        return
    qhash = question_hash(question)
    try:
        with _pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE cached_lessons SET pinned = true WHERE question_hash = %s",
                (qhash,),
            )
    except Exception:
        logger.exception("Pin cached lesson failed")


def delete_cached_lesson(question: str) -> None:
    """Remove a cached lesson so it will be regenerated on next request."""
    if _pool is None:
        return
    qhash = question_hash(question)
    try:
        with _pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "DELETE FROM cached_lessons WHERE question_hash = %s",
                (qhash,),
            )
        logger.info("Deleted stale cache entry for: %s", question[:80])
    except Exception:
        logger.exception("Cache delete failed")


def save_cached_lesson(question: str, lesson: dict[str, Any]) -> None:
    if _pool is None:
        return
    qhash = question_hash(question)
    try:
        with _pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO cached_lessons (question_hash, question, lesson)
                VALUES (%s, %s, %s)
                ON CONFLICT (question_hash) DO NOTHING
                """,
                (qhash, question[:500], Jsonb(lesson)),
            )
    except Exception:
        logger.exception("Cache write failed")


def today_spend_usd() -> float:
    """Sum of cost_usd across all api_calls in the last 24h. 0.0 on failure."""
    if _pool is None:
        return 0.0
    try:
        with _pool.connection() as conn, conn.cursor() as cur:
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
        logger.exception("Budget query failed")
        return 0.0


def ip_calls_last_hour(ip: str | None, endpoints: tuple[str, ...]) -> int:
    """Count of api_calls from this IP hitting given endpoints in the last hour."""
    if _pool is None or not ip:
        return 0
    try:
        with _pool.connection() as conn, conn.cursor() as cur:
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
        logger.exception("IP rate-limit query failed")
        return 0
