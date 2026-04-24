"""Postgres storage — feedback and API call telemetry."""
import logging
import os

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
  cost_usd       NUMERIC(10, 6) NOT NULL DEFAULT 0,
  duration_ms    INTEGER NOT NULL DEFAULT 0,
  success        BOOLEAN NOT NULL,
  error          TEXT,
  ip_address     TEXT,
  city           TEXT,
  region         TEXT,
  country        TEXT,
  user_agent     TEXT,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS api_calls_created_at_idx ON api_calls (created_at DESC);

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
"""


def init_db() -> bool:
    """Create the pool and ensure schema exists. Returns True if DB is available."""
    global _pool
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        logger.warning("DATABASE_URL not set — DB storage disabled")
        return False
    try:
        _pool = ConnectionPool(dsn, min_size=1, max_size=4, kwargs={"autocommit": True})
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
    with _pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO feedback
              (question, rating, comment, ip_address, city, region, country, user_agent)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                question[:500],
                rating,
                (comment or "")[:2000] or None,
                (ip_address or "")[:64] or None,
                (city or "")[:128] or None,
                (region or "")[:128] or None,
                (country or "")[:128] or None,
                (user_agent or "")[:500] or None,
            ),
        )


def record_api_call(
    *,
    endpoint: str,
    question: str,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float = 0.0,
    duration_ms: int = 0,
    success: bool,
    error: str | None = None,
    ip_address: str | None = None,
    city: str | None = None,
    region: str | None = None,
    country: str | None = None,
    user_agent: str | None = None,
) -> None:
    if _pool is None:
        return
    try:
        with _pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO api_calls
                  (endpoint, question, model, input_tokens, output_tokens,
                   cost_usd, duration_ms, success, error,
                   ip_address, city, region, country, user_agent)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    endpoint[:100],
                    question[:500],
                    model[:100],
                    input_tokens,
                    output_tokens,
                    cost_usd,
                    duration_ms,
                    success,
                    (error or "")[:1000] or None,
                    (ip_address or "")[:64] or None,
                    (city or "")[:128] or None,
                    (region or "")[:128] or None,
                    (country or "")[:128] or None,
                    (user_agent or "")[:500] or None,
                ),
            )
    except Exception:
        logger.exception("Failed to record api_call")
