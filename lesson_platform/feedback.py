"""Feedback storage — Postgres via psycopg."""
import logging
import os

import psycopg
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)

_pool: ConnectionPool | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
  id         SERIAL PRIMARY KEY,
  question   TEXT NOT NULL,
  rating     TEXT NOT NULL CHECK (rating IN ('up', 'down')),
  comment    TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


def _dsn() -> str | None:
    return os.getenv("DATABASE_URL") or None


def init_db() -> bool:
    """Create the pool and ensure schema exists. Returns True if DB is available."""
    global _pool
    dsn = _dsn()
    if not dsn:
        logger.warning("DATABASE_URL not set — feedback storage disabled")
        return False
    try:
        _pool = ConnectionPool(dsn, min_size=1, max_size=4, kwargs={"autocommit": True})
        with _pool.connection() as conn, conn.cursor() as cur:
            cur.execute(SCHEMA)
        logger.info("Feedback DB ready")
        return True
    except Exception:
        logger.exception("Feedback DB init failed")
        _pool = None
        return False


def save_feedback(question: str, rating: str, comment: str | None) -> None:
    if _pool is None:
        raise RuntimeError("feedback storage not configured")
    if rating not in ("up", "down"):
        raise ValueError("rating must be 'up' or 'down'")
    with _pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO feedback (question, rating, comment) VALUES (%s, %s, %s)",
            (question[:500], rating, (comment or "")[:2000] or None),
        )
