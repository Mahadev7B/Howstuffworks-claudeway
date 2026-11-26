"""init_db must succeed on a database that does not already have the asker tables.

Skipped unless DATABASE_URL points at a Postgres this process can empty.
"""

import os

import psycopg
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL is not set",
)


def test_init_db_succeeds_on_an_empty_database():
    from lesson_platform import db

    dsn = os.environ["DATABASE_URL"]
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("DROP SCHEMA public CASCADE")
        conn.execute("CREATE SCHEMA public")
    if db._pool is not None:
        db._pool.close()
        db._pool = None
    assert db.init_db() is True
    with psycopg.connect(dsn, autocommit=True) as conn:
        columns = {
            row[0]
            for row in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'asker_questions'"
            ).fetchall()
        }
    assert {"email", "question_hash", "generated", "ip_hash"} <= columns
    if db._pool is not None:
        db._pool.close()
        db._pool = None
