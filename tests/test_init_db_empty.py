"""init_db must succeed on a database that does not already have the asker tables.

Skipped unless TEST_DATABASE_URL is a local database whose name contains 'test'.
DATABASE_URL is never read. A set but unsafe TEST_DATABASE_URL is refused
before any connection is opened.
"""

import os

import psycopg
import pytest

from tests.pg_guard import local_test_database_url

pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"),
    reason="TEST_DATABASE_URL is not set",
)


def test_init_db_succeeds_on_an_empty_database():
    from lesson_platform import db

    dsn = local_test_database_url()
    if dsn is None:
        pytest.skip("TEST_DATABASE_URL is not set")
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("DROP SCHEMA public CASCADE")
        conn.execute("CREATE SCHEMA public")
    if db._pool is not None:
        db._pool.close()
        db._pool = None
    assert db.init_db(dsn) is True
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
