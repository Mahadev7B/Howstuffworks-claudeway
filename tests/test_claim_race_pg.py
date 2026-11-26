"""The free-lesson reservation holds its lock until the insert commits.

Skipped unless TEST_DATABASE_URL is a local database whose name contains 'test'.
DATABASE_URL is never read.
"""

import os
import threading
import time

import psycopg
import pytest

from lesson_platform import askers, db
from tests.pg_guard import local_test_database_url

pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"),
    reason="TEST_DATABASE_URL is not set",
)


def _prepare_empty_asker_tables(dsn: str) -> None:
    """The schema script alters asker_questions before it creates the table.

    That order is fixed in a separate change. Creating the table first lets
    this test run init_db against a database that does not already have it.
    """
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS askers (
              email TEXT PRIMARY KEY,
              subscription_status TEXT NOT NULL DEFAULT 'none',
              created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
              last_question_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS asker_questions (
              email TEXT NOT NULL REFERENCES askers(email) ON DELETE CASCADE,
              question_hash TEXT NOT NULL,
              question TEXT NOT NULL,
              created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
              PRIMARY KEY (email, question_hash)
            )
            """
        )


@pytest.fixture(scope="module")
def pg():
    dsn = local_test_database_url()
    if dsn is None:
        pytest.skip("TEST_DATABASE_URL is not set")
    _prepare_empty_asker_tables(dsn)
    assert db.init_db(dsn)
    yield dsn
    if db._pool is not None:
        db._pool.close()
        db._pool = None


def _reset():
    with db._pool.connection() as conn:
        conn.execute("DELETE FROM asker_questions")
        conn.execute("DELETE FROM askers")


def _race(n: int, delay: float) -> tuple[int, int]:
    ip = askers.ip_hash("203.0.113.7")
    original = askers._decide

    def slow(*args, **kwargs):
        time.sleep(delay)
        return original(*args, **kwargs)

    askers._decide = slow
    results = []
    barrier = threading.Barrier(n)

    def worker(number):
        barrier.wait()
        try:
            decision, reserved = askers.claim_question(
                f"p{number}@example.com", f"How do magnets work {number}?", ip_hash=ip)
            results.append((decision["allowed"], reserved, decision["reason"]))
        except Exception as exc:
            results.append(("ERR", type(exc).__name__, str(exc)[:120]))

    threads = [threading.Thread(target=worker, args=(number,)) for number in range(n)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    askers._decide = original
    with db._pool.connection() as conn:
        rows = conn.execute(
            "SELECT COUNT(DISTINCT question_hash) FROM asker_questions WHERE ip_hash = %s",
            (ip,),
        ).fetchone()[0]
    reserved = sum(1 for item in results if item[1] is True)
    errors = [item for item in results if item[0] == "ERR"]
    assert errors == []
    return reserved, int(rows)


def test_advisory_lock_is_held_until_the_transaction_ends(pg):
    with db._pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(%s)", (12345,))
        with psycopg.connect(pg, autocommit=True) as other:
            held = other.execute("SELECT pg_try_advisory_xact_lock(12345)").fetchone()[0]
        assert held is False
    with psycopg.connect(pg, autocommit=True) as other:
        assert other.execute("SELECT pg_try_advisory_xact_lock(12345)").fetchone()[0] is True


@pytest.mark.parametrize("run", [1, 2, 3])
def test_eight_askers_reserve_exactly_three_with_no_latency(pg, run):
    _reset()
    reserved, rows = _race(8, 0.0)
    assert reserved == 3
    assert rows == 3


@pytest.mark.parametrize("run", [1, 2, 3])
def test_eight_askers_reserve_exactly_three_with_latency_between_read_and_insert(pg, run):
    _reset()
    reserved, rows = _race(8, 0.05)
    assert reserved == 3
    assert rows == 3


def test_four_askers_still_reserve_exactly_three(pg):
    _reset()
    reserved, rows = _race(4, 0.05)
    assert reserved == 3
    assert rows == 3


def test_a_reservation_older_than_ten_minutes_does_not_count(pg):
    _reset()
    ip = askers.ip_hash("203.0.113.7")
    with db._pool.connection() as conn:
        conn.execute("INSERT INTO askers (email) VALUES ('old@example.com')")
        conn.execute(
            "INSERT INTO asker_questions "
            "(email, question_hash, question, generated, ip_hash, settled, created_at) "
            "VALUES ('old@example.com', 'old-hash', 'old question', false, %s, false, "
            "NOW() - INTERVAL '11 minutes')",
            (ip,),
        )
    decision, reserved = askers.claim_question(
        "new@example.com", "How do birds fly?", ip_hash=ip)
    assert decision["used"] == 0
    assert reserved is True
    with db._pool.connection() as conn:
        left = conn.execute(
            "SELECT COUNT(*) FROM asker_questions WHERE question_hash = 'old-hash'"
        ).fetchone()[0]
    assert left == 0
