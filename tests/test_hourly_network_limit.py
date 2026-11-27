"""The hourly lesson limit counts the free-lesson network code, not a raw address."""

import time

import app as app_module
from lesson_platform import askers, db


def _store(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(db, "_direct_connect", lambda: (_ for _ in ()).throw(AssertionError("use the fake connection")))
    rows = []
    statements = []

    class Cur:
        def __init__(self):
            self.sql = ""
            self.count = 0

        def execute(self, sql, params=None):
            self.sql = " ".join(sql.split())
            statements.append((self.sql, params))
            if self.sql.startswith("SELECT COUNT"):
                digest, endpoints = params
                self.count = sum(
                    1 for row in rows
                    if row["ip_hash"] and row["ip_hash"] == digest
                    and row["endpoint"] in endpoints and row["age_h"] < 1
                )
            elif self.sql.startswith("UPDATE"):
                table = "api_calls" if "UPDATE api_calls" in self.sql else "asker_questions"
                hours = params[0]
                for row in rows:
                    if row.get("table", "api_calls") == table and row["ip_hash"] and row["age_h"] > hours:
                        row["ip_hash"] = None

        def fetchone(self):
            if self.sql.startswith("SELECT pg_try_advisory_xact_lock"):
                return (True,)
            if self.sql.startswith("SELECT COUNT"):
                return (self.count,)
            return None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class Conn:
        def cursor(self):
            return Cur()

        def transaction(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def close(self):
            return None

    monkeypatch.setattr(db, "_direct_connect", lambda: Conn())
    monkeypatch.setattr(db, "_pool", type("Pool", (), {"connection": lambda self, timeout=None: Conn()})())
    db._IP_RATE_CACHE.clear()
    askers.reset_memory()
    askers._last_purge_at = time.monotonic()
    return rows, statements


def test_one_hash_trips_the_hourly_limit_and_another_does_not(monkeypatch):
    rows, statements = _store(monkeypatch)
    monkeypatch.setenv("FREE_WINDOW_HOURS", "6")
    one = askers.ip_hash("203.0.113.9")
    other = askers.ip_hash("198.51.100.7")
    for _ in range(3):
        rows.append({"ip_hash": one, "ip_address": None, "endpoint": "/api/lesson", "age_h": 0.2})
    rows.append({"ip_hash": None, "ip_address": "203.0.113.9", "endpoint": "/api/lesson", "age_h": 0.1})
    rows.append({"ip_hash": None, "ip_address": None, "endpoint": "/api/lesson", "age_h": 0.1})
    monkeypatch.setattr(app_module, "today_spend_usd", lambda: 0.0)
    monkeypatch.setattr(app_module.settings, "per_ip_hourly_limit", 3)
    monkeypatch.setattr(app_module.ratelimit, "allow", lambda *args, **kwargs: True)

    assert db.ip_calls_last_hour(None, ("/api/lesson",)) == 0
    assert db.ip_calls_last_hour(one, ("/api/lesson",)) == 3
    db._IP_RATE_CACHE.clear()
    assert db.ip_calls_last_hour(other, ("/api/lesson",)) == 0
    sql = " ".join(text for text, _params in statements)
    assert "ip_hash = %s" in sql and "ip_address = %s" not in sql
    assert all("203.0.113.9" not in repr(params) for _text, params in statements)

    with app_module.app.test_request_context("/", headers={"X-Forwarded-For": "203.0.113.9"}):
        db._IP_RATE_CACHE.clear()
        assert app_module._guardrail_check("/api/lesson", {}) == app_module.RATE_LIMIT_ERROR
    with app_module.app.test_request_context("/", headers={"X-Forwarded-For": "198.51.100.7"}):
        db._IP_RATE_CACHE.clear()
        assert app_module._guardrail_check("/api/lesson", {}) is None


def test_recent_codes_still_count_after_a_purge(monkeypatch):
    rows, _statements = _store(monkeypatch)
    monkeypatch.setenv("FREE_WINDOW_HOURS", "6")
    digest = askers.ip_hash("203.0.113.10")
    for _ in range(2):
        rows.append({"ip_hash": digest, "ip_address": None, "endpoint": "/api/lesson", "age_h": 0.4})
    rows.append({"ip_hash": digest, "ip_address": None, "endpoint": "/api/lesson", "age_h": 7})
    assert askers.purge_ip_hashes() == "purged"
    assert rows[0]["ip_hash"] == digest and rows[1]["ip_hash"] == digest
    assert rows[2]["ip_hash"] is None
    db._IP_RATE_CACHE.clear()
    assert db.ip_calls_last_hour(digest, ("/api/lesson",)) == 2


def test_purge_nulls_old_api_call_codes_and_the_hourly_limit_still_trips(monkeypatch):
    rows, statements = _store(monkeypatch)
    monkeypatch.setenv("FREE_WINDOW_HOURS", "6")
    digest = askers.ip_hash("203.0.113.11")
    rows.append({"table": "api_calls", "ip_hash": digest, "endpoint": "/api/lesson", "age_h": 8})
    for age in (0.2, 0.4, 0.6):
        rows.append({"table": "api_calls", "ip_hash": digest, "endpoint": "/api/lesson", "age_h": age})
    rows.append({"table": "asker_questions", "ip_hash": digest, "endpoint": "/api/lesson", "age_h": 9})
    assert askers.purge_ip_hashes() == "purged"
    updates = [sql for sql, _params in statements if sql.startswith("UPDATE")]
    assert updates == [
        "UPDATE asker_questions SET ip_hash = NULL WHERE ip_hash IS NOT NULL AND created_at < NOW() - (%s * INTERVAL '1 hour')",
        "UPDATE api_calls SET ip_hash = NULL WHERE ip_hash IS NOT NULL AND created_at < NOW() - (%s * INTERVAL '1 hour')",
    ]
    assert rows[0]["ip_hash"] is None and rows[4]["ip_hash"] is None
    assert [row["ip_hash"] for row in rows[1:4]] == [digest, digest, digest]
    db._IP_RATE_CACHE.clear()
    assert db.ip_calls_last_hour(digest, ("/api/lesson",)) == 3
    monkeypatch.setattr(app_module, "today_spend_usd", lambda: 0.0)
    monkeypatch.setattr(app_module.settings, "per_ip_hourly_limit", 3)
    monkeypatch.setattr(app_module.ratelimit, "allow", lambda *args, **kwargs: True)
    with app_module.app.test_request_context("/", headers={"X-Forwarded-For": "203.0.113.11"}):
        db._IP_RATE_CACHE.clear()
        assert app_module._guardrail_check("/api/lesson", {}) == app_module.RATE_LIMIT_ERROR


def test_new_api_call_stores_the_network_code_and_not_the_address(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    captured = []
    monkeypatch.setattr(db, "_pool", object())
    monkeypatch.setattr(db, "_bg", lambda fn, params: captured.append(params))
    digest = askers.ip_hash("203.0.113.9")
    db.record_api_call(
        endpoint="/api/lesson", question="Why?", model="m", success=True,
        ip_address="203.0.113.9", ip_hash=digest,
    )
    assert "203.0.113.9" not in repr(captured[0])
    assert digest in captured[0]
