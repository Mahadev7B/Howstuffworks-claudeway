"""Address handling, offline country, and where a country is allowed to be stored."""

import importlib
import logging
import socket
import urllib.request

from lesson_platform import askers, db, ratelimit
from lesson_platform.geo import _client_ip, country_code, lookup
from lesson_platform.privacy_log import install_address_redaction, scrub_sentry_event


ADDRESS = "203.0.113.50"


def test_known_public_private_and_garbage_addresses_do_not_call_out(monkeypatch):
    calls = []
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: calls.append("urlopen"))
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: calls.append("connect"))
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: calls.append("dns"))
    # No registry table is bundled, so a public address has no country either.
    assert country_code("8.8.8.8") is None
    assert country_code("1.1.1.1") is None
    assert country_code("2001:4860:4860::8888") is None
    assert lookup("8.8.8.8") == (None, None, None)
    for address in ("10.1.2.3", "192.168.0.1", "127.0.0.1", "::1", "not-an-ip", "", None, "javascript:alert(1)"):
        assert country_code(address) is None
        assert lookup(address) == (None, None, None)
    assert calls == []


def test_leftmost_forwarded_address_and_a_supplied_header_do_not_win():
    assert _client_ip("198.51.100.7, 203.0.113.9", "10.0.0.1", "192.0.2.9") == "203.0.113.9"
    assert _client_ip("203.0.113.9", "10.0.0.1", "198.51.100.7") == "203.0.113.9"


def test_a_lesson_request_does_not_open_a_location_connection(monkeypatch):
    calls = []
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: calls.append("urlopen") or (_ for _ in ()).throw(AssertionError("urlopen")))
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: calls.append("connect"))
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: calls.append("dns"))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    app_module = importlib.import_module("app")
    app_module.app.config.update(TESTING=True)
    ratelimit.reset()
    askers.reset_memory()
    monkeypatch.setattr(app_module.askers.db, "_pool", None)

    def fake_track(endpoint, question, ctx):
        from flask import g
        g.lesson_generated = True
        assert ctx.get("city", None) is None
        assert ctx.get("region", None) is None
        assert ADDRESS not in repr(ctx)
        return {"title": question, "slides": []}, None

    monkeypatch.setattr(app_module, "_track_lesson", fake_track)
    monkeypatch.setattr(app_module, "check_question", lambda q: None)
    client = app_module.app.test_client()
    response = client.post(
        "/api/lesson",
        json={"question": "Why is the sky blue?", "email": "mum@example.com", "parent_consent": True},
        headers={"X-Forwarded-For": f"198.51.100.7, {ADDRESS}", "CF-Connecting-IP": "192.0.2.55"},
    )
    assert response.status_code == 200
    assert calls == []
    askers.reset_memory()
    ratelimit.reset()


def test_lesson_rows_can_hold_a_country_and_nothing_that_links_an_asker():
    sql = db._INSERT_API_CALL.lower()
    assert "country" in sql
    for forbidden in ("city", "region", "email", "ip_hash", "asker", "client_id"):
        assert forbidden not in sql
    captured = []
    pool = object()
    # record_api_call returns immediately when the pool is missing, so set one and capture the params.
    import lesson_platform.db as db_module
    original = db_module._pool
    db_module._pool = pool
    try:
        db_module._bg = lambda fn, params: captured.append(params)
        db_module.record_api_call(
            endpoint="/api/lesson", question="Why?", model="m", success=True,
            ip_address=ADDRESS, city="Austin", region="Texas", country="US",
            user_agent="test",
        )
    finally:
        db_module._pool = original
    assert captured
    blob = repr(captured[0])
    assert ADDRESS not in blob
    assert "Austin" not in blob and "Texas" not in blob
    assert "US" in captured[0]
    assert None in captured[0]  # the address column is written as null


def test_feedback_asker_and_spam_rows_get_no_country_city_region_or_address(monkeypatch, caplog, tmp_path):
    import lesson_platform.db as db_module
    import lesson_platform.spam_log as spam_log
    captured = []
    original = db_module._pool
    db_module._pool = object()
    try:
        db_module._bg = lambda fn, params: captured.append(params)
        db_module.save_feedback(
            "Why?", "up", None, client_id="visit-0009", ip_address=ADDRESS,
            city="Austin", region="Texas", country="US", user_agent="test",
        )
    finally:
        db_module._pool = original
    feedback_sql = db_module._INSERT_FEEDBACK.lower()
    assert "country" not in feedback_sql and "city" not in feedback_sql and "region" not in feedback_sql
    assert ADDRESS not in repr(captured) and "Austin" not in repr(captured) and "US" not in repr(captured)

    class Cursor:
        def __init__(self):
            self.queries = []

        def execute(self, sql, params=None):
            self.queries.append((sql, params))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class Conn:
        def __init__(self):
            self.cur = Cursor()

        def cursor(self):
            return self.cur

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    conn = Conn()

    class Pool:
        def connection(self, timeout=None):
            return conn

    monkeypatch.setattr(askers.db, "_pool", Pool())
    askers.record_question("mum@example.com", "Why is the sky blue?", ip_hash="abc123")
    joined = " ".join(sql.lower() for sql, _params in conn.cur.queries)
    assert "insert into askers" in joined and "insert into asker_questions" in joined
    assert "country" not in joined and "city" not in joined and "region" not in joined
    assert ADDRESS not in repr(conn.cur.queries)
    assert all(params is None or ADDRESS not in repr(params) for _sql, params in conn.cur.queries)

    spam_log._PATH = tmp_path / "refused.jsonl"
    monkeypatch.setattr(spam_log, "_submit", lambda fn, *args: fn(*args))
    install_address_redaction()
    with caplog.at_level(logging.INFO):
        spam_log.record(
            endpoint="/api/lesson", question="Why?", reason="link", ip=ADDRESS,
            forwarded_for=ADDRESS, remote_addr="10.0.0.8", user_agent="test",
        )
    text = spam_log._PATH.read_text(encoding="utf-8")
    assert ADDRESS not in text and "Austin" not in text and "Texas" not in text
    assert '"country"' not in text and '"city"' not in text and '"region"' not in text
    assert ADDRESS not in caplog.text and "Austin" not in caplog.text and "Texas" not in caplog.text


def test_logged_addresses_and_error_tracker_context_are_scrubbed(caplog):
    install_address_redaction()
    log = logging.getLogger("asklilowl.test")
    with caplog.at_level(logging.WARNING):
        log.warning("visitor %s reached us", ADDRESS)
        log.warning("header 2001:db8::1")
    assert ADDRESS not in caplog.text
    assert "2001:db8::1" not in caplog.text
    assert "[redacted]" in caplog.text
    event = scrub_sentry_event({
        "message": f"failed for {ADDRESS}",
        "request": {"headers": {"X-Forwarded-For": ADDRESS, "User-Agent": "test"}, "env": {"REMOTE_ADDR": ADDRESS}},
        "user": {"ip_address": ADDRESS, "id": "parent"},
        "extra": {"city": "Austin", "region": "Texas"},
    })
    blob = repr(event)
    assert ADDRESS not in blob
    assert "env" not in event["request"]
    assert event["request"]["headers"]["X-Forwarded-For"] == "[redacted]"
    assert event["user"]["ip_address"] == "[redacted]"

