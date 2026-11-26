"""Log ids are keyed, and database errors do not echo a kid's question."""

import json
import logging
import os

import psycopg
import pytest

from lesson_platform import askers, db, spam_log
from lesson_platform.db import content_log_id, lesson_log_id, question_hash
from lesson_platform.gunicorn_log import RedactingLogger, strip_query

QUESTION = "my name is Emma Doe emma.doe@example.com"


def _leak(message=None):
    text = message or f'Key (k)=({QUESTION}) DETAIL: {{"question": "{QUESTION}"}}'
    raise psycopg.errors.UniqueViolation(text)


class _Cursor:
    rowcount = 1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, *args, **kwargs):
        _leak()

    def fetchone(self):
        _leak()

    def fetchall(self):
        _leak()


class _Conn:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def cursor(self):
        return _Cursor()

    def close(self):
        pass


class _Pool:
    def connection(self, timeout=None):
        return _Conn()


class _QuietCursor:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, *args, **kwargs):
        return None

    def fetchone(self):
        return (0, 0, 0, 0)

    def fetchall(self):
        return []


class _QuietConn(_Conn):
    def cursor(self):
        return _QuietCursor()


def _hidden(caplog, capsys, step):
    captured = capsys.readouterr()
    blob = caplog.text + captured.err + captured.out
    assert QUESTION not in blob, step
    assert "emma.doe@example.com" not in blob, step


def test_log_id_is_keyed_hmac_not_the_cache_hash(monkeypatch):
    question = "Why is the sky blue?"
    monkeypatch.setenv("LOG_ID_KEY", "key-one")
    first = lesson_log_id(question)
    prompt = content_log_id("a cheerful tree")
    assert first == lesson_log_id(question)
    assert first != question_hash(question)
    assert len(first) == 16 and len(prompt) == 16
    assert set(first) <= set("0123456789abcdef")
    monkeypatch.setenv("LOG_ID_KEY", "key-two")
    assert lesson_log_id(question) != first
    assert content_log_id("a cheerful tree") != prompt
    monkeypatch.setenv("LOG_ID_KEY", "key-one")
    assert lesson_log_id(question) == first

    monkeypatch.delenv("LOG_ID_KEY", raising=False)
    monkeypatch.setenv("FLASK_SECRET_KEY", "flask-secret-a")
    derived = lesson_log_id(question)
    monkeypatch.setenv("FLASK_SECRET_KEY", "flask-secret-b")
    assert lesson_log_id(question) != derived
    monkeypatch.setenv("LOG_ID_KEY", "key-one")
    monkeypatch.setenv("FLASK_SECRET_KEY", "flask-secret-b")
    assert lesson_log_id(question) == first


def test_gunicorn_error_line_drops_the_query_string(caplog):
    question = "My name is Jane Doe"
    assert strip_query(f"/lesson?q={question}") == "/lesson"
    assert strip_query(f"GET /lesson?q={question} HTTP/1.1") == "GET /lesson HTTP/1.1"
    log = object.__new__(RedactingLogger)
    named = logging.getLogger("gunicorn.error.privacy-test")
    named.handlers.clear()
    named.propagate = True
    named.setLevel(logging.DEBUG)
    log.error_log = named
    caplog.set_level(logging.DEBUG, logger="gunicorn.error.privacy-test")
    try:
        raise RuntimeError("worker failed")
    except RuntimeError:
        log.exception("Error handling request %s", f"/lesson?q={question}")
    log.warning("Invalid Request Line '%s'", f"GET /lesson?q={question} HTTP/1.1")
    assert question not in caplog.text
    assert "?q=" not in caplog.text
    assert "Error handling request /lesson" in caplog.text


def test_psycopg_errors_do_not_log_question_text(monkeypatch, caplog, capsys):
    import importlib
    app_module = importlib.import_module("app")
    caplog.set_level(logging.DEBUG)
    real_direct = db._direct_connect

    def connect(*args, **kwargs):
        _leak()

    original_getenv = os.getenv

    def getenv(name, default=None):
        if name == "DATABASE_URL":
            return "postgresql://127.0.0.1:9/not-connected"
        return original_getenv(name, default)

    monkeypatch.setattr(db.os, "getenv", getenv)
    monkeypatch.setattr(db.psycopg, "connect", connect)
    assert real_direct() is None
    _hidden(caplog, capsys, "direct connect")

    class Dead:
        def submit(self, *args, **kwargs):
            _leak("schedule " + QUESTION)

    monkeypatch.setattr(db, "_bg_writer", Dead())
    db._bg(lambda: None)
    _hidden(caplog, capsys, "schedule")

    monkeypatch.setattr(db, "_direct_connect", lambda: _Conn())
    monkeypatch.setattr(db, "_pool", _Pool())
    monkeypatch.setattr(db, "_budget_cache", None)
    steps = [
        ("cache read", lambda: db.get_cached_lesson(QUESTION)),
        ("pin", lambda: db._do_pin_cached_lesson("hash")),
        ("delete", lambda: db._do_delete_cached_lesson("hash", "logid")),
        ("exists", lambda: db.cached_lesson_exists(QUESTION)),
        ("voice lookup", lambda: db.get_saved_tts("hash")),
        ("voice save", lambda: db._do_save_tts("hash", b"mp3")),
        ("cache write", lambda: db._do_save_cached_lesson("hash", QUESTION, {"title": QUESTION})),
        ("feedback", lambda: db._do_save_feedback((QUESTION, "up", None, None, None, None, None, None, None))),
        ("api call", lambda: db._do_record_api_call((QUESTION,))),
        ("share", lambda: db.create_share_link(QUESTION, "slug", {"slides": [{"title": QUESTION}]})),
        ("slug", lambda: db.get_lesson_by_slug("slug")),
        ("budget", lambda: db.today_spend_usd()),
        ("admin query", lambda: db._safe_query(None, lambda cur: cur.execute("SELECT question"))),
        ("delete asker", lambda: askers.delete_asker("mum@example.com")),
    ]
    for name, call in steps:
        call()
        reason = db.get_write_stats().get("last_failure_reason") or ""
        assert QUESTION not in reason, name
        _hidden(caplog, capsys, name)

    def boom(*args, **kwargs):
        _leak()

    monkeypatch.setattr(db, "_pool", object())
    monkeypatch.setattr(db, "_direct_connect", lambda: _QuietConn())
    monkeypatch.setattr(db, "_admin_feedback", boom)
    monkeypatch.setattr(db, "_admin_askers", boom)
    db.admin_load_all()
    _hidden(caplog, capsys, "admin load")

    app_module.app.config.update(TESTING=True)
    monkeypatch.setattr(app_module, "db_enabled", True)
    monkeypatch.setattr(app_module.verification, "required", lambda: False)
    monkeypatch.setattr(db, "_pool", _Pool())
    from lesson_platform import ratelimit
    ratelimit.reset()
    client = app_module.app.test_client()
    body = {"email": "mum@example.com", "parent_consent": True}
    client.post("/api/more-lessons", json=body)
    _hidden(caplog, capsys, "more lessons")
    client.post("/api/waitlist", json={**body, "choice": "subscribe"})
    _hidden(caplog, capsys, "waitlist")
    with app_module.app.test_request_context(
        "/api/trial", method="POST", data=json.dumps(body), content_type="application/json",
    ):
        app_module.api_trial_start()
    _hidden(caplog, capsys, "trial")
    monkeypatch.setattr(app_module.billing, "subscription_change", lambda payload, signature: {
        "email": "mum@example.com", "status": "active", "customer_id": "cus", "subscription_id": "sub",
    })
    client.post("/stripe/webhook", data=b"{}", headers={"Stripe-Signature": "t"})
    _hidden(caplog, capsys, "stripe")
    monkeypatch.setenv("ADMIN_SECRET", "unit-secret")
    client.post("/admin/clear-cache", headers={"X-Admin-Secret": "unit-secret"})
    _hidden(caplog, capsys, "clear cache")
    with client.session_transaction() as sess:
        sess["admin"] = True
    client.get("/admin/cached-lessons")
    _hidden(caplog, capsys, "cached lessons")

    def explode(*args, **kwargs):
        raise OSError(f"write {QUESTION}")

    monkeypatch.setattr(spam_log, "_submit", explode)
    spam_log.record(endpoint="/api/lesson", question=QUESTION, reason="too long", ip=None,
                    forwarded_for=None, remote_addr=None, user_agent="SecretBrowser/9.9")
    _hidden(caplog, capsys, "schedule refused note")
    monkeypatch.setattr("builtins.open", explode)
    spam_log._write({"at": 1, "endpoint": "/api/lesson", "reason": "too long", "question": QUESTION,
                     "user_agent": "SecretBrowser/9.9", "device": "device-tag-zz"}, None, None)
    _hidden(caplog, capsys, "write refused note")
    assert "SecretBrowser/9.9" not in caplog.text
    assert "device-tag-zz" not in caplog.text
