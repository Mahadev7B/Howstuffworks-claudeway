"""Live-site hardening: private status, admin session, trusted client IP, paid-call caps."""

import hashlib
import hmac
import importlib

import pytest

from lesson_platform import ratelimit
from lesson_platform.geo import _client_ip


@pytest.fixture
def site(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("ADMIN_SECRET", raising=False)
    monkeypatch.delenv("RENDER", raising=False)
    app_module = importlib.import_module("app")
    app_module.app.config.update(TESTING=True)
    ratelimit.reset()
    monkeypatch.setattr(app_module.library, "list_published_cards", lambda: [])
    client = app_module.app.test_client()
    yield app_module, client
    ratelimit.reset()


def test_healthz_is_only_ok_and_debug_needs_a_secret_or_admin_session(site, monkeypatch):
    app_module, client = site
    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.get_json() == {"ok": True}
    assert "no-store" in health.headers["Cache-Control"]
    assert b"today_spend" not in health.data

    assert client.get("/debug/status").status_code == 403
    monkeypatch.setenv("ADMIN_SECRET", "s3cret-value")
    assert client.get("/debug/status", headers={"X-Admin-Secret": "nope"}).status_code == 403
    opened = client.get("/debug/status", headers={"X-Admin-Secret": "s3cret-value"})
    assert opened.status_code == 200
    assert "git_commit" in opened.get_json()

    with client.session_transaction() as session:
        session["admin"] = True
    assert client.get("/debug/status").status_code == 200


def test_clear_cache_uses_the_header_secret_and_does_not_leak_errors(site, monkeypatch):
    app_module, client = site
    monkeypatch.setenv("ADMIN_SECRET", "s3cret-value")
    monkeypatch.setattr(app_module, "db_enabled", False)
    assert client.post("/admin/clear-cache").status_code == 403
    blocked = client.post("/admin/clear-cache", headers={"X-Admin-Secret": "s3cret-value-wrong"})
    assert blocked.status_code == 403
    safe = client.post("/admin/clear-cache", headers={"X-Admin-Secret": "s3cret-value"})
    assert safe.status_code == 503
    assert safe.get_json()["error"] == "DB not enabled"


def test_admin_login_locks_out_guesses_and_sets_a_fresh_cookie(site, monkeypatch):
    app_module, client = site
    monkeypatch.setattr(app_module, "_admin_password", lambda: "correct-horse")
    app_module.app.config["SESSION_COOKIE_SECURE"] = True
    with client.session_transaction() as session:
        session["planted"] = "keep-me-out"
    headers = {"X-Forwarded-For": "203.0.113.40", "X-Forwarded-Proto": "https"}
    for _ in range(10):
        wrong = client.post("/admin/login", data={"password": "nope"}, headers=headers)
        assert wrong.status_code == 401
    locked = client.post("/admin/login", data={"password": "nope"}, headers=headers)
    assert locked.status_code == 429
    signed = client.post("/admin/login", data={"password": "correct-horse"}, headers=headers)
    assert signed.status_code == 302
    cookie = signed.headers["Set-Cookie"]
    assert "HttpOnly" in cookie and "SameSite=Lax" in cookie and "Secure" in cookie
    with client.session_transaction() as session:
        assert session.get("admin") is True
        assert "planted" not in session
    assert client.get("/admin/logout").status_code == 405
    assert client.post("/admin/logout").status_code == 302


def test_saved_lesson_delete_needs_the_admin_csrf_token(site, monkeypatch):
    app_module, client = site
    monkeypatch.setattr(app_module.app, "secret_key", "test-secret-not-dev")
    monkeypatch.setattr(app_module, "db_enabled", False)
    with client.session_transaction() as session:
        session["admin"] = True
        session["library_csrf"] = "good-token"
    assert client.post("/admin/delete-cached-lesson", json={"question": "Why?"}).status_code == 403
    allowed = client.post("/admin/delete-cached-lesson",
                          json={"question": "Why?", "csrf_token": "good-token"})
    assert allowed.status_code == 503


def test_public_pages_send_framing_and_sniff_headers(site):
    _, client = site
    page = client.get("/", headers={"X-Forwarded-Proto": "https"})
    assert page.status_code == 200
    assert page.headers["X-Content-Type-Options"] == "nosniff"
    assert page.headers["X-Frame-Options"] == "SAMEORIGIN"
    assert "frame-ancestors 'self'" in page.headers["Content-Security-Policy"]
    assert page.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert "max-age=15552000" in page.headers["Strict-Transport-Security"]
    robots = client.get("/robots.txt").get_data(as_text=True)
    assert "Disallow: /admin" in robots and "Disallow: /debug" in robots


def test_forwarded_for_trusts_the_address_render_appended():
    assert _client_ip("1.1.1.1, 203.0.113.9", "10.0.0.1") == "203.0.113.9"
    assert _client_ip("203.0.113.9, not-an-ip", None) == "203.0.113.9"
    assert _client_ip("", "127.0.0.1") == "127.0.0.1"
    assert _client_ip("javascript:alert(1)", "10.0.0.8") == "10.0.0.8"
    # A supplied CF-Connecting-IP must not replace the appended address.
    assert _client_ip("1.1.1.1, 8.8.8.8", "10.0.0.1", "203.0.113.9") == "8.8.8.8"
    assert _client_ip("203.0.113.9", "10.0.0.1", "not-an-ip") == "203.0.113.9"
    # The header is used only when X-Forwarded-For has no valid address.
    assert _client_ip("", None, "203.0.113.9") == "203.0.113.9"
    assert _client_ip("not-an-ip", "10.0.0.8", "198.51.100.7") == "198.51.100.7"


def test_missing_flask_secret_does_not_use_the_published_dev_key(monkeypatch):
    askers = importlib.import_module("lesson_platform.askers")
    published = hmac.new(b"dev-secret-key", b"free-lesson-ip|203.0.113.9", hashlib.sha256).hexdigest()
    monkeypatch.setenv("FLASK_SECRET_KEY", "dev-secret-key")
    assert askers.ip_hash("203.0.113.9") != published
    monkeypatch.setenv("FLASK_SECRET_KEY", "a-real-secret")
    assert askers.ip_hash("203.0.113.9") == hmac.new(
        b"a-real-secret", b"free-lesson-ip|203.0.113.9", hashlib.sha256).hexdigest()


def test_quiz_stops_at_the_daily_budget_and_tts_stops_at_the_hourly_cap(site, monkeypatch):
    app_module, client = site
    called = []
    monkeypatch.setattr(app_module, "today_spend_usd", lambda: app_module.settings.daily_budget_usd)
    monkeypatch.setattr(app_module, "generate_quiz", lambda *a, **k: called.append(1))
    quiz = client.post("/api/quiz", json={"lesson": {"slides": [{"title": "Sky"}]}},
                       headers={"X-Forwarded-For": "203.0.113.70"})
    assert quiz.status_code == 503 and called == []

    monkeypatch.setattr(app_module, "today_spend_usd", lambda: 0)
    monkeypatch.setattr(app_module.settings, "openai_api_key", "test-key")
    ip = "203.0.113.71"
    for _ in range(60):
        assert ratelimit.allow("tts", ip, 60)
    audio = client.post("/api/tts", json={"text": "Hello"}, headers={"X-Forwarded-For": ip})
    assert audio.status_code == 429


def test_quiz_answers_are_escaped_before_they_reach_the_page(site):
    app_module, _ = site
    with app_module.app.test_request_context("/"):
        from flask import render_template
        page = render_template("lesson.html", question='<script>', lesson={"title": "T", "slides": [
            {"title": "S", "explanation": "E", "fun_fact": "F", "narration": "N",
             "image_data_url": "data:image/png;base64,aA=="}]})
    assert "${q.question}" not in page and "${opt}" not in page
    assert "function escQuiz" in page
    assert "&lt;script&gt;" in page
