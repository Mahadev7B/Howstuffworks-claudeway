"""Site-wide password gate. Off unless SITE_PASSWORD is set."""

import logging
import time
from urllib.parse import parse_qs, urlsplit

import pytest

from lesson_platform import ratelimit


PASSWORD = "correct-horse-battery"
WRONG = "nope-nope-nope"


@pytest.fixture
def site(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("SITE_PASSWORD", raising=False)
    monkeypatch.delenv("RENDER", raising=False)
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    app_module = __import__("app")
    app_module.app.config.update(TESTING=True)
    monkeypatch.setitem(app_module.app.config, "SESSION_COOKIE_SECURE", False)
    app_module.reset_site_login_limits()
    ratelimit.reset()
    monkeypatch.setattr(app_module.library, "list_published_cards", lambda: [])
    monkeypatch.setattr(app_module, "generate_lesson", lambda *a, **k: pytest.fail("paid lesson call"))
    client = app_module.app.test_client()
    yield app_module, client
    app_module.reset_site_login_limits()


def _location(response):
    return urlsplit(response.headers["Location"])


def _session_cookie(response):
    matches = [value for value in response.headers.getlist("Set-Cookie") if value.startswith("session=")]
    assert matches
    return matches[0]


def test_gate_is_off_when_the_password_is_unset_or_blank(site, monkeypatch, value=None):
    app_module, client = site
    if value is not None:
        monkeypatch.setenv("SITE_PASSWORD", value)
    home = client.get("/")
    assert home.status_code == 200
    assert "X-Robots-Tag" not in home.headers
    privacy = client.get("/privacy")
    assert privacy.status_code == 200
    assert "X-Robots-Tag" not in privacy.headers
    robots = client.get("/robots.txt")
    assert robots.status_code == 200
    assert robots.mimetype == "text/plain"
    body = robots.get_data(as_text=True)
    assert "Allow: /" in body and "Disallow: /admin" in body and "Disallow: /debug" in body
    assert "X-Robots-Tag" not in robots.headers
    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.get_json() == {"ok": True}
    assert "X-Robots-Tag" not in health.headers
    feedback = client.post("/api/feedback", json={})
    assert feedback.status_code == 503
    assert feedback.get_json()["error"] != "password required"
    assert client.get("/debug/status").status_code == 403
    assert client.get("/terms").status_code == 404
    opened = client.get("/login")
    assert opened.status_code == 302
    assert _location(opened).path == "/"
    assert "X-Robots-Tag" not in opened.headers


@pytest.mark.parametrize("value", ["", "   "])
def test_blank_site_password_leaves_routes_unchanged(site, monkeypatch, value):
    test_gate_is_off_when_the_password_is_unset_or_blank(site, monkeypatch, value)


def test_gate_blocks_pages_and_json_until_the_password_is_entered(site, monkeypatch):
    app_module, client = site
    monkeypatch.setenv("SITE_PASSWORD", PASSWORD)
    home = client.get("/")
    assert home.status_code == 302
    assert _location(home).path == "/login"
    assert parse_qs(_location(home).query)["next"] == ["/"]
    assert home.headers["X-Robots-Tag"] == "noindex, nofollow"

    privacy = client.get("/privacy")
    assert parse_qs(_location(privacy).query)["next"] == ["/privacy"]
    explore = client.get("/explore/sky/thumbnail.webp")
    assert explore.status_code == 302
    assert parse_qs(_location(explore).query)["next"] == ["/explore/sky/thumbnail.webp"]
    share = client.get("/s/some-share")
    assert share.status_code == 302
    assert parse_qs(_location(share).query)["next"] == ["/s/some-share"]
    admin = client.get("/admin")
    assert _location(admin).path == "/login"
    assert parse_qs(_location(admin).query)["next"] == ["/admin"]
    missing_page = client.get("/terms")
    assert missing_page.status_code == 302
    assert _location(missing_page).path == "/login"

    lesson = client.post("/api/lesson", json={"question": "Why is the sky blue?"})
    assert lesson.status_code == 401
    assert lesson.get_json() == {"ok": False, "error": "password required"}
    assert lesson.headers["X-Robots-Tag"] == "noindex, nofollow"
    unknown = client.get("/api/does-not-exist")
    assert unknown.status_code == 401
    assert unknown.get_json() == {"ok": False, "error": "password required"}
    status = client.get("/debug/status")
    assert status.status_code == 401
    assert status.get_json() == {"ok": False, "error": "password required"}
    webhook = client.post("/stripe/webhook", json={"type": "ping"})
    assert webhook.status_code == 401
    assert webhook.get_json() == {"ok": False, "error": "password required"}
    private = client.post("/api/admin/private-lessons", json={})
    assert private.status_code == 401
    assert private.get_json() == {"ok": False, "error": "password required"}

    wrong = client.post("/login", data={"password": WRONG, "next": "/privacy"})
    assert wrong.status_code == 401
    assert WRONG.encode() not in wrong.data
    assert b"That password" in wrong.data and b"Please try again" in wrong.data
    assert wrong.headers["X-Robots-Tag"] == "noindex, nofollow"
    with client.session_transaction() as session:
        assert session.get("site_ok") is not True
    assert client.get("/privacy").status_code == 302

    signed = client.post("/login", data={"password": PASSWORD, "next": "/privacy"})
    assert signed.status_code == 302
    assert _location(signed).path == "/privacy"
    cookie = _session_cookie(signed)
    assert "HttpOnly" in cookie
    assert "SameSite=Lax" in cookie
    assert "Max-Age=2592000" in cookie
    assert "Secure" not in cookie
    with client.session_transaction() as session:
        assert session.get("site_ok") is True
        assert session.permanent is True
    assert client.get("/privacy").status_code == 200
    assert client.get("/").status_code == 200
    authed = client.get("/explore")
    assert authed.status_code == 200
    assert authed.headers["X-Robots-Tag"] == "noindex, nofollow"

    admin_page = client.get("/admin/login")
    assert admin_page.status_code == 200
    assert b"Admin login" in admin_page.data
    logged_out = client.get("/logout")
    assert logged_out.status_code == 302
    assert _location(logged_out).path == "/login"
    assert client.get("/privacy").status_code == 302


def test_secure_cookie_follows_the_existing_render_flag(site, monkeypatch):
    app_module, client = site
    monkeypatch.setenv("SITE_PASSWORD", PASSWORD)
    monkeypatch.setitem(app_module.app.config, "SESSION_COOKIE_SECURE", True)
    signed = client.post("/login", data={"password": PASSWORD},
                         headers={"X-Forwarded-Proto": "https"})
    cookie = _session_cookie(signed)
    assert "Secure" in cookie
    assert "HttpOnly" in cookie and "SameSite=Lax" in cookie


def test_failed_logins_are_limited_per_hashed_ip(site, monkeypatch, caplog):
    app_module, client = site
    monkeypatch.setenv("SITE_PASSWORD", PASSWORD)
    ip = "203.0.113.55"
    other = "203.0.113.56"
    headers = {"X-Forwarded-For": ip}
    caplog.set_level(logging.DEBUG)
    for _ in range(10):
        failed = client.post("/login", data={"password": WRONG}, headers=headers)
        assert failed.status_code == 401
    blocked = client.post("/login", data={"password": WRONG}, headers=headers)
    assert blocked.status_code == 429
    assert b"Too many tries" in blocked.data
    assert WRONG.encode() not in blocked.data
    still = client.post("/login", data={"password": WRONG}, headers=headers)
    assert still.status_code == 429
    other_try = client.post("/login", data={"password": WRONG}, headers={"X-Forwarded-For": other})
    assert other_try.status_code == 401
    assert ip not in caplog.text and other not in caplog.text
    assert PASSWORD not in caplog.text and WRONG not in caplog.text
    assert ip not in repr(app_module._site_login_failures)
    assert other not in repr(app_module._site_login_failures)
    assert all(len(key) == 64 for key in app_module._site_login_failures)

    later = time.time() + app_module._SITE_LOGIN_WINDOW_S + 5
    monkeypatch.setattr(app_module.time, "time", lambda: later)
    expired = client.post("/login", data={"password": WRONG}, headers=headers)
    assert expired.status_code == 401
    assert ip not in caplog.text and PASSWORD not in caplog.text and WRONG not in caplog.text


def test_exempt_paths_stay_open_and_robots_hides_the_site(site, monkeypatch):
    _, client = site
    monkeypatch.setenv("SITE_PASSWORD", PASSWORD)
    login = client.get("/login")
    assert login.status_code == 200
    assert b"Enter the site password" in login.data
    assert b'name="password"' in login.data
    assert login.headers["X-Robots-Tag"] == "noindex, nofollow"
    static = client.get("/static/favicon.svg")
    assert static.status_code == 200
    assert static.data
    assert static.headers["X-Robots-Tag"] == "noindex, nofollow"
    missing = client.get("/static/does-not-exist.css")
    assert missing.status_code == 404
    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.get_json() == {"ok": True}
    assert health.headers["X-Robots-Tag"] == "noindex, nofollow"
    robots = client.get("/robots.txt")
    assert robots.status_code == 200
    assert robots.mimetype == "text/plain"
    assert robots.get_data(as_text=True) == "User-agent: *\nDisallow: /\n"
    assert robots.headers["X-Robots-Tag"] == "noindex, nofollow"


@pytest.mark.parametrize("target", [
    "https://evil.example/phish",
    "http://evil.example",
    "//evil.example",
    "///evil.example",
    "/\\evil.example",
    "https://www.asklilowl.com/privacy",
    "javascript:alert(1)",
])
def test_next_redirect_rejects_external_urls(site, monkeypatch, target):
    _, client = site
    monkeypatch.setenv("SITE_PASSWORD", PASSWORD)
    signed = client.post("/login", data={"password": PASSWORD, "next": target})
    assert signed.status_code == 302
    location = signed.headers["Location"]
    assert _location(signed).path == "/"
    assert "evil" not in location
    assert "javascript" not in location


def test_next_redirect_allows_a_same_site_path(site, monkeypatch):
    _, client = site
    monkeypatch.setenv("SITE_PASSWORD", PASSWORD)
    signed = client.post("/login", data={"password": PASSWORD, "next": "/explore?q=sky"})
    assert signed.status_code == 302
    assert _location(signed).path == "/explore"
    assert parse_qs(_location(signed).query)["q"] == ["sky"]
    form = client.get("/login?next=https://evil.example/phish")
    assert form.status_code == 302
    assert _location(form).path == "/"
    assert "evil" not in form.headers["Location"]


def test_admin_login_stays_behind_the_site_gate(site, monkeypatch):
    app_module, client = site
    monkeypatch.setenv("SITE_PASSWORD", PASSWORD)
    monkeypatch.setattr(app_module, "_admin_password", lambda: "admin-secret")
    client.post("/login", data={"password": PASSWORD, "next": "/admin"})
    denied = client.post("/admin/login", data={"password": "admin-secret"})
    assert denied.status_code == 302
    assert _location(denied).path == "/admin"
    assert client.get("/privacy").status_code == 200
    with client.session_transaction() as session:
        assert session.get("admin") is True
        assert session.get("site_ok") is True
    client.post("/admin/logout")
    assert client.get("/privacy").status_code == 200
    assert client.get("/admin").status_code == 302
    assert _location(client.get("/admin")).path == "/admin/login"
    client.get("/logout")
    assert client.get("/admin/login").status_code == 302
    assert _location(client.get("/admin/login")).path == "/login"
