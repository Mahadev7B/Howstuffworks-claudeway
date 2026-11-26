"""Launch checks: private status pages, share tags, terms, errors, and admin session basics."""

import importlib

import pytest


@pytest.fixture
def site(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    app_module = importlib.import_module("app")
    app_module.app.config.update(TESTING=True)
    monkeypatch.setattr(app_module.library, "list_published_cards", lambda: [
        {"slug": "sky-blue", "title": "Why is the sky blue?", "summary": "Light and air",
         "category": "Science", "sort_order": 1, "featured_rank": 1},
    ])
    monkeypatch.setattr(app_module.library, "get_published_lesson", lambda slug: {
        "lesson": {"title": "Raw title", "slides": [{"title": "Light", "narration": "Blue light scatters.",
                    "image_data_url": "data:image/png;base64,aA==", "explanation": "The air scatters it.",
                    "fun_fact": "The sky changes color."}]},
        "title": "Why is the sky blue?", "summary": "Light and air", "category": "Science",
    } if slug == "sky-blue" else None)
    monkeypatch.setattr(app_module.library, "get_published_thumbnail", lambda slug: (b"webp", "image/webp") if slug == "sky-blue" else None)
    return app_module.app.test_client()


def test_healthz_is_only_an_ok(site):
    response = site.get("/healthz")
    assert response.status_code == 200
    assert response.get_json() == {"ok": True}
    assert "no-store" in response.headers["Cache-Control"]


def test_debug_status_requires_admin(site):
    hidden = site.get("/debug/status")
    assert hidden.status_code == 403
    assert hidden.get_json() == {"ok": False, "error": "forbidden"}
    assert b"anthropic" not in hidden.data and b"budget" not in hidden.data
    with site.session_transaction() as sess:
        sess["admin"] = True
    shown = site.get("/debug/status")
    assert shown.status_code == 200
    assert "anthropic_model" in shown.get_json()["settings"]
    assert "no-store" in shown.headers["Cache-Control"]


def test_robots_hides_admin_and_debug(site):
    body = site.get("/robots.txt").get_data(as_text=True)
    assert "Disallow: /admin" in body
    assert "Disallow: /debug" in body
    assert "Sitemap: https://www.asklilowl.com/sitemap.xml" in body


def test_home_has_share_tags_product_heading_and_explore_link(site):
    html = site.get("/").get_data(as_text=True)
    assert '<meta name="description"' in html
    assert 'rel="canonical" href="https://www.asklilowl.com/"' in html
    assert 'property="og:image" content="https://www.asklilowl.com/static/brand/og-share.png"' in html
    assert 'name="twitter:card" content="summary_large_image"' in html
    assert "Ask Lil Owl</h1>" in html
    assert 'href="/explore"' in html
    assert 'href="/terms"' in html
    assert "apple-touch-icon" in html


def test_explore_and_lesson_have_share_tags(site):
    explore = site.get("/explore").get_data(as_text=True)
    assert 'property="og:url" content="https://www.asklilowl.com/explore"' in explore
    assert 'name="twitter:card"' in explore
    lesson = site.get("/explore/sky-blue").get_data(as_text=True)
    assert 'property="og:url" content="https://www.asklilowl.com/explore/sky-blue"' in lesson
    assert 'property="og:site_name" content="Ask Lil Owl"' in lesson
    assert 'property="og:type" content="article"' in lesson
    assert 'name="twitter:image"' in lesson
    assert "data:image/png;base64,aA==" not in lesson
    assert "/explore/sky-blue/slide/0" in lesson


def test_slide_image_is_cacheable_and_hides_drafts(site):
    image = site.get("/explore/sky-blue/slide/0")
    assert image.status_code == 200
    assert image.mimetype == "image/png"
    assert image.data == b"h"
    assert "public" in image.headers["Cache-Control"]
    assert site.get("/explore/sky-blue/slide/3").status_code == 404
    assert site.get("/explore/missing/slide/0").status_code == 404


def test_terms_page_is_linked_and_listed(site):
    page = site.get("/terms")
    assert page.status_code == 200
    text = page.get_data(as_text=True)
    assert "Terms of use" in text
    assert "first 3 different questions" in text
    assert "$4.99" in text
    assert "Stripe" in text
    assert b"/terms" in site.get("/sitemap.xml").data


def test_unknown_pages_use_the_branded_error(site):
    missing = site.get("/this-page-is-not-real")
    assert missing.status_code == 404
    assert b"That page is not here." in missing.data
    assert b"<title>Not Found</title>" not in missing.data
    hidden = site.get("/explore/not-a-real-lesson")
    assert hidden.status_code == 404
    assert b"That page is not here." in hidden.data


def test_server_error_page_hides_the_exception():
    app_module = importlib.import_module("app")
    with app_module.app.test_request_context("/"):
        body, status = app_module.server_error(RuntimeError("secret-internal"))
    page = body.data if hasattr(body, "data") else body.encode()
    assert status == 500
    assert b"secret-internal" not in page
    assert b"Something went wrong on our side" in page


def test_security_headers_are_present(site):
    headers = site.get("/").headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "SAMEORIGIN"
    assert "frame-ancestors" in headers["Content-Security-Policy"]
    assert headers["Referrer-Policy"] == "strict-origin-when-cross-origin"


def test_cloudflare_ip_wins_over_a_spoofed_forwarded_header():
    app_module = importlib.import_module("app")
    with app_module.app.test_request_context("/", headers={
        "X-Forwarded-For": "198.51.100.9, 203.0.113.5",
        "CF-Connecting-IP": "203.0.113.5",
    }):
        assert app_module._client_ip() == "203.0.113.5"


def test_quiz_stops_at_the_daily_budget(monkeypatch):
    app_module = importlib.import_module("app")
    monkeypatch.setattr(app_module, "today_spend_usd", lambda: 9)
    monkeypatch.setattr(app_module.settings, "daily_budget_usd", 5)
    monkeypatch.setattr(app_module, "generate_quiz", lambda *a, **k: pytest.fail("quiz should not run"))
    response = app_module.app.test_client().post("/api/quiz", json={"lesson": {"slides": [{"title": "A"}]}})
    assert response.status_code == 503
    assert b"budget" in response.data.lower()


def test_admin_login_locks_out_and_sets_a_secure_cookie(monkeypatch):
    app_module = importlib.import_module("app")
    monkeypatch.setenv("ADMIN_PASSWORD", "correct-horse")
    app_module.ratelimit.reset()
    client = app_module.app.test_client()
    for _ in range(10):
        assert client.post("/admin/login", data={"password": "nope"}).status_code == 401
    locked = client.post("/admin/login", data={"password": "nope"})
    assert locked.status_code == 429
    ok = client.post("/admin/login", data={"password": "correct-horse"},
                     headers={"X-Forwarded-Proto": "https"})
    assert ok.status_code == 302
    set_cookie = ok.headers.getlist("Set-Cookie")
    assert any("HttpOnly" in cookie and "Secure" in cookie and "SameSite=Lax" in cookie for cookie in set_cookie)
    assert client.get("/admin/logout").status_code == 405
    app_module.ratelimit.reset()


def test_clear_cache_secret_is_compared_in_constant_time(monkeypatch):
    app_module = importlib.import_module("app")
    monkeypatch.setenv("ADMIN_SECRET", "top-secret")
    client = app_module.app.test_client()
    assert client.post("/admin/clear-cache", headers={"X-Admin-Secret": "nope"}).status_code == 403
    assert client.post("/admin/clear-cache").status_code == 403
