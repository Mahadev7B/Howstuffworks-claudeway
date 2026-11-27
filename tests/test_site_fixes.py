"""Homepage tags, the 13 description updates, share tags, and the sitemap.

The site password stays in front of these pages. Tests cover the gate off
and a signed-in session with the gate on.
"""

import json
import re
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from lesson_platform.site_copy import (
    DESCRIPTIONS_BY_SLUG,
    HOME_DESCRIPTION,
    HOME_H1,
    HOME_TITLE,
    LIBRARY_DESCRIPTIONS,
    PLACEHOLDER_PREFIX,
    PLACEHOLDER_UPDATE_SQL,
    apply_placeholder_summaries,
    placeholder_update_params,
    replacement_summary,
)

ROOT = Path(__file__).resolve().parents[1]
PASSWORD = "correct-horse-battery"
W3C_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}Z)?$")


def _h1s(html: str) -> list[str]:
    return re.findall(r"<h1\b[^>]*>.*?</h1>", html, flags=re.S)


@pytest.fixture
def site(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    monkeypatch.delenv("SITE_PASSWORD", raising=False)
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.delenv("FAL_API_KEY", raising=False)
    monkeypatch.setenv("FLASK_SECRET_KEY", "test-site-gate-secret")
    import app as app_module
    app_module.app.config.update(TESTING=True)
    monkeypatch.setitem(app_module.app.config, "SESSION_COOKIE_SECURE", False)
    app_module.reset_site_login_limits()
    monkeypatch.setattr(app_module, "generate_lesson", lambda *a, **k: pytest.fail("paid lesson call"))
    monkeypatch.setattr(app_module, "_attach_slide_images", lambda *a, **k: pytest.fail("paid image call"))
    monkeypatch.setattr(app_module, "synthesize", lambda *a, **k: pytest.fail("paid voice call"))
    cards = [{
        "slug": "what-is-a-dosa",
        "title": "What Is a Dosa?",
        "summary": DESCRIPTIONS_BY_SLUG["what-is-a-dosa"].summary,
        "category": "World & Culture",
        "sort_order": 17,
        "featured_rank": 1,
        "published_at": datetime(2026, 11, 20, 15, 4, 5, tzinfo=timezone.utc),
    }, {
        "slug": "why-do-flowers-exist",
        "title": "Why Do Flowers Exist?",
        "summary": DESCRIPTIONS_BY_SLUG["why-do-flowers-exist"].summary,
        "category": "Earth & Nature",
        "sort_order": 20,
        "featured_rank": None,
        "published_at": None,
    }]
    monkeypatch.setattr(app_module.library, "list_published_cards", lambda: cards)
    monkeypatch.setattr(app_module.library, "get_published_lesson", lambda slug: {
        "lesson": {"title": cards[0]["title"], "slides": [{
            "title": "Batter", "narration": "Rice and lentils.",
            "explanation": "The batter cooks on a hot pan.", "fun_fact": "Chutney is common.",
            "image_data_url": "data:image/png;base64,aA==",
        }]},
        "title": cards[0]["title"],
        "summary": cards[0]["summary"],
        "category": cards[0]["category"],
    } if slug == "what-is-a-dosa" else None)
    monkeypatch.setattr(
        app_module.library, "get_published_thumbnail",
        lambda slug: (b"webp-bytes", "image/webp") if slug == "what-is-a-dosa" else None,
    )
    return app_module, app_module.app.test_client()


def _sign_in(client):
    posted = client.post("/login", data={"password": PASSWORD, "next": "/"})
    assert posted.status_code == 302
    assert urlsplit(posted.headers["Location"]).path == "/"


def test_homepage_has_one_h1_and_the_exact_tags(site, monkeypatch):
    app_module, client = site
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://preview.example")
    html = client.get("/", headers={"Host": "evil.example"}).get_data(as_text=True)
    assert _h1s(html) == [f'<h1 class="cine-product-title">{HOME_H1}</h1>']
    assert "What Is a Dosa?</h1>" not in html
    assert '<h2 class="cine-feature-title">What Is a Dosa?</h2>' in html
    assert f"<title>{HOME_TITLE}</title>" in html
    assert f'<meta name="description" content="{HOME_DESCRIPTION}">' in html
    assert '<link rel="canonical" href="https://preview.example/">' in html
    assert f'<meta property="og:title" content="{HOME_TITLE}">' in html
    assert f'<meta property="og:description" content="{HOME_DESCRIPTION}">' in html
    assert '<meta property="og:type" content="website">' in html
    assert '<meta property="og:url" content="https://preview.example/">' in html
    assert '<meta property="og:image" content="https://preview.example/static/brand/owl-cutout.png">' in html
    assert '<meta name="twitter:card" content="summary_large_image">' in html
    assert f'<meta name="twitter:title" content="{HOME_TITLE}">' in html
    assert f'<meta name="twitter:description" content="{HOME_DESCRIPTION}">' in html
    assert html.count('property="og:title"') == 1
    assert "evil.example" not in html
    assert "localhost" not in html
    css = (ROOT / "static" / "home.css").read_text(encoding="utf-8")
    assert ".cine-hero h2.cine-feature-title" in css
    assert "font-size: 54px" in css
    assert ".cine-hero h1" not in css


def test_homepage_keeps_one_product_h1_when_no_lesson_is_live(site, monkeypatch):
    app_module, client = site
    monkeypatch.setattr(app_module.library, "list_published_cards", lambda: [])
    html = client.get("/").get_data(as_text=True)
    assert _h1s(html) == [f'<h1 class="cine-product-title">{HOME_H1}</h1>']
    assert "<h2>New lessons are on the way</h2>" in html


def test_placeholder_seed_is_idempotent_and_leaves_other_rows_alone():
    queue = json.loads((ROOT / "assets" / "editorial" / "lesson-queue.json").read_text(encoding="utf-8"))
    assert len(queue) == 30
    by_slug = {item["slug"]: item for item in queue}
    assert set(DESCRIPTIONS_BY_SLUG) <= set(by_slug)
    assert len(LIBRARY_DESCRIPTIONS) == 13
    for item in LIBRARY_DESCRIPTIONS:
        assert by_slug[item.slug]["title"] == item.title
        assert item.summary == replacement_summary(item.slug, PLACEHOLDER_PREFIX + " " + item.title)
    assert "\u2019" in DESCRIPTIONS_BY_SLUG["indias-amazing-languages"].title
    set_clause = PLACEHOLDER_UPDATE_SQL.split("SET", 1)[1].split("FROM", 1)[0]
    assert set_clause.strip() == "summary = v.summary"
    for forbidden in ("title", "lesson", "active", "slug", "thumbnail", "audio", "question", "free"):
        assert forbidden not in set_clause
    params = placeholder_update_params()
    assert params[-1] == PLACEHOLDER_PREFIX + "%"
    assert params[:-1:2] == tuple(item.slug for item in LIBRARY_DESCRIPTIONS)
    assert params[1:-1:2] == tuple(item.summary for item in LIBRARY_DESCRIPTIONS)

    rows = []
    for item in queue:
        summary = PLACEHOLDER_PREFIX + " " + item["title"]
        if item["slug"] not in DESCRIPTIONS_BY_SLUG:
            summary = "Already a real description."
        rows.append({
            "slug": item["slug"],
            "title": item["title"],
            "summary": summary,
            "active": True,
            "free": True,
            "lesson": {"slides": ["unchanged"]},
            "audio": b"mp3",
        })
    dosa = next(row for row in rows if row["slug"] == "what-is-a-dosa")
    dosa["summary"] = "Kept by hand."
    eggs = next(row for row in rows if row["slug"] == "why-some-eggs-have-baby-chicks")
    eggs["summary"] = PLACEHOLDER_PREFIX + " chicks"

    class Cursor:
        def __init__(self, stored):
            self.stored = stored
            self.rowcount = 0

        def execute(self, sql, values):
            assert sql == PLACEHOLDER_UPDATE_SQL
            pairs = list(zip(values[0:-1:2], values[1:-1:2]))
            assert values[-1] == PLACEHOLDER_PREFIX + "%"
            wanted = dict(pairs)
            self.rowcount = 0
            for row in self.stored:
                new = wanted.get(row["slug"])
                if new is None or not str(row["summary"]).startswith(PLACEHOLDER_PREFIX):
                    continue
                row["summary"] = new
                self.rowcount += 1

    cursor = Cursor(rows)
    assert apply_placeholder_summaries(cursor) == 12
    assert dosa["summary"] == "Kept by hand."
    assert eggs["summary"].startswith(PLACEHOLDER_PREFIX)
    assert eggs["lesson"] == {"slides": ["unchanged"]} and eggs["audio"] == b"mp3"
    for item in LIBRARY_DESCRIPTIONS:
        if item.slug == "what-is-a-dosa":
            continue
        row = next(entry for entry in rows if entry["slug"] == item.slug)
        assert row["summary"] == item.summary
        assert row["title"] == item.title
        assert row["active"] is True and row["free"] is True
        assert row["lesson"] == {"slides": ["unchanged"]} and row["audio"] == b"mp3"
    assert len(rows) == 30
    assert sum(1 for row in rows if row["active"] and row["free"]) == 30
    assert apply_placeholder_summaries(cursor) == 0
    assert replacement_summary("what-is-a-dosa", "Kept by hand.") is None
    assert replacement_summary("not-a-library-lesson", PLACEHOLDER_PREFIX + " other") is None


def test_schema_init_runs_the_placeholder_update():
    source = (ROOT / "lesson_platform" / "db.py").read_text(encoding="utf-8")
    start = source.index("def init_db")
    body = source[start:source.index("\ndef enabled", start)]
    assert "apply_placeholder_summaries" in body
    assert body.index("cur.execute(SCHEMA)") < body.index("apply_placeholder_summaries")


def test_explore_cards_and_lesson_meta_use_the_new_descriptions(site):
    _, client = site
    explore = unescape(client.get("/explore").get_data(as_text=True))
    dosa = DESCRIPTIONS_BY_SLUG["what-is-a-dosa"].summary
    flowers = DESCRIPTIONS_BY_SLUG["why-do-flowers-exist"].summary
    assert dosa in explore and flowers in explore
    lesson = unescape(client.get("/explore/what-is-a-dosa").get_data(as_text=True))
    assert f'<meta name="description" content="{dosa}">' in lesson
    assert f'<meta property="og:description" content="{dosa}">' in lesson
    assert '<meta property="og:title" content="What Is a Dosa?">' in lesson
    assert '<meta property="og:type" content="article">' in lesson
    assert '<meta property="og:url" content="https://www.asklilowl.com/explore/what-is-a-dosa">' in lesson
    assert '<meta property="og:image" content="https://www.asklilowl.com/explore/what-is-a-dosa/thumbnail.webp">' in lesson
    assert '<meta name="twitter:card" content="summary_large_image">' in lesson
    assert '<meta name="twitter:title" content="What Is a Dosa?">' in lesson
    assert f'<meta name="twitter:description" content="{dosa}">' in lesson
    assert "localhost" not in lesson


def test_thumbnail_is_cacheable_and_share_without_one_uses_the_brand_image(site):
    app_module, client = site
    thumb = client.get("/explore/what-is-a-dosa/thumbnail.webp")
    assert thumb.status_code == 200
    assert thumb.data == b"webp-bytes"
    cache = thumb.headers["Cache-Control"]
    assert "public" in cache and "no-store" not in cache
    assert int(cache.split("max-age=")[1].split(",")[0]) >= 86400
    etag = thumb.headers["ETag"]
    assert etag
    again = client.get("/explore/what-is-a-dosa/thumbnail.webp", headers={"If-None-Match": etag})
    assert again.status_code == 304
    assert again.headers["ETag"] == etag
    assert "public" in again.headers["Cache-Control"]
    missing = client.get("/explore/missing/thumbnail.webp")
    assert missing.status_code == 404
    brand = client.get("/static/brand/owl-cutout.png")
    assert brand.status_code == 200 and brand.data
    with app_module.app.test_request_context("/s/shared-lesson"):
        from flask import render_template
        html = render_template(
            "share.html", question="Why is rain wet?", slug="shared-lesson",
            lesson={"title": "Rain", "slides": [{
                "title": "Clouds", "narration": "Water falls.",
                "explanation": "Clouds get heavy.", "fun_fact": "Rain is fresh.",
            }]},
        )
    assert 'content="https://www.asklilowl.com/static/brand/owl-cutout.png"' in html
    assert '<meta property="og:url" content="https://www.asklilowl.com/s/shared-lesson">' in html
    assert '<meta name="twitter:card" content="summary_large_image">' in html
    assert "evil.example" not in html


def test_share_text_does_not_use_the_asker_question():
    import app as app_module
    title, description = app_module._lesson_share_text({
        "title": "How Do Cars Work",
        "question": "why does the blue car go",
        "summary": DESCRIPTIONS_BY_SLUG["how-do-cars-work"].summary,
        "slides": [{"explanation": "Unused when a summary is saved."}],
    })
    assert title == "How Do Cars Work"
    assert description == DESCRIPTIONS_BY_SLUG["how-do-cars-work"].summary
    assert "blue car" not in description
    title, description = app_module._lesson_share_text({
        "title": "Rain",
        "question": "why does the blue car go",
        "slides": [{"explanation": "Clouds get heavy."}],
    })
    assert description == "Clouds get heavy."
    assert "blue car" not in description


def test_video_download_page_has_share_tags(site, tmp_path):
    app_module, client = site
    video = tmp_path / "reel.mp4"
    video.write_bytes(b"mp4")
    description = DESCRIPTIONS_BY_SLUG["how-do-cars-work"].summary
    app_module._VIDEO_JOBS["job-share"] = {
        "status": "done",
        "path": str(video),
        "created_at": 10**10,
        "share_title": "How Do Cars Work",
        "share_description": description,
    }
    html = unescape(client.get("/api/export-video/job-share/stream").get_data(as_text=True))
    assert "<title>How Do Cars Work — Ask Lil Owl</title>" in html
    assert f'<meta property="og:title" content="How Do Cars Work — Ask Lil Owl">' in html
    assert f'<meta property="og:description" content="{description}">' in html
    assert '<meta property="og:type" content="article">' in html
    assert '<meta property="og:url" content="https://www.asklilowl.com/api/export-video/job-share/stream">' in html
    assert '<meta property="og:image" content="https://www.asklilowl.com/static/brand/owl-cutout.png">' in html
    assert '<meta name="twitter:card" content="summary_large_image">' in html
    assert f'<meta name="twitter:description" content="{description}">' in html
    assert 'src="/api/export-video/job-share/file"' in html
    app_module._VIDEO_JOBS.pop("job-share", None)


def test_sitemap_lists_public_pages_and_active_lessons_with_lastmod(site, monkeypatch):
    app_module, client = site
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://www.asklilowl.com")
    body = client.get("/sitemap.xml", headers={"Host": "evil.example"}).get_data(as_text=True)
    assert body.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    locs = re.findall(r"<loc>(.*?)</loc>", body)
    lastmods = re.findall(r"<lastmod>(.*?)</lastmod>", body)
    assert locs == [
        "https://www.asklilowl.com/",
        "https://www.asklilowl.com/explore",
        "https://www.asklilowl.com/lesson",
        "https://www.asklilowl.com/privacy",
        "https://www.asklilowl.com/explore/what-is-a-dosa",
        "https://www.asklilowl.com/explore/why-do-flowers-exist",
    ]
    assert lastmods[0] == app_module.deploy_lastmod()
    assert lastmods[3] == "2026-11-26"
    assert lastmods[4] == "2026-11-20T15:04:05Z"
    assert lastmods[5] == app_module.deploy_lastmod()
    assert all(W3C_DATE.fullmatch(value) for value in lastmods)
    assert len(locs) == len(lastmods)
    joined = " ".join(locs)
    for blocked in ("/admin", "/api/", "/login", "/logout", "/debug", "evil.example", "hidden"):
        assert blocked not in joined
    assert app_module.public_sitemap_paths() == ["/", "/explore", "/lesson", "/privacy"]


def test_gate_still_blocks_the_new_pages_and_the_exempt_list_is_unchanged(site, monkeypatch):
    app_module, client = site
    monkeypatch.setenv("SITE_PASSWORD", PASSWORD)
    home = client.get("/")
    assert home.status_code == 302
    assert urlsplit(home.headers["Location"]).path == "/login"
    assert parse_qs(urlsplit(home.headers["Location"]).query)["next"] == ["/"]
    sitemap = client.get("/sitemap.xml")
    assert sitemap.status_code == 302
    assert parse_qs(urlsplit(sitemap.headers["Location"]).query)["next"] == ["/sitemap.xml"]
    lesson = client.get("/explore/what-is-a-dosa")
    assert lesson.status_code == 302
    video = client.get("/api/export-video/job-share/stream")
    assert video.status_code == 401
    assert client.get("/login").status_code == 200
    assert client.get("/static/favicon.svg").status_code == 200
    assert client.get("/static/brand/owl-cutout.png").status_code == 200
    assert client.get("/healthz").status_code == 200
    assert client.get("/favicon.ico").status_code == 200
    robots = client.get("/robots.txt")
    assert robots.status_code == 200
    assert robots.get_data(as_text=True) == "User-agent: *\nDisallow: /\n"
    _sign_in(client)
    signed_in = client.get("/").get_data(as_text=True)
    assert _h1s(signed_in) == [f'<h1 class="cine-product-title">{HOME_H1}</h1>']
    assert f"<title>{HOME_TITLE}</title>" in signed_in
    assert f'<meta name="description" content="{HOME_DESCRIPTION}">' in signed_in
    signed_sitemap = client.get("/sitemap.xml").get_data(as_text=True)
    assert "<lastmod>" in signed_sitemap
    assert "https://www.asklilowl.com/explore/what-is-a-dosa" in signed_sitemap
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    gate = source[source.index("def _require_site_password"):source.index("app.before_request")]
    assert '{"/healthz", "/robots.txt", "/favicon.ico"}' in gate or "{'/healthz', '/robots.txt', '/favicon.ico'}" in gate
    assert "/sitemap.xml" not in gate.split("path == \"/login\"", 1)[0]
