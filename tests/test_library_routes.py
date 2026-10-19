"""Public browsing must be read-only and hide unpublished records."""

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
        "lesson": {"title": "Raw cached lesson title", "slides": [{"title": "Light", "narration": "Blue light scatters.",
                    "image_data_url": "data:image/png;base64,aA==", "explanation": "The air scatters it.",
                    "fun_fact": "The sky changes color."}]},
        "title": "Why is the sky blue?", "summary": "Light and air", "category": "Science",
    } if slug == "sky-blue" else None)
    monkeypatch.setattr(app_module.library, "get_published_thumbnail", lambda slug: (b"webp", "image/webp") if slug == "sky-blue" else None)
    monkeypatch.setattr(app_module.library, "get_published_audio", lambda slug, index: b"mp3" if slug == "sky-blue" and index == 0 else None)
    monkeypatch.setattr(app_module, "generate_lesson", lambda *a, **k: pytest.fail("generation on GET"))
    monkeypatch.setattr(app_module, "_attach_slide_images", lambda *a, **k: pytest.fail("image generation on GET"))
    monkeypatch.setattr(app_module, "synthesize", lambda *a, **k: pytest.fail("voice generation on GET"))
    return app_module.app.test_client()


def test_explore_lists_only_published_cards(site):
    response = site.get("/explore")
    assert response.status_code == 200
    assert b"Why is the sky blue?" in response.data
    assert b"/explore/sky-blue" in response.data
    assert b"Featured lesson" in response.data
    assert b"data-library-rail" in response.data
    assert b"data-scroll" in response.data
    assert b"fetch(" not in response.data


def test_home_keeps_ask_form_and_features_curated_lesson(site):
    response = site.get("/")
    assert response.status_code == 200
    assert b'id="askForm"' in response.data
    assert b'name="question"' in response.data
    assert b"/explore/sky-blue" in response.data


def test_published_detail_uses_saved_audio(site):
    response = site.get("/explore/sky-blue")
    assert response.status_code == 200
    assert b"AI-generated narration" in response.data
    assert b"/explore/sky-blue/audio/0.mp3" in response.data
    assert b"/api/tts" not in response.data
    assert b'rel="canonical"' in response.data


def test_published_detail_uses_curated_metadata(site):
    response = site.get("/explore/sky-blue")
    html = response.get_data(as_text=True)
    assert "<title>Why is the sky blue? — Ask Lil Owl</title>" in html
    assert '<meta name="description" content="Light and air">' in html
    assert '<meta property="og:description" content="Light and air">' in html
    assert '<h1>Why is the sky blue?</h1>' in html


def test_explore_featured_row_follows_editorial_rank(site, monkeypatch):
    from app import library
    monkeypatch.setattr(library, "list_published_cards", lambda: [
        {"slug": "first", "title": "First", "summary": "One", "category": "Science", "sort_order": 1, "featured_rank": 2},
        {"slug": "second", "title": "Second", "summary": "Two", "category": "Animals", "sort_order": 2, "featured_rank": 1},
    ])
    html = site.get("/explore").get_data(as_text=True)
    featured_section = html.split('id="featured-lessons"', 1)[1].split('id="all-lessons"', 1)[0]
    assert featured_section.index('/explore/second') < featured_section.index('/explore/first')


def test_existing_share_player_keeps_its_voice_path(site):
    from flask import render_template
    with site.application.test_request_context("/s/share-old"):
        html = render_template("share.html", question="Why sky?", slug="share-old",
                               lesson={"title": "Why sky?", "slides": [{"title": "Light",
                               "narration": "Blue light scatters.", "explanation": "Air scatters it.",
                               "fun_fact": "Sky colors change."}]})
    assert "/api/tts" in html
    assert "Someone shared this lesson" in html
    assert "Air scatters it." in html
    assert "Sky colors change." in html


@pytest.mark.parametrize("url", ["/explore/hidden", "/explore/hidden/thumbnail.webp",
                                "/explore/hidden/audio/0.mp3"])
def test_hidden_slug_is_invisible(site, url):
    assert site.get(url).status_code == 404


def test_saved_media_and_sitemap(site):
    thumb = site.get("/explore/sky-blue/thumbnail.webp")
    voice = site.get("/explore/sky-blue/audio/0.mp3")
    sitemap = site.get("/sitemap.xml")
    assert thumb.mimetype == "image/webp" and b"webp" == thumb.data
    assert voice.mimetype == "audio/mpeg" and b"mp3" == voice.data
    assert "no-store" in thumb.headers["Cache-Control"]
    assert "no-store" in voice.headers["Cache-Control"]
    assert b"/explore/sky-blue" in sitemap.data
    assert b"/explore/hidden" not in sitemap.data
