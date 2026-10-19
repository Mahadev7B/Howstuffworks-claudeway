"""Publishing is an explicit authenticated, CSRF-protected editorial action."""

import importlib

import pytest


@pytest.fixture
def admin_site(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    module = importlib.import_module("app")
    module.app.config.update(TESTING=True)
    monkeypatch.setattr(module.app, "secret_key", "test-only-private-session-key")
    monkeypatch.setattr(module, "db_enabled", True)
    monkeypatch.setattr(module.library, "list_editorial_publications", lambda: [])
    return module, module.app.test_client()


def test_stage_requires_admin_and_csrf(admin_site, monkeypatch):
    module, client = admin_site
    monkeypatch.setattr(module.library, "stage_cached_lesson", lambda *a: pytest.fail("unauthorized write"))
    assert client.post("/admin/published-lessons", data={}).status_code == 403
    with client.session_transaction() as session:
        session["admin"] = True
        session["library_csrf"] = "good-token"
    assert client.post("/admin/published-lessons", data={}).status_code == 403


def test_editorial_page_lists_drafts(admin_site, monkeypatch):
    module, client = admin_site
    monkeypatch.setattr(module.library, "list_editorial_publications", lambda: [
        {"slug": "sky", "title": "Why sky?", "active": False, "slide_count": 2, "clip_count": 1}])
    with client.session_transaction() as session:
        session["admin"] = True
    response = client.get("/admin/published-lessons")
    assert response.status_code == 200
    assert b"Why sky?" in response.data
    assert b"1 / 2" in response.data
    assert b'href="/admin/published-lessons/sky/preview"' in response.data


def test_private_full_lesson_preview_requires_admin(admin_site, monkeypatch):
    module, client = admin_site
    monkeypatch.setattr(module.library, "get_editorial_publication",
                        lambda slug: pytest.fail("unauthorized draft read"))

    assert client.get("/admin/published-lessons/sky/preview").status_code == 403


def test_admin_full_lesson_preview_uses_saved_versioned_audio(admin_site, monkeypatch):
    module, client = admin_site
    lesson = {"title": "Blue sky", "slides": [{
        "title": "Air scatters light", "subtitle": "Not a roof",
        "image_data_url": "data:image/png;base64,abc", "explanation": "Light meets air.",
        "narration": "Sunlight meets tiny bits of air.", "fun_fact": "Air is all around us.",
    }]}
    monkeypatch.setattr(module.library, "get_editorial_publication", lambda slug: {
        "lesson": lesson, "title": "Why is the sky blue?", "summary": "Light and air",
        "category": "Science", "active": False,
    })
    monkeypatch.setattr(module.library, "get_audio_metadata", lambda slug: {
        0: {"model": "gpt-4o-mini-tts", "voice": "marin", "source_text_hash": "fresh-hash"}
    })
    with client.session_transaction() as session:
        session["admin"] = True

    response = client.get("/admin/published-lessons/sky/preview")

    assert response.status_code == 200
    assert b"Private draft preview" in response.data
    assert b"Air scatters light" in response.data
    assert b"Air is all around us." in response.data
    assert (b'<p class="explanation publication-narration">Sunlight meets tiny bits of air.</p>'
            in response.data)
    assert (b'<div class="fun-fact publication-fun-fact">' in response.data)
    assert b"<strong>Fun fact:</strong> Air is all around us." in response.data
    assert b"Light meets air." not in response.data
    assert b"/admin/published-lessons/sky/audio/0.mp3?v=fresh-hash" in response.data
    assert b'id="autoBtn" class="primary" disabled' not in response.data


def test_editorial_page_shows_each_fun_fact_beside_its_audio(admin_site, monkeypatch):
    module, client = admin_site
    monkeypatch.setattr(module.library, "list_editorial_publications", lambda: [
        {"slug": "sky", "title": "Why sky?", "active": False, "slide_count": 2,
         "clip_count": 2, "cover_slide_index": 0,
         "fun_facts": ["Raindrops can separate sunlight.", "Air is all around us."]}])
    with client.session_transaction() as session:
        session["admin"] = True

    response = client.get("/admin/published-lessons")

    assert response.status_code == 200
    assert b"Raindrops can separate sunlight." in response.data
    assert b"Air is all around us." in response.data


def test_valid_stage_is_inactive(admin_site, monkeypatch):
    module, client = admin_site
    staged = []
    monkeypatch.setattr(module.library, "stage_cached_lesson", lambda *a: staged.append(a) or {"slug": "sky"})
    with client.session_transaction() as session:
        session["admin"] = True
        session["library_csrf"] = "good-token"
    response = client.post("/admin/published-lessons", data={
        "csrf_token": "good-token", "question": "Why sky?", "slug": "sky", "title": "Why sky?",
        "summary": "Light and air", "category": "Science", "sort_order": "1", "featured_rank": "1",
        "cover_slide_number": "2"})
    assert response.status_code == 302
    assert staged[0][1] == "sky"
    assert staged[0][-1] == 1


@pytest.mark.parametrize("path", ["/admin/published-lessons/sky/audio",
                                    "/admin/published-lessons/sky/visibility",
                                    "/admin/published-lessons/sky/cover"])
def test_audio_and_visibility_require_csrf(admin_site, monkeypatch, path):
    module, client = admin_site
    with client.session_transaction() as session:
        session["admin"] = True
        session["library_csrf"] = "good-token"
    assert client.post(path, data={"csrf_token": "wrong"}).status_code == 403


def test_publish_requires_review_confirmation(admin_site, monkeypatch):
    module, client = admin_site
    monkeypatch.setattr(module.library, "set_publication_active", lambda *a: pytest.fail("review not confirmed"))
    with client.session_transaction() as session:
        session["admin"] = True
        session["library_csrf"] = "good-token"
    response = client.post("/admin/published-lessons/sky/visibility",
                           data={"csrf_token": "good-token", "active": "1"})
    assert response.status_code == 400


def test_admin_can_preview_and_update_draft_cover(admin_site, monkeypatch):
    module, client = admin_site
    chosen = []
    monkeypatch.setattr(module.library, "get_editorial_thumbnail", lambda slug: (b"cover", "image/webp"))
    monkeypatch.setattr(module.library, "update_publication_cover", lambda slug, index: chosen.append(index) or True)
    assert client.get("/admin/published-lessons/sky/thumbnail.webp").status_code == 403
    with client.session_transaction() as session:
        session["admin"] = True
        session["library_csrf"] = "good-token"
    assert client.get("/admin/published-lessons/sky/thumbnail.webp").data == b"cover"
    response = client.post("/admin/published-lessons/sky/cover",
                           data={"csrf_token": "good-token", "cover_slide_number": "2"})
    assert response.status_code == 302
    assert chosen == [1]
