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
                                    "/admin/published-lessons/sky/approval",
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


def test_admin_can_mark_private_draft_approved_without_publishing(admin_site, monkeypatch):
    module, client = admin_site
    changes = []
    monkeypatch.setattr(module.library, "set_publication_approved",
                        lambda slug, approved: changes.append((slug, approved)) or True)
    with client.session_transaction() as session:
        session["admin"] = True
        session["library_csrf"] = "good-token"

    response = client.post("/admin/published-lessons/sky/approval",
                           data={"csrf_token": "good-token", "approved": "1"})

    assert response.status_code == 302
    assert changes == [("sky", True)]


def test_editorial_page_distinguishes_approval_from_publication(admin_site, monkeypatch):
    module, client = admin_site
    monkeypatch.setattr(module.library, "list_editorial_publications", lambda: [{
        "slug": "sky", "title": "Why sky?", "active": False,
        "approved_for_publish": True, "slide_count": 2, "clip_count": 2,
        "cover_slide_index": 0, "fun_facts": ["One", "Two"],
    }])
    with client.session_transaction() as session:
        session["admin"] = True

    response = client.get("/admin/published-lessons")

    assert response.status_code == 200
    assert b"Approved for publication" in response.data
    assert b"Private draft" in response.data


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


def _admin_session(client):
    with client.session_transaction() as session:
        session["admin"] = True
        session["library_csrf"] = "good-token"


def test_publish_all_requires_csrf_and_review_confirmation(admin_site, monkeypatch):
    module, client = admin_site
    monkeypatch.setattr(module.library, "publish_all_approved", lambda: pytest.fail("unconfirmed publish"))
    assert client.post("/admin/published-lessons/publish-all", data={"reviewed_approved": "1"}).status_code == 403
    _admin_session(client)
    assert client.post("/admin/published-lessons/publish-all", data={"csrf_token": "good-token"}).status_code == 400


def test_publish_all_reports_lessons_that_could_not_go_live(admin_site, monkeypatch):
    module, client = admin_site
    _admin_session(client)
    monkeypatch.setattr(module.library, "publish_all_approved", lambda: {"published": ["sky"], "failed": ["rain"]})
    response = client.post("/admin/published-lessons/publish-all",
                           data={"csrf_token": "good-token", "reviewed_approved": "1"})
    assert response.status_code == 302
    assert "rain" in response.headers["Location"]


def test_unpublish_all_requires_csrf_and_takes_lessons_down(admin_site, monkeypatch):
    module, client = admin_site
    called = []
    monkeypatch.setattr(module.library, "unpublish_all", lambda: called.append(True) or ["sky"])
    assert client.post("/admin/published-lessons/unpublish-all", data={}).status_code == 403
    _admin_session(client)
    assert client.post("/admin/published-lessons/unpublish-all", data={"csrf_token": "good-token"}).status_code == 302
    assert called == [True]


def test_admin_page_offers_publish_all_without_held_lesson(admin_site, monkeypatch):
    module, client = admin_site
    _admin_session(client)
    row = {"title": "T", "active": False, "approved_for_publish": True, "slide_count": 4, "clip_count": 4,
           "cover_slide_index": 0, "fun_facts": ["a", "b", "c", "d"]}
    monkeypatch.setattr(module.library, "list_editorial_publications", lambda: [
        {**row, "slug": "sky"}, {**row, "slug": "solve-a-2x2-rubiks-cube"}, {**row, "slug": "rain", "active": True}])
    html = client.get("/admin/published-lessons").get_data(as_text=True)
    assert "Publish all 1 approved" in html
    assert "/admin/published-lessons/unpublish-all" in html


def _clip_upload(*names):
    from io import BytesIO
    return {"csrf_token": "good-token", "clips": [(BytesIO(b"mp3-" + name.encode()), name) for name in names]}


def test_replace_audio_requires_admin_and_csrf(admin_site, monkeypatch):
    module, client = admin_site
    monkeypatch.setattr(module.library_import, "replace_private_audio", lambda *a: pytest.fail("unauthorized write"))
    assert client.post("/admin/published-lessons/sky/replace-audio", data={},
                       content_type="multipart/form-data").status_code == 403
    with client.session_transaction() as session:
        session["admin"] = True
        session["library_csrf"] = "good-token"
    assert client.post("/admin/published-lessons/sky/replace-audio", data={"csrf_token": "bad"},
                       content_type="multipart/form-data").status_code == 403


def test_replace_audio_maps_files_to_slides_by_name(admin_site, monkeypatch):
    module, client = admin_site
    _admin_session(client)
    saved = {}
    monkeypatch.setattr(module.library_import, "replace_private_audio", lambda slug, clips: saved.update(slug=slug, clips=clips))
    names = [f"slide-0{i}-kokoro-af-heart-v1.mp3" for i in (2, 1, 4, 3)]
    response = client.post("/admin/published-lessons/sky/replace-audio", data=_clip_upload(*names),
                           content_type="multipart/form-data")
    assert response.status_code == 302 and "error" not in response.headers["Location"]
    assert saved["slug"] == "sky" and sorted(saved["clips"]) == [0, 1, 2, 3]
    assert saved["clips"][1] == b"mp3-slide-02-kokoro-af-heart-v1.mp3"


@pytest.mark.parametrize("names", [["voice.mp3"], ["slide-01.wav"], ["slide-01.mp3", "slide-01-copy.mp3"], ["slide-00.mp3"]])
def test_replace_audio_rejects_unclear_files(admin_site, monkeypatch, names):
    module, client = admin_site
    _admin_session(client)
    monkeypatch.setattr(module.library_import, "replace_private_audio", lambda *a: pytest.fail("bad files saved"))
    response = client.post("/admin/published-lessons/sky/replace-audio", data=_clip_upload(*names),
                           content_type="multipart/form-data")
    assert response.status_code == 302 and "error=" in response.headers["Location"]


def test_replace_audio_reports_live_lesson_refusal(admin_site, monkeypatch):
    module, client = admin_site
    _admin_session(client)
    def refuse(slug, clips):
        raise ValueError("Only a private draft's audio can be changed")
    monkeypatch.setattr(module.library_import, "replace_private_audio", refuse)
    response = client.post("/admin/published-lessons/sky/replace-audio", data=_clip_upload("slide-01.mp3"),
                           content_type="multipart/form-data")
    assert response.status_code == 302 and "error=" in response.headers["Location"]


def test_admin_page_offers_clip_replacement_only_for_private_lessons(admin_site, monkeypatch):
    module, client = admin_site
    _admin_session(client)
    row = {"title": "T", "approved_for_publish": True, "slide_count": 4, "clip_count": 4,
           "cover_slide_index": 0, "fun_facts": ["a", "b", "c", "d"]}
    monkeypatch.setattr(module.library, "list_editorial_publications", lambda: [
        {**row, "slug": "private-one", "active": False}, {**row, "slug": "live-one", "active": True}])
    html = client.get("/admin/published-lessons").get_data(as_text=True)
    assert "/admin/published-lessons/private-one/replace-audio" in html
    assert "/admin/published-lessons/live-one/replace-audio" not in html
