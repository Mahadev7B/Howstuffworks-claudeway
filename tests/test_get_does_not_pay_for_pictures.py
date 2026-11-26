"""Opening a page must not call the paid picture service."""

import importlib


def _lesson():
    return {
        "title": "Bees",
        "slides": [
            {"title": "Missing", "explanation": "Needs a picture.", "fun_fact": "New.",
             "narration": "Needs a picture.", "image_prompt": "a friendly bee"},
        ],
    }


def _install(monkeypatch):
    app_module = importlib.import_module("app")
    app_module._RECENT_LESSONS.clear()
    monkeypatch.setattr(app_module.settings, "image_provider", "flux")
    monkeypatch.setattr(app_module.settings, "fal_api_key", "test-key")
    monkeypatch.setattr(app_module, "db_enabled", True)
    monkeypatch.setattr(app_module, "save_cached_lesson", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "_client_context", lambda: {})
    monkeypatch.setattr(app_module, "_render_with_matplotlib", lambda slide: slide.__setitem__(
        "image_data_url", "data:image/png;base64,NEW") or True)
    monkeypatch.setattr(app_module, "_compress_image", lambda raw: (raw, "image/png"))
    paid = []

    def batch(prompts, settings, negative=None):
        paid.append("batch")
        return [(b"png", "image/png", 0.01) for _ in prompts]

    def one(prompt, settings, negative=None):
        paid.append("one")
        return b"png", "image/png", 0.01

    monkeypatch.setattr(app_module, "generate_images_batch", batch)
    monkeypatch.setattr(app_module, "generate_image", one)
    monkeypatch.setattr(app_module.library, "list_published_cards", lambda: [])
    monkeypatch.setattr(app_module.library, "get_published_lesson", lambda slug: {
        "lesson": _lesson(), "summary": "Bees", "title": "Bees",
    } if slug == "bees" else None)
    monkeypatch.setattr(app_module.library, "get_published_slide_image",
                        lambda slug, index: (b"h", "image/png") if slug == "bees" and index == 0 else None)
    monkeypatch.setattr(app_module.library, "get_published_thumbnail",
                        lambda slug: (b"webp", "image/webp") if slug == "bees" else None)
    monkeypatch.setattr(app_module, "get_lesson_by_slug", lambda slug: {
        **_lesson(), "_share_question": "How do bees fly?",
    } if slug == "share-bees" else None)
    return app_module, paid


def test_get_routes_do_not_call_the_paid_picture_service(monkeypatch):
    app_module, paid = _install(monkeypatch)
    client = app_module.app.test_client()
    question = "How do bees fly?"
    app_module._recent_put(question, _lesson())
    assert client.get("/lesson", query_string={"q": question}).status_code == 200
    app_module._RECENT_LESSONS.clear()
    assert client.get("/s/share-bees").status_code == 200
    assert client.get("/explore/bees").status_code == 200
    assert client.get("/explore/bees/slide/0").status_code == 200
    assert client.get("/explore/bees/thumbnail.webp").status_code == 200
    assert paid == []


def test_admin_redraw_is_the_only_paid_picture_path_and_it_checks_budget(monkeypatch):
    app_module, paid = _install(monkeypatch)
    monkeypatch.setattr(app_module.app, "secret_key", "test-secret-not-dev")
    monkeypatch.setattr(app_module, "today_spend_usd", lambda: 0)
    monkeypatch.setattr(app_module, "db_enabled", False)
    client = app_module.app.test_client()
    question = "How do bees fly?"
    app_module._recent_put(question, _lesson())
    with client.session_transaction() as session:
        session["admin"] = True
        session["library_csrf"] = "good-token"
    blocked = client.post("/admin/missing-pictures", data={"question": question})
    assert blocked.status_code == 403
    assert paid == []
    monkeypatch.setattr(app_module, "today_spend_usd", lambda: app_module.settings.daily_budget_usd)
    over = client.post("/admin/missing-pictures", data={"question": question, "csrf_token": "good-token"})
    assert over.status_code == 503
    assert paid == []
    monkeypatch.setattr(app_module, "today_spend_usd", lambda: 0)
    drawn = client.post("/admin/missing-pictures", data={"question": question, "csrf_token": "good-token"})
    assert drawn.status_code == 302
    assert paid == ["batch"]
