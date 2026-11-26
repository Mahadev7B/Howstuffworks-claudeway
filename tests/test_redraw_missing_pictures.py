"""Opening a saved or shared lesson draws only the slides that have no picture."""

import importlib


def _lesson():
    return {
        "title": "Bees",
        "slides": [
            {"title": "Kept", "explanation": "Already drawn.", "fun_fact": "Kept.",
             "narration": "Already drawn.", "image_data_url": "data:image/png;base64,KEEP"},
            {"title": "Missing", "explanation": "Needs a picture.", "fun_fact": "New.",
             "narration": "Needs a picture."},
        ],
    }


def _install(monkeypatch):
    app_module = importlib.import_module("app")
    app_module._RECENT_LESSONS.clear()
    monkeypatch.setattr(app_module.settings, "image_provider", "matplotlib")
    monkeypatch.setattr(app_module, "db_enabled", True)
    drawn = []
    saved = []

    def draw(slide):
        drawn.append(slide.get("title"))
        if slide.get("title") == "Fails":
            return False
        slide["image_data_url"] = "data:image/png;base64,NEW"
        return True

    monkeypatch.setattr(app_module, "_render_with_matplotlib", draw)
    monkeypatch.setattr(app_module, "save_cached_lesson", lambda question, lesson: saved.append((question, lesson)))
    return app_module, drawn, saved


def test_saved_and_shared_pages_draw_only_missing_pictures(monkeypatch):
    app_module, drawn, saved = _install(monkeypatch)
    question = "How do bees fly?"
    app_module._recent_put(question, _lesson())
    saved_page = app_module.app.test_client().get("/lesson", query_string={"q": question})
    html = saved_page.get_data(as_text=True)
    assert saved_page.status_code == 200
    assert drawn == ["Missing"]
    assert "data:image/png;base64,KEEP" in html
    assert "data:image/png;base64,NEW" in html
    assert "Illustration unavailable" not in html
    assert saved and saved[0][0] == question
    assert saved[0][1]["slides"][0]["image_data_url"].endswith("KEEP")
    assert saved[0][1]["slides"][1]["image_data_url"].endswith("NEW")

    drawn.clear()
    saved.clear()
    app_module._RECENT_LESSONS.clear()
    shared_lesson = _lesson()
    shared_lesson["_share_question"] = question
    monkeypatch.setattr(app_module, "get_lesson_by_slug", lambda slug: shared_lesson)
    shared_page = app_module.app.test_client().get("/s/share-bees")
    shared_html = shared_page.get_data(as_text=True)
    assert shared_page.status_code == 200
    assert drawn == ["Missing"]
    assert "data:image/png;base64,KEEP" in shared_html
    assert "data:image/png;base64,NEW" in shared_html
    assert saved and saved[0][0] == question


def test_failed_redraw_still_shows_the_placeholder(monkeypatch):
    app_module, drawn, saved = _install(monkeypatch)
    question = "How do bees fly?"
    lesson = _lesson()
    lesson["slides"][1]["title"] = "Fails"
    app_module._recent_put(question, lesson)
    page = app_module.app.test_client().get("/lesson", query_string={"q": question})
    html = page.get_data(as_text=True)
    assert page.status_code == 200
    assert drawn == ["Fails"]
    assert "data:image/png;base64,KEEP" in html
    assert "Illustration unavailable" in html
    assert saved == []
