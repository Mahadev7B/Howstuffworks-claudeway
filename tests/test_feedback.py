"""A thumb and a written comment from one visit are both saved, to the same feedback entry."""

import importlib

import pytest

from lesson_platform import ratelimit


@pytest.fixture
def site(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    app_module = importlib.import_module("app")
    app_module.app.config.update(TESTING=True)
    ratelimit.reset()
    saved = []
    monkeypatch.setattr(app_module, "db_enabled", True)
    monkeypatch.setattr(app_module, "save_feedback", lambda question, rating, comment, **kw: saved.append(
        {"question": question, "rating": rating, "comment": comment, **kw}))
    monkeypatch.setattr(app_module, "pin_cached_lesson", lambda q: None)
    monkeypatch.setattr(app_module, "save_cached_lesson", lambda q, d: None)
    monkeypatch.setattr(app_module, "record_api_call", lambda **kw: None)
    monkeypatch.setattr(app_module, "_client_context", lambda: {"country": "United States", "user_agent": "test"})
    client = app_module.app.test_client()
    client.saved = saved
    yield client
    ratelimit.reset()


def send(client, **body):
    return client.post("/api/feedback", json={"question": "Why is the sky blue?", **body})


def test_thumb_then_comment_share_one_entry(site):
    assert send(site, rating="up", client_id="abc12345-visit").status_code == 200
    assert send(site, rating="up", comment="  Loved the pictures  ", client_id="abc12345-visit").status_code == 200
    assert [s["client_id"] for s in site.saved] == ["abc12345-visit", "abc12345-visit"]
    assert site.saved[0]["comment"] is None
    assert site.saved[1]["comment"] == "Loved the pictures"
    assert site.saved[1]["rating"] == "up"


def test_real_save_feedback_accepts_the_request_context(monkeypatch):
    """The route used to pass ip_hash into save_feedback and every post returned 500."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    app_module = importlib.import_module("app")
    from lesson_platform import askers, db
    app_module.app.config.update(TESTING=True, PROPAGATE_EXCEPTIONS=False)
    ratelimit.reset()
    captured = []
    monkeypatch.setattr(app_module, "db_enabled", True)
    monkeypatch.setattr(db, "_pool", object())
    monkeypatch.setattr(db, "_bg", lambda fn, params: captured.append(params))
    monkeypatch.setattr(app_module, "pin_cached_lesson", lambda q: None)
    monkeypatch.setattr(app_module, "save_cached_lesson", lambda q, d: None)
    monkeypatch.setattr(app_module, "record_api_call", lambda **kw: None)
    client = app_module.app.test_client()
    response = client.post(
        "/api/feedback",
        json={"question": "Why is the sky blue?", "rating": "up", "client_id": "visit-0099"},
        headers={"X-Forwarded-For": "203.0.113.9"},
    )
    assert response.status_code == 200
    digest = askers.ip_hash("203.0.113.9")
    assert captured and digest not in captured[0]
    assert "203.0.113.9" not in repr(captured[0])
    ratelimit.reset()


def test_comment_without_a_thumb_is_saved(site):
    response = send(site, comment="Please add more space lessons", client_id="visit-0001")
    assert response.status_code == 200
    assert site.saved == [{"question": "Why is the sky blue?", "rating": None, "comment": "Please add more space lessons",
                           "client_id": "visit-0001", "country": "United States", "user_agent": "test"}]
    assert "city" not in site.saved[0] and "region" not in site.saved[0]


@pytest.mark.parametrize("body", [{}, {"rating": "maybe"}, {"comment": "   "}, {"rating": "sideways", "comment": "x"}])
def test_needs_a_valid_thumb_or_some_text(site, body):
    assert send(site, **body).status_code == 400
    assert site.saved == []


def test_odd_client_ids_are_ignored(site):
    send(site, rating="down", client_id="x'); DROP TABLE feedback;--")
    send(site, rating="down", client_id="short")
    assert [s["client_id"] for s in site.saved] == [None, None]


def test_lesson_page_keeps_comment_box_open_after_a_thumb():
    app_module = importlib.import_module("app")
    with app_module.app.test_request_context("/"):
        from flask import render_template
        page = render_template("lesson.html", question="Why?", lesson={"title": "T", "slides": [
            {"title": "S", "explanation": "E", "fun_fact": "F", "narration": "N", "image_data_url": "data:image/png;base64,aA=="}]})
    assert "client_id: fbId" in page
    assert "You can add a comment too" in page
    assert "Send comment" in page


def test_save_feedback_accepts_comment_only_and_passes_client_id(monkeypatch):
    db = importlib.import_module("lesson_platform.db")
    captured = []
    monkeypatch.setattr(db, "_pool", object())
    monkeypatch.setattr(db, "_bg", lambda fn, params: captured.append(params))
    db.save_feedback("Why?", None, "More dinosaurs", client_id="visit-0002",
                     country="United States")
    assert captured[0][1] is None and captured[0][2] == "More dinosaurs" and captured[0][-1] == "visit-0002"
    assert "United States" in captured[0]
    with pytest.raises(ValueError):
        db.save_feedback("Why?", None, None)
