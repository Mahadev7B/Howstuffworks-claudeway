"""While PRIVATE_ASKING is on, only a browser logged in to /admin can ask new questions or make quizzes."""

import importlib

import pytest
from flask import g

from lesson_platform import askers, ratelimit


@pytest.fixture
def site(monkeypatch):
    monkeypatch.setenv("PRIVATE_ASKING", "on")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    for name in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    app_module = importlib.import_module("app")
    app_module.app.config.update(TESTING=True)
    monkeypatch.setattr(app_module.askers.db, "_pool", None)
    askers.reset_memory()
    ratelimit.reset()
    made = []

    def fake_track(endpoint, question, ctx):
        made.append(question)
        g.lesson_generated = True
        return {"title": question, "slides": []}, None

    monkeypatch.setattr(app_module, "_track_lesson", fake_track)
    monkeypatch.setattr(app_module, "check_question", lambda q: None)
    monkeypatch.setattr(app_module, "_client_context", lambda: {})
    monkeypatch.setattr(app_module, "generate_quiz", lambda lesson, settings: made.append("quiz") or {"questions": []})
    monkeypatch.setattr(app_module.library, "list_published_cards", lambda: [])
    yield app_module, app_module.app.test_client(), made
    askers.reset_memory()
    ratelimit.reset()


def ask(client):
    return client.post("/api/lesson", json={"question": "Why?", "email": "mum@example.com", "parent_consent": True})


def quiz(client):
    return client.post("/api/quiz", json={"lesson": {"title": "T", "slides": [{"number": 1, "title": "t", "explanation": "e"}]}})


def test_visitors_are_told_asking_is_paused_and_nothing_is_made(site):
    _, client, made = site
    reply = ask(client)
    assert reply.status_code == 403 and reply.get_json()["code"] == "private"
    assert quiz(client).status_code == 403
    assert client.post("/lesson", data={"question": "Why?", "email": "mum@example.com", "parent_consent": "yes"}).status_code == 403
    assert made == []


def test_typing_the_admin_email_is_not_enough(site):
    _, client, made = site
    reply = client.post("/api/lesson", json={"question": "Why?", "email": "Asklilowl@gmail.com", "parent_consent": True})
    assert reply.status_code == 403 and made == []


def test_logged_in_admin_can_ask_and_make_quizzes(site):
    _, client, made = site
    with client.session_transaction() as session:
        session["admin"] = True
    assert ask(client).status_code == 200
    assert quiz(client).status_code == 200
    assert made == ["Why?", "quiz"]


def test_home_page_shows_the_paused_note_only_to_visitors(site):
    _, client, _ = site
    assert b'id="privateNote"' in client.get("/").data
    with client.session_transaction() as session:
        session["admin"] = True
    assert b'id="privateNote"' not in client.get("/").data


def test_switching_it_off_opens_asking_to_everyone(site, monkeypatch):
    _, client, made = site
    monkeypatch.setenv("PRIVATE_ASKING", "off")
    assert ask(client).status_code == 200 and made == ["Why?"]
    assert b'id="privateNote"' not in client.get("/").data


def test_ready_made_library_pages_still_open(site):
    _, client, _ = site
    assert client.get("/privacy").status_code == 200
