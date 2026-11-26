"""3 free lessons, then a subscription waitlist question; saved copies and repeats stay free."""

import importlib

import pytest
from flask import g, render_template

from lesson_platform import askers, ratelimit


@pytest.fixture
def site(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    for name in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "STRIPE_SECRET_KEY", "STRIPE_PRICE_ID"):
        monkeypatch.delenv(name, raising=False)
    app_module = importlib.import_module("app")
    app_module.app.config.update(TESTING=True)
    monkeypatch.setattr(app_module.askers.db, "_pool", None)
    askers.reset_memory()
    ratelimit.reset()
    saved_questions = {"Already asked by someone?"}

    def fake_track(endpoint, question, ctx):
        if question not in saved_questions:
            g.lesson_generated = True
        return {"title": question, "slides": []}, None

    monkeypatch.setattr(app_module, "_track_lesson", fake_track)
    monkeypatch.setattr(app_module, "_question_is_saved", lambda q: q in saved_questions)
    monkeypatch.setattr(app_module, "check_question", lambda q: None)
    monkeypatch.setattr(app_module, "_client_context", lambda: {})
    yield app_module.app.test_client()
    askers.reset_memory()
    ratelimit.reset()


def ask(client, question):
    return client.post("/api/lesson", json={"question": question, "email": "mum@example.com", "parent_consent": True})


def answer(client, choice, email="mum@example.com"):
    return client.post("/api/waitlist", json={"email": email, "parent_consent": True, "choice": choice})


def test_three_free_lessons_then_the_waitlist_question(site):
    assert [ask(site, f"Lesson {n}?").get_json()["free_left"] for n in range(3)] == [2, 1, 0]
    blocked = ask(site, "Lesson 4?")
    assert blocked.status_code == 402
    body = blocked.get_json()
    assert body["code"] == "waitlist" and body["waitlist_choice"] is None
    assert "free month" not in body["error"]


def test_a_saved_lesson_uses_one_free_question_the_first_time(site):
    reply = ask(site, "Already asked by someone?")
    assert reply.status_code == 200
    assert reply.get_json()["free_left"] == 2


def test_saved_copies_and_repeats_stay_free_after_three(site):
    for n in range(3):
        ask(site, f"Lesson {n}?")
    assert ask(site, "Already asked by someone?").status_code == 200
    assert ask(site, "Lesson 0?").status_code == 200


@pytest.mark.parametrize("choice", ["subscribe", "browsing", "price"])
def test_each_answer_is_saved_and_reported_back(site, choice):
    assert answer(site, choice).get_json() == {"ok": True, "choice": choice}
    for n in range(3):
        ask(site, f"Lesson {n}?")
    assert ask(site, "Lesson 4?").get_json()["waitlist_choice"] == choice


def test_answer_can_be_changed(site):
    answer(site, "browsing")
    answer(site, "subscribe")
    assert askers.allowance("mum@example.com", "x")["waitlist_choice"] == "subscribe"


@pytest.mark.parametrize("body,status", [({"email": "mum@example.com", "parent_consent": True, "choice": "free stuff"}, 400),
                                         ({"email": "mum@example.com", "choice": "subscribe"}, 400),
                                         ({"parent_consent": True, "choice": "subscribe"}, 400)])
def test_bad_answers_are_refused(site, body, status):
    assert site.post("/api/waitlist", json=body).status_code == status


def test_free_month_cannot_be_started_from_the_site(site):
    assert site.post("/api/trial/start", json={"email": "mum@example.com", "parent_consent": True}).status_code == 404


def test_lesson_page_asks_when_the_lesson_finishes(site):
    app_module = importlib.import_module("app")
    with app_module.app.test_request_context("/"):
        page = render_template("lesson.html", question="Why?", lesson={"title": "T", "slides": [
            {"title": "S", "explanation": "E", "fun_fact": "F", "narration": "N", "image_data_url": "data:image/png;base64,aA=="}]})
    assert 'id="waitlistBox"' in page and "waitlist.js" in page
    assert "lessonFinished()" in page
    assert "lessons are free" in page


def test_home_says_three_free_lessons_without_a_free_month(site, monkeypatch):
    app_module = importlib.import_module("app")
    monkeypatch.setattr(app_module.library, "list_published_cards", lambda: [])
    page = site.get("/").data
    assert b"Your first 3 lessons are free" in page
    assert b"free month" not in page.lower()
    assert b"waitlist.js" in page


def test_two_more_lessons_after_review_and_the_question_comes_back(site):
    answer(site, "browsing")
    for n in range(3):
        ask(site, f"Lesson {n}?")
    assert ask(site, "Lesson 4?").status_code == 402
    assert askers.grant_bonus("mum@example.com", 2) is True
    reply = ask(site, "Lesson 4?")
    assert reply.status_code == 200
    body = reply.get_json()
    assert body["free_left"] == 1 and body["free_limit"] == 5
    assert body["waitlist_choice"] is None  # Cleared, so the lesson page asks again.
    assert ask(site, "Lesson 5?").get_json()["free_left"] == 0
    blocked = ask(site, "Lesson 6?").get_json()
    assert blocked["code"] == "waitlist" and "5 free lessons" in blocked["error"]


def test_grant_needs_an_existing_email(site):
    assert askers.grant_bonus("nobody@example.com", 2) is False


def test_parent_can_ask_for_more_and_approval_clears_the_request(site):
    for n in range(3):
        ask(site, f"Lesson {n}?")
    blocked = ask(site, "Lesson 4?").get_json()
    assert blocked["code"] == "waitlist" and blocked["more_requested"] is False
    sent = site.post("/api/more-lessons", json={"email": "mum@example.com", "parent_consent": True})
    assert sent.get_json() == {"ok": True}
    assert ask(site, "Lesson 4?").get_json()["more_requested"] is True
    askers.grant_bonus("mum@example.com", 2)
    assert askers.allowance("mum@example.com", "x")["more_requested"] is False
    assert ask(site, "Lesson 4?").status_code == 200


def test_more_lessons_needs_consent_and_a_known_email(site):
    assert site.post("/api/more-lessons", json={"email": "mum@example.com"}).status_code == 400
    assert site.post("/api/more-lessons", json={"email": "new@example.com", "parent_consent": True}).status_code == 404


def test_limit_screen_has_the_ask_for_more_button(site, monkeypatch):
    app_module = importlib.import_module("app")
    monkeypatch.setattr(app_module.library, "list_published_cards", lambda: [])
    page = site.get("/").data
    assert b'id="moreBtn"' in page and b"Ask for 2 more free lessons" in page


def test_question_box_has_a_minus_button_to_close_it(site, monkeypatch):
    app_module = importlib.import_module("app")
    monkeypatch.setattr(app_module.library, "list_published_cards", lambda: [])
    page = site.get("/").get_data(as_text=True)
    assert 'id="owlMinimize"' in page and 'aria-label="Close the question box"' in page
