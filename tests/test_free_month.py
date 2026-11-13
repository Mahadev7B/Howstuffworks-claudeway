"""After 3 free questions: one free month, up to 5 new lessons in any 24 hours; saved copies stay free."""

import importlib
import time

import pytest
from flask import g

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
            g.lesson_generated = True  # A brand-new lesson was made.
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


def start_trial(client):
    return client.post("/api/trial/start", json={"email": "mum@example.com", "parent_consent": True})


def use_free_questions(client):
    for number in range(3):
        assert ask(client, f"Free question {number}?").status_code == 200


def test_free_month_is_offered_after_three_questions_and_started_once(site):
    use_free_questions(site)
    offer = ask(site, "Question 4?")
    assert offer.status_code == 402 and offer.get_json()["code"] == "trial_available"
    started = start_trial(site)
    assert started.status_code == 200 and started.get_json()["daily"] == 5
    assert start_trial(site).status_code == 409  # Only one free month per email.
    reply = ask(site, "Question 4?")
    assert reply.status_code == 200 and reply.get_json()["new_left_today"] == 4


def test_five_new_lessons_per_24_hours_then_wait(site):
    use_free_questions(site)
    start_trial(site)
    for number in range(5):
        assert ask(site, f"New question {number}?").status_code == 200
    blocked = ask(site, "One too many?")
    assert blocked.status_code == 429
    body = blocked.get_json()
    assert body["code"] == "daily_limit" and "24 hours" in body["error"] and "about" in body["error"]


def test_saved_copies_and_repeats_dont_count_toward_the_daily_limit(site):
    use_free_questions(site)
    start_trial(site)
    for number in range(5):
        ask(site, f"New question {number}?")
    assert ask(site, "Already asked by someone?").status_code == 200
    assert ask(site, "New question 0?").status_code == 200  # The family's own repeat.


def test_limit_is_a_rolling_24_hours_not_a_calendar_day(site):
    use_free_questions(site)
    start_trial(site)
    for number in range(5):
        ask(site, f"New question {number}?")
    assert ask(site, "Later question?").status_code == 429
    with askers._memory_lock:  # Pretend those five were asked 25 hours ago.
        for key, (ts, generated) in list(askers._memory_asked["mum@example.com"].items()):
            if generated:
                askers._memory_asked["mum@example.com"][key] = (ts - 25 * 3600, generated)
    assert ask(site, "Later question?").status_code == 200


def test_free_month_ends_after_30_days(site):
    use_free_questions(site)
    start_trial(site)
    with askers._memory_lock:
        askers._memory_trial["mum@example.com"] = time.time() - 31 * 24 * 3600
    ended = ask(site, "After the month?")
    assert ended.status_code == 402
    body = ended.get_json()
    assert body["code"] == "subscription_required" and body["subscribe_available"] is False
    assert "free month has ended" in body["error"]


def test_free_month_needs_consent(site):
    response = site.post("/api/trial/start", json={"email": "mum@example.com"})
    assert response.status_code == 400 and response.get_json()["code"] == "consent_required"


def test_redrawn_pictures_are_saved_once(monkeypatch):
    app_module = importlib.import_module("app")
    saved = []
    monkeypatch.setattr(app_module, "db_enabled", True)
    monkeypatch.setattr(app_module, "save_cached_lesson", lambda q, lesson: saved.append(q))
    lesson = {"slides": [{"image_data_url": "data:image/png;base64,aA=="}]}
    app_module._keep_redrawn_pictures("Why?", lesson, had_images=False)
    app_module._keep_redrawn_pictures("Why?", lesson, had_images=True)
    app_module._keep_redrawn_pictures("Why?", {"slides": [{"image_data_url": None}]}, had_images=False)
    assert saved == ["Why?"]


def test_voice_is_reused_from_the_database(monkeypatch):
    app_module = importlib.import_module("app")
    stored = {}
    monkeypatch.setattr(app_module, "db_enabled", True)
    monkeypatch.setattr(app_module, "save_tts", lambda key, audio: stored.__setitem__(key, audio))
    monkeypatch.setattr(app_module, "get_saved_tts", lambda key: stored.get(key))
    app_module._TTS_CACHE.clear()
    app_module._tts_cache_put("Hello owls", b"mp3-bytes")
    app_module._TTS_CACHE.clear()  # As after a restart or an hour later.
    assert app_module._tts_cache_get("Hello owls") == b"mp3-bytes"
    assert app_module._tts_cache_get("Never said") is None
