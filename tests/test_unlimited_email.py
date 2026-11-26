"""Temporary: asklilowl@gmail.com can keep asking, past the free cap and the device lock."""

import importlib

import pytest
from flask import g

from lesson_platform import askers, ratelimit

ADMIN = "asklilowl@gmail.com"


@pytest.fixture
def site(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DEVICE_EMAIL_LOCK", raising=False)
    monkeypatch.delenv("UNLIMITED_EMAILS", raising=False)
    for name in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    app_module = importlib.import_module("app")
    app_module.app.config.update(TESTING=True)
    monkeypatch.setattr(app_module.askers.db, "_pool", None)
    askers.reset_memory()
    ratelimit.reset()

    def fake_track(endpoint, question, ctx):
        g.lesson_generated = True
        return {"title": question, "slides": []}, None

    monkeypatch.setattr(app_module, "_track_lesson", fake_track)
    monkeypatch.setattr(app_module, "check_question", lambda q: None)
    monkeypatch.setattr(app_module, "_client_context", lambda: {})
    yield app_module, app_module.app.test_client()
    askers.reset_memory()
    ratelimit.reset()


def ask(client, email, question="Why?"):
    return client.post("/api/lesson", json={"question": question, "email": email, "parent_consent": True})


def test_admin_email_skips_the_free_limit_and_a_normal_email_does_not(site):
    app_module, client = site
    for number in range(4):
        response = ask(client, " AskLilOwl@Gmail.com ", f"Admin question {number}?")
        assert response.status_code == 200
        assert response.get_json()["ok"] is True
    # Still recorded, so the admin page and cost path keep seeing these asks.
    assert len(app_module.askers._memory_asked[ADMIN]) == 4

    other = app_module.app.test_client()
    for number in range(3):
        assert ask(other, "mum@example.com", f"Lesson {number}?").status_code == 200
    blocked = ask(other, "mum@example.com", "Lesson 4?")
    assert blocked.status_code == 402
    assert blocked.get_json()["code"] == "waitlist"


def test_admin_email_skips_the_device_lock_and_a_normal_email_does_not(site):
    _, client = site
    assert ask(client, "mum@example.com", "Why?").status_code == 200
    admin = ask(client, ADMIN, "How do owls fly?")
    assert admin.status_code == 200
    refused = ask(client, "neighbour@example.com", "When?")
    assert refused.status_code == 403
    assert refused.get_json()["code"] == "device_email"


def test_unlimited_emails_env_replaces_the_code_default(site, monkeypatch):
    app_module, _ = site
    monkeypatch.setenv("UNLIMITED_EMAILS", "helper@example.com")
    helper = app_module.app.test_client()
    for number in range(4):
        assert ask(helper, "helper@example.com", f"Helper {number}?").status_code == 200

    owner = app_module.app.test_client()
    for number in range(3):
        assert ask(owner, ADMIN, f"Owner {number}?").status_code == 200
    blocked = ask(owner, ADMIN, "Owner 4?")
    assert blocked.status_code == 402
    assert blocked.get_json()["code"] == "waitlist"

    shared = app_module.app.test_client()
    assert ask(shared, "mum@example.com", "Why?").status_code == 200
    assert ask(shared, "helper@example.com", "How?").status_code == 200
    still_locked = ask(shared, "dad@example.com", "When?")
    assert still_locked.status_code == 403
    assert still_locked.get_json()["code"] == "device_email"
