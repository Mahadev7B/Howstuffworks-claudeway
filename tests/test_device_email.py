"""While email codes are off, one device can use only one email for free lessons."""

import importlib

import pytest
from flask import g

from lesson_platform import askers, ratelimit


@pytest.fixture
def site(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DEVICE_EMAIL_LOCK", raising=False)
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
    return client.post("/api/lesson", json={"question": question, "email": email, "parent_consent": True},
                       headers={"X-Forwarded-Proto": "https"})


def test_second_email_on_the_same_device_is_refused(site):
    _, client = site
    first = ask(client, "mum@example.com")
    assert first.status_code == 200
    cookie = first.headers.get("Set-Cookie", "")
    assert "lilowl_device=" in cookie and "HttpOnly" in cookie and "Secure" in cookie
    assert "mum@example.com" not in cookie  # Signed and encoded, not the plain address.
    refused = ask(client, "neighbour@example.com", "How?")
    assert refused.status_code == 403
    body = refused.get_json()
    assert body["code"] == "device_email" and "mu***@example.com" in body["error"]
    assert ask(client, "MUM@example.com", "When?").status_code == 200  # Same email, any case, still fine.


def test_another_device_can_use_another_email(site):
    app_module, client = site
    ask(client, "mum@example.com")
    other_device = app_module.app.test_client()
    assert ask(other_device, "neighbour@example.com").status_code == 200


def test_typo_before_any_lesson_does_not_lock_the_device(site):
    _, client = site
    blocked = client.post("/api/lesson", json={"question": "Why?", "email": "mum@example.com"})  # No consent.
    assert blocked.status_code == 400 and "lilowl_device=" not in blocked.headers.get("Set-Cookie", "")
    assert ask(client, "dad@example.com").status_code == 200


def test_a_forged_cookie_is_ignored(site):
    _, client = site
    client.set_cookie("lilowl_device", "mum@example.com")
    assert ask(client, "dad@example.com").status_code == 200


def test_more_lessons_request_is_also_locked(site):
    _, client = site
    ask(client, "mum@example.com")
    reply = client.post("/api/more-lessons", json={"email": "neighbour@example.com", "parent_consent": True})
    assert reply.status_code == 403 and reply.get_json()["code"] == "device_email"


def test_lock_is_off_when_email_codes_are_on_or_switched_off(site, monkeypatch):
    app_module, client = site
    ask(client, "mum@example.com")
    monkeypatch.setenv("DEVICE_EMAIL_LOCK", "off")
    assert ask(client, "dad@example.com").status_code == 200
    monkeypatch.delenv("DEVICE_EMAIL_LOCK")
    monkeypatch.setattr(app_module.verification, "required", lambda: True)
    assert app_module._device_lock_on() is False


def test_privacy_page_mentions_the_device_cookie(site):
    _, client = site
    assert b"links the device to that email" in client.get("/privacy").data
