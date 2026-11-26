"""Free lessons are 3 per client IP. A second email on that network is not refused."""

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


def ask(client, email, question="Why?", ip="203.0.113.9", cf=None):
    headers = {"X-Forwarded-For": ip, "X-Forwarded-Proto": "https"}
    if cf:
        headers["CF-Connecting-IP"] = cf
    return client.post("/api/lesson", json={"question": question, "email": email, "parent_consent": True},
                       headers=headers)


def test_a_second_email_is_allowed_and_no_device_cookie_is_set(site):
    _, client = site
    first = ask(client, "mum@example.com", "Why?")
    assert first.status_code == 200
    assert "lilowl_device=" not in first.headers.get("Set-Cookie", "")
    second = ask(client, "dad@example.com", "How?")
    assert second.status_code == 200
    assert second.get_json()["ok"] is True
    client.set_cookie("lilowl_device", "mum@example.com")
    assert ask(client, "neighbour@example.com", "When?").status_code == 200


def test_three_free_lessons_are_shared_by_every_email_on_one_ip(site):
    _, client = site
    for number, email in enumerate(("mum@example.com", "dad@example.com", "kid@example.com")):
        assert ask(client, email, f"Question {number}?").status_code == 200
    blocked = ask(client, "neighbour@example.com", "Question 3?")
    assert blocked.status_code == 402
    assert blocked.get_json()["code"] == "waitlist"


def test_another_ip_gets_its_own_three(site):
    _, client = site
    for number in range(3):
        assert ask(client, "mum@example.com", f"Question {number}?", ip="203.0.113.9").status_code == 200
    assert ask(client, "dad@example.com", "Question 3?", ip="203.0.113.9").status_code == 402
    assert ask(client, "dad@example.com", "Question 3?", ip="198.51.100.7").status_code == 200


def test_cloudflare_connecting_ip_beats_a_spoofed_forwarded_header(site):
    _, client = site
    for number in range(3):
        assert ask(client, "mum@example.com", f"Question {number}?",
                   ip="1.1.1.1, 8.8.8.8", cf="203.0.113.9").status_code == 200
    assert ask(client, "dad@example.com", "Question 3?", ip="9.9.9.9", cf="203.0.113.9").status_code == 402
    assert ask(client, "dad@example.com", "A new one?", ip="203.0.113.9", cf="198.51.100.7").status_code == 200


def test_the_proxy_appended_address_is_the_client(site):
    _, client = site
    # The leftmost address is visitor-supplied. Render appends the real client at the end.
    for number in range(3):
        assert ask(client, "mum@example.com", f"Question {number}?", ip="198.51.100.7, 203.0.113.9").status_code == 200
    assert ask(client, "dad@example.com", "Question 3?", ip="1.2.3.4, 203.0.113.9").status_code == 402
    assert ask(client, "dad@example.com", "A new one?", ip="203.0.113.9, 198.51.100.7").status_code == 200


def test_the_same_question_counts_once_for_the_ip(site):
    _, client = site
    assert ask(client, "mum@example.com", "Why is the sky blue?").status_code == 200
    assert ask(client, "dad@example.com", "Why is the sky blue?").status_code == 200
    assert ask(client, "dad@example.com", "How do birds fly?").status_code == 200
    assert ask(client, "mum@example.com", "Why do we dream?").status_code == 200
    assert ask(client, "kid@example.com", "What is lightning?").status_code == 402


def test_only_a_hash_of_the_ip_is_remembered(site):
    app_module, client = site
    assert ask(client, "mum@example.com").status_code == 200
    digest = app_module.askers.ip_hash("203.0.113.9")
    assert digest and digest != "203.0.113.9"
    assert "203.0.113.9" not in repr(app_module.askers._memory_ip)
    assert digest in app_module.askers._memory_ip
    assert digest != app_module.askers.ip_hash("198.51.100.7")
    assert app_module.askers.ip_hash("203.0.113.9") == digest
    assert app_module.askers.ip_hash(None) is None


def test_a_subscriber_does_not_use_up_the_networks_free_lessons(site):
    app_module, client = site
    for number in range(3):
        assert ask(client, "mum@example.com", f"Question {number}?").status_code == 200
    app_module.askers.set_subscription("dad@example.com", "active")
    assert ask(client, "dad@example.com", "An extra one?").status_code == 200
    assert len(app_module.askers._memory_ip[app_module.askers.ip_hash("203.0.113.9")]) == 3
    assert ask(client, "kid@example.com", "Question 3?").status_code == 402


def test_asking_for_more_is_not_blocked_by_the_device(site):
    _, client = site
    ask(client, "mum@example.com")
    reply = client.post("/api/more-lessons", json={"email": "dad@example.com", "parent_consent": True},
                        headers={"X-Forwarded-For": "203.0.113.9"})
    assert reply.status_code != 403
    assert reply.get_json().get("code") != "device_email"


def test_privacy_page_keeps_the_address_out_and_describes_the_code(site):
    _, client = site
    page = client.get("/privacy").data
    assert b"We don't store internet (IP) addresses." in page
    assert b"one-way code" in page
    assert b"links the device to that email" not in page
