"""Refused questions are noted in a temporary file (not the database) and shown on the admin page."""

import copy
import importlib

import pytest

from lesson_platform import ratelimit, spam_log


@pytest.fixture
def site(monkeypatch, tmp_path):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    app_module = importlib.import_module("app")
    app_module.app.config.update(TESTING=True)
    ratelimit.reset()
    monkeypatch.setattr(spam_log, "_PATH", tmp_path / "refused.jsonl")
    monkeypatch.setattr(spam_log, "_submit", lambda fn, *args: fn(*args))  # Write at once in tests.
    yield app_module, app_module.app.test_client()
    ratelimit.reset()


def rows():
    return spam_log.summary("America/New_York")["rows"]


def test_link_spam_is_noted_without_making_a_lesson_or_keeping_the_ip(site, monkeypatch):
    app_module, client = site
    monkeypatch.setattr(app_module, "_track_lesson", lambda *a: pytest.fail("no lesson should be made"))
    reply = client.post("/api/lesson", json={"question": "SCAM SERVICE https://spam.cc/"},
                        headers={"User-Agent": "BadBot/1.0"}, environ_base={"REMOTE_ADDR": "203.0.113.9"})
    assert reply.status_code == 400
    [entry] = rows()
    assert entry["reason"] == "link" and entry["question"] == "SCAM SERVICE https://spam.cc/"
    assert entry["user_agent"] == "BadBot/1.0" and entry["place"] == "—"
    stored = spam_log._PATH.read_text(encoding="utf-8")
    assert "Austin" not in stored and "Texas" not in stored and "United States" not in stored
    assert len(entry["device"]) == 6
    assert "203.0.113.9" not in spam_log._PATH.read_text(encoding="utf-8")


def test_form_post_too_long_and_rude_words_are_noted(site):
    _, client = site
    client.post("/lesson", data={"question": "go to www.spam.net"})
    client.post("/api/lesson", json={"question": "x" * 201})
    client.post("/api/lesson", json={"question": "what is porn"})
    assert sorted(r["reason"] for r in rows()) == ["blocked word", "link", "too long"]


def test_normal_questions_are_not_noted(site, monkeypatch):
    app_module, client = site
    monkeypatch.setattr(app_module, "_asker_gate", lambda *a: (None, None, ("x", "stop here", 400)))
    client.post("/api/lesson", json={"question": "Why is the sky blue?"})
    assert rows() == []


def test_one_device_can_only_add_so_many_and_is_grouped(site):
    _, client = site
    for n in range(30):
        client.post("/api/lesson", json={"question": f"spam {n} https://spam.cc/"},
                    environ_base={"REMOTE_ADDR": "203.0.113.9"})
    summary = spam_log.summary("America/New_York")
    assert summary["total"] == 20 and summary["links"] == 20
    assert summary["devices"][0]["count"] == 20 and len(summary["devices"]) == 1


def test_device_tag_is_the_same_for_one_device_and_differs_between_devices():
    assert spam_log.device_tag("203.0.113.9") == spam_log.device_tag("203.0.113.9")
    assert spam_log.device_tag("203.0.113.9") != spam_log.device_tag("198.51.100.4")
    assert spam_log.device_tag(None) == "—"


def test_admin_page_shows_blocked_questions_escaped(site, monkeypatch):
    app_module, client = site
    client.post("/api/lesson", json={"question": "<a href=https://spam.cc>win</a>"}, headers={"User-Agent": "BadBot"},
                environ_base={"REMOTE_ADDR": "203.0.113.9"})
    monkeypatch.setattr(app_module, "db_enabled", True)
    monkeypatch.setattr(app_module, "admin_load_all", lambda **kw: copy.deepcopy(app_module.askers.db._ADMIN_DEFAULTS))
    with client.session_transaction() as session:
        session["admin"] = True
    page = client.get("/admin").get_data(as_text=True)
    assert 'id="blocked"' in page and "Web link" in page and "BadBot" in page and "Most active devices" in page
    assert "&lt;a href=https://spam.cc&gt;" in page and "<a href=https://spam.cc>" not in page
