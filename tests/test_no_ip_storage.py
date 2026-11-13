"""Visitors' IP addresses are used briefly for location and limits, never stored or logged."""

import importlib
import logging

from lesson_platform import db, ratelimit


def test_client_context_keeps_location_but_not_the_address(monkeypatch):
    app_module = importlib.import_module("app")
    monkeypatch.setattr(app_module, "extract_and_lookup", lambda fwd, remote: ("203.0.113.9", "Austin", "Texas", "United States"))
    with app_module.app.test_request_context("/", headers={"X-Forwarded-For": "203.0.113.9"}):
        ctx = app_module._client_context()
    assert "ip_address" not in ctx
    assert "203.0.113.9" not in repr(ctx)
    assert ctx["city"] == "Austin"


def test_database_writes_never_include_an_address(monkeypatch):
    captured = []
    monkeypatch.setattr(db, "_pool", object())
    monkeypatch.setattr(db, "_bg", lambda fn, params: captured.append(params))
    db.record_api_call(endpoint="/api/lesson", question="Why?", model="m", success=True, ip_address="203.0.113.9")
    db.save_feedback("Why?", "up", None, ip_address="203.0.113.9")
    assert captured and all("203.0.113.9" not in repr(params) for params in captured)


def test_hourly_limit_counts_in_memory_without_keeping_the_address():
    ratelimit.reset()
    assert all(ratelimit.allow("lesson", "203.0.113.9", 3) for _ in range(3))
    assert ratelimit.allow("lesson", "203.0.113.9", 3) is False
    assert ratelimit.allow("lesson", "198.51.100.7", 3) is True
    assert ratelimit.allow("feedback", "203.0.113.9", 3) is True
    assert "203.0.113.9" not in repr(ratelimit._hits)
    assert ratelimit.allow("lesson", None, 0) is True
    ratelimit.reset()


def test_rate_limit_message_does_not_log_the_address(monkeypatch, caplog):
    app_module = importlib.import_module("app")
    ratelimit.reset()
    monkeypatch.setattr(app_module, "today_spend_usd", lambda: 0.0)
    monkeypatch.setattr(app_module.settings, "per_ip_hourly_limit", 0)
    with app_module.app.test_request_context("/", headers={"X-Forwarded-For": "203.0.113.9"}):
        with caplog.at_level(logging.INFO):
            result = app_module._guardrail_check("/api/lesson", {})
    assert result == app_module.RATE_LIMIT_ERROR
    assert "203.0.113.9" not in caplog.text
    ratelimit.reset()


def test_privacy_page_says_addresses_are_not_stored():
    app_module = importlib.import_module("app")
    page = app_module.app.test_client().get("/privacy").data
    assert b"We don't store internet (IP) addresses." in page
