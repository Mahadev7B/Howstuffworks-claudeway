"""New rows keep a country and do not store a city or a region."""

import copy
import importlib
import logging
import re
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse

from lesson_platform import askers, db, spam_log, verification


class _Body:
    def __init__(self, payload: bytes):
        self.payload = payload

    def read(self):
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _patch_ipwho(monkeypatch, payload: bytes):
    geo = importlib.import_module("lesson_platform.geo")
    seen = []

    def urlopen(req, timeout=None):
        seen.append({"url": req.full_url, "data": req.data, "headers": dict(req.header_items()), "timeout": timeout})
        return _Body(payload)

    monkeypatch.setattr(geo.urllib.request, "urlopen", urlopen)
    return geo, seen


def _assert_country_only_request(call):
    parsed = urlparse(call["url"])
    query = parse_qs(parsed.query, keep_blank_values=True)
    assert query.get("fields") == ["success,country"]
    blob = call["url"] + repr(call["data"]) + repr(call["headers"])
    assert "city" not in blob.lower()
    assert "region" not in blob.lower()


def test_lookup_requests_only_success_and_country(monkeypatch):
    geo, seen = _patch_ipwho(monkeypatch, b'{"success": true, "country": "Canada", "city": "Toronto", "region": "Ontario"}')
    assert geo.lookup("8.8.8.8") == "Canada"
    assert geo.extract_and_lookup("8.8.8.8", "10.0.0.1") == ("8.8.8.8", "Canada")
    assert seen and all("ipwho.is" in call["url"] for call in seen)
    for call in seen:
        _assert_country_only_request(call)
    seen.clear()
    assert geo.lookup("10.0.0.1") is None
    assert seen == []


class _Cursor:
    def __init__(self):
        self.statements = []
        self.params = []
        self._sql = ""

    def execute(self, sql, params=None):
        self._sql = " ".join(sql.split())
        self.statements.append(self._sql)
        self.params.append(params)

    def fetchone(self):
        if "trial_started_at" in self._sql:
            return (datetime.now(timezone.utc),)
        return (1,)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    rowcount = 1


class _Conn:
    def __init__(self, cursor):
        self.cursor_obj = cursor

    def cursor(self):
        return self.cursor_obj

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def close(self):
        return None


class _Pool:
    def __init__(self, cursor):
        self.cursor_obj = cursor

    def connection(self, timeout=None):
        return _Conn(self.cursor_obj)


def _no_city_or_region(sql: str) -> None:
    assert re.search(r"\bcity\b", sql, re.IGNORECASE) is None
    assert re.search(r"\bregion\b", sql, re.IGNORECASE) is None


def test_feedback_and_api_calls_never_write_city_or_region(monkeypatch):
    """api_calls is the activity record. There is no separate activity table."""
    executed = []

    class Cur:
        def execute(self, sql, params=None):
            executed.append((" ".join(sql.split()), params))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    conn = _Conn(Cur())

    def direct():
        return conn

    monkeypatch.setattr(db, "_pool", object())
    monkeypatch.setattr(db, "_direct_connect", direct)
    monkeypatch.setattr(db, "_bg", lambda fn, params: fn(params))
    db.save_feedback("Why?", "up", "Thanks", client_id="visit-0001",
                     city="Austin", region="Texas", country="Canada", user_agent="test")
    db.record_api_call(endpoint="/api/lesson", question="Why?", model="m", success=True,
                       city="Austin", region="Texas", country="Canada", user_agent="test")
    assert len(executed) == 2
    tables = []
    for sql, params in executed:
        _no_city_or_region(sql)
        assert "country" in sql
        assert sql.count("%s") == len(params)
        assert "Austin" not in repr(params) and "Texas" not in repr(params)
        assert "Canada" in params
        tables.append("feedback" if "INTO feedback" in sql else "api_calls" if "INTO api_calls" in sql else sql)
    assert tables == ["feedback", "api_calls"]


def test_asker_inserts_never_write_city_or_region(monkeypatch):
    cursor = _Cursor()
    monkeypatch.setattr(db, "_pool", _Pool(cursor))
    monkeypatch.setattr(verification.mailer, "send", lambda *args, **kwargs: None)
    monkeypatch.setattr(verification.hmac, "compare_digest", lambda left, right: True)
    askers.record_question("parent@example.com", "Why is the sky blue?")
    askers.start_trial("parent@example.com")
    askers.set_waitlist("parent@example.com", "browsing")
    askers.set_subscription("parent@example.com", "active", "cus_123", "sub_123")
    verification.start("parent@example.com")
    assert verification.check("parent@example.com", "123456")
    joined = "\n".join(cursor.statements)
    assert "INSERT INTO askers" in joined
    assert "INSERT INTO asker_questions" in joined
    assert "INSERT INTO email_codes" in joined
    for sql in cursor.statements:
        _no_city_or_region(sql)
    for params in cursor.params:
        assert params is None or ("Austin" not in repr(params) and "Texas" not in repr(params))


def test_refused_question_log_keeps_country_only(monkeypatch, tmp_path, caplog):
    geo, seen = _patch_ipwho(monkeypatch, b'{"success": true, "country": "Canada", "city": "Toronto", "region": "Ontario"}')
    monkeypatch.setattr(spam_log, "_PATH", tmp_path / "refused.jsonl")
    monkeypatch.setattr(spam_log, "_submit", lambda fn, *args: fn(*args))
    with caplog.at_level(logging.WARNING, logger="lesson_platform.spam_log"):
        spam_log.record(endpoint="/api/lesson", question="buy this now", reason="link",
                        ip="8.8.8.8", forwarded_for="8.8.8.8", remote_addr="10.0.0.1",
                        user_agent="BadBot/1.0")
    assert seen
    for call in seen:
        _assert_country_only_request(call)
    stored = (tmp_path / "refused.jsonl").read_text(encoding="utf-8")
    assert '"place": "Canada"' in stored
    assert "Toronto" not in stored and "Ontario" not in stored
    assert "city" not in stored and "region" not in stored
    assert "Toronto" not in caplog.text and "Ontario" not in caplog.text
    assert "Canada" in caplog.text
    assert geo.lookup("8.8.8.8") == "Canada"


def test_admin_chart_renders_country_with_unknown_and_no_city_or_region(monkeypatch):
    app_module = importlib.import_module("app")
    app_module.app.config.update(TESTING=True)
    data = copy.deepcopy(db._ADMIN_DEFAULTS)
    data["overview"].update(saved_copies_today=0, saved_copies_all_time=0)
    data["lessons_by_country"] = [
        {"country": "Canada", "count": 4},
        {"country": "Unknown", "count": 3},
    ]
    monkeypatch.setattr(app_module, "db_enabled", True)
    monkeypatch.setattr(app_module, "admin_load_all", lambda **kwargs: data)
    client = app_module.app.test_client()
    with client.session_transaction() as session:
        session["admin"] = True
    page = client.get("/admin").get_data(as_text=True)
    assert "chartByCountry" in page
    assert "New lessons by country" in page
    assert "r.country !== 'Unknown'" in page
    assert '"country": "Unknown"' in page or '"country": "Canada"' in page
    assert "chartByRegion" not in page
    assert "State or region" not in page
    assert "State/Region" not in page
    assert "filterRegion" not in page
    assert ">City<" not in page
    charts = db.aggregate_lessons_by_location([
        {"country": "Canada", "region": "Ontario", "city": "Toronto"},
        {"country": None, "region": "Texas", "city": "Austin"},
        {"country": "", "city": "Paris"},
    ])
    assert [row["country"] for row in charts["lessons_by_country"]] == ["Canada", "Unknown"]
    assert charts["lessons_by_country"][-1]["count"] == 2
    assert "lessons_by_region" not in charts
    assert all("city" not in row and "region" not in row for row in charts["lessons_by_country"])


def test_privacy_page_estimates_country_only():
    app_module = importlib.import_module("app")
    page = app_module.app.test_client().get("/privacy").get_data(as_text=True)
    assert "estimated country" in page
    assert page.count("We send the visitor's IP address to a location service (ipwho.is) to estimate the country only.") == 2
    assert "removing this" not in page
    assert "the scrambled network code stored with those questions" not in page
    assert "city, region" not in page
    assert "estimated city, region and country" not in page
    assert "to estimate country, region and city" not in page
    assert "country, region and city" not in page
