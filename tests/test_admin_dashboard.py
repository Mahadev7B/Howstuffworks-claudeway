"""The admin dashboard shows askers' emails, escapes visitor text and fills quiet days."""

import copy
import importlib

import pytest


def fake_data(db):
    data = copy.deepcopy(db._ADMIN_DEFAULTS)
    data["overview"].update(total_lessons=12, lessons_today=2, saved_copies_today=5, saved_copies_all_time=40,
                            cost_today=1.5, cost_all_time=20.0, today_start_iso="2026-11-12T00:00:00-05:00")
    data["askers"] = {"total": 2, "new_today": 1, "new_7d": 2, "subscribed": 0, "at_limit": 1, "questions": 4,
                      "available": True, "rows": [
                          {"email": "mum@example.com", "first_seen": "2026-11-12 09:00", "last_question": "2026-11-12 09:30",
                           "questions": 3, "free_left": 0, "status": "none"},
                          {"email": "<script>x</script>@example.com", "first_seen": "2026-11-12 10:00",
                           "last_question": "2026-11-12 10:00", "questions": 1, "free_left": 2, "status": "none"},
                      ]}
    data["top_questions"] = [{"question": "<img src=x onerror=alert(1)>", "count": 3}]
    return data


@pytest.fixture
def admin(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    app_module = importlib.import_module("app")
    app_module.app.config.update(TESTING=True)
    monkeypatch.setattr(app_module, "db_enabled", True)
    monkeypatch.setattr(app_module, "admin_load_all", lambda **kwargs: fake_data(app_module.askers.db))
    client = app_module.app.test_client()
    with client.session_transaction() as session:
        session["admin"] = True
    return client, app_module


def test_dashboard_requires_login(admin):
    client, _ = admin
    with client.session_transaction() as session:
        session.clear()
    assert client.get("/admin").status_code == 302
    assert client.get("/admin/askers.csv").status_code == 302


def test_dashboard_lists_unique_emails_and_escapes_visitor_text(admin):
    client, _ = admin
    page = client.get("/admin").data
    assert b"Unique emails" in page
    assert b"mum@example.com" in page
    assert b"Free used up" in page
    assert b"Download all as CSV" in page
    assert b"<script>x</script>" not in page
    assert b"<img src=x" not in page
    assert b"HowStuffWorks" not in page
    assert b"AskLilOwl" in page


def test_csv_download_has_every_asker_and_neutralises_formulas(admin, monkeypatch):
    client, app_module = admin
    monkeypatch.setattr(app_module.askers.db, "admin_askers_export", lambda: [
        {"email": "mum@example.com", "first_seen": "a", "last_question": "b", "questions": 3, "free_left": 0, "status": "none"},
        {"email": "=cmd@example.com", "first_seen": "a", "last_question": "b", "questions": 1, "free_left": None, "status": "active"},
    ])
    response = client.get("/admin/askers.csv")
    assert response.status_code == 200
    assert response.mimetype == "text/csv"
    text = response.data.decode()
    assert text.splitlines()[0] == "email,first_seen,last_question,questions_asked,free_left,subscription_status"
    assert "mum@example.com,a,b,3,0,none" in text
    assert "'=cmd@example.com" in text


def test_quiet_days_are_filled_with_zero(monkeypatch):
    db = importlib.import_module("lesson_platform.db")
    monkeypatch.setattr(db, "_local_days", lambda days: ["2026-11-10", "2026-11-11", "2026-11-12"])
    filled = db._fill_days([{"date": "2026-11-11", "count": 4}], 3, {"count": 0})
    assert filled == [{"date": "2026-11-10", "count": 0}, {"date": "2026-11-11", "count": 4},
                      {"date": "2026-11-12", "count": 0}]


class FakeCursor:
    def __init__(self, results):
        self.results = list(results)

    def execute(self, sql, params=None):
        self.current = self.results.pop(0)

    def fetchone(self):
        return self.current

    def fetchall(self):
        return self.current


def test_asker_summary_counts_limits_and_subscribers():
    db = importlib.import_module("lesson_platform.db")
    cursor = FakeCursor([
        (3, 1, 2, 1),
        [("a@example.com", "d1", "d2", "none", 3),
         ("b@example.com", "d1", "d2", "none", 1),
         ("c@example.com", "d1", "d2", "active", 7)],
    ])
    summary = db._admin_askers(cursor, db.today_start_local(), limit=2)
    assert summary["total"] == 3 and summary["subscribed"] == 1
    assert summary["at_limit"] == 1
    assert summary["questions"] == 11
    assert [r["free_left"] for r in summary["rows"]] == [0, 2]
    assert len(summary["rows"]) == 2


def test_admin_delete_needs_csrf_and_removes_email(admin, monkeypatch):
    client, app_module = admin
    monkeypatch.setattr(app_module.app, "secret_key", "test-secret-not-dev")
    with client.session_transaction() as session:  # A new key invalidates the earlier sign-in.
        session["admin"] = True
    removed = []
    monkeypatch.setattr(app_module.askers, "delete_asker", lambda email: removed.append(email) or True)
    assert client.post("/admin/askers/delete", data={"email": "mum@example.com"}).status_code == 403
    assert removed == []
    page = client.get("/admin").data.decode()
    token = page.split('name="csrf_token" value="', 1)[1].split('"', 1)[0]
    response = client.post("/admin/askers/delete", data={"email": " Mum@Example.com ", "csrf_token": token})
    assert response.status_code == 302
    assert "asker=deleted" in response.headers["Location"]
    assert removed == ["mum@example.com"]
