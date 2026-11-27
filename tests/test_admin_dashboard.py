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
    data["lessons_by_country"] = [{"country": "Canada", "count": 3}, {"country": "Unknown", "count": 2}]
    data["feedback"] = {"total": 2, "up": 1, "down": 1, "comments": 1, "last_30d": 2, "available": True, "rows": [
        {"created_at": "2026-11-12 09:00", "rating": "down", "question": "Why?", "comment": "<b>too long</b>", "place": "Texas, United States"},
        {"created_at": "2026-11-12 08:00", "rating": "up", "question": "How?", "comment": None, "place": "—"},
    ]}
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
    assert text.splitlines()[0] == "email,first_seen,last_question,questions_asked,free_left,subscription_status,waitlist,asked_for_more"
    assert "mum@example.com,a,b,3,0,none,," in text
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
        (0,),
        [("subscribe", 2)],
        [("a@example.com", "d1", "d2", "none", 3, "subscribe", 0, "2026-11-15 09:00"),
         ("b@example.com", "d1", "d2", "none", 1, None, 0, None),
         ("c@example.com", "d1", "d2", "active", 7, "subscribe", 0, None)],
    ])
    summary = db._admin_askers(cursor, db.today_start_local(), limit=2)
    assert summary["total"] == 3 and summary["subscribed"] == 1
    assert summary["at_limit"] == 1
    assert summary["questions"] == 11
    assert [r["free_left"] for r in summary["rows"]] == [0, 2]
    assert len(summary["rows"]) == 2
    assert summary["waitlist"] == {"subscribe": 2, "browsing": 0, "price": 0}
    assert summary["rows"][0]["waitlist"] == "subscribe"
    assert summary["more_requests"] == 1 and summary["rows"][0]["requested_more"] == "2026-11-15 09:00"


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


def test_dashboard_shows_feedback_comments_escaped(admin):
    client, _ = admin
    page = client.get("/admin").data
    assert b'id="feedback"' in page
    assert b"Written comments" in page
    assert b"&lt;b&gt;too long&lt;/b&gt;" in page
    assert b"<b>too long</b>" not in page


def test_asker_summary_counts_bonus_lessons():
    db = importlib.import_module("lesson_platform.db")
    cursor = FakeCursor([(1, 0, 1, 0), (0,), [], [("a@example.com", "d1", "d2", "none", 3, None, 2, None)]])
    summary = db._admin_askers(cursor, db.today_start_local(), limit=5)
    assert summary["rows"][0]["free_left"] == 2 and summary["rows"][0]["bonus"] == 2
    assert summary["at_limit"] == 0


def _counts(rows, *keys):
    return {tuple(row[key] for key in keys): row["count"] for row in rows}


def test_blank_locations_count_as_unknown_and_totals_match():
    """Null, blank, and missing place keys share the Unknown bucket."""
    db = importlib.import_module("lesson_platform.db")
    lessons = [
        {"country": "Canada", "region": "Ontario"},
        {"country": "  Canada  ", "region": "Ontario"},
        {"country": "United States", "region": "Texas"},
        {"country": None, "region": None},
        {"country": "", "region": "   "},
        {},
        {"country": "Canada"},
        {"region": "Texas"},
        {"country": "Canada", "region": None},
    ]
    charts = db.aggregate_lessons_by_location(lessons)
    by_country = charts["lessons_by_country"]

    assert "lessons_by_region" not in charts
    assert _counts(by_country, "country") == {
        ("Canada",): 4,
        ("United States",): 1,
        ("Unknown",): 4,
    }
    assert all("region" not in row and "city" not in row for row in by_country)
    assert sum(row["count"] for row in by_country) == len(lessons)
    assert [row["country"] for row in by_country][-1] == "Unknown"
    assert by_country[-1]["count"] == 4


def test_unknown_bucket_stays_last_even_when_it_is_largest():
    db = importlib.import_module("lesson_platform.db")
    charts = db.aggregate_lessons_by_location([
        {"country": None, "region": None, "count": 37},
        {"country": "", "region": "", "count": 2},
        {"country": "Canada", "region": "Ontario", "count": 5},
        {"country": "Mexico", "region": "Jalisco", "count": 1},
    ])
    assert [row["country"] for row in charts["lessons_by_country"]] == ["Canada", "Mexico", "Unknown"]
    assert charts["lessons_by_country"][-1]["count"] == 39
    assert "lessons_by_region" not in charts
    assert sum(row["count"] for row in charts["lessons_by_country"]) == 45


def test_located_lessons_do_not_invent_an_unknown_bucket():
    db = importlib.import_module("lesson_platform.db")
    charts = db.aggregate_lessons_by_location([
        {"country": "Canada", "region": "Ontario"},
        {"country": "Canada", "region": "Quebec"},
    ])
    assert "Unknown" not in {row["country"] for row in charts["lessons_by_country"]}
    assert "lessons_by_region" not in charts
    assert sum(row["count"] for row in charts["lessons_by_country"]) == 2


class RecordingCursor(FakeCursor):
    def __init__(self, results):
        super().__init__(results)
        self.statements = []
        self.params = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=None):
        self.statements.append(" ".join(sql.split()))
        self.params = params
        super().execute(sql, params)


def test_location_query_keeps_null_and_blank_rows():
    db = importlib.import_module("lesson_platform.db")
    cursor = RecordingCursor([[
        ("Canada", 6),
        (None, 3),
        ("", 1),
    ]])
    charts = db._admin_location_charts(cursor, ["/api/lesson", "/lesson"], "AND (city IS NULL)", ["Ashburn"])
    sql = cursor.statements[0]
    assert "country IS NOT NULL" not in sql
    assert "region" not in sql
    assert "SELECT country, COUNT(*) AS n" in sql
    assert "GROUP BY country" in sql
    assert cursor.params[0] == ["/api/lesson", "/lesson"]
    assert cursor.params[1:] == ("Ashburn",)
    assert _counts(charts["lessons_by_country"], "country") == {("Canada",): 6, ("Unknown",): 4}
    assert sum(row["count"] for row in charts["lessons_by_country"]) == 10
    assert "lessons_by_region" not in charts
    assert charts["lessons_by_country"][-1]["country"] == "Unknown"


class _Conn:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_unknown_place_filter_matches_blanks_and_labels_them(monkeypatch):
    db = importlib.import_module("lesson_platform.db")
    cursor = RecordingCursor([[(
        "2026-11-01 09:00", "Why is the sky blue?", None, 0.01, 1200, "Mozilla",
    )]])
    conn = _Conn(cursor)
    monkeypatch.setattr(db, "_direct_connect", lambda: conn)

    rows = db.admin_lessons_filtered(country="Unknown")
    sql = cursor.statements[0]
    selected = sql.split(" FROM ", 1)[0]
    assert "COALESCE(NULLIF(BTRIM(country), ''), %s) = %s" in sql
    assert "region" not in sql
    assert "city" not in selected and "region" not in selected
    assert cursor.params.count("Unknown") == 2
    assert rows == [{
        "created_at": "2026-11-01 09:00",
        "question": "Why is the sky blue?",
        "country": "Unknown",
        "cost": 0.01,
        "duration_ms": 1200,
        "device": "desktop",
    }]
    assert "region" not in rows[0] and "city" not in rows[0]

    named = RecordingCursor([[]])
    conn._cursor = named
    db.admin_lessons_filtered(country="Canada")
    named_sql = named.statements[0]
    named_selected = named_sql.split(" FROM ", 1)[0]
    assert "country = %s" in named_sql
    assert "region" not in named_sql
    assert "city" not in named_selected
    assert "BTRIM(country)" not in named_sql
    assert "Canada" in named.params


def test_dashboard_location_chart_script_keeps_unknown_last(admin):
    client, _ = admin
    page = client.get("/admin").data.decode()
    assert "chartByCountry" in page
    assert "chartByRegion" not in page
    assert "r.country !== 'Unknown'" in page
    assert '"country": "Unknown"' in page
    assert "State or region" not in page and "State/Region" not in page
    assert "filterRegion" not in page
    assert ">City<" not in page


def test_admin_grant_needs_csrf_and_gives_two_lessons(admin, monkeypatch):
    client, app_module = admin
    monkeypatch.setattr(app_module.app, "secret_key", "test-secret-not-dev")
    with client.session_transaction() as session:
        session["admin"] = True
    granted = []
    monkeypatch.setattr(app_module.askers, "grant_bonus", lambda email, lessons: granted.append((email, lessons)) or True)
    assert client.post("/admin/askers/grant", data={"email": "mum@example.com"}).status_code == 403
    page = client.get("/admin").data.decode()
    assert "Give 2 more lessons" in page
    token = page.split('name="csrf_token" value="', 1)[1].split('"', 1)[0]
    response = client.post("/admin/askers/grant", data={"email": "Mum@Example.com", "csrf_token": token})
    assert response.status_code == 302 and "asker=granted" in response.headers["Location"]
    assert granted == [("mum@example.com", 2)]
