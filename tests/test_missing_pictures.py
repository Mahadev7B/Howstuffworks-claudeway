"""The admin list of saved lessons that are missing pictures is read-only."""

import importlib

import pytest


class _Cursor:
    def __init__(self, pool):
        self.pool = pool

    def execute(self, sql, params=None):
        self.pool.sql = sql
        self.pool.params = params

    def fetchall(self):
        return [("How do rockets fly?", "Rockets", "shareslug", "2026-11-01 09:00", 4, 1)]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _Connection:
    def __init__(self, pool):
        self.pool = pool

    def cursor(self):
        return _Cursor(self.pool)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _Pool:
    def __init__(self):
        self.sql = ""
        self.params = None

    def connection(self, timeout=None):
        return _Connection(self)


@pytest.fixture
def admin(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    app_module = importlib.import_module("app")
    app_module.app.config.update(TESTING=True)
    monkeypatch.setattr(app_module.askers.db, "_pool", None)
    app_module._RECENT_LESSONS.clear()
    client = app_module.app.test_client()
    with client.session_transaction() as session:
        session["admin"] = True
    return client, app_module


def test_missing_pictures_requires_admin(admin):
    client, app_module = admin
    anonymous = app_module.app.test_client()
    assert anonymous.get("/admin/missing-pictures").status_code == 302
    assert "/admin/login" in anonymous.get("/admin/missing-pictures").headers["Location"]


def test_memory_list_counts_gaps_without_copying_pictures_or_offering_actions(admin):
    client, app_module = admin
    huge = "data:image/png;base64," + ("A" * 8000)
    app_module._recent_put("complete lesson", {
        "title": "Complete",
        "slides": [{"image_data_url": huge}, {"image_data_url": huge}],
    })
    app_module._recent_put("partial lesson", {
        "title": "Partial",
        "slides": [{"image_data_url": huge}, {"title": "No art"}, {"image_data_url": ""}, "not-a-slide"],
    })
    page = client.get("/admin/missing-pictures")
    html = page.get_data(as_text=True)
    assert page.status_code == 200
    assert "1 saved lesson is missing pictures" in html
    assert "partial lesson" in html and "Partial" in html
    assert "complete lesson" not in html
    assert "3 of 4" in html
    assert "No share link" in html
    assert huge not in html
    assert "Delete" not in html and "Generate" not in html
    assert client.get("/admin").get_data(as_text=True).count("/admin/missing-pictures") >= 1


def test_database_list_uses_counts_only(admin, monkeypatch):
    client, app_module = admin
    pool = _Pool()
    monkeypatch.setattr(app_module.askers.db, "_pool", pool)
    page = client.get("/admin/missing-pictures")
    html = page.get_data(as_text=True)
    assert page.status_code == 200
    assert "1 saved lesson is missing pictures" in html
    assert "How do rockets fly?" in html and "Rockets" in html
    assert 'href="/s/shareslug"' in html
    assert "1 of 4" in html
    compact = " ".join(pool.sql.split())
    assert "FROM cached_lessons" in compact
    assert "SELECT lesson" not in compact
    assert "->> 'image_data_url'" not in compact and "->>'image_data_url'" not in compact
    assert "image_data_url" in compact
    assert client.post("/admin/missing-pictures").status_code == 405
