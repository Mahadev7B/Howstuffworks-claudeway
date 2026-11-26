"""The saved-lesson list waits on the admin checkout, long enough for a full table."""

import importlib

from psycopg_pool import PoolTimeout


class _Cursor:
    def __init__(self, rows):
        self.rows = rows
        self.sql = ""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql):
        self.sql = sql

    def fetchall(self):
        return self.rows


class _Connection:
    def __init__(self, rows):
        self.cursor_obj = _Cursor(rows)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def cursor(self):
        return self.cursor_obj


class _Pool:
    def __init__(self, rows):
        self.connection_obj = _Connection(rows)
        self.timeouts = []

    def connection(self, timeout):
        self.timeouts.append(timeout)
        if timeout < 30:
            raise PoolTimeout("checkout")
        return self.connection_obj


def test_cached_lesson_list_uses_the_30s_admin_timeout(monkeypatch):
    app_module = importlib.import_module("app")
    db = importlib.import_module("lesson_platform.db")
    rows = [
        (f"Question {number}?", number, False, "2026-11-01 12:00", "2026-11-01 12:05")
        for number in range(176)
    ]
    pool = _Pool(rows)
    monkeypatch.setattr(app_module, "db_enabled", True)
    monkeypatch.setattr(db, "_pool", pool)
    client = app_module.app.test_client()
    assert client.get("/admin/cached-lessons").status_code == 403
    with client.session_transaction() as session:
        session["admin"] = True
    response = client.get("/admin/cached-lessons")
    assert response.status_code == 200
    body = response.get_json()
    assert body["ok"] is True
    assert len(body["lessons"]) == 176
    assert body["lessons"][0]["question"] == "Question 0?"
    assert pool.timeouts == [db._ADMIN_POOL_TIMEOUT_S]
    assert db._ADMIN_POOL_TIMEOUT_S == 30.0
    assert "image_data_url" not in pool.connection_obj.cursor_obj.sql
