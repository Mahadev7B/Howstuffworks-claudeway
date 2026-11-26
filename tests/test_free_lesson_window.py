"""The free-lesson network code expires on one setting, and the purge leaves everything else."""

import importlib

from lesson_platform import askers, ratelimit


def _site(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    app_module = importlib.import_module("app")
    app_module.app.config.update(TESTING=True)
    monkeypatch.setattr(app_module.askers.db, "_pool", None)
    askers.reset_memory()
    ratelimit.reset()

    def fake_track(endpoint, question, ctx):
        from flask import g
        g.lesson_generated = True
        return {"title": question, "slides": []}, None

    monkeypatch.setattr(app_module, "_track_lesson", fake_track)
    monkeypatch.setattr(app_module, "check_question", lambda q: None)
    return app_module.app.test_client()


def _ask(client, question, ip="203.0.113.9"):
    return client.post(
        "/api/lesson",
        json={"question": question, "email": "mum@example.com", "parent_consent": True},
        headers={"X-Forwarded-For": ip},
    )


def test_three_free_lessons_reset_after_the_window_and_not_before(monkeypatch):
    monkeypatch.setenv("FREE_LESSON_WINDOW_HOURS", "6")
    client = _site(monkeypatch)
    clock = {"now": 1_700_000_000.0}
    monkeypatch.setattr(askers.time, "time", lambda: clock["now"])
    for number in range(3):
        assert _ask(client, f"Question {number}?").status_code == 200
    assert _ask(client, "Question 3?").status_code == 402
    clock["now"] += 5 * 3600
    assert _ask(client, "Question 4?").status_code == 402
    clock["now"] += 3600 + 1
    assert _ask(client, "Question 5?").status_code == 200
    askers.reset_memory()
    ratelimit.reset()


def test_the_window_setting_changes_the_count_and_the_purge_together(monkeypatch):
    monkeypatch.setenv("FREE_LESSON_WINDOW_HOURS", "2")
    assert askers.free_lesson_window_hours() == 2
    client = _site(monkeypatch)
    clock = {"now": 1_700_000_000.0}
    monkeypatch.setattr(askers.time, "time", lambda: clock["now"])
    for number in range(3):
        assert _ask(client, f"Question {number}?").status_code == 200
    clock["now"] += 2 * 3600 + 1
    assert _ask(client, "Question 3?").status_code == 200

    class Cursor:
        def __init__(self):
            self.queries = []
            self._rows = [
                ("none", None, None, 0, False),
                (0, False, 0, None),
                (0, False),
            ]

        def execute(self, sql, params=None):
            self.queries.append((sql, params))

        def fetchone(self):
            return self._rows.pop(0)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class Conn:
        def __init__(self):
            self.cur = Cursor()

        def cursor(self):
            return self.cur

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    conn = Conn()

    class Pool:
        def connection(self, timeout=None):
            return conn

    askers.reset_memory()
    monkeypatch.setattr(askers.db, "_pool", Pool())
    lesson_row = {"country": "US", "question": "Why?"}
    askers.allowance("mum@example.com", "Why is the sky blue?", ip_hash="network-hash")
    updates = [item for item in conn.cur.queries if item[0].lstrip().upper().startswith("UPDATE")]
    counts = [item for item in conn.cur.queries if "COUNT(DISTINCT question_hash)" in item[0]]
    assert len(updates) == 1 and updates[0][1] == (2,)
    assert len(counts) == 1 and counts[0][1][-1] == 2
    purge_sql = updates[0][0].lower()
    assert "ip_hash = null" in purge_sql
    assert "delete" not in purge_sql
    assert "country" not in purge_sql and "api_calls" not in purge_sql
    assert lesson_row == {"country": "US", "question": "Why?"}
    askers.reset_memory()
    ratelimit.reset()


def test_purge_drops_only_the_network_hash_and_is_throttled(monkeypatch):
    askers.reset_memory()
    askers._memory_asked["mum@example.com"] = {"kept": (100.0, True)}
    askers._memory_ip["old-hash"] = {"kept": (100.0, True)}
    monkeypatch.setattr(askers.time, "time", lambda: 100.0 + 6 * 3600 + 1)
    monkeypatch.setenv("FREE_LESSON_WINDOW_HOURS", "6")
    askers._purge_memory_ip_hashes()
    assert "old-hash" not in askers._memory_ip
    assert askers._memory_asked["mum@example.com"]["kept"][0] == 100.0

    class Cursor:
        def __init__(self):
            self.queries = []

        def execute(self, sql, params=None):
            self.queries.append(sql)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class Conn:
        def __init__(self):
            self.cur = Cursor()

        def cursor(self):
            return self.cur

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    conn = Conn()

    class Pool:
        def connection(self, timeout=None):
            return conn

    clock = {"m": 50_000.0}
    monkeypatch.setattr(askers.time, "monotonic", lambda: clock["m"])
    monkeypatch.setattr(askers.db, "_pool", Pool())
    askers.reset_memory()
    askers.maybe_purge_ip_hashes()
    askers.maybe_purge_ip_hashes()
    assert len(conn.cur.queries) == 1
    clock["m"] += 15 * 60
    askers.maybe_purge_ip_hashes()
    assert len(conn.cur.queries) == 2
    assert all("country" not in sql.lower() for sql in conn.cur.queries)
    askers.reset_memory()


def test_a_bad_window_setting_stays_on_the_default(monkeypatch):
    monkeypatch.setenv("FREE_LESSON_WINDOW_HOURS", "nope")
    assert askers.free_lesson_window_hours() == 6
    monkeypatch.setenv("FREE_LESSON_WINDOW_HOURS", "0")
    assert askers.free_lesson_window_hours() == 6
