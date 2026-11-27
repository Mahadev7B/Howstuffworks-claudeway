"""The free-lesson cap and the network-code purge share FREE_WINDOW_HOURS."""

import threading
import time

from lesson_platform import askers, db


def _no_database(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(db, "_pool", None)

    def refuse_direct():
        raise AssertionError("purge must not open DATABASE_URL")

    monkeypatch.setattr(db, "_direct_connect", refuse_direct)
    askers.reset_memory()
    askers._last_purge_at = time.monotonic()


def _backdate(digest: str, age_s: float) -> None:
    when = time.time() - age_s
    for qhash, pair in list(askers._memory_ip[digest].items()):
        askers._memory_ip[digest][qhash] = (when, pair[1])


def test_free_window_hours_defaults_to_six(monkeypatch):
    monkeypatch.delenv("FREE_WINDOW_HOURS", raising=False)
    assert askers.free_window_hours() == 6
    monkeypatch.setenv("FREE_WINDOW_HOURS", "nope")
    assert askers.free_window_hours() == 6
    monkeypatch.setenv("FREE_WINDOW_HOURS", "0")
    assert askers.free_window_hours() == 6
    monkeypatch.setenv("FREE_WINDOW_HOURS", "4")
    assert askers.free_window_hours() == 4


def test_purge_drops_codes_older_than_the_window_and_keeps_newer_ones(monkeypatch):
    _no_database(monkeypatch)
    monkeypatch.setenv("FREE_WINDOW_HOURS", "6")
    old_net = "old-network"
    new_net = "new-network"
    askers.record_question("mum@example.com", "Why old?", ip_hash=old_net)
    askers.record_question("dad@example.com", "Why new?", ip_hash=new_net)
    _backdate(old_net, 7 * 3600)
    _backdate(new_net, 30 * 60)
    assert askers.purge_ip_hashes() == "memory"
    assert old_net not in askers._memory_ip
    assert new_net in askers._memory_ip
    # The question itself stays with the email. Only the network code is removed.
    assert db.question_hash("Why old?") in askers._memory_asked["mum@example.com"]


def test_cap_resets_after_the_window(monkeypatch):
    _no_database(monkeypatch)
    monkeypatch.setenv("FREE_WINDOW_HOURS", "6")
    digest = "shared-network"
    for question in ("one", "two", "three"):
        askers.record_question("mum@example.com", question, ip_hash=digest)
    blocked = askers.allowance("dad@example.com", "four", ip_hash=digest)
    assert blocked["used"] == 3 and blocked["reason"] == "waitlist"
    _backdate(digest, 7 * 3600)
    fresh = askers.allowance("kid@example.com", "four", ip_hash=digest)
    assert fresh["used"] == 0 and fresh["reason"] == "free"
    # This email's own question is still a repeat. The network slot is not.
    repeat = askers.allowance("mum@example.com", "one", ip_hash=digest)
    assert repeat["repeat"] is True and repeat["reason"] == "repeat"


def test_changing_free_window_hours_changes_the_cap_and_the_purge(monkeypatch):
    _no_database(monkeypatch)
    monkeypatch.setenv("FREE_WINDOW_HOURS", "6")
    digest = "timed-network"
    askers.record_question("mum@example.com", "one", ip_hash=digest)
    _backdate(digest, 3 * 3600)
    assert askers.allowance("dad@example.com", "two", ip_hash=digest)["used"] == 1
    askers.purge_ip_hashes()
    assert digest in askers._memory_ip

    monkeypatch.setenv("FREE_WINDOW_HOURS", "2")
    assert askers.free_window_hours() == 2
    assert askers.allowance("dad@example.com", "two", ip_hash=digest)["used"] == 0
    askers.purge_ip_hashes()
    assert digest not in askers._memory_ip

    captured = []

    class Cur:
        def execute(self, sql, params=None):
            self.sql = " ".join(sql.split())
            captured.append((self.sql, params))

        def fetchone(self):
            if "pg_try_advisory_lock" in self.sql:
                return (True,)
            if "subscription_status" in self.sql:
                return ("none", None, None, 0, False)
            if "WHERE email = %s" in self.sql:
                return (0, False, 0, None)
            if "ip_hash = %s" in self.sql and "UPDATE" not in self.sql:
                return (0, False)
            return (True,)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class Conn:
        def __init__(self):
            self.cur = Cur()

        def cursor(self):
            return self.cur

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class Pool:
        def connection(self, timeout=None):
            return Conn()

    monkeypatch.setattr(db, "_pool", Pool())
    monkeypatch.setenv("FREE_WINDOW_HOURS", "5")
    askers._last_purge_at = time.monotonic()
    askers.allowance("mum@example.com", "three", ip_hash="db-code")
    counted = [item for item in captured if "ip_hash = %s" in item[0] and item[0].startswith("SELECT")]
    assert counted and counted[0][1][-1] == 5
    assert "INTERVAL '1 hour'" in counted[0][0]

    captured.clear()
    assert askers.purge_ip_hashes() == "purged"
    updates = [sql for sql, _params in captured if sql.startswith("UPDATE")]
    assert len(updates) == 1
    assert "SET ip_hash = NULL" in updates[0]
    assert "asker_questions" in updates[0]
    assert "ip_address" not in updates[0]
    assert "DELETE" not in updates[0]
    assert captured[-2][1] == (5,)  # the UPDATE's hours, before the unlock


def test_second_purge_skips_when_the_advisory_lock_is_held(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(db, "_direct_connect", lambda: (_ for _ in ()).throw(AssertionError("no direct connect")))
    askers.reset_memory()
    state = {"held": False}
    guard = threading.Lock()
    started = threading.Event()
    release = threading.Event()
    statements = []
    statements_lock = threading.Lock()

    class Cur:
        def execute(self, sql, params=None):
            self.sql = " ".join(sql.split())
            with statements_lock:
                statements.append(self.sql)
            if self.sql.startswith("UPDATE"):
                started.set()
                assert release.wait(timeout=3)

        def fetchone(self):
            if "pg_try_advisory_lock" in self.sql:
                with guard:
                    if state["held"]:
                        return (False,)
                    state["held"] = True
                    return (True,)
            if "pg_advisory_unlock" in self.sql:
                with guard:
                    state["held"] = False
                return (True,)
            return None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class Conn:
        def cursor(self):
            return Cur()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class Pool:
        def connection(self, timeout=None):
            return Conn()

    monkeypatch.setattr(db, "_pool", Pool())
    results = []

    def first():
        results.append(askers.purge_ip_hashes())

    def second():
        assert started.wait(timeout=3)
        results.append(askers.purge_ip_hashes())
        release.set()

    left = threading.Thread(target=first)
    right = threading.Thread(target=second)
    left.start()
    right.start()
    left.join(timeout=5)
    right.join(timeout=5)
    assert not left.is_alive() and not right.is_alive()
    assert sorted(results) == ["purged", "skipped"]
    assert sum(sql.startswith("UPDATE") for sql in statements) == 1


def test_privacy_page_renders_the_window(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("FREE_WINDOW_HOURS", "4")
    app_module = __import__("app")
    page = app_module.app.test_client().get("/privacy").get_data(as_text=True)
    assert "We delete the scrambled network code within 4 hours." in page
    assert "We do not currently delete the scrambled network code on a short timer." not in page
