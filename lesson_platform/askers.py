"""Who is asking: an email per asker, a few free questions, a free month, then a subscription.

* The first FREE_QUESTIONS different questions from one network are free inside
  FREE_WINDOW_HOURS (default 6), whichever email asks. After that the network
  code is cleared. The questions and the email stay.
* Then a parent can start one free month: up to TRIAL_DAILY_NEW new lessons in
  any 24 hours (a rolling window, so time zones don't matter) for TRIAL_DAYS days. Questions someone already asked (a saved copy) and
  the family's own repeats never count.
* After that, a paid subscription is needed.

Without a database (local previews and tests) everything lives in memory.
"""
import hashlib
import hmac
import logging
import os
import re
import secrets
import threading
import time
from datetime import datetime, timedelta, timezone

from lesson_platform import db

logger = logging.getLogger(__name__)

FREE_QUESTIONS = db.ASKER_FREE_QUESTIONS
TRIAL_DAYS = 30
TRIAL_DAILY_NEW = 5
WAITLIST_INSTEAD_OF_TRIAL = True
WAITLIST_CHOICES = ("subscribe", "browsing", "price")
WINDOW_S = 24 * 3600
# How often a worker may clear expired network codes. Several workers on Render
# also take a Postgres advisory lock so only one of them deletes at a time.
_PURGE_INTERVAL_S = 15 * 60
_PURGE_ADVISORY_LOCK_KEY = 815204116
_last_purge_at = 0.0
_purge_interval_lock = threading.Lock()
_purge_timer_started = False
_purge_timer_guard = threading.Lock()
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_ACTIVE = ("active", "trialing")  # Same as billing.ACTIVE_STATES.

_memory_lock = threading.Lock()
_memory_asked: dict[str, dict[str, tuple[float, bool]]] = {}  # email -> {question hash: (time, newly made)}
_memory_ip: dict[str, dict[str, tuple[float, bool]]] = {}  # ip hash -> {question hash: (time, newly made)}
_memory_status: dict[str, str] = {}
_memory_trial: dict[str, float] = {}
_memory_waitlist: dict[str, str] = {}
_memory_bonus: dict[str, int] = {}
_memory_requests: dict[str, float] = {}


class AskerUnavailable(RuntimeError):
    """The allowance could not be checked, so the question must not be generated."""


def normalize_email(raw: str | None) -> str | None:
    email = (raw or "").strip().lower()
    if len(email) > 254 or not _EMAIL.match(email):
        return None
    return email


# Used only when FLASK_SECRET_KEY is missing or still the published dev default.
# It is not saved, so a restart gives each network a fresh set of free lessons.
_PROCESS_IP_SECRET = secrets.token_bytes(32)


def _ip_hmac_key() -> bytes:
    secret = os.getenv("FLASK_SECRET_KEY", "").strip()
    if secret and secret != "dev-secret-key":
        return secret.encode("utf-8")
    return _PROCESS_IP_SECRET


def ip_hash(ip: str | None) -> str | None:
    """One-way code for a client IP. The address cannot be read back from it."""
    if not ip:
        return None
    return hmac.new(_ip_hmac_key(), b"free-lesson-ip|" + ip.encode("utf-8"), hashlib.sha256).hexdigest()


# The hourly lesson count reads api_calls.ip_hash for the past hour. A purge
# window under 1 hour would delete that code before the count could see it, so
# a configured value below 1 is ignored and the default of 6 is used instead.
_MIN_FREE_WINDOW_HOURS = 1
_warned_short_window = False


def free_window_hours() -> int:
    """Hours a network code counts toward the free lessons and how long it is kept.

    ``FREE_WINDOW_HOURS`` is the only setting. The cap and the purge both call this.
    A missing or unusable value, including anything below 1 hour, means 6.
    """
    global _warned_short_window
    raw = os.getenv("FREE_WINDOW_HOURS", "6").strip()
    try:
        hours = int(raw)
    except ValueError:
        hours = None
    if hours is None or hours < _MIN_FREE_WINDOW_HOURS:
        if not _warned_short_window:
            _warned_short_window = True
            logger.warning(
                "FREE_WINDOW_HOURS is below 1; using 6 so the hourly network-code count still sees the last hour"
            )
        return 6
    return hours


def _window_cutoff(now: float | None = None) -> float:
    moment = time.time() if now is None else now
    return moment - free_window_hours() * 3600


def _purge_memory_ip_hashes(now: float | None = None) -> None:
    """Drop in-memory network codes older than the window. Question history stays."""
    cutoff = _window_cutoff(now)
    with _memory_lock:
        for digest in list(_memory_ip):
            kept = {qhash: pair for qhash, pair in _memory_ip[digest].items() if pair[0] >= cutoff}
            if kept:
                _memory_ip[digest] = kept
            else:
                del _memory_ip[digest]


def _purge_db_ip_hashes() -> bool:
    """Set expired network codes to null.

    The code is stored on asker_questions and on api_calls. Both updates run in
    the same transaction as pg_try_advisory_xact_lock, so a rollback releases the
    lock. The askers table has no ip_hash column. Raw ip_address columns are not
    touched, and existing rows are not rewritten except to clear an old code.
    """
    hours = free_window_hours()
    with db._pool.connection(timeout=db._POOL_TIMEOUT_S) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute("SET LOCAL statement_timeout = '5s'")
                cur.execute("SET LOCAL lock_timeout = '2s'")
                cur.execute("SELECT pg_try_advisory_xact_lock(%s)", (_PURGE_ADVISORY_LOCK_KEY,))
                row = cur.fetchone()
                if not row or not row[0]:
                    return False
                for table in ("asker_questions", "api_calls"):
                    cur.execute(
                        f"UPDATE {table} SET ip_hash = NULL "
                        "WHERE ip_hash IS NOT NULL "
                        "AND created_at < NOW() - (%s * INTERVAL '1 hour')",
                        (hours,),
                    )
    return True


def purge_ip_hashes() -> str:
    """Clear network codes older than free_window_hours().

    Returns "memory" when there is no database, "skipped" when another worker
    holds the lock, or "purged" when this worker cleared the database column.
    """
    _purge_memory_ip_hashes()
    if db._pool is None:
        return "memory"
    return "purged" if _purge_db_ip_hashes() else "skipped"


def maybe_purge_ip_hashes() -> None:
    """Purge at most once every 15 minutes in this process. Failures are logged."""
    global _last_purge_at
    now = time.monotonic()
    with _purge_interval_lock:
        if now - _last_purge_at < _PURGE_INTERVAL_S:
            return
        _last_purge_at = now
    try:
        purge_ip_hashes()
    except Exception as exc:
        logger.error("Could not remove expired network codes: %s", type(exc).__name__)


class _StopPurgeTimer(BaseException):
    """Stops a test loop. It is not an Exception, so the timer does not swallow it."""


def _spawn_purge_daemon(loop) -> None:
    threading.Thread(target=loop, name="ip-hash-purge", daemon=True).start()


def start_purge_timer(*, interval_s: float | None = None, spawn=None, sleep=None) -> bool:
    """Start one purge loop in this process. A second call does nothing.

    ``spawn`` receives the loop and starts it. The default is a daemon thread.
    ``sleep`` replaces time.sleep so tests can run one tick without waiting.
    """
    global _purge_timer_started
    with _purge_timer_guard:
        if _purge_timer_started:
            return False
        _purge_timer_started = True
    interval = _PURGE_INTERVAL_S if interval_s is None else interval_s
    sleeper = time.sleep if sleep is None else sleep

    def loop() -> None:
        while True:
            sleeper(interval)
            try:
                maybe_purge_ip_hashes()
            except Exception as exc:
                logger.error("Network-code purge timer failed: %s", type(exc).__name__)

    (spawn or _spawn_purge_daemon)(loop)
    return True


def _trial_state(started: datetime | None, now: datetime) -> tuple[str, datetime | None]:
    if started is None:
        return "none", None
    ends = started + timedelta(days=TRIAL_DAYS)
    return ("active" if now < ends else "ended"), ends


def allowance(email: str, question: str, is_saved=None, ip_hash: str | None = None) -> dict:
    """Decide whether this email may ask this question now, and why.

    The 3 free lessons are shared by everyone on the same network inside
    free_window_hours() (`ip_hash` is a one-way code, never the address). A repeat
    of a question this email already asked, a repeat this network asked inside
    the window, and a lesson saved from an earlier ask, stay free.

    `is_saved` is an optional callable that says whether a saved lesson already
    exists for the question; it's only consulted when the daily limit is reached.
    """
    hours = free_window_hours()
    qhash = db.question_hash(question)
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(seconds=WINDOW_S)
    ip_cutoff = _window_cutoff()
    if db._pool is None:
        with _memory_lock:
            asked = _memory_asked.get(email, {})
            by_ip_all = _memory_ip.get(ip_hash, {}) if ip_hash else {}
            by_ip = {qhash_key: pair for qhash_key, pair in by_ip_all.items() if pair[0] >= ip_cutoff}
            status = _memory_status.get(email, "none")
            started_ts = _memory_trial.get(email)
            choice = _memory_waitlist.get(email)
            bonus = _memory_bonus.get(email, 0)
            requested = email in _memory_requests
            used = len(by_ip)
            repeat = qhash in asked or qhash in by_ip
            # Only lessons made during the free month count toward its 24-hour limit.
            since = max(window_start.timestamp(), started_ts or 0)
            recent = [ts for ts, generated in asked.values() if generated and ts >= since]
            new_today = len(recent)
            oldest = datetime.fromtimestamp(min(recent), timezone.utc) if recent else None
        started = datetime.fromtimestamp(started_ts, timezone.utc) if started_ts else None
    else:
        try:
            with db._pool.connection(timeout=db._POOL_TIMEOUT_S) as conn, conn.cursor() as cur:
                cur.execute("SELECT subscription_status, trial_started_at, waitlist_choice, bonus_lessons, "
                            "more_requested_at IS NOT NULL FROM askers WHERE email = %s", (email,))
                row = cur.fetchone()
                status, started, choice, bonus, requested = row if row else ("none", None, None, 0, False)
                since = max(window_start, started) if started else window_start
                cur.execute(
                    "SELECT COUNT(*), BOOL_OR(question_hash = %s), "
                    "COUNT(*) FILTER (WHERE generated AND created_at >= %s), "
                    "MIN(created_at) FILTER (WHERE generated AND created_at >= %s) "
                    "FROM asker_questions WHERE email = %s",
                    (qhash, since, since, email),
                )
                _email_used, email_repeat, new_today, oldest = cur.fetchone()
                if ip_hash:
                    cur.execute(
                        "SELECT COUNT(DISTINCT question_hash), BOOL_OR(question_hash = %s) "
                        "FROM asker_questions WHERE ip_hash = %s "
                        "AND created_at >= NOW() - (%s * INTERVAL '1 hour')",
                        (qhash, ip_hash, hours),
                    )
                    used, ip_repeat = cur.fetchone()
                else:
                    used, ip_repeat = 0, False
                repeat = bool(email_repeat or ip_repeat)
        except Exception as exc:
            logger.exception("Asker allowance check failed")
            raise AskerUnavailable(str(exc)) from exc
    used, new_today = int(used or 0), int(new_today or 0)
    free_limit = FREE_QUESTIONS + int(bonus or 0)
    subscribed = status in _ACTIVE
    trial, trial_ends = _trial_state(started, now)

    if subscribed:
        reason = "subscribed"
    elif repeat:
        reason = "repeat"
    elif used < free_limit:
        reason = "free"
    elif trial == "active":
        if new_today < TRIAL_DAILY_NEW:
            reason = "trial"
        elif is_saved is not None and is_saved():
            reason = "saved_copy"
        else:
            reason = "daily_limit"
    elif is_saved is not None and is_saved():
        reason = "saved_copy"
    elif trial == "ended" and not WAITLIST_INSTEAD_OF_TRIAL:
        reason = "subscription_required"
    else:
        # The free-month offer is switched off for now; families answer the waitlist question instead.
        reason = "waitlist"
    return {
        "used": used,
        "free_left": max(free_limit - used, 0),
        "free_limit": free_limit,
        "subscribed": subscribed,
        "repeat": repeat,
        "trial": trial,
        "trial_ends": trial_ends.isoformat() if trial_ends else None,
        "new_today": new_today,
        "new_left_today": max(TRIAL_DAILY_NEW - new_today, 0) if trial == "active" else None,
        # Hours until the oldest of the last 24 hours' new lessons frees up a slot.
        "next_slot_hours": (max(1, round(((oldest + timedelta(seconds=WINDOW_S)) - now).total_seconds() / 3600))
                            if oldest is not None and new_today >= TRIAL_DAILY_NEW else None),
        "reason": reason,
        "waitlist_choice": choice,
        "more_requested": bool(requested),
        "allowed": reason in ("subscribed", "repeat", "free", "trial", "saved_copy"),
    }


def record_question(email: str, question: str, generated: bool = True, ip_hash: str | None = None) -> None:
    """Count a question that was answered. `generated` is False when a saved copy was reused.

    `ip_hash` is the one-way code for the client IP. Pass None for a subscriber so
    their lessons do not use up that network's free ones.
    """
    qhash = db.question_hash(question)
    if db._pool is None:
        with _memory_lock:
            email_asked = _memory_asked.setdefault(email, {})
            if qhash in email_asked:
                return  # A replay keeps its original network slot.
            email_asked[qhash] = (time.time(), generated)
            if ip_hash:
                _memory_ip.setdefault(ip_hash, {}).setdefault(qhash, (time.time(), generated))
        return
    try:
        with db._pool.connection(timeout=db._POOL_TIMEOUT_S) as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO askers (email, consented_at) VALUES (%s, NOW()) "
                "ON CONFLICT (email) DO UPDATE SET last_question_at = NOW(), "
                "consented_at = COALESCE(askers.consented_at, NOW())",
                (email,),
            )
            cur.execute(
                "INSERT INTO asker_questions (email, question_hash, question, generated, ip_hash) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (email, question_hash) DO NOTHING",
                (email, qhash, question[:500], generated, ip_hash),
            )
    except Exception:
        logger.exception("Failed to record asker question")


def start_trial(email: str) -> tuple[bool, str | None]:
    """Start this email's one free month. Returns (started_now, ends_at_iso)."""
    if db._pool is None:
        with _memory_lock:
            if email in _memory_trial:
                started = datetime.fromtimestamp(_memory_trial[email], timezone.utc)
                return False, (started + timedelta(days=TRIAL_DAYS)).isoformat()
            _memory_trial[email] = time.time()
            started = datetime.fromtimestamp(_memory_trial[email], timezone.utc)
        return True, (started + timedelta(days=TRIAL_DAYS)).isoformat()
    with db._pool.connection(timeout=db._POOL_TIMEOUT_S) as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO askers (email, consented_at) VALUES (%s, NOW()) ON CONFLICT (email) DO NOTHING", (email,))
        cur.execute(
            "UPDATE askers SET trial_started_at = NOW() WHERE email = %s AND trial_started_at IS NULL "
            "RETURNING trial_started_at",
            (email,),
        )
        row = cur.fetchone()
        fresh = row is not None
        if not fresh:
            cur.execute("SELECT trial_started_at FROM askers WHERE email = %s", (email,))
            row = cur.fetchone()
    return fresh, (row[0] + timedelta(days=TRIAL_DAYS)).isoformat()


def set_waitlist(email: str, choice: str) -> None:
    """Record the grown-up's answer to the subscription waitlist question."""
    if choice not in WAITLIST_CHOICES:
        raise ValueError("unknown waitlist choice")
    if db._pool is None:
        with _memory_lock:
            _memory_waitlist[email] = choice
        return
    with db._pool.connection(timeout=db._POOL_TIMEOUT_S) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO askers (email, consented_at, waitlist_choice, waitlist_at) VALUES (%s, NOW(), %s, NOW()) "
            "ON CONFLICT (email) DO UPDATE SET waitlist_choice = EXCLUDED.waitlist_choice, waitlist_at = NOW()",
            (email, choice),
        )


def request_more(email: str) -> bool:
    """A parent who used their free lessons asks for more. Returns False if the email is unknown."""
    if db._pool is None:
        with _memory_lock:
            if email not in _memory_asked:
                return False
            _memory_requests.setdefault(email, time.time())
            return True
    with db._pool.connection(timeout=db._POOL_TIMEOUT_S) as conn, conn.cursor() as cur:
        cur.execute("UPDATE askers SET more_requested_at = COALESCE(more_requested_at, NOW()) WHERE email = %s", (email,))
        return cur.rowcount > 0


def grant_bonus(email: str, lessons: int = 2) -> bool:
    """Give an email more free lessons and clear its waitlist answer so the question is asked again."""
    if db._pool is None:
        with _memory_lock:
            if email not in _memory_asked and email not in _memory_waitlist:
                return False
            _memory_bonus[email] = _memory_bonus.get(email, 0) + lessons
            _memory_waitlist.pop(email, None)
            _memory_requests.pop(email, None)
            return True
    with db._pool.connection(timeout=db._POOL_TIMEOUT_S) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE askers SET bonus_lessons = bonus_lessons + %s, waitlist_choice = NULL, waitlist_at = NULL, "
            "more_requested_at = NULL WHERE email = %s",
            (lessons, email),
        )
        return cur.rowcount > 0


def set_subscription(email: str, status: str, customer_id: str | None = None,
                     subscription_id: str | None = None) -> None:
    """Record what Stripe says about this email's subscription."""
    if db._pool is None:
        with _memory_lock:
            _memory_status[email] = status
        return
    with db._pool.connection(timeout=db._POOL_TIMEOUT_S) as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO askers (email, subscription_status, stripe_customer_id, stripe_subscription_id,
                                consented_at, subscription_updated_at)
            VALUES (%s, %s, %s, %s, NOW(), NOW())
            ON CONFLICT (email) DO UPDATE SET
              subscription_status = EXCLUDED.subscription_status,
              stripe_customer_id = COALESCE(EXCLUDED.stripe_customer_id, askers.stripe_customer_id),
              stripe_subscription_id = COALESCE(EXCLUDED.stripe_subscription_id, askers.stripe_subscription_id),
              subscription_updated_at = NOW()
            """,
            (email, status, customer_id, subscription_id),
        )


def delete_asker(email: str) -> bool:
    """Remove an email and its questions. Returns True when something was deleted."""
    if db._pool is None:
        with _memory_lock:
            existed = email in _memory_asked or email in _memory_status or email in _memory_trial
            hashes = set(_memory_asked.get(email, {}))
            still_used = set()
            for other, asked in _memory_asked.items():
                if other != email:
                    still_used.update(asked)
            for asked in _memory_ip.values():
                for qhash in hashes - still_used:
                    asked.pop(qhash, None)
            _memory_waitlist.pop(email, None)
            _memory_bonus.pop(email, None)
            _memory_requests.pop(email, None)
            _memory_asked.pop(email, None)
            _memory_status.pop(email, None)
            _memory_trial.pop(email, None)
            return existed
    try:
        with db._pool.connection(timeout=db._POOL_TIMEOUT_S) as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM askers WHERE email = %s", (email,))
            return cur.rowcount > 0
    except Exception:
        logger.exception("Failed to delete asker")
        return False


def reset_memory() -> None:
    """Test helper: forget in-memory askers."""
    global _last_purge_at
    with _memory_lock:
        _memory_asked.clear()
        _memory_ip.clear()
        _memory_status.clear()
        _memory_trial.clear()
        _memory_waitlist.clear()
        _memory_bonus.clear()
        _memory_requests.clear()
    with _purge_interval_lock:
        _last_purge_at = 0.0
