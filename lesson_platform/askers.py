"""Who is asking: an email per asker, a few free questions, a free month, then a subscription.

* The first FREE_QUESTIONS different questions from one client IP are free, whichever email asks.
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
# A reservation whose worker died should stop counting. Ten minutes is longer
# than lesson generation, and short enough that a stuck row does not hold a slot.
RESERVATION_TTL_S = 10 * 60
# Bound how long a claim may wait on the network lock or sit on a pool connection.
_LOCK_TIMEOUT = "2s"
_STATEMENT_TIMEOUT = "5s"
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


def _trial_state(started: datetime | None, now: datetime) -> tuple[str, datetime | None]:
    if started is None:
        return "none", None
    ends = started + timedelta(days=TRIAL_DAYS)
    return ("active" if now < ends else "ended"), ends


def _decide(status, started, choice, bonus, requested, used, repeat, new_today, oldest, now, saved: bool) -> dict:
    """Turn one consistent snapshot into the same allowance dict the routes already use."""
    used, new_today = int(used or 0), int(new_today or 0)
    free_limit = FREE_QUESTIONS + int(bonus or 0)
    subscribed = status in _ACTIVE
    trial, trial_ends = _trial_state(started, now)
    repeat = bool(repeat)

    if subscribed:
        reason = "subscribed"
    elif repeat:
        reason = "repeat"
    elif used < free_limit:
        reason = "free"
    elif trial == "active":
        if new_today < TRIAL_DAILY_NEW:
            reason = "trial"
        elif saved:
            reason = "saved_copy"
        else:
            reason = "daily_limit"
    elif saved:
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


def _slot_is_live(slot: tuple, now_ts: float) -> bool:
    """A reservation (generated is None) older than the TTL no longer counts."""
    generated = slot[1] if len(slot) > 1 else True
    if generated is None and now_ts - slot[0] >= RESERVATION_TTL_S:
        return False
    return True


def _sweep_stale_memory(now_ts: float) -> None:
    """Caller must hold _memory_lock. Drop reservations whose worker did not finish."""
    for bucket in list(_memory_asked.values()) + list(_memory_ip.values()):
        for qhash, slot in list(bucket.items()):
            if not _slot_is_live(slot, now_ts):
                bucket.pop(qhash, None)


def _sweep_stale_db(cur) -> None:
    cur.execute(
        "DELETE FROM asker_questions WHERE settled = false "
        "AND created_at < NOW() - INTERVAL '10 minutes'"
    )


def _memory_snapshot(email: str, qhash: str, now: datetime, ip_hash: str | None):
    """Caller must hold _memory_lock. `used` is this network's count, not the email's."""
    _sweep_stale_memory(now.timestamp())
    asked = _memory_asked.get(email, {})
    by_ip = _memory_ip.get(ip_hash, {}) if ip_hash else {}
    status = _memory_status.get(email, "none")
    started_ts = _memory_trial.get(email)
    choice = _memory_waitlist.get(email)
    bonus = _memory_bonus.get(email, 0)
    requested = email in _memory_requests
    used = len(by_ip)
    repeat = qhash in asked or qhash in by_ip
    window_start = (now - timedelta(seconds=WINDOW_S)).timestamp()
    # Only lessons made during the free month count toward its 24-hour limit.
    since = max(window_start, started_ts or 0)
    recent = [ts for ts, generated in asked.values() if generated and ts >= since]
    new_today = len(recent)
    oldest = datetime.fromtimestamp(min(recent), timezone.utc) if recent else None
    started = datetime.fromtimestamp(started_ts, timezone.utc) if started_ts else None
    return status, started, choice, bonus, requested, used, repeat, new_today, oldest


def _network_lock_key(ip_hash: str) -> int:
    """A signed bigint for pg_advisory_xact_lock, taken from the IP code."""
    return int(ip_hash[:15], 16)


def _db_snapshot(cur, email: str, qhash: str, now: datetime, ip_hash: str | None, lock: bool):
    """Read one allowance snapshot. When locking, the network lock is taken before the email row."""
    if lock and ip_hash:
        cur.execute("SELECT pg_advisory_xact_lock(%s)", (_network_lock_key(ip_hash),))
    if lock:
        cur.execute(
            "INSERT INTO askers (email, consented_at) VALUES (%s, NOW()) "
            "ON CONFLICT (email) DO NOTHING",
            (email,),
        )
        cur.execute(
            "SELECT subscription_status, trial_started_at, waitlist_choice, bonus_lessons, "
            "more_requested_at IS NOT NULL FROM askers WHERE email = %s FOR UPDATE",
            (email,),
        )
    else:
        cur.execute(
            "SELECT subscription_status, trial_started_at, waitlist_choice, bonus_lessons, "
            "more_requested_at IS NOT NULL FROM askers WHERE email = %s",
            (email,),
        )
    row = cur.fetchone()
    status, started, choice, bonus, requested = row if row else ("none", None, None, 0, False)
    window_start = now - timedelta(seconds=WINDOW_S)
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
            "AND (settled OR created_at >= NOW() - INTERVAL '10 minutes')",
            (qhash, ip_hash),
        )
        used, ip_repeat = cur.fetchone()
    else:
        used, ip_repeat = 0, False
    repeat = bool(email_repeat or ip_repeat)
    return status, started, choice, bonus, requested, int(used or 0), repeat, int(new_today or 0), oldest


def _stored_ip(decision: dict, ip_hash: str | None) -> str | None:
    """Subscribers do not consume the network's free lessons."""
    if decision.get("subscribed"):
        return None
    return ip_hash


def allowance(email: str, question: str, is_saved=None, ip_hash: str | None = None) -> dict:
    """Decide whether this email may ask this question now, and why.

    The 3 free lessons are shared by everyone on the same client IP (`ip_hash` is
    a one-way code, never the address). A repeat of a question this email or this
    IP already asked, and a lesson saved from an earlier ask, stay free.

    `is_saved` is an optional callable that says whether a saved lesson already
    exists for the question; it's only consulted when the daily limit is reached.
    This read does not reserve a slot. Asking goes through claim_question.
    """
    qhash = db.question_hash(question)
    now = datetime.now(timezone.utc)
    saved = bool(is_saved()) if is_saved is not None else False
    if db._pool is None:
        with _memory_lock:
            snapshot = _memory_snapshot(email, qhash, now, ip_hash)
    else:
        try:
            with db._pool.connection(timeout=db._POOL_TIMEOUT_S) as conn, conn.cursor() as cur:
                _sweep_stale_db(cur)
                snapshot = _db_snapshot(cur, email, qhash, now, ip_hash, lock=False)
        except Exception as exc:
            logger.exception("Asker allowance check failed")
            raise AskerUnavailable(str(exc)) from exc
    return _decide(*snapshot, now, saved)


def claim_question(email: str, question: str, is_saved=None, ip_hash: str | None = None) -> tuple[dict, bool]:
    """Decide and, when this question newly counts, record it before returning.

    The insert happens in the same locked step as the count, so two requests
    cannot both take the last free question. The lock covers the network, not
    only this email: everyone on that IP shares the three lessons. Returns
    (allowance, reserved). `reserved` is True only when this call inserted the
    question. The allowance numbers are from before that insert.
    """
    qhash = db.question_hash(question)
    now = datetime.now(timezone.utc)
    saved = bool(is_saved()) if is_saved is not None else False
    if db._pool is None:
        with _memory_lock:
            decision = _decide(*_memory_snapshot(email, qhash, now, ip_hash), now, saved)
            reserved = False
            if decision["allowed"] and not decision["repeat"]:
                # None means reserved and not finished. False would look like a saved copy.
                _memory_asked.setdefault(email, {})[qhash] = (time.time(), None)
                stored_ip = _stored_ip(decision, ip_hash)
                if stored_ip:
                    _memory_ip.setdefault(stored_ip, {})[qhash] = (time.time(), None)
                reserved = True
            return decision, reserved
    try:
        # The pool connection is autocommit, so the lock and the insert have to
        # share one transaction. Otherwise each statement commits and releases
        # pg_advisory_xact_lock before the next asker is counted.
        with db._pool.connection(timeout=db._POOL_TIMEOUT_S) as conn, conn.transaction(), conn.cursor() as cur:
            cur.execute(f"SET LOCAL lock_timeout = '{_LOCK_TIMEOUT}'")
            cur.execute(f"SET LOCAL statement_timeout = '{_STATEMENT_TIMEOUT}'")
            _sweep_stale_db(cur)
            decision = _decide(*_db_snapshot(cur, email, qhash, now, ip_hash, lock=True), now, saved)
            reserved = False
            if decision["allowed"] and not decision["repeat"]:
                cur.execute(
                    "UPDATE askers SET last_question_at = NOW(), "
                    "consented_at = COALESCE(consented_at, NOW()) WHERE email = %s",
                    (email,),
                )
                cur.execute(
                    "INSERT INTO asker_questions "
                    "(email, question_hash, question, generated, ip_hash, settled) "
                    "VALUES (%s, %s, %s, false, %s, false) "
                    "ON CONFLICT (email, question_hash) DO NOTHING RETURNING question_hash",
                    (email, qhash, question[:500], _stored_ip(decision, ip_hash)),
                )
                reserved = cur.fetchone() is not None
            return decision, reserved
    except Exception as exc:
        logger.exception("Asker allowance check failed")
        raise AskerUnavailable(str(exc)) from exc


def finish_question(email: str, question: str, ip_hash: str | None, generated: bool) -> None:
    """Record whether a reserved question was newly made or a saved copy."""
    qhash = db.question_hash(question)
    if db._pool is None:
        with _memory_lock:
            slot = _memory_asked.get(email, {}).get(qhash)
            if slot is not None:
                _memory_asked[email][qhash] = (slot[0], bool(generated))
            if ip_hash and qhash in _memory_ip.get(ip_hash, {}):
                _memory_ip[ip_hash][qhash] = (_memory_ip[ip_hash][qhash][0], bool(generated))
        return
    try:
        with db._pool.connection(timeout=db._POOL_TIMEOUT_S) as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE asker_questions SET generated = %s, settled = true "
                "WHERE email = %s AND question_hash = %s",
                (bool(generated), email, qhash),
            )
    except Exception:
        logger.exception("Failed to finish asker question")


def release_question(email: str, question: str, ip_hash: str | None = None) -> None:
    """Drop a reservation when the lesson was not actually made."""
    qhash = db.question_hash(question)
    if db._pool is None:
        with _memory_lock:
            asked = _memory_asked.get(email)
            if asked is not None:
                asked.pop(qhash, None)
            still_used = any(qhash in rows for rows in _memory_asked.values())
            if not still_used:
                for bucket in _memory_ip.values():
                    bucket.pop(qhash, None)
        return
    try:
        with db._pool.connection(timeout=db._POOL_TIMEOUT_S) as conn, conn.cursor() as cur:
            cur.execute(
                "DELETE FROM asker_questions WHERE email = %s AND question_hash = %s",
                (email, qhash),
            )
    except Exception:
        logger.exception("Failed to release asker question")


def record_question(email: str, question: str, generated: bool = True, ip_hash: str | None = None) -> None:
    """Count a question that was answered. `generated` is False when a saved copy was reused.

    `ip_hash` is the one-way code for the client IP. Pass None for a subscriber so
    their lessons do not use up that network's free ones. A question already reserved
    by claim_question is left as it is; finish_question sets whether it was newly made.
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
                "INSERT INTO asker_questions "
                "(email, question_hash, question, generated, ip_hash, settled) "
                "VALUES (%s, %s, %s, %s, %s, true) "
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
    with _memory_lock:
        _memory_asked.clear()
        _memory_ip.clear()
        _memory_status.clear()
        _memory_trial.clear()
        _memory_waitlist.clear()
        _memory_bonus.clear()
        _memory_requests.clear()
