"""Who is asking: an email per asker, a few free questions, a free month, then a subscription.

* The first FREE_QUESTIONS different questions are free.
* Then a parent can start one free month: up to TRIAL_DAILY_NEW new lessons in
  any 24 hours (a rolling window, so time zones don't matter) for TRIAL_DAYS days. Questions someone already asked (a saved copy) and
  the family's own repeats never count.
* After that, a paid subscription is needed.

Without a database (local previews and tests) everything lives in memory.
"""
import logging
import re
import threading
import time
from datetime import datetime, timedelta, timezone

from lesson_platform import db

logger = logging.getLogger(__name__)

FREE_QUESTIONS = db.ASKER_FREE_QUESTIONS
TRIAL_DAYS = 30
TRIAL_DAILY_NEW = 5
WINDOW_S = 24 * 3600
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_ACTIVE = ("active", "trialing")  # Same as billing.ACTIVE_STATES.

_memory_lock = threading.Lock()
_memory_asked: dict[str, dict[str, tuple[float, bool]]] = {}  # email -> {question hash: (time, newly made)}
_memory_status: dict[str, str] = {}
_memory_trial: dict[str, float] = {}


class AskerUnavailable(RuntimeError):
    """The allowance could not be checked, so the question must not be generated."""


def normalize_email(raw: str | None) -> str | None:
    email = (raw or "").strip().lower()
    if len(email) > 254 or not _EMAIL.match(email):
        return None
    return email


def _trial_state(started: datetime | None, now: datetime) -> tuple[str, datetime | None]:
    if started is None:
        return "none", None
    ends = started + timedelta(days=TRIAL_DAYS)
    return ("active" if now < ends else "ended"), ends


def allowance(email: str, question: str, is_saved=None) -> dict:
    """Decide whether this email may ask this question now, and why.

    `is_saved` is an optional callable that says whether a saved lesson already
    exists for the question; it's only consulted when the daily limit is reached.
    """
    qhash = db.question_hash(question)
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(seconds=WINDOW_S)
    if db._pool is None:
        with _memory_lock:
            asked = _memory_asked.get(email, {})
            status = _memory_status.get(email, "none")
            started_ts = _memory_trial.get(email)
            used, repeat = len(asked), qhash in asked
            # Only lessons made during the free month count toward its 24-hour limit.
            since = max(window_start.timestamp(), started_ts or 0)
            recent = [ts for ts, generated in asked.values() if generated and ts >= since]
            new_today = len(recent)
            oldest = datetime.fromtimestamp(min(recent), timezone.utc) if recent else None
        started = datetime.fromtimestamp(started_ts, timezone.utc) if started_ts else None
    else:
        try:
            with db._pool.connection(timeout=db._POOL_TIMEOUT_S) as conn, conn.cursor() as cur:
                cur.execute("SELECT subscription_status, trial_started_at FROM askers WHERE email = %s", (email,))
                row = cur.fetchone()
                status, started = (row[0], row[1]) if row else ("none", None)
                since = max(window_start, started) if started else window_start
                cur.execute(
                    "SELECT COUNT(*), COUNT(*) FILTER (WHERE question_hash = %s), "
                    "COUNT(*) FILTER (WHERE generated AND created_at >= %s), "
                    "MIN(created_at) FILTER (WHERE generated AND created_at >= %s) "
                    "FROM asker_questions WHERE email = %s",
                    (qhash, since, since, email),
                )
                used, repeat, new_today, oldest = cur.fetchone()
                repeat = bool(repeat)
        except Exception as exc:
            logger.exception("Asker allowance check failed")
            raise AskerUnavailable(str(exc)) from exc
    used, new_today = int(used), int(new_today)
    subscribed = status in _ACTIVE
    trial, trial_ends = _trial_state(started, now)

    if subscribed:
        reason = "subscribed"
    elif repeat:
        reason = "repeat"
    elif used < FREE_QUESTIONS:
        reason = "free"
    elif trial == "active":
        if new_today < TRIAL_DAILY_NEW:
            reason = "trial"
        elif is_saved is not None and is_saved():
            reason = "saved_copy"
        else:
            reason = "daily_limit"
    elif trial == "none":
        reason = "trial_available"
    else:
        reason = "subscription_required"
    return {
        "used": used,
        "free_left": max(FREE_QUESTIONS - used, 0),
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
        "allowed": reason in ("subscribed", "repeat", "free", "trial", "saved_copy"),
    }


def record_question(email: str, question: str, generated: bool = True) -> None:
    """Count a question that was answered. `generated` is False when a saved copy was reused."""
    qhash = db.question_hash(question)
    if db._pool is None:
        with _memory_lock:
            _memory_asked.setdefault(email, {}).setdefault(qhash, (time.time(), generated))
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
                "INSERT INTO asker_questions (email, question_hash, question, generated) VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (email, question_hash) DO NOTHING",
                (email, qhash, question[:500], generated),
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
        _memory_status.clear()
        _memory_trial.clear()
