"""Who is asking: an email per asker, a few free questions, then a subscription.

Asking the same question again never uses up another free question. Without a
database (local previews and tests) the counts live in memory.
"""
import logging
import re
import threading

from lesson_platform import db

logger = logging.getLogger(__name__)

FREE_QUESTIONS = db.ASKER_FREE_QUESTIONS
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_ACTIVE = ("active", "trialing")

_memory_lock = threading.Lock()
_memory_asked: dict[str, set[str]] = {}
_memory_status: dict[str, str] = {}


class AskerUnavailable(RuntimeError):
    """The allowance could not be checked, so the question must not be generated."""


def normalize_email(raw: str | None) -> str | None:
    email = (raw or "").strip().lower()
    if len(email) > 254 or not _EMAIL.match(email):
        return None
    return email


def allowance(email: str, question: str) -> dict:
    """Return used, free_left, subscribed and allowed for this email and question."""
    qhash = db.question_hash(question)
    if db._pool is None:
        with _memory_lock:
            asked = _memory_asked.get(email, set())
            status = _memory_status.get(email, "none")
            used, repeat = len(asked), qhash in asked
    else:
        try:
            with db._pool.connection(timeout=db._POOL_TIMEOUT_S) as conn, conn.cursor() as cur:
                cur.execute("SELECT subscription_status FROM askers WHERE email = %s", (email,))
                row = cur.fetchone()
                status = row[0] if row else "none"
                cur.execute(
                    "SELECT COUNT(*), COUNT(*) FILTER (WHERE question_hash = %s) FROM asker_questions WHERE email = %s",
                    (qhash, email),
                )
                used, repeat = cur.fetchone()
                repeat = bool(repeat)
        except Exception as exc:
            logger.exception("Asker allowance check failed")
            raise AskerUnavailable(str(exc)) from exc
    subscribed = status in _ACTIVE
    return {
        "used": int(used),
        "free_left": max(FREE_QUESTIONS - int(used), 0),
        "subscribed": subscribed,
        "repeat": repeat,
        "allowed": subscribed or repeat or used < FREE_QUESTIONS,
    }


def record_question(email: str, question: str) -> None:
    """Count a question that was answered. Repeats of the same question are free."""
    qhash = db.question_hash(question)
    if db._pool is None:
        with _memory_lock:
            _memory_asked.setdefault(email, set()).add(qhash)
        return
    try:
        with db._pool.connection(timeout=db._POOL_TIMEOUT_S) as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO askers (email) VALUES (%s) ON CONFLICT (email) DO UPDATE SET last_question_at = NOW()",
                (email,),
            )
            cur.execute(
                "INSERT INTO asker_questions (email, question_hash, question) VALUES (%s, %s, %s) "
                "ON CONFLICT (email, question_hash) DO NOTHING",
                (email, qhash, question[:500]),
            )
    except Exception:
        logger.exception("Failed to record asker question")


def reset_memory() -> None:
    """Test helper: forget in-memory askers."""
    with _memory_lock:
        _memory_asked.clear()
        _memory_status.clear()
